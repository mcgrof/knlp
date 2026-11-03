#!/usr/bin/env python3
"""Check that a packed training sequence computes what its parts compute alone.

cas_packing_check.py shows that packing keeps every teacher target at
the right token. This script checks the forward and backward pass:
that the Cartridges library's packed sequence, with its per-conversation
mask and restarted positions, gives each conversation the same
log-probabilities, the same loss and the same cartridge gradient as
running that conversation alone with the same cartridge, and that no
conversation can see another.

For each tested pack (the one holding the most conversations, and the
last one, which ends in padding) it reports:

- the run-to-run floor: the packed forward twice, compared;
- each conversation alone (padded to the same length, so the compiled
  kernels see one shape) against its span of the pack: largest
  difference in the per-position log-normalizer and in the top-k
  target log-probabilities the loss reads;
- the placement floor: each conversation alone at offset 0 against
  the same conversation alone at its pack offset behind filler it
  cannot attend to. This is the same computation in different kernel
  tiles, so it measures bf16 rounding with no packing involved; a
  packed-versus-alone gap of the same size is rounding, not a mask or
  position error. The shifted conversations are also compared with
  the pack directly;
- the loss: the pack's mean over all its top-k entries against the sum
  of each conversation's entry sums divided by the pack's entry count;
- the cartridge gradient of that pack loss against the sum of the
  isolated gradients, as relative L2 error and cosine, with a check
  that it is finite and nonzero and that the frozen sink gets none;
- leakage: the first conversation's tokens are shuffled inside the
  pack and every other conversation's outputs are compared with the
  unshuffled pack (the shuffled one must change);
- that the reader's weights are bit-identical after all backward passes.

Env: DATA_PARQUET, RECORD (text for the first-tokens cartridge), MODEL
(Qwen/Qwen3-0.6b), PACKED_LEN (8192), SEED (42), LIMIT (rows, default
64), KV_TOKENS (128), DTYPE (bf16 as in training, or fp32: the reader
and cartridge in float32 with no autocast, which separates a masking
or position error from bf16 rounding), OUT_JSON (optional),
CARTRIDGES_DIR.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.environ.get("CARTRIDGES_DIR", "cartridges"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.initialization import KVFromText
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import CacheAndModel

DATA = os.environ["DATA_PARQUET"]
RECORD = os.environ["RECORD"]
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6b")
S = int(os.environ.get("PACKED_LEN", "8192"))
SEED = int(os.environ.get("SEED", "42"))
LIMIT = int(os.environ.get("LIMIT", "64"))
KV_TOKENS = int(os.environ.get("KV_TOKENS", "128"))
DTYPE = os.environ.get("DTYPE", "bf16")
assert DTYPE in ("bf16", "fp32"), "DTYPE: bf16 or fp32"
DEV = "cuda"


def run(wrapped, ids, seq, pos, tidx, tids, tlps, grad):
    """Library forward and loss terms: per-position log-normalizer,
    top-k log-probs at the target entries, and per-entry cross-entropy."""
    with torch.set_grad_enabled(grad):
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=DTYPE == "bf16"
        ):
            out = wrapped(
                input_ids=ids.to(DEV), seq_ids=seq.to(DEV), position_ids=pos.to(DEV)
            )
            logits = out.logits
            lp = F.log_softmax(logits, dim=-1)[0, tidx.to(DEV) - 1, tids.to(DEV)]
            ce = -tlps.to(DEV).exp() * lp
        lse = torch.logsumexp(logits[0].float(), dim=-1).detach()
    return lse, lp.detach().float(), ce


def grads(cache):
    return torch.cat(
        [
            p.grad.float().flatten()
            for p in list(cache.trainable_keys) + list(cache.trainable_values)
        ]
    )


def weights_digest(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def isolated(batch, s, t, at=0):
    """One conversation alone, padded like a one-conversation pack.

    With at > 0 it starts at that offset behind a filler segment of
    its own sequence id, which it cannot attend to: the same
    computation at a different place in the kernel's tiles."""
    n = t - s
    ids = torch.zeros(S, dtype=torch.long)
    ids[at : at + n] = batch.input_ids[s:t]
    pos = torch.zeros(S, dtype=torch.long)
    pos[at : at + n] = batch.position_ids[s:t]
    seq = torch.zeros(S, dtype=torch.long)
    seq[:at] = 1
    m = (batch.topk_token_idxs >= s) & (batch.topk_token_idxs < t)
    return ids, seq, pos, m


def check_pack(wrapped, cache, ds, b):
    elems = [ds.elements[i] for i in ds.batches[b]]
    batch = ds.collate(elems)
    spans, off = [], 0
    for e in elems:
        spans.append((off, off + len(e.input_ids)))
        off += len(e.input_ids)
    tidx, tids, tlps = batch.topk_token_idxs, batch.topk_token_ids, batch.topk_logprobs
    args = (batch.input_ids, batch.element_ids, batch.position_ids, tidx, tids, tlps)

    cache.zero_grad(set_to_none=True)
    lse_p, lp_p, ce_p = run(wrapped, *args, grad=True)
    loss_p = ce_p.mean()
    loss_p.backward()
    g_pack = grads(cache)
    sink_grad = [p.grad for p in list(cache.frozen_keys) + list(cache.frozen_values)]
    lse_p2, lp_p2, _ = run(wrapped, *args, grad=False)
    floor = dict(
        lse=float((lse_p2 - lse_p).abs().max()),
        topk_logprob=float((lp_p2 - lp_p).abs().max()),
    )

    n_total = len(tidx)

    def alone(at_pack_offset):
        cache.zero_grad(set_to_none=True)
        loss_sum, outs = 0.0, []
        for s, t in spans:
            at = s if at_pack_offset else 0
            ids, seq, pos, m = isolated(batch, s, t, at)
            lse_i, lp_i, ce_i = run(
                wrapped, ids, seq, pos, tidx[m] - s + at, tids[m], tlps[m], grad=True
            )
            (ce_i.sum() / n_total).backward()
            loss_sum += float(ce_i.detach().sum())
            outs.append((lse_i[at : at + t - s], lp_i, m))
        return loss_sum, outs, grads(cache)

    loss_sum, iso, g_iso = alone(False)
    _, shifted, g_shift = alone(True)
    per = []
    for (s, t), (lse_i, lp_i, m), (lse_s, lp_s, _) in zip(spans, iso, shifted):
        mm = m.to(lp_p.device)
        per.append(
            dict(
                tokens=t - s,
                targets=int(m.sum()),
                lse=float((lse_i - lse_p[s:t]).abs().max()),
                topk_logprob=float((lp_i - lp_p[mm]).abs().max()),
                placement_lse=float((lse_i - lse_s).abs().max()),
                placement_topk_logprob=float((lp_i - lp_s).abs().max()),
            )
        )

    shuf = batch.input_ids.clone()
    s0, t0 = spans[0]
    g = torch.Generator().manual_seed(SEED)
    shuf[s0 + 1 : t0] = shuf[s0 + 1 : t0][torch.randperm(t0 - s0 - 1, generator=g)]
    lse_x, lp_x, _ = run(wrapped, shuf, *args[1:], grad=False)
    m0 = (tidx >= s0) & (tidx < t0)
    others = slice(t0, spans[-1][1])
    leak = dict(
        other_lse=float((lse_x[others] - lse_p[others]).abs().max()),
        other_topk_logprob=float((lp_x[~m0.to(DEV)] - lp_p[~m0.to(DEV)]).abs().max()),
        shuffled_own_lse=float((lse_x[s0:t0] - lse_p[s0:t0]).abs().max()),
    )
    return dict(
        pack=b,
        conversations=len(elems),
        real_tokens=off,
        padding=S - off,
        targets=n_total,
        floor=floor,
        per_conversation=per,
        loss_pack=float(loss_p),
        loss_isolated=loss_sum / n_total,
        loss_rel_diff=abs(loss_sum / n_total - float(loss_p)) / float(loss_p),
        grad_rel_l2=float((g_iso - g_pack).norm() / g_pack.norm()),
        grad_cosine=float(F.cosine_similarity(g_iso, g_pack, dim=0)),
        placement_grad_rel_l2=float((g_iso - g_shift).norm() / g_iso.norm()),
        placement_grad_cosine=float(F.cosine_similarity(g_iso, g_shift, dim=0)),
        shifted_vs_pack_grad_rel_l2=float((g_shift - g_pack).norm() / g_pack.norm()),
        grad_norm=float(g_pack.norm()),
        grad_finite=bool(torch.isfinite(g_pack).all()),
        sink_grad_none=all(x is None for x in sink_grad),
        leakage=leak,
    )


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    dt = torch.bfloat16 if DTYPE == "bf16" else torch.float32
    # the cartridge is built in bf16 as in training (the library's text
    # initializer always runs under bf16 autocast); fp32 mode casts after
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEV).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )
    cache = (
        KVFromText.Config(max_tokens=KV_TOKENS, text_source=RECORD, num_frozen_tokens=1)
        .instantiate()
        .initialize_kv_cache(tokenizer=tok, model=model, attn_config=attn)
        .to(DEV)
        .to(dt)
    )
    model = model.to(dt)
    before = weights_digest(model)
    wrapped = CacheAndModel(cache, model)
    ds = TrainDataset.Config(
        data_sources=[DataSource(path=DATA, type="local", limit=LIMIT)],
        top_k_logits=20,
        packed_seq_length=S,
        packing_mode="pad",
    ).instantiate(tokenizer=tok, seed=SEED)
    most = max(range(len(ds.batches)), key=lambda b: len(ds.batches[b]))
    packs = sorted({most, len(ds.batches) - 1})
    results = [check_pack(wrapped, cache, ds, b) for b in packs]
    rec = dict(
        model=MODEL,
        dtype=DTYPE,
        data=os.path.abspath(DATA),
        packed_len=S,
        rows=LIMIT,
        packs_total=len(ds.batches),
        param_dtype=str(cache.trainable_keys[0].dtype),
        reader_unchanged=weights_digest(model) == before,
        packs=results,
    )
    if os.environ.get("OUT_JSON"):
        with open(os.environ["OUT_JSON"], "w") as f:
            json.dump(rec, f, indent=1)
    print("PACKSEM " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
