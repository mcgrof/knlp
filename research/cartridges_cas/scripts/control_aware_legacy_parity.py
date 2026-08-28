#!/usr/bin/env python3
"""Prove (or disprove) that the screen's legacy arm is the historical
objective.

The screen's parity gate shows legacy_raw == unique + anchor.  That is
an internal identity; it says nothing about whether legacy_raw is the
loss the earlier continuation trainer actually optimized.  If the
reproduction control fails to reproduce, this is the first thing to
check, because a difference in aggregation or masking would look
exactly like a mechanism.

Computes, on the same elements and the same cartridge:

    historical  opcart_train.stored_topk_loss: mean of p*nll over the
                valid stored entries
    screen      control_aware arm_loss(legacy_raw): sum of p*nll over
                valid entries divided by the valid legacy entry count

and compares values and cartridge gradients.  Any disagreement is a
provenance defect in the screen, not a finding about cartridges.

Env: MODEL, PATIENT, DATA_PARQUET, CART_INIT, N (elements, default 4).
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/legacy_parity")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    sys.path.insert(0, str(_d))
from control_aware.targets import build_target_set, parse_element  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PATIENT = os.environ.get("PATIENT", "patient_02")
DATA_PARQUET = os.environ["DATA_PARQUET"]
CART_INIT = os.environ["CART_INIT"]
N = int(os.environ.get("N", "4"))
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/legacy_parity/legacy_parity.json")
DEVICE = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
EOT = tok.convert_tokens_to_ids("<|im_end|>")
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)

ck = torch.load(CART_INIT, map_location="cpu", weights_only=False)


def t(p):
    return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)


tk = [t(p) for p in ck["trainable_keys"]]
tv = [t(p) for p in ck["trainable_values"]]
fk = ck.get("frozen_keys") or []
fv = ck.get("frozen_values") or []
if fk:
    ik = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
    iv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
    nfrozen = t(fk[0]).shape[2]
else:
    ik, iv, nfrozen = tk, tv, 0
cache = TrainableCache(
    config=ac, init_keys=ik, init_values=iv, num_frozen_tokens=nfrozen
).to(DEVICE)

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=0)
VOCAB = int(model.config.vocab_size)


def forward(el):
    ids = el.input_ids.to(DEVICE)
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=ids,
            seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
            position_ids=torch.arange(ids.shape[0], device=DEVICE),
            use_cache=True,
            past_key_values=cache,
        )
    return out.logits, ids.shape[0]


def historical_loss(el):
    """Verbatim shape of opcart_train.stored_topk_loss."""
    ids = el.input_ids.to(DEVICE)
    idxs = el.topk_token_idxs.to(DEVICE)
    tids = el.topk_token_ids.to(DEVICE)
    tlp = el.topk_logprobs.to(DEVICE)
    logits, seq_len = forward(el)
    vocab = logits.shape[-1]
    gi = idxs - 1
    valid = (gi >= 0) & (gi < ids.shape[0]) & (tids >= 0) & (tids < vocab)
    if int(valid.sum()) == 0:
        return None
    lp = F.log_softmax(logits.float(), dim=-1)[0, gi[valid], tids[valid]]
    return (-tlp[valid].exp() * lp).mean()


def screen_loss(el, et):
    logits, seq_len = forward(el)
    vocab = logits.shape[-1]
    ts = build_target_set(et, "legacy_raw")
    gi, tids, probs = ts.tensors(device=DEVICE)
    gi = gi - 1
    valid = (gi >= 0) & (gi < seq_len) & (tids >= 0) & (tids < vocab)
    denom = int(valid.sum())
    if denom == 0:
        return None
    lp = F.log_softmax(logits.float(), dim=-1)[0, gi[valid], tids[valid]]
    return -(probs[valid] * lp).sum() / denom


def grads(loss):
    for p in cache.parameters():
        p.grad = None
    loss.backward()
    return torch.cat(
        [
            p.grad.detach().float().flatten()
            for p in cache.parameters()
            if p.grad is not None
        ]
    )


def main():
    rows = []
    for i in range(N):
        el = dataset.elements[i]
        et = parse_element(el.topk_token_idxs, el.topk_token_ids, el.topk_logprobs, EOT)
        h = historical_loss(el)
        gh = grads(h)
        s = screen_loss(el, et)
        gs = grads(s)
        rel = abs(float(h) - float(s)) / max(abs(float(h)), 1e-12)
        gden = gh.abs().max().clamp_min(1e-12)
        grel = float((gh - gs).abs().max() / gden)
        rows.append(
            dict(
                element=i,
                historical=float(h),
                screen=float(s),
                loss_rel=rel,
                grad_max_rel=grel,
                n_entries=int(el.topk_token_ids.shape[0]),
            )
        )
        print(
            f"[legacy-parity] element {i}: historical={float(h):.8f} "
            f"screen={float(s):.8f} rel={rel:.2e} grad_rel={grel:.2e}",
            flush=True,
        )
    ok = all(r["loss_rel"] < 1e-5 and r["grad_max_rel"] < 1e-2 for r in rows)
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(dict(ok=ok, results=rows), indent=1))
    print(f"LEGACY_PARITY_{'OK' if ok else 'MISMATCH'} {OUT_JSON}")


if __name__ == "__main__":
    main()
