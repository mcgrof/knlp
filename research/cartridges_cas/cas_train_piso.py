#!/usr/bin/env python3
"""Faithful per-sample P_iso joint cartridge trainer (the real CAS rescue).

The earlier joint trainer (cas_train_joint.py + cas_joint.KVFromCarts) is a
*fixed-distractor* approximation: it freezes the other patients' cartridges as
always-visible (seq_id=-1) distractors and trains one target against them. That
is not what Cartridges-at-Scale actually does. CAS co-trains *all* N cartridges
in one cache and, per training sample, hides each non-target cartridge with
probability P_iso (mixed-visibility). The target is always visible; a distractor
is revealed with prob 1-P_iso. Gradients flow into every revealed cartridge, so
each one learns both to answer its own document and to not corrupt the others
when co-loaded -- the mechanism that turns collapse-under-coload into a rescue.

The single-cartridge equality/global mask the library ships
(create_block_mask_w_cache: attend iff key is global -1 or key seq_id == query
seq_id) can only express two regimes (visible-to-all or visible-to-one), so it
cannot represent a per-sample random reveal. This trainer therefore monkeypatches
the mask builder with a reveal-vector mask: request tokens of a sample targeting
cartridge j attend to cartridge c iff reveal[c] is set (target always set; other
carts Bernoulli(1-P_iso)), and causally to their own request tokens.

All N cartridges live in ONE TrainableCache as trainable params (num_frozen=0),
laid out contiguously [cart_0 | cart_1 | ... ] with exactly KV_TOKENS tokens each,
so cartridge id = kv_idx // KV_TOKENS. Each cartridge is initialized from its
patient record (same truncation init as KVFromText). Loss is the library's exact
distillation cross-entropy against the teacher's sparse top-k logprobs. We drive
one sample per forward and accumulate gradients over a batch (expected gradient
identical to packing, far simpler mask), then step.

Env:
  PATIENTS      space-sep patient ids co-trained in one cache (>=2)
  RECORDS_DIR   dir of <patient>.txt record files (cart init source)
  DATA_DIR      dir holding per-patient synth output (searched for parquet)
  P_ISO         prob a sample sees the gold cartridge ALONE; else gold +
                k~U(1,N-1) sampled distractors (CAS section 2.1; default 0.75)
  KV_TOKENS     tokens per cartridge (default 1024)
  STEPS         optimizer steps (default 600)
  ACCUM         samples accumulated per optimizer step (default 16)
  LR            AdamW lr on cartridge params (default 2e-2)
  OUT_DIR       where carts_piso/<patient>.pt land
Compare the resulting carts_piso/*.pt against isolated and fixed-distractor
joint carts with the existing combine-eval to see whether faithful P_iso rescues.
"""
import os, glob, random, time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", os.environ.get("CARTRIDGES_DIR", "/home/mcgrof/cartridges"))
# cartridges.__init__ requires CARTRIDGES_OUTPUT_DIR before import; mirror OUT_DIR.
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR",
                      os.environ.get("OUT_DIR", "/home/mcgrof/cas_out/cart_out"))
# eager flex-attention: the reveal pattern changes every sample, so a compiled
# flex would recompile per pattern. Correctness over speed for this validation.
os.environ["CARTRIDGES_COMPILE_FLEX"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER
import cartridges.models.qwen.modeling_qwen3 as mq
from torch.nn.attention.flex_attention import create_block_mask

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PATIENTS = os.environ.get("PATIENTS", "patient_01 patient_02 patient_03").split()
RECORDS_DIR = os.environ.get("RECORDS_DIR", "/home/mcgrof/cas_out/records")
DATA_DIR = os.environ.get("DATA_DIR", "/home/mcgrof/cas_out/synth")
P_ISO = float(os.environ.get("P_ISO", "0.75"))
KV_TOKENS = int(os.environ.get("KV_TOKENS", "1024"))
STEPS = int(os.environ.get("STEPS", "600"))
ACCUM = int(os.environ.get("ACCUM", "16"))
LR = float(os.environ.get("LR", "2e-2"))
OUT_DIR = os.environ.get("OUT_DIR", "/home/mcgrof/cas_out/cart_out")
DEVICE = os.environ.get("DEVICE", "cuda:0")
SEED = int(os.environ.get("SEED", "42"))

N = len(PATIENTS)
assert N >= 2, "P_iso joint training needs >=2 co-trained cartridges"
di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
torch.cuda.set_device(di)
torch.manual_seed(SEED)
rng = random.Random(SEED)

# ---- per-sample reveal mask (monkeypatched into the model's forward) ----------
_STATE = {"reveal": None}  # (N,) float {0,1} on device, set before each forward


def piso_block_mask(cache, seq_ids, device):
    """Reveal-vector block mask. A request query attends to a cartridge token iff
    that cart is revealed for the current sample, and causally to request tokens
    (single sample => same-sequence is implicit). Visibility is a full-length
    per-kv lookup (kv_vis[kv_idx]) -- the same 1-D index pattern the library uses
    for kv_seq_ids -- so no data-dependent index arithmetic runs inside the mask."""
    cache_len = cache.num_cartridge_tokens()  # N*KV_TOKENS (train mode: no append)
    R = seq_ids.shape[0]
    reveal = _STATE["reveal"].to(device)                       # (N,) float {0,1}
    # per-kv cartridge visibility over the whole [carts | request] axis
    kv_vis = torch.zeros(cache_len + R, dtype=torch.bool, device=device)
    kv_vis[:cache_len] = reveal.repeat_interleave(KV_TOKENS).bool()
    cl = cache_len

    def mask_func(_b, _h, q_idx, kv_idx):
        is_cart = kv_idx < cl
        cart_ok = kv_vis[kv_idx]
        req_ok = (q_idx + cl) >= kv_idx  # library causal form (request region)
        return torch.where(is_cart, cart_ok, req_ok)

    return create_block_mask(mask_func, B=1, H=1, Q_LEN=R,
                             KV_LEN=R + cl, device=device)


def find_parquet(patient):
    pstr = patient.replace("patient_", "p")
    hits = glob.glob(f"{DATA_DIR}/**/synth_qwen3_8b_lh_{pstr}_n*/artifact/dataset.parquet",
                     recursive=True)
    hits = hits or glob.glob(f"{DATA_DIR}/**/*{pstr}*/**/dataset.parquet", recursive=True)
    assert hits, f"no synth parquet for {patient} under {DATA_DIR}"
    return sorted(hits)[-1]


@torch.no_grad()
def init_cart(model, tokenizer, record_path, theta_dev):
    """Truncation init of one cartridge from a patient record (KVFromText path).
    Returns per-layer (1,H,KV_TOKENS,D) key/value tensors on CPU."""
    content = Path(record_path).read_text()
    tok_fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]
    ids = tok_fn(tokenizer=tokenizer, content=content, max_tokens=KV_TOKENS).squeeze(0)
    assert ids.shape[0] >= KV_TOKENS, (
        f"{record_path}: only {ids.shape[0]} tokens (< KV_TOKENS={KV_TOKENS}); "
        "use a smaller KV_TOKENS so every cart is exactly KV_TOKENS")
    ids = ids[:KV_TOKENS].to(theta_dev)
    tmp = TrainableCache(config=model_attn_config)
    seq_ids = torch.full_like(ids, 0, dtype=torch.long)
    pos = torch.arange(ids.shape[-1], dtype=torch.long, device=theta_dev)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(input_ids=ids, seq_ids=seq_ids, position_ids=pos,
              use_cache=True, past_key_values=tmp, mode="generate")
    k = [t.detach().float().cpu() for t in tmp._keys]
    v = [t.detach().float().cpu() for t in tmp._values]
    return k, v


def main():
    global model_attn_config
    print(f"P_iso trainer: N={N} patients {PATIENTS} P_iso={P_ISO} "
          f"KV_TOKENS={KV_TOKENS} STEPS={STEPS} ACCUM={ACCUM} LR={LR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    model.requires_grad_(False)
    cfg = model.config
    model_attn_config = AttnConfig(
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )

    # 1. build N cartridges (truncation init), concatenate [cart_0|...|cart_{N-1}]
    per_k, per_v = [], []
    for p in PATIENTS:
        k, v = init_cart(model, tokenizer, os.path.join(RECORDS_DIR, f"{p}.txt"), DEVICE)
        per_k.append(k); per_v.append(v)
        print(f"  init cart {p}: {k[0].shape}")
    init_keys, init_values = [], []
    for li in range(cfg.num_hidden_layers):
        init_keys.append(torch.cat([per_k[j][li] for j in range(N)], dim=2).to(torch.bfloat16))
        init_values.append(torch.cat([per_v[j][li] for j in range(N)], dim=2).to(torch.bfloat16))
    cache = TrainableCache(config=model_attn_config, init_keys=init_keys,
                           init_values=init_values, num_frozen_tokens=0).to(DEVICE)
    assert cache.num_cartridge_tokens() == N * KV_TOKENS

    # 2. monkeypatch the mask builder used inside the model forward
    mq.create_block_mask_w_cache = piso_block_mask

    # 3. per-patient datasets (one parquet each -> target cart = patient index)
    datasets = []
    for j, p in enumerate(PATIENTS):
        pq = find_parquet(p)
        ds = TrainDataset.Config(
            data_sources=[DataSource(path=pq, type="local")],
            top_k_logits=20, packed_seq_length=2048, packing_mode="truncate",
        ).instantiate(tokenizer=tokenizer, seed=SEED)
        datasets.append(ds)
        print(f"  dataset {p}: {len(ds.elements)} elements  ({pq.split('/')[-3]})")

    opt = torch.optim.AdamW(cache.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    reveal_base = torch.zeros(N, device=DEVICE)

    # 4. training: one sample per forward, ACCUM samples per optimizer step
    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad()
        accum_loss = 0.0
        for _ in range(ACCUM):
            j = rng.randrange(N)                      # target cartridge
            ds = datasets[j]
            el = ds.elements[rng.randrange(len(ds.elements))]
            # CAS section 2.1 mixed visibility: w.p. P_iso the gold cartridge
            # is visible ALONE; otherwise gold plus k ~ U(1, N-1) sampled
            # distractor cartridges (not independent per-distractor masking).
            # Paper value: P_iso = 0.75.
            reveal = reveal_base.clone()
            reveal[j] = 1.0
            if rng.random() > P_ISO:
                k = rng.randint(1, N - 1)
                for c in rng.sample([c for c in range(N) if c != j], k):
                    reveal[c] = 1.0
            _STATE["reveal"] = reveal

            ids = el.input_ids.to(DEVICE)
            R = ids.shape[0]
            sids = torch.zeros(R, dtype=torch.long, device=DEVICE)   # single sample
            # Per-sample positions must reflect the VISIBLE prefix, not the
            # physical one: the model forward adds num_cartridge_tokens()
            # (= N*KV_TOKENS, all resident carts) to position_ids, but a
            # gold-alone sample should see the request at position
            # KV_TOKENS+i -- the geometry a solo-loaded cart serves at
            # inference. Without this, every cart trains exclusively at the
            # co-load offset and fails when loaded alone (observed: oracle
            # 0.22 vs co-load 0.45). Offset by (visible - total) so the
            # forward's +total nets to +visible.
            visible_tokens = int(reveal.sum().item()) * KV_TOKENS
            pos = (torch.arange(R, dtype=torch.long, device=DEVICE)
                   + visible_tokens - N * KV_TOKENS)
            idxs = el.topk_token_idxs.to(DEVICE)
            tids = el.topk_token_ids.to(DEVICE)
            tlp = el.topk_logprobs.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, seq_ids=sids, position_ids=pos,
                            use_cache=True, past_key_values=cache)
                V = out.logits.shape[-1]
                gi = idxs - 1
                # drop sparse target entries that fall outside the sequence or
                # vocab (boundary/sentinel artifacts of the synth top-k encoding;
                # the library's collate normalizes these, the raw element does not)
                valid = (gi >= 0) & (gi < R) & (tids >= 0) & (tids < V)
                nv = int(valid.sum())
                if not _STATE.get("diag") and nv < valid.numel():
                    _STATE["diag"] = True
                    print(f"  [diag] R={R} idx_max={int(idxs.max())} tid[min="
                          f"{int(tids.min())},max={int(tids.max())}] V={V} "
                          f"dropped={valid.numel() - nv}/{valid.numel()}", flush=True)
                if nv == 0:
                    continue
                lp = F.log_softmax(out.logits, dim=-1)[0, gi[valid], tids[valid]]
                loss = (-tlp[valid].exp() * lp).mean() / ACCUM
            loss.backward()
            accum_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(cache.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == STEPS - 1:
            print(f"  step {step:>4} loss={accum_loss:.4f} "
                  f"ppl={np.exp(accum_loss):.2f} t={time.time()-t0:.0f}s", flush=True)

    # 5. save each cartridge separately (slice its KV_TOKENS block), library format
    out = Path(OUT_DIR) / "carts_piso"; out.mkdir(parents=True, exist_ok=True)
    tk = [p.detach() for p in cache.trainable_keys]
    tv = [p.detach() for p in cache.trainable_values]
    for j, p in enumerate(PATIENTS):
        s, e = j * KV_TOKENS, (j + 1) * KV_TOKENS
        torch.save({
            "trainable_keys": [t[:, :, s:e].contiguous().cpu() for t in tk],
            "trainable_values": [t[:, :, s:e].contiguous().cpu() for t in tv],
            "frozen_keys": None, "frozen_values": None,
        }, out / f"{p}.pt")
        print(f"CAS_PISO_SAVED {p} -> {out / f'{p}.pt'}")
    print(f"CAS_PISO_DONE N={N} P_iso={P_ISO} steps={STEPS} "
          f"wall={time.time()-t0:.0f}s out={out}")


if __name__ == "__main__":
    main()
