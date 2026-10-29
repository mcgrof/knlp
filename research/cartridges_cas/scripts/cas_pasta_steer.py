#!/usr/bin/env python3
"""Reader-side attention steering toward the cartridge slots, scored by
the held-out distillation loss, with an oracle trigger as the ceiling.

PASTA (Zhang et al., arXiv 2311.02262) emphasizes a marked span of the
input at inference by multiplying the post-softmax attention to every
other token by a small factor alpha in selected heads and renormalizing.
For a cartridge the span is fixed: the trainable KV slots in front of
every conversation. This script applies that steering to a frozen
reader with a trained cartridge and measures what it does to the
trainer's own objective on the held-out split, on the positions where
the document changes the teacher's answer (informative) and on the
rest, so a steering choice can be judged by the same number the
cartridge was trained on.

Three policies are scored at each layer and alpha:

* ``all``: steer every query position (plain PASTA).
* ``oracle``: steer only the informative positions, which are known
  under teacher forcing. No inference-time trigger can beat this, so
  it bounds every requery policy from above.
* control: the same manual attention path with alpha=1, which measures
  the numerical difference between the manual fp32 path and the
  library's compiled kernel, so a steering effect is read against it
  and not against the kernel.

After the per-layer sweep the layers whose oracle informative loss
improved most over their alpha=1 control are steered jointly.

Env: MODEL, PARQUET, LIMIT, SEED (as cas_cart_loss.py); CART (one
cartridge file, the trained one); ALPHAS (comma list, default
``0.1,0.01``); LAYERS (comma list or ``all``); TOPK_JOINT (default 5);
OUT_JSON; DEVICE.

Prints ``STEER,<layers>,<alpha>,<policy>,<loss>,<loss_info>,<loss_rest>``
per pass and ``STEER_DONE`` at the end.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_pasta_steer")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import torch
from transformers import AutoTokenizer

import cartridges.models.qwen.modeling_qwen3 as mq
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_cart_init import load_cart  # noqa: E402
from cas_cart_loss import (  # noqa: E402
    SEQ_LEN,
    build_dataset,
    score,
    summarize,
    teacher_argmax,
)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ["PARQUET"]
SEED = int(os.environ.get("SEED", "42"))
CART = os.environ["CART"]
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.1,0.01").split(",")]
LAYERS_ENV = os.environ.get("LAYERS", "all")
TOPK_JOINT = int(os.environ.get("TOPK_JOINT", "5"))
OUT_JSON = os.environ.get("OUT_JSON", "pasta_steer.json")
DEVICE = os.environ.get("DEVICE", "cuda")

# Sweep state read by the attention wrapper. ``batch`` counts batches by
# watching layer 0, so cas_cart_loss.score() can be reused unchanged.
STATE = {
    "active": False,
    "layers": set(),
    "log_alpha": 0.0,
    "policy": "all",
    "cache_len": 0,
    "elem_ids": [],
    "info_rows": [],
    "batch": -1,
    "mask": None,
    "rows": None,
}

_orig_attention = mq.flex_attention_forward


def dense_mask(seq_ids, cache_len, device):
    q = seq_ids.to(device)
    kv = torch.cat([torch.full((cache_len,), -1, dtype=q.dtype, device=device), q])
    qi = torch.arange(len(q), device=device)[:, None]
    ki = torch.arange(len(kv), device=device)[None, :]
    return (kv[None, :] == -1) | ((q[:, None] == kv[None, :]) & (qi + cache_len >= ki))


@torch.no_grad()
def steered_attention(query, key, value, scaling):
    """Manual masked attention with the conversation keys down-weighted
    by log_alpha on the selected query rows; returns [1, Q, Hq, d] in
    the query dtype, the layout the library's kernel path returns."""
    q = query[0]
    k = key[0]
    v = value[0]
    hq, hkv = q.shape[0], k.shape[0]
    groups = hq // hkv
    cl = STATE["cache_len"]
    scale = scaling if scaling is not None else q.shape[-1] ** -0.5
    mask = STATE["mask"]
    rows = STATE["rows"]  # bool [Q] or None for all rows
    out = torch.empty_like(q, dtype=torch.float32)
    for h in range(hkv):
        qh = q[h * groups : (h + 1) * groups].float()
        kh = k[h].float()
        vh = v[h].float()
        s = torch.matmul(qh, kh.transpose(0, 1)) * scale  # [g, Q, K]
        if STATE["log_alpha"] != 0.0:
            if rows is None:
                s[:, :, cl:] += STATE["log_alpha"]
            else:
                s[:, rows, cl:] += STATE["log_alpha"]
        s.masked_fill_(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[h * groups : (h + 1) * groups] = torch.matmul(p, vh)
        del s, p
    return out.to(query.dtype).transpose(0, 1).contiguous()[None]


def wrapped_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    if STATE["active"] and STATE["cache_len"] > 0:
        if module.layer_idx == 0:
            STATE["batch"] += 1
            b = STATE["batch"]
            STATE["mask"] = dense_mask(STATE["elem_ids"][b], STATE["cache_len"], DEVICE)
            if STATE["policy"] == "oracle":
                STATE["rows"] = STATE["info_rows"][b].to(DEVICE)
            else:
                STATE["rows"] = None
        if module.layer_idx in STATE["layers"]:
            assert key.shape[2] == STATE["cache_len"] + query.shape[2]
            return steered_attention(query, key, value, scaling)
    return _orig_attention(
        module, query, key, value, attention_mask, scaling=scaling, **kw
    )


mq.flex_attention_forward = wrapped_attention


def run_pass(model, dataset, cache, layers, alpha, policy, info_mask):
    STATE.update(
        active=True,
        layers=set(layers),
        log_alpha=math.log(alpha) if alpha != 1.0 else 0.0,
        policy=policy,
        batch=-1,
    )
    ce, n, _ = score(model, dataset, cache)
    STATE["active"] = False
    r = summarize(ce, n, info_mask)
    return r, ce


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dataset = build_dataset(tok)
    n_batches = len(dataset)
    print(
        f"[steer] {PARQUET}: {len(dataset.elements)} rows -> "
        f"{n_batches} packed sequences of {SEQ_LEN}",
        flush=True,
    )
    model = (
        FlexQwen3ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    attn = AttnConfig(
        n_layers=n_layers, n_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim
    )
    print(f"[steer] model up in {time.time() - t0:.0f}s", flush=True)

    # informative positions from the no-cartridge pass (kernel path)
    n_pos = n_batches * SEQ_LEN
    t_arg = np.full(n_pos, -1, dtype=np.int64)
    elem_ids = []
    for b in range(n_batches):
        batch = dataset[b]
        t_arg[b * SEQ_LEN : (b + 1) * SEQ_LEN] = teacher_argmax(batch, SEQ_LEN).numpy()
        elem_ids.append(batch.element_ids.clone())
    ce0, n0, am0 = score(model, dataset, None)
    has = n0 > 0
    info = has & (am0 != t_arg)
    STATE["elem_ids"] = elem_ids
    STATE["info_rows"] = [
        torch.from_numpy(info[b * SEQ_LEN : (b + 1) * SEQ_LEN])
        for b in range(n_batches)
    ]
    print(
        f"[steer] informative positions {int(info.sum())}/{int(has.sum())}", flush=True
    )

    keys, values, nfrozen = load_cart(CART, device=DEVICE)
    cache = TrainableCache(
        config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
    ).to(DEVICE)
    STATE["cache_len"] = cache.num_tokens()

    results = {
        "model": MODEL,
        "cart": os.path.abspath(CART),
        "parquet": os.path.abspath(PARQUET),
        "seed": SEED,
        "n_packed": n_batches,
        "n_positions": int(has.sum()),
        "n_info_positions": int(info.sum()),
        "alphas": ALPHAS,
        "passes": [],
    }
    arrays = {"info_mask": info}

    def record(layers, alpha, policy, r):
        name = "+".join(str(l) for l in layers) if layers else "none"
        row = {"layers": list(layers), "alpha": alpha, "policy": policy, **r}
        results["passes"].append(row)
        print(
            f"STEER,{name},{alpha},{policy},{r['loss']:.6f},{r['loss_info']:.6f},"
            f"{r['loss_rest']:.6f}",
            flush=True,
        )
        return row

    # baseline through the kernel path
    r, ce = run_pass(model, dataset, cache, [], 1.0, "none", info)
    record([], 1.0, "none", r)
    arrays["base_ce"] = ce

    layers = (
        list(range(n_layers))
        if LAYERS_ENV == "all"
        else [int(x) for x in LAYERS_ENV.split(",")]
    )
    gain = {}
    for L in layers:
        t1 = time.time()
        ctl, _ = run_pass(model, dataset, cache, [L], 1.0, "all", info)
        record([L], 1.0, "control", ctl)
        best = 0.0
        for a in ALPHAS:
            for policy in ("all", "oracle"):
                r, _ = run_pass(model, dataset, cache, [L], a, policy, info)
                record([L], a, policy, r)
                if policy == "oracle":
                    best = max(best, ctl["loss_info"] - r["loss_info"])
        gain[L] = best
        print(f"[steer] layer {L} done in {time.time() - t1:.0f}s", flush=True)

    top = sorted(gain, key=lambda L: gain[L], reverse=True)[:TOPK_JOINT]
    top = [L for L in top if gain[L] > 0]
    results["joint_layers"] = top
    if top:
        ctl, _ = run_pass(model, dataset, cache, top, 1.0, "all", info)
        record(top, 1.0, "control", ctl)
        for a in ALPHAS:
            for policy in ("all", "oracle"):
                r, ce = run_pass(model, dataset, cache, top, a, policy, info)
                record(top, a, policy, r)
                arrays[f"joint_a{a}_{policy}_ce"] = ce

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(str(Path(OUT_JSON).with_suffix(".npz")), **arrays)
    print(f"STEER_DONE {OUT_JSON} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
