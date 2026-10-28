#!/usr/bin/env python3
"""Measure how much attention the reader pays to the cartridge slots,
split by whether the position is document-informative.

A cartridge is a trained KV prefix that every conversation token can
attend to. Whether reader-side attention steering toward the cartridge
(a PASTA-style intervention) has anything to work with depends on a
number nobody has measured: at the positions where the document changes
the teacher's answer (the informative positions), how much of each
head's attention already lands on the cartridge, compared with the
positions where the document does not matter (the rest). If the
informative positions already attend heavily to the cartridge, steering
cannot help and the limit is what the slots contain. If the mass is thin
exactly where the document matters, steering is a candidate.

The script runs the same packed held-out slice and the same forward
pass as ``cas_cart_loss.py``, wraps the library's attention call, and
recomputes the softmax over all allowed keys for every head and query
position from the queries and keys the kernel sees (after RoPE and
q/k norm, with the library's own block-mask rule). It records the
attention mass on the frozen sink slot and on the trainable slots
separately, because the sink absorbs attention by construction and
only the trainable slots carry document content.

Env (as ``cas_cart_loss.py``): MODEL, PARQUET, LIMIT, SEED, CARTS,
OUT_JSON, DEVICE. The no-cartridge pass that defines the informative
positions always runs.

Prints per cartridge
``ATTN_MASS,<name>,all,<info_slots>,<rest_slots>,<info_sink>,<rest_sink>``
(means over heads and positions), one ``ATTN_MASS_LAYER`` line per
layer, the heads with the largest informative-minus-rest slot mass, and
``ATTN_MASS_DONE``. Per-layer, per-head sums and the per-position
head-mean slot mass per layer go to the ``.npz`` next to the JSON.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_attn_mass")
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
    parse_carts,
    score,
    teacher_argmax,
)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ["PARQUET"]
SEED = int(os.environ.get("SEED", "42"))
CARTS = [c for c in os.environ.get("CARTS", "").split(",") if c]
OUT_JSON = os.environ.get("OUT_JSON", "attn_mass.json")
DEVICE = os.environ.get("DEVICE", "cuda")
TOPN = int(os.environ.get("TOPN", "20"))

# Per-batch state the attention wrapper reads: the dense allowed-key
# mask, the cartridge geometry, and the per-layer accumulators.
STATE = {"mask": None, "cache_len": 0, "n_frozen": 0, "sink": None, "slots": None}

_orig_attention = mq.flex_attention_forward


def dense_mask(seq_ids, cache_len, device):
    """Boolean [Q, K] mask with the library's rule: cartridge keys are
    visible to every query; conversation keys are visible to queries of
    the same packed element at or after them."""
    q = seq_ids.to(device)
    kv = torch.cat([torch.full((cache_len,), -1, dtype=q.dtype, device=device), q])
    qi = torch.arange(len(q), device=device)[:, None]
    ki = torch.arange(len(kv), device=device)[None, :]
    return (kv[None, :] == -1) | ((q[:, None] == kv[None, :]) & (qi + cache_len >= ki))


@torch.no_grad()
def measure(layer_idx, query, key, scaling):
    """Attention mass on the sink and on the trainable slots for every
    head and query position of this layer, written into STATE."""
    mask = STATE["mask"]
    cl, nf = STATE["cache_len"], STATE["n_frozen"]
    q = query[0]  # [Hq, Q, d]
    k = key[0]  # [Hkv, K, d]
    hq, hkv = q.shape[0], k.shape[0]
    groups = hq // hkv
    scale = scaling if scaling is not None else q.shape[-1] ** -0.5
    assert k.shape[1] == cl + q.shape[1], (k.shape, cl, q.shape)
    for h in range(hkv):
        qh = q[h * groups : (h + 1) * groups].float()
        kh = k[h].float()
        s = torch.matmul(qh, kh.transpose(0, 1)) * scale  # [g, Q, K]
        s.masked_fill_(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        STATE["sink"][layer_idx, h * groups : (h + 1) * groups] = p[..., :nf].sum(-1)
        STATE["slots"][layer_idx, h * groups : (h + 1) * groups] = p[..., nf:cl].sum(-1)
        del s, p


def wrapped_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    out = _orig_attention(
        module, query, key, value, attention_mask, scaling=scaling, **kw
    )
    if STATE["mask"] is not None and STATE["cache_len"] > 0:
        measure(module.layer_idx, query, key, scaling)
    return out


mq.flex_attention_forward = wrapped_attention


@torch.no_grad()
def attn_pass(model, dataset, cache, n_frozen, n_layers, n_heads):
    """One pass over the packed dataset with the cartridge; returns the
    per-position head-mean slot mass per layer [L, n_pos] and per-layer,
    per-head sums split by the caller's masks later."""
    n_batches = len(dataset)
    n_pos = n_batches * SEQ_LEN
    slot_pos = np.zeros((n_layers, n_pos), dtype=np.float32)
    sink_pos = np.zeros((n_layers, n_pos), dtype=np.float32)
    slot_head = np.zeros((n_layers, n_heads, n_pos), dtype=np.float16)
    cl = cache.num_tokens()
    for b in range(n_batches):
        batch = dataset[b]
        cache.clear()
        STATE["mask"] = dense_mask(batch.element_ids, cl, DEVICE)
        STATE["cache_len"], STATE["n_frozen"] = cl, n_frozen
        STATE["sink"] = torch.zeros(n_layers, n_heads, SEQ_LEN, device=DEVICE)
        STATE["slots"] = torch.zeros(n_layers, n_heads, SEQ_LEN, device=DEVICE)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(
                input_ids=batch.input_ids.to(DEVICE),
                seq_ids=batch.element_ids.to(DEVICE),
                position_ids=batch.position_ids.to(DEVICE),
                use_cache=True,
                past_key_values=cache,
            )
        base = b * SEQ_LEN
        slot_pos[:, base : base + SEQ_LEN] = STATE["slots"].mean(1).cpu().numpy()
        sink_pos[:, base : base + SEQ_LEN] = STATE["sink"].mean(1).cpu().numpy()
        slot_head[:, :, base : base + SEQ_LEN] = (
            STATE["slots"].to(torch.float16).cpu().numpy()
        )
        if b % 20 == 0:
            print(f"[attn-mass] batch {b}/{n_batches}", flush=True)
    STATE["mask"] = None
    return slot_pos, sink_pos, slot_head


def split_means(arr_pos, has, info):
    """Means of a [.., n_pos] array over informative and rest positions."""
    i = has & info
    r = has & ~info
    return arr_pos[..., i].astype(np.float64).mean(-1), arr_pos[..., r].astype(
        np.float64
    ).mean(-1)


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    dataset = build_dataset(tok)
    print(
        f"[attn-mass] {PARQUET}: {len(dataset.elements)} rows -> "
        f"{len(dataset)} packed sequences of {SEQ_LEN}",
        flush=True,
    )
    model = (
        FlexQwen3ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    model.eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    attn = AttnConfig(
        n_layers=n_layers, n_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim
    )
    print(f"[attn-mass] model up in {time.time() - t0:.0f}s", flush=True)

    # informative positions from the no-cartridge pass, as cas_cart_loss
    n_pos = len(dataset) * SEQ_LEN
    t_arg = np.full(n_pos, -1, dtype=np.int64)
    for b in range(len(dataset)):
        t_arg[b * SEQ_LEN : (b + 1) * SEQ_LEN] = teacher_argmax(
            dataset[b], SEQ_LEN
        ).numpy()
    ce0, n0, am0 = score(model, dataset, None)
    has = n0 > 0
    info = has & (am0 != t_arg)
    print(
        f"[attn-mass] informative positions {int(info.sum())}/{int(has.sum())}",
        flush=True,
    )

    results = {
        "model": MODEL,
        "parquet": os.path.abspath(PARQUET),
        "seed": SEED,
        "n_rows": len(dataset.elements),
        "n_packed": len(dataset),
        "n_positions": int(has.sum()),
        "n_info_positions": int(info.sum()),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "carts": {},
    }
    arrays = {"info_mask": info, "has_targets": has}
    for name, path in parse_carts(CARTS):
        t1 = time.time()
        keys, values, nfrozen = load_cart(path, device=DEVICE)
        cache = TrainableCache(
            config=attn, init_keys=keys, init_values=values, num_frozen_tokens=nfrozen
        ).to(DEVICE)
        slot_pos, sink_pos, slot_head = attn_pass(
            model, dataset, cache, nfrozen, n_layers, n_heads
        )
        slot_i, slot_r = split_means(slot_pos, has, info)  # [L]
        sink_i, sink_r = split_means(sink_pos, has, info)
        head_i, head_r = split_means(slot_head, has, info)  # [L, H]
        diff = head_i - head_r
        order = np.argsort(diff.ravel())[::-1][:TOPN]
        top = [
            {
                "layer": int(ix // n_heads),
                "head": int(ix % n_heads),
                "info": float(head_i.ravel()[ix]),
                "rest": float(head_r.ravel()[ix]),
            }
            for ix in order
        ]
        r = {
            "path": os.path.abspath(path),
            "trainable": int(keys[0].shape[2] - nfrozen),
            "frozen": int(nfrozen),
            "slots_info": float(slot_i.mean()),
            "slots_rest": float(slot_r.mean()),
            "sink_info": float(sink_i.mean()),
            "sink_rest": float(sink_r.mean()),
            "per_layer": {
                "slots_info": slot_i.tolist(),
                "slots_rest": slot_r.tolist(),
                "sink_info": sink_i.tolist(),
                "sink_rest": sink_r.tolist(),
            },
            "top_heads_info_minus_rest": top,
            "seconds": time.time() - t1,
        }
        results["carts"][name] = r
        arrays[f"{name}__slot_pos"] = slot_pos.astype(np.float16)
        arrays[f"{name}__head_info"] = head_i
        arrays[f"{name}__head_rest"] = head_r
        print(
            f"ATTN_MASS,{name},all,{r['slots_info']:.5f},{r['slots_rest']:.5f},"
            f"{r['sink_info']:.5f},{r['sink_rest']:.5f}",
            flush=True,
        )
        for L in range(n_layers):
            print(
                f"ATTN_MASS_LAYER,{name},{L},{slot_i[L]:.5f},{slot_r[L]:.5f},"
                f"{sink_i[L]:.5f},{sink_r[L]:.5f}",
                flush=True,
            )
        for t in top[:10]:
            print(
                f"ATTN_MASS_HEAD,{name},L{t['layer']}H{t['head']},"
                f"{t['info']:.5f},{t['rest']:.5f}",
                flush=True,
            )
        del cache, keys, values, slot_head
        torch.cuda.empty_cache()

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(str(Path(OUT_JSON).with_suffix(".npz")), **arrays)
    print(f"ATTN_MASS_DONE {OUT_JSON} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
