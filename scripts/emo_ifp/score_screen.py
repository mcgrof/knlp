#!/usr/bin/env python3
"""Paired screen of full EMO against a fixed expert pool (GPU).

Scores every screening item by answer likelihood: for each choice, the sum
of log probabilities of the continuation tokens given the context. The
registered primary metric picks the choice with the highest log probability
per continuation character; the secondary metric uses the raw sum. Context
and continuation are tokenized separately and concatenated, so every choice
shares the same context tokens.

The full model is scored first. The pool is then applied in place, the
removed experts are freed, and the same items are scored again. The script
also records resident weight bytes and per-layer prefill times for both
models.

Usage:
    score_screen.py MODEL_DIR SCREEN.json POOL.json OUT_DIR
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    apply_pool,
    load_model,
    mem_snapshot,
    runtime_identity,
    write_json,
)  # noqa
from pilot import FILLER  # noqa


@torch.no_grad()
def score_item(model, tok, item):
    ctx = tok(item["context"], add_special_tokens=False)["input_ids"]
    conts = [tok(c, add_special_tokens=False)["input_ids"] for c in item["choices"]]
    seqs = [ctx + c for c in conts]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    out = []
    for i, c in enumerate(conts):
        pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(c), device="cuda")
        tgt = torch.tensor(c, device="cuda")
        lp = logp[i, pos, tgt]
        out.append(
            {
                "sum_logprob": float(lp.sum()),
                "n_tokens": len(c),
                "n_chars": len(item["choices"][i]),
            }
        )
    raw = max(range(len(out)), key=lambda j: out[j]["sum_logprob"])
    pc = max(range(len(out)), key=lambda j: out[j]["sum_logprob"] / out[j]["n_chars"])
    g = item["gold"]
    return {
        "id": item["id"],
        "gold": g,
        "context_tokens": len(ctx),
        "choices": out,
        "pred_per_char": pc,
        "pred_raw": raw,
        "correct_per_char": pc == g,
        "correct_raw": raw == g,
        "gold_logprob_per_token": out[g]["sum_logprob"] / out[g]["n_tokens"],
    }


def score_all(model, tok, items, tag):
    t0 = time.time()
    res = []
    for it in items:
        res.append(score_item(model, tok, it))
    torch.cuda.synchronize()
    el = time.time() - t0
    acc = sum(r["correct_per_char"] for r in res)
    print(f"{tag}: {acc}/{len(res)} per-char correct, {el:.1f} s", flush=True)
    return res, el


@torch.no_grad()
def layer_prefill_times(model, tok, n, reps=5):
    """Median per-layer prefill time (CUDA events) for an n-token prompt."""
    ids = tok(FILLER * (n // 40 + 2), add_special_tokens=False)["input_ids"][:n]
    ids = torch.tensor([ids], device="cuda")
    layers = model.model.layers
    ev = {}
    hs = []
    for li, layer in enumerate(layers):

        def pre(mod, inp, li=li):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev.setdefault(li, []).append([e, None])

        def post(mod, inp, out, li=li):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[li][-1][1] = e

        hs.append(layer.register_forward_pre_hook(pre))
        hs.append(layer.register_forward_hook(post))
    for _ in range(2):
        model(input_ids=ids)
    torch.cuda.synchronize()
    ev.clear()
    tot = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(input_ids=ids)
        torch.cuda.synchronize()
        tot.append(time.perf_counter() - t0)
    for h in hs:
        h.remove()
    per = [
        statistics.median(a.elapsed_time(b) / 1e3 for a, b in ev[li])
        for li in range(len(layers))
    ]
    return {
        "tokens": n,
        "total_wall_median_s": statistics.median(tot),
        "per_layer_median_s": per,
        "before_last_layer_s": sum(per[:-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("screen")
    ap.add_argument("pool")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.screen) as f:
        items = json.load(f)["items"]
    with open(args.pool) as f:
        pool = json.load(f)

    res = {"runtime": runtime_identity(), "pool_file": os.path.abspath(args.pool)}
    model, tok = load_model(args.model_dir)
    pb = lambda: sum(p.numel() * p.element_size() for p in model.parameters())  # noqa
    res["full_param_bytes"] = pb()
    res["full_mem"] = mem_snapshot()
    full, t_full = score_all(model, tok, items, "full")
    res["full_layer_prefill"] = {
        n: layer_prefill_times(model, tok, n) for n in (128, 1024)
    }

    apply_pool(model, pool)
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    res["pool_param_bytes"] = pb()
    res["pool_mem"] = mem_snapshot()
    sub, t_sub = score_all(model, tok, items, "pool")
    res["pool_layer_prefill"] = {
        n: layer_prefill_times(model, tok, n) for n in (128, 1024)
    }

    rows = []
    for a, b in zip(full, sub):
        assert a["id"] == b["id"]
        rows.append({"id": a["id"], "full": a, "pool": b})

    def paired(key):
        fw = sum(1 for r in rows if r["full"][key] and not r["pool"][key])
        pw = sum(1 for r in rows if r["pool"][key] and not r["full"][key])
        return {
            "full_correct": sum(r["full"][key] for r in rows),
            "pool_correct": sum(r["pool"][key] for r in rows),
            "full_only": fw,
            "pool_only": pw,
            "net_losses": fw - pw,
        }

    dl = [
        r["pool"]["gold_logprob_per_token"] - r["full"]["gold_logprob_per_token"]
        for r in rows
    ]
    res["n"] = len(rows)
    res["primary_per_char"] = paired("correct_per_char")
    res["secondary_raw"] = paired("correct_raw")
    res["gold_logprob_per_token_delta"] = {
        "mean": statistics.mean(dl),
        "median": statistics.median(dl),
        "min": min(dl),
        "max": max(dl),
    }
    res["max_context_tokens"] = max(r["full"]["context_tokens"] for r in rows)
    res["score_seconds"] = {"full": t_full, "pool": t_sub}
    nl = res["primary_per_char"]["net_losses"]
    res["screen_rule"] = "<=4 pass; 5-8 unresolved (inspect, widen once); >8 gross harm"
    res["screen_outcome"] = (
        "pass" if nl <= 4 else ("unresolved" if nl <= 8 else "gross_harm")
    )
    write_json(os.path.join(args.out_dir, "screen-summary.json"), res)
    with open(os.path.join(args.out_dir, "screen-per-example.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "primary_per_char",
                    "secondary_raw",
                    "screen_outcome",
                    "full_param_bytes",
                    "pool_param_bytes",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
