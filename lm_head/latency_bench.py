#!/usr/bin/env python3
"""Latency microbenchmark for the certified shadow-bound LM-head decode.

Bytes are a lower bound on data movement; this measures wall-clock at batch 1
(the regime where the LM head is latency-critical -- at larger batch the dense
head amortizes and the motivation weakens). It times each component at real
qwen-7b dimensions and composes three end-to-end numbers:

  dense_bf16        the full head GEMV -- the lossless baseline it replaces
  certified_naive   shadow + segmax + sort + a HOST-DRIVEN sequential expansion
                    (one launch + sync per opened slab) -- pessimistic GPU case
  certified_ideal   shadow + segmax + sort + the fetched fraction as ONE batched
                    GEMV -- the fused / on-device upper bound (what addressable
                    hardware would approach)

The decisive number is the per-slab launch+sync overhead: it is what separates
naive (loses) from fused (wins). Mean/p95 slab counts come from the replay
(mean fetch 6.7% ~ 17 slabs, p95 42% ~ 107 slabs at C=256).

Data values do not affect timing, so random tensors of the right shape/dtype are
used (no model load needed).

Usage:
  python3 lm_head/latency_bench.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def timed(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def timed_with_sync(fn, iters, warmup):
    """Per-call cost INCLUDING a host sync each call -- models the serialized
    open->check->open dependency of the expansion loop."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--V", type=int, default=152064)
    ap.add_argument("--d", type=int, default=3584)
    ap.add_argument("--r", type=int, default=1280)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--mean-fetch", type=float, default=0.067)
    ap.add_argument("--p95-fetch", type=float, default=0.42)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="/tmp/latency_bench.json")
    args = ap.parse_args()

    dev = torch.device(args.device)
    V, d, r, C = args.V, args.d, args.r, args.C
    slab = (V + C - 1) // C
    it, wu = args.iters, args.warmup

    # tensors (values irrelevant to timing)
    W = torch.randn(V, d, device=dev, dtype=torch.bfloat16)  # dense head
    h = torch.randn(d, device=dev, dtype=torch.bfloat16)
    B = torch.randn(d, r, device=dev, dtype=torch.bfloat16)
    a = torch.randn(V, r, device=dev, dtype=torch.float16)  # shadow (fp16 timed)
    delta = torch.randn(V, device=dev, dtype=torch.float16)
    cof = torch.randint(0, C, (V,), device=dev)
    hf = h.float()
    Wslab = torch.randn(slab, d, device=dev, dtype=torch.bfloat16)
    trivial = torch.zeros(1, device=dev)

    res = {"V": V, "d": d, "r": r, "C": C, "slab": slab}

    # --- components ---
    L_dense = timed(lambda: torch.mv(W, h), it, wu, dev)

    def shadow():
        q = hf @ B.float()  # [r]
        recon = q @ B.float().t()
        rho = (hf - recon).norm()
        qn = q.norm()
        U = a.float() @ q + delta.float() * rho + 0.0 * qn  # [V]
        return U

    L_shadow = timed(shadow, it, wu, dev)

    def segmax():
        U = torch.randn(V, device=dev)
        out = torch.full((C,), float("-inf"), device=dev)
        return out.scatter_reduce_(0, cof, U, reduce="amax", include_self=True)

    L_segmax = timed(segmax, it, wu, dev)
    L_sort = timed(
        lambda: torch.randn(C, device=dev).argsort(descending=True), it, wu, dev
    )
    L_slab = timed(lambda: torch.mv(Wslab, h), it, wu, dev)

    # per-call cost with a host sync (serialized-step model)
    L_slab_sync = timed_with_sync(lambda: torch.mv(Wslab, h), it, wu)
    L_sync_only = timed_with_sync(lambda: trivial.add_(1.0), it, wu)
    sync_overhead = L_slab_sync - L_slab  # extra cost of one host round-trip

    # shadow at int8 ~ half the bytes of fp16 (bandwidth-bound estimate)
    L_shadow_int8 = L_shadow * 0.5

    fixed = L_shadow_int8 + L_segmax + L_sort
    n_mean = args.mean_fetch * C
    n_p95 = args.p95_fetch * C
    cert_naive_mean = fixed + n_mean * L_slab_sync
    cert_naive_p95 = fixed + n_p95 * L_slab_sync
    cert_ideal_mean = fixed + args.mean_fetch * L_dense  # fetched as one GEMV
    cert_ideal_p95 = fixed + args.p95_fetch * L_dense

    res.update(
        {
            "ms": {
                "dense_bf16": L_dense,
                "shadow_fp16": L_shadow,
                "shadow_int8_est": L_shadow_int8,
                "segmax": L_segmax,
                "sort": L_sort,
                "one_slab_gemv": L_slab,
                "one_slab_gemv_with_sync": L_slab_sync,
                "host_sync_overhead": sync_overhead,
                "trivial_with_sync": L_sync_only,
            },
            "end_to_end_ms": {
                "dense_bf16": L_dense,
                "certified_naive_mean": cert_naive_mean,
                "certified_naive_p95": cert_naive_p95,
                "certified_ideal_mean": cert_ideal_mean,
                "certified_ideal_p95": cert_ideal_p95,
            },
            "speedup_vs_dense": {
                "naive_mean": L_dense / cert_naive_mean,
                "naive_p95": L_dense / cert_naive_p95,
                "ideal_mean": L_dense / cert_ideal_mean,
                "ideal_p95": L_dense / cert_ideal_p95,
            },
        }
    )

    print(f"=== latency @ batch1, V={V} d={d} r={r} C={C} slab={slab} ===")
    for k, v in res["ms"].items():
        print(f"  {k:28s} {v*1000:8.1f} us")
    print("  --- end-to-end (lower is better) ---")
    for k, v in res["end_to_end_ms"].items():
        sp = res["speedup_vs_dense"].get(
            k.replace("certified_", "").replace("_bf16", ""), None
        )
        tag = ""
        if k != "dense_bf16":
            tag = f"  ({L_dense / v:.2f}x vs dense)"
        print(f"  {k:28s} {v*1000:8.1f} us{tag}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
