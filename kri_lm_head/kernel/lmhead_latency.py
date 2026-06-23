#!/usr/bin/env python3
"""Proper latency sweep for the LM-head certified-decode kernel on prune (W7900).

CUDA-event timed (HIP), warmup + many iters, p50/p95. Sweeps decode batch B (the
number of sequences each emitting one token this step). Compares the dense head
GEMV against the shadow stage-1 kernel (the clean memory-bound win) and an
estimate of the full certified path (stage-1 + a fixed-budget fetch at the
measured mean fetched fraction). Uses the real qwen-7b shapes; tensors are
synthetic at those shapes (latency depends on shape/dtype, not values).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import shadow_slab_bounds


def cuda_time(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    times.sort()
    return {
        "p50": statistics.median(times),
        "p95": times[int(0.95 * len(times)) - 1],
        "mean": sum(times) / len(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--V", type=int, default=152064)
    ap.add_argument("--d", type=int, default=3584)
    ap.add_argument("--r", type=int, default=1280)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--fetched-frac", type=float, default=0.074)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    V, d, r, C = args.V, args.d, args.r, args.C
    S = V // C
    print(f"GPU={torch.cuda.get_device_name(0)} V={V} d={d} r={r} C={C} S={S}", flush=True)

    # qwen-7b-shaped tensors (synthetic; latency is shape/dtype-driven)
    W_U = torch.randn(V, d, device=dev, dtype=torch.bfloat16)
    WUt = W_U.t().contiguous()
    aq = torch.randint(-127, 127, (V, r), device=dev, dtype=torch.int8)
    scale = torch.rand(r, device=dev) * 0.01
    delta = torch.rand(V, device=dev)
    B = torch.randn(d, r, device=dev)
    aq_l2 = aq.float().pow(2).sum(1).sqrt().contiguous()  # offline shadow-head prop
    k_fetch = max(1, round(args.fetched_frac * C))

    rows = {}
    for bs in [int(x) for x in args.batches.split(",")]:
        Hh = torch.randn(bs, d, device=dev, dtype=torch.bfloat16)
        Hf = Hh.float()

        def dense():
            return Hh @ WUt  # [bs, V]

        def stage1():
            # BATCHED slab-fused shadow bound: aq read once across the batch (like
            # dense reuses W_U); outputs U_b[bs,C] directly (slab-max fused).
            Qb = Hf @ B  # [bs, r]
            rho = (Hf - Qb @ B.t()).norm(dim=1)  # [bs]
            _ = shadow_slab_bounds(aq, scale, delta, Qb, rho, 0.0087, S, aq_l2=aq_l2)

        def fetch_est():
            # fixed-budget exact fetch proxy: open k_fetch slabs, GEMV their rows
            for i in range(bs):
                _ = W_U[: k_fetch * S].float() @ Hf[i]

        t_dense = cuda_time(dense)
        t_stage1 = cuda_time(stage1)
        t_fetch = cuda_time(fetch_est)
        full = t_stage1["p50"] + t_fetch["p50"]
        rows[bs] = {
            "dense_p50": t_dense["p50"],
            "dense_p95": t_dense["p95"],
            "stage1_p50": t_stage1["p50"],
            "stage1_p95": t_stage1["p95"],
            "fetch_est_p50": t_fetch["p50"],
            "full_est_p50": full,
            "speedup_stage1": t_dense["p50"] / t_stage1["p50"],
            "speedup_full": t_dense["p50"] / full,
        }
        print(
            f"  B={bs:>2}  dense={t_dense['p50']:.3f}ms (p95 {t_dense['p95']:.3f})  "
            f"stage1={t_stage1['p50']:.3f}ms ({t_dense['p50']/t_stage1['p50']:.2f}x)  "
            f"fetch~{t_fetch['p50']:.3f}  full~{full:.3f}ms "
            f"({t_dense['p50']/full:.2f}x)",
            flush=True,
        )

    Path(args.out).write_text(
        json.dumps(
            {"V": V, "d": d, "r": r, "C": C, "fetched_frac": args.fetched_frac,
             "by_batch": rows}, indent=2)
    )
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
