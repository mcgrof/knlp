#!/usr/bin/env python3
"""Proper latency sweep for the low-rank-V decode kernel on prune (W7900).

CUDA-event timed (HIP), warmup + iters, p50/p95. Sweeps (n_heads, context T) so
the bandwidth-bound regime is reached -- the 4x V-read reduction only shows up as
latency once there is enough parallel work to saturate memory and enough context
that the V read dominates. Decode = 1 query per head (nq=1). proj stored bf16
(deployable coeff). Synthetic tensors at realistic shapes (latency is shape-driven).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vlowrank_kernel import dense_decode_triton, lowrank_decode


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
        times.append(s.elapsed_time(e))
    times.sort()
    return {
        "p50": statistics.median(times),
        "p95": times[int(0.95 * len(times)) - 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--heads", default="8,32,128")
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--nq", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    dh, r, nq = args.head_dim, args.r, args.nq
    print(f"GPU={torch.cuda.get_device_name(0)} head_dim={dh} r={r} nq={nq}", flush=True)

    rows = []
    for H in [int(x) for x in args.heads.split(",")]:
        for T in [int(x) for x in args.contexts.split(",")]:
            Q = torch.randn(H, nq, dh, device=dev, dtype=torch.bfloat16)
            K = torch.randn(H, T, dh, device=dev, dtype=torch.bfloat16)
            V = torch.randn(H, T, dh, device=dev, dtype=torch.bfloat16)
            proj = torch.randn(H, T, r, device=dev, dtype=torch.bfloat16)
            Bbasis = torch.randn(H, dh, r, device=dev, dtype=torch.float32)

            t_dense = cuda_time(lambda: dense_decode_triton(Q, K, V))
            t_low = cuda_time(lambda: lowrank_decode(Q, K, proj, Bbasis))
            sp = t_dense["p50"] / t_low["p50"]
            # traffic: dense reads K+V (2*T*dh*2); lowrank reads K + proj bf16
            dense_kv = 2 * T * dh * 2
            low_kv = T * dh * 2 + T * r * 2
            rows.append({
                "heads": H, "T": T,
                "dense_p50": t_dense["p50"], "dense_p95": t_dense["p95"],
                "lowrank_p50": t_low["p50"], "lowrank_p95": t_low["p95"],
                "speedup": sp,
                "kv_traffic_ratio": low_kv / dense_kv,
            })
            print(
                f"  H={H:>3} T={T:>6}  dense={t_dense['p50']:.3f}ms "
                f"lowrank={t_low['p50']:.3f}ms  speedup={sp:.2f}x  "
                f"kv-traffic={low_kv/dense_kv*100:.0f}%",
                flush=True,
            )

    Path(args.out).write_text(
        json.dumps({"head_dim": dh, "r": r, "nq": nq, "rows": rows}, indent=2)
    )
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
