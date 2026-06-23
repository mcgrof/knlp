#!/usr/bin/env python3
"""gamma microbench (ChatGPT-Pro's decisive H100 measurement).

gamma = sparse effective bandwidth / dense effective bandwidth, for the
slab-major exact stage. The certified batch path reads a UNION of L idblock
slabs (each S=594 CONTIGUOUS rows of W_U, ~4 MB) scattered across the vocab,
vs the dense head's one contiguous V*d read. This times a slab-major Triton
kernel (reads each opened slab's rows contiguously, dots with all B tokens,
per-token max) against the dense GEMM, over a sweep of L (union slab counts)
matching the offline replay's u_all_cert. Model-free: W_U is synthetic (only
the bandwidth matters). Plug the measured gamma into tau_shadow + u_all_cert/gamma.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _slabmax(wu_ptr, slabs_ptr, h_ptr, out_ptr, L, S, d, Bn,
            BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_B: tl.constexpr):
    li = tl.program_id(0)
    sub = tl.program_id(1)
    slab = tl.load(slabs_ptr + li)
    rloc = sub * BLOCK_R + tl.arange(0, BLOCK_R)
    row = slab * S + rloc
    mask_r = rloc < S
    b = tl.arange(0, BLOCK_B)
    mask_b = b < Bn
    acc = tl.zeros([BLOCK_R, BLOCK_B], dtype=tl.float32)
    for d0 in range(0, d, BLOCK_D):
        dj = d0 + tl.arange(0, BLOCK_D)
        md = dj < d
        w = tl.load(wu_ptr + row[:, None] * d + dj[None, :],
                    mask=mask_r[:, None] & md[None, :], other=0.0)      # [BR,BD]
        h = tl.load(h_ptr + b[:, None] * d + dj[None, :],
                    mask=mask_b[:, None] & md[None, :], other=0.0)      # [BB,BD]
        acc += tl.dot(w, tl.trans(h), out_dtype=tl.float32)            # [BR,BB]
    acc = tl.where(mask_r[:, None], acc, -float("inf"))
    m = tl.max(acc, axis=0)                                            # [BB]
    tl.atomic_max(out_ptr + b, m, mask=mask_b)


def slabmax(W_U, slabs, H, S, BLOCK_R=64, BLOCK_D=64, BLOCK_B=32):
    L = slabs.numel()
    V, d = W_U.shape
    Bn = H.shape[0]
    out = torch.full((Bn,), -float("inf"), device=W_U.device, dtype=torch.float32)
    grid = (L, triton.cdiv(S, BLOCK_R))
    _slabmax[grid](W_U, slabs, H.contiguous(), out, L, S, d, Bn,
                   BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D, BLOCK_B=max(16, BLOCK_B))
    return out


def t_evt(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--V", type=int, default=152064)
    ap.add_argument("--d", type=int, default=3584)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--Ls", default="4,8,16,32,52,64,86,128")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    V, d, C = args.V, args.d, args.C
    S = V // C
    g = torch.Generator(device=dev).manual_seed(0)
    W_U = torch.randn(V, d, device=dev, dtype=torch.bfloat16)
    H = torch.randn(args.batch, d, device=dev, dtype=torch.bfloat16)
    WUt = W_U.t().contiguous()
    print(f"V={V} d={d} C={C} S={S} B={args.batch} headGB={V*d*2/1e9:.2f}", flush=True)

    # dense effective bandwidth (the full contiguous head read)
    t_dense = t_evt(lambda: H @ WUt)
    dense_bytes = V * d * 2
    dense_bw = dense_bytes / (t_dense / 1e3) / 1e9     # GB/s
    print(f"[dense] {t_dense:.3f} ms  {dense_bw:.0f} GB/s", flush=True)

    rows = []
    for L in [int(x) for x in args.Ls.split(",")]:
        slabs = torch.randperm(C, generator=g, device=dev)[:L].to(torch.int32)
        t = t_evt(lambda: slabmax(W_U, slabs, H, S))
        sp_bytes = L * S * d * 2
        sp_bw = sp_bytes / (t / 1e3) / 1e9
        gamma = sp_bw / dense_bw
        union_frac = L * S / V
        rows.append({"L": L, "union_frac": union_frac, "ms": t,
                     "sparse_bw_GBs": sp_bw, "gamma": gamma})
        print(f"L={L:>3} union={100*union_frac:>4.1f}% {t:.3f}ms "
              f"{sp_bw:.0f} GB/s  gamma={gamma:.2f}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"V": V, "d": d, "C": C, "S": S, "batch": args.batch,
         "dense_ms": t_dense, "dense_bw_GBs": dense_bw, "rows": rows}, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
