#!/usr/bin/env python3
"""Validate + benchmark the Triton certified-decode kernel against the dense head.

Loads the qwen-7b shadow artifact (gen_artifact.py), then:
  1. correctness: kernel upper bound == torch reference; bound is VALID
     (U_v >= true logit w_v.h for every v on sampled tokens);
  2. losslessness: certified_decode argmax == dense argmax (gt) for all tokens,
     and mean fetched-row fraction (should match the replay's ~6.7% at r=1280);
  3. latency: single-token dense GEMV vs the shadow-bound kernel + greedy fetch;
  4. bytes: dense (V*d*bf16) vs shadow (V*r*int8 + delta + fetched*V*d*bf16).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from certdecode_kernel import (
    certified_decode,
    shadow_upper_bound,
    shadow_upper_bound_ref,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-latency", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    A = Path(args.artifact)
    meta = json.loads((A / "meta.json").read_text())
    V, d, r, C = meta["V"], meta["d"], meta["r"], meta["C"]
    S = V // C
    aq = torch.load(A / "aq.pt").to(dev)
    scale = torch.load(A / "scale.pt").to(dev)
    delta = torch.load(A / "delta.pt").to(dev)
    B = torch.load(A / "B.pt").to(dev)
    cof = torch.load(A / "cof.pt").to(dev)
    H = torch.load(A / "H.pt").to(dev)
    gt = torch.load(A / "gt.pt").to(dev)
    W_U = torch.load(A / "W_U.pt").to(dev)
    aq_err = meta["aq_err_norm"]
    print(f"V={V} d={d} r={r} C={C} S={S} N={H.shape[0]}", flush=True)

    # idblock must be contiguous equal slabs for the view-based slab max
    cof_expected = torch.arange(V, device=dev) // S
    assert torch.equal(cof, cof_expected.to(cof.dtype)), "idblock not contiguous S"

    # --- 1. correctness: kernel == ref, bound valid ---
    h0 = H[0].float()
    q0 = h0 @ B
    rho0 = (h0 - q0 @ B.t()).norm()
    U_k = shadow_upper_bound(aq, scale, delta, q0, rho0, aq_err)
    U_r = shadow_upper_bound_ref(aq, scale, delta, q0, rho0, aq_err)
    max_abs = (U_k - U_r).abs().max().item()
    rel = max_abs / (U_r.abs().max().item() + 1e-9)
    # bound validity on a few tokens: U_v >= true logit for all v
    n_valid = min(8, H.shape[0])
    worst_violation = -1e30
    for i in range(n_valid):
        hi = H[i].float()
        qi = hi @ B
        rhoi = (hi - qi @ B.t()).norm()
        Ui = shadow_upper_bound(aq, scale, delta, qi, rhoi, aq_err)
        true_logits = W_U.float() @ hi
        viol = (true_logits - Ui).max().item()  # should be <= 0
        worst_violation = max(worst_violation, viol)
    print(
        f"[correctness] kernel-vs-ref maxabs={max_abs:.3e} rel={rel:.3e}  "
        f"worst bound violation={worst_violation:.3e} (<=0 = valid)",
        flush=True,
    )

    # --- 2. losslessness + fetched fraction ---
    n = min(args.n_eval, H.shape[0])
    ok = 0
    fetched_fracs = []
    for i in range(n):
        bid, fetched, _ = certified_decode(
            H[i], B, aq, scale, delta, W_U, S, aq_err, use_kernel=True
        )
        ok += int(bid == int(gt[i]))
        fetched_fracs.append(fetched / V)
    ff = torch.tensor(fetched_fracs)
    argmax_match = ok / n
    print(
        f"[lossless] argmax_match={argmax_match:.4f}  fetched%: "
        f"mean={ff.mean()*100:.2f} median={ff.median()*100:.2f} "
        f"p95={ff.quantile(0.95)*100:.2f}",
        flush=True,
    )

    # --- 3. latency: dense GEMV vs shadow kernel + greedy fetch ---
    def timeit(fn, iters):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1e3  # ms

    WUt = W_U.t().contiguous()  # [d,V] bf16 dense head
    hbf = H[0].to(W_U.dtype)

    def dense_gemv():
        return hbf @ WUt

    def shadow_stage1():
        q = H[0].float() @ B
        rho = (H[0].float() - q @ B.t()).norm()
        U = shadow_upper_bound(aq, scale, delta, q, rho, aq_err)
        return U.view(C, S).amax(1)

    def shadow_full():
        return certified_decode(
            H[0], B, aq, scale, delta, W_U, S, aq_err, use_kernel=True
        )

    it = args.n_latency
    t_dense = timeit(dense_gemv, it)
    t_stage1 = timeit(shadow_stage1, it)
    t_full = timeit(shadow_full, it)
    print(
        f"[latency ms] dense_gemv={t_dense:.3f}  shadow_stage1={t_stage1:.3f}  "
        f"shadow_full(cert)={t_full:.3f}  speedup_vs_dense={t_dense/t_full:.2f}x",
        flush=True,
    )

    # --- 4. bytes ---
    dense_bytes = V * d * 2
    shadow_bytes = V * r * 1 + V * 4  # aq int8 + delta fp32
    fetch_bytes = float(ff.mean()) * V * d * 2
    total_bytes = shadow_bytes + fetch_bytes
    print(
        f"[bytes/token] dense={dense_bytes/1e6:.0f}MB shadow={shadow_bytes/1e6:.0f}MB "
        f"fetch={fetch_bytes/1e6:.0f}MB total={total_bytes/1e6:.0f}MB "
        f"ratio={total_bytes/dense_bytes*100:.1f}%",
        flush=True,
    )

    result = {
        "meta": meta,
        "S": S,
        "kernel_vs_ref_maxabs": max_abs,
        "worst_bound_violation": worst_violation,
        "argmax_match": argmax_match,
        "fetched_mean": float(ff.mean()),
        "fetched_p95": float(ff.quantile(0.95)),
        "latency_ms": {
            "dense_gemv": t_dense,
            "shadow_stage1": t_stage1,
            "shadow_full_cert": t_full,
            "speedup_vs_dense": t_dense / t_full,
        },
        "bytes_per_token": {
            "dense": dense_bytes,
            "shadow": shadow_bytes,
            "fetch": fetch_bytes,
            "total": total_bytes,
            "ratio": total_bytes / dense_bytes,
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
