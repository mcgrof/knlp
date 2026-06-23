#!/usr/bin/env python3
"""Stage-2 fix: wave-based fused gather-GEMV vs the per-slab Python loop.

Times the REAL certified decode three ways over real Qwen-7B hidden states --
dense head, per-slab greedy (the old loop), and wave-based gather-GEMV (open `wave`
slabs/wave, one gathered GEMV) -- and verifies the wave path is still lossless
(argmax == dense gt). Reports latency mean/p50/p95/p99 so the tail (which made the
per-slab mean 2.3x slower than dense) is visible.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import certified_decode_waves, shadow_upper_bound


@torch.no_grad()
def perslab(h, B, aq, scale, delta, W_U, S, aq_err):
    V, d = W_U.shape
    C = V // S
    hf = h.float()
    q = hf @ B
    rho = torch.sqrt((hf * hf).sum() - (q * q).sum()).clamp_min(0)
    U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err)
    U_b = U.view(C, S).amax(1)
    order = U_b.argsort(descending=True)
    ell = torch.tensor(float("-inf"), device=h.device)
    best, fetched = -1, 0
    for i in range(C):
        b = int(order[i])
        if bool(ell > U_b[b]):
            break
        lo = b * S
        m, j = (W_U[lo:lo + S].float() @ hf).max(0)
        if bool(m > ell):
            ell = m
            best = lo + int(j)
        fetched += S
    return best, fetched


def time_token(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out


def dist(ts):
    ts = sorted(ts)
    return {"mean": sum(ts) / len(ts), "p50": statistics.median(ts),
            "p95": ts[int(0.95 * len(ts)) - 1], "p99": ts[int(0.99 * len(ts)) - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=400)
    ap.add_argument("--waves", default="8,16,32")
    ap.add_argument("--fallbacks", default="0.30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    A = Path(args.artifact)
    meta = json.loads((A / "meta.json").read_text())
    V, d, C = meta["V"], meta["d"], meta["C"]
    S = V // C
    aq = torch.load(A / "aq.pt").to(dev)
    scale = torch.load(A / "scale.pt").to(dev)
    delta = torch.load(A / "delta.pt").to(dev)
    B = torch.load(A / "B.pt").to(dev)
    H = torch.load(A / "H.pt").to(dev)
    W_U = torch.load(A / "W_U.pt").to(dev)
    gt = torch.load(A / "gt.pt").to(dev)
    aq_err = meta["aq_err_norm"]
    n = min(args.n_tokens, H.shape[0])
    WUt = W_U.t().contiguous()
    waves = [int(x) for x in args.waves.split(",")]
    print(f"V={V} d={d} S={S} tokens={n} waves={waves}", flush=True)

    # warmup
    for _ in range(5):
        _ = H[0].to(W_U.dtype) @ WUt
        _ = certified_decode_waves(H[0], B, aq, scale, delta, W_U, S, aq_err, 16)
    torch.cuda.synchronize()

    dense_ts = []
    for i in range(n):
        h = H[i].to(W_U.dtype)
        t, _ = time_token(lambda: h @ WUt)
        dense_ts.append(t)
    dense = dist(dense_ts)

    ps_ts, ps_ok = [], 0
    for i in range(n):
        t, out = time_token(lambda: perslab(H[i], B, aq, scale, delta, W_U, S, aq_err))
        ps_ts.append(t)
        ps_ok += int(out[0] == int(gt[i]))
    ps = dist(ps_ts)

    res = {"model": meta["model"], "n": n, "S": S,
           "dense_ms": dense, "perslab_ms": ps,
           "perslab_argmax_match": ps_ok / n, "waves": {}}
    print(f"[dense]   mean={dense['mean']:.3f} p50={dense['p50']:.3f} "
          f"p95={dense['p95']:.3f}", flush=True)
    print(f"[perslab] mean={ps['mean']:.3f} p50={ps['p50']:.3f} p95={ps['p95']:.3f} "
          f"p99={ps['p99']:.3f}  match={ps_ok/n:.3f}  "
          f"vs dense: mean {dense['p50']/ps['mean']:.2f}x", flush=True)

    fbs = [float(x) for x in args.fallbacks.split(",")]
    for w in waves:
        for fb in fbs:
            wt, ok = [], 0
            for i in range(n):
                t, out = time_token(
                    lambda: certified_decode_waves(
                        H[i], B, aq, scale, delta, W_U, S, aq_err, w,
                        fallback_frac=fb, WUt=WUt))
                wt.append(t)
                ok += int(out[0] == int(gt[i]))
            wd = dist(wt)
            res["waves"][f"w{w}_fb{fb}"] = {
                "ms": wd, "argmax_match": ok / n,
                "speedup_vs_dense_mean": dense["p50"] / wd["mean"],
                "speedup_vs_perslab_mean": ps["mean"] / wd["mean"]}
            print(f"[wave {w:>2} fb {fb:.2f}] mean={wd['mean']:.3f} "
                  f"p50={wd['p50']:.3f} p95={wd['p95']:.3f} p99={wd['p99']:.3f}  "
                  f"match={ok/n:.3f}  dense {dense['p50']/wd['mean']:.2f}x  "
                  f"perslab {ps['mean']/wd['mean']:.2f}x", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
