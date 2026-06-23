#!/usr/bin/env python3
"""Stage-2 refinement: fused gather-GEMV vs the per-wave host loop.

The per-wave loop in certified_decode_waves syncs the device every wave
(bool(ell > ...) / bool(m > ...)) and materializes W_U[idx].float() (row-copy +
bf16->fp32 cast) per wave -- launch/sync/copy-bound, not byte-bound (14B median
7 ms for ~7% bytes = ~0.14 ms of actual reads). This bench times the sync-free
one-shot path (fixed top-K budget, ONE fused gather-GEMV, ONE cert check, dense
fallback) two ways -- fused Triton gather (reads W_U rows directly, no copy) and
a torch gather baseline -- vs dense and the old wave loop. Reports latency
distribution, lossless argmax match, fetched fraction, and the fallback rate
(tokens that did not certify within the budget) so the byte/latency trade is
explicit. Sweeps k_frac.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import (
    certified_decode_oneshot,
    certified_decode_waves,
)


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
    ap.add_argument("--kfracs", default="0.06,0.09,0.125,0.18")
    ap.add_argument("--wave", type=int, default=4)
    ap.add_argument("--fallback", type=float, default=0.20)
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
    kfracs = [float(x) for x in args.kfracs.split(",")]
    print(f"V={V} d={d} S={S} C={C} tokens={n} kfracs={kfracs}", flush=True)

    # warmup all paths
    for _ in range(6):
        _ = H[0].to(W_U.dtype) @ WUt
        _ = certified_decode_waves(H[0], B, aq, scale, delta, W_U, S, aq_err,
                                   args.wave, fallback_frac=args.fallback, WUt=WUt)
        _ = certified_decode_oneshot(H[0], B, aq, scale, delta, W_U, S, aq_err,
                                     WUt, k_frac=0.125, fused=True)
    torch.cuda.synchronize()

    # dense baseline
    dense_ts = []
    for i in range(n):
        h = H[i].to(W_U.dtype)
        t, _ = time_token(lambda: h @ WUt)
        dense_ts.append(t)
    dense = dist(dense_ts)
    print(f"[dense] mean={dense['mean']:.3f} p50={dense['p50']:.3f} "
          f"p95={dense['p95']:.3f}", flush=True)

    # old wave loop (reference for the regression we are fixing)
    wt, wok = [], 0
    for i in range(n):
        t, out = time_token(lambda: certified_decode_waves(
            H[i], B, aq, scale, delta, W_U, S, aq_err, args.wave,
            fallback_frac=args.fallback, WUt=WUt))
        wt.append(t)
        wok += int(out[0] == int(gt[i]))
    wd = dist(wt)
    print(f"[wave{args.wave} fb{args.fallback}] mean={wd['mean']:.3f} "
          f"p50={wd['p50']:.3f} p99={wd['p99']:.3f} match={wok/n:.3f} "
          f"dense {dense['p50']/wd['mean']:.2f}x", flush=True)

    res = {"model": meta["model"], "n": n, "S": S, "C": C,
           "dense_ms": dense,
           "wave_ms": wd, "wave_match": wok / n,
           "oneshot": {}}

    for fused in (True, False):
        tag = "fused" if fused else "torch"
        for kf in kfracs:
            ts, ok, ncert, fetched = [], 0, 0, 0
            for i in range(n):
                t, out = time_token(lambda: certified_decode_oneshot(
                    H[i], B, aq, scale, delta, W_U, S, aq_err, WUt,
                    k_frac=kf, fused=fused))
                ts.append(t)
                ok += int(out[0] == int(gt[i]))
                ncert += int(out[2])
                fetched += out[1]
            dd = dist(ts)
            key = f"{tag}_k{kf}"
            res["oneshot"][key] = {
                "ms": dd, "argmax_match": ok / n,
                "cert_rate": ncert / n, "fetched_frac_mean": fetched / n / V,
                "speedup_vs_dense_mean": dense["p50"] / dd["mean"],
                "speedup_vs_dense_median": dense["p50"] / dd["p50"]}
            print(f"[1shot {tag} k{kf}] mean={dd['mean']:.3f} p50={dd['p50']:.3f} "
                  f"p99={dd['p99']:.3f} match={ok/n:.3f} cert={ncert/n:.3f} "
                  f"fetch={100*fetched/n/V:.1f}% dense {dense['p50']/dd['mean']:.2f}x "
                  f"(med {dense['p50']/dd['p50']:.2f}x)", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
