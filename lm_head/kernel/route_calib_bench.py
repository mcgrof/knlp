#!/usr/bin/env python3
"""Calibrate + benchmark predictive routing for the certified decode (Codex thread
019eb249). Phase 1: per real token collect rho/||h||, the U_b top-K gap gK, and the
EXACT certified fetch fraction; label hard = fetch_frac > f_bad; sweep (tau_rho,
tau_gap) for <0.5% false negatives (a missed hard token recreates the tail) at
lowest simulated latency. Phase 2: time certified_decode_routed with the chosen
thresholds vs the dense head and the plain wave+fallback path.

Latency model for calibration (measured constants, ms): easy certified 1.06,
stage1 0.48, dense 1.68; a hard token that slips into the certified path hits the
tail ~5.0. Routing: rho_veto -> dense (1.68, no stage1); gap_route -> stage1+dense
(2.16); certified easy -> 1.06; certified hard-slip (false neg) -> 5.0.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import (
    certified_decode,
    certified_decode_routed,
    shadow_upper_bound,
)

T_EASY, T_S1, T_DENSE, T_TAIL = 1.06, 0.48, 1.68, 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-cal", type=int, default=1000)
    ap.add_argument("--n-bench", type=int, default=400)
    ap.add_argument("--f-bad", type=float, default=0.15)
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--tau-gap", type=float, default=None, help="override (skip search)")
    ap.add_argument("--tau-rho", type=float, default=None)
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
    WUt = W_U.t().contiguous()
    nC = min(args.n_cal, H.shape[0])
    K = min(args.K, C)
    print(f"V={V} S={S} K={K} f_bad={args.f_bad} cal={nC}", flush=True)

    # ---- Phase 1: collect features + exact fetch fraction ----
    rho_n, gKs, fracs = [], [], []
    for i in range(nC):
        h = H[i]
        hf = h.float()
        q = hf @ B
        hn = torch.sqrt((hf * hf).sum()).clamp_min(1e-9)
        rho = torch.sqrt((hf * hf).sum() - (q * q).sum()).clamp_min(0)
        U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err)
        Ub = U.view(C, S).amax(1)
        Ub_s, _ = Ub.sort(descending=True)
        _, fetched, _ = certified_decode(h, B, aq, scale, delta, W_U, S, aq_err)
        rho_n.append(float(rho / hn))
        gKs.append(float(Ub_s[0] - Ub_s[K - 1]))
        fracs.append(fetched / V)
    rho_n = torch.tensor(rho_n)
    gKs = torch.tensor(gKs)
    fracs = torch.tensor(fracs)
    hard = fracs > args.f_bad
    n_hard = int(hard.sum())
    print(f"  hard tokens (fetch>{args.f_bad}): {n_hard}/{nC} ({100*n_hard/nC:.1f}%)",
          flush=True)

    # ---- sweep (tau_rho, tau_gap) ----
    def sim_latency(rt, gt_):
        lat = torch.empty(nC)
        veto = rho_n >= rt
        gap = (~veto) & (gKs <= gt_)
        cert = ~veto & ~gap
        lat[veto] = T_DENSE
        lat[gap] = T_S1 + T_DENSE
        easy = cert & (~hard)
        slip = cert & hard               # false negatives -> tail
        lat[easy] = T_EASY
        lat[slip] = T_TAIL
        fn = int(slip.sum())
        return lat.mean().item(), fn, int(veto.sum()), int(gap.sum())

    def finite(xs):
        return [x for x in xs if x == x and abs(x) != float("inf")]

    if args.tau_gap is not None:  # override: skip the search, use given thresholds
        tau_rho = args.tau_rho if args.tau_rho is not None else float("inf")
        tau_gap = args.tau_gap
        mlat, _, nv, ng = sim_latency(tau_rho, tau_gap)
        fn_frac = (((rho_n < tau_rho) & (gKs > tau_gap) & hard).sum() / nC).item()
    else:
        rho_cands = [float("inf")] + finite(
            torch.quantile(rho_n, torch.linspace(0.80, 0.995, 10)).tolist())
        gap_cands = finite(
            torch.quantile(gKs, torch.linspace(0.01, 0.60, 40)).tolist())
        FN_MAX = 0.025  # practical target (the gK predictor floors ~2% FN)
        best = None
        for rt in rho_cands:
            for gt_ in gap_cands:
                mlat, fn, nv, ng = sim_latency(rt, gt_)
                if fn / nC <= FN_MAX and (best is None or mlat < best[0]):
                    best = (mlat, rt, gt_, fn / nC, nv, ng)
        if best is None:
            for rt in rho_cands:
                for gt_ in gap_cands:
                    mlat, fn, nv, ng = sim_latency(rt, gt_)
                    if best is None or mlat < best[0]:
                        best = (mlat, rt, gt_, fn / nC, nv, ng)
        mlat, tau_rho, tau_gap, fn_frac, nv, ng = best
    print(f"  chosen: tau_rho={tau_rho:.4f} tau_gap={tau_gap:.4f}  "
          f"sim_mean={mlat:.3f}ms FN={fn_frac*100:.2f}% "
          f"veto={100*nv/nC:.0f}% gap={100*ng/nC:.0f}%", flush=True)

    # ---- Phase 2: real timing with chosen thresholds ----
    def timed(fn):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record(); out = fn(); e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e), out

    def dist(ts):
        ts = sorted(ts)
        return {"mean": sum(ts) / len(ts), "p50": statistics.median(ts),
                "p95": ts[int(0.95 * len(ts)) - 1], "p99": ts[int(0.99 * len(ts)) - 1]}

    nB = min(args.n_bench, H.shape[0])
    for _ in range(5):
        _ = H[0].to(W_U.dtype) @ WUt
        _ = certified_decode_routed(H[0], B, aq, scale, delta, W_U, S, aq_err, WUt,
                                    tau_gap=tau_gap, tau_rho=tau_rho, K=K)
    torch.cuda.synchronize()
    dense_ts = [timed(lambda: H[i].to(W_U.dtype) @ WUt)[0] for i in range(nB)]
    rt_ts, ok, routes = [], 0, {"rho_veto": 0, "gap_route": 0, "certified": 0}
    for i in range(nB):
        t, out = timed(lambda: certified_decode_routed(
            H[i], B, aq, scale, delta, W_U, S, aq_err, WUt,
            tau_gap=tau_gap, tau_rho=tau_rho, K=K))
        rt_ts.append(t)
        ok += int(out[0] == int(gt[i]))
        routes[out[1]] += 1
    de, rd = dist(dense_ts), dist(rt_ts)
    print(f"[dense ] mean={de['mean']:.3f} p50={de['p50']:.3f} p95={de['p95']:.3f}",
          flush=True)
    print(f"[routed] mean={rd['mean']:.3f} p50={rd['p50']:.3f} p95={rd['p95']:.3f} "
          f"p99={rd['p99']:.3f}  match={ok/nB:.3f}  "
          f"dense {de['p50']/rd['mean']:.2f}x", flush=True)
    print(f"  routes: {routes}", flush=True)

    Path(args.out).write_text(json.dumps({
        "meta": meta, "f_bad": args.f_bad, "K": K,
        "n_hard_frac": n_hard / nC,
        "tau_rho": tau_rho, "tau_gap": tau_gap, "sim_mean_ms": mlat,
        "fn_frac": fn_frac, "veto_frac": nv / nC, "gap_frac": ng / nC,
        "dense_ms": de, "routed_ms": rd, "argmax_match": ok / nB, "routes": routes,
    }, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
