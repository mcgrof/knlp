#!/usr/bin/env python3
"""REAL certified-fetch latency over real hidden states (replaces the fixed-budget
proxy in lmhead_latency.py). Times the actual data-dependent certified greedy
decode -- variable slabs/token, the long p95 tail, non-contiguous opened slabs --
per token, and reports the latency DISTRIBUTION (mean/p50/p95/p99), not a single
fixed-budget number. Also a batched-union pass: per-token opened sets, the union
across the batch, one gather-GEMV.

Uses the qwen-7b artifact (real H, true W_U, shadow head). Needs a CLEAN GPU
(timing under contention is meaningless). rho uses sqrt(||h||^2 - ||q||^2) (B
orthonormal) per Codex -- one basis read, not two.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import shadow_upper_bound


def cuda_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts


def pct(ts, p):
    return ts[min(len(ts) - 1, int(p * len(ts)))]


@torch.no_grad()
def certified_decode_timed(h, B, aq, scale, delta, W_U, S, aq_err, dh):
    """One token: stage-1 bound + REAL data-dependent greedy fetch. Returns
    (best_id, fetched_rows, n_slabs). rho via sqrt(||h||^2-||q||^2)."""
    V, _ = W_U.shape
    C = V // S
    hf = h.float()
    q = hf @ B
    rho = torch.sqrt((hf * hf).sum() - (q * q).sum()).clamp_min(0)
    U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err)
    U_b = U.view(C, S).amax(1)
    order = U_b.argsort(descending=True)
    ell = torch.tensor(float("-inf"), device=h.device)
    best, fetched, nsl = -1, 0, 0
    for i in range(C):
        b = int(order[i])
        if ell > U_b[b]:
            break
        lo = b * S
        logits = W_U[lo:lo + S].float() @ hf
        m, j = logits.max(0)
        if m > ell:
            ell = m
            best = lo + int(j)
        fetched += S
        nsl += 1
    return best, fetched, nsl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=400)
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
    H = torch.load(A / "H.pt").to(dev)
    W_U = torch.load(A / "W_U.pt").to(dev)
    aq_err = meta["aq_err_norm"]
    n = min(args.n_tokens, H.shape[0])
    WUt = W_U.t().contiguous()
    print(f"V={V} d={d} r={r} S={S} tokens={n}", flush=True)

    # dense per-token head GEMV latency (the baseline)
    h0 = H[0].to(W_U.dtype)
    dense_ts = cuda_ms(lambda: h0 @ WUt, 10, 100)
    dense_p50 = statistics.median(dense_ts)

    # REAL certified decode per token: latency + fetched fraction distribution
    lat, frac, nslabs = [], [], []
    for i in range(n):
        h = H[i]
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        _, fetched, nsl = certified_decode_timed(
            h, B, aq, scale, delta, W_U, S, aq_err, d)
        e.record()
        torch.cuda.synchronize()
        lat.append(s.elapsed_time(e))
        frac.append(fetched / V)
        nslabs.append(nsl)
    lat.sort()
    frac.sort()

    res = {
        "model": meta["model"], "V": V, "r": r, "S": S, "n_tokens": n,
        "dense_head_p50_ms": dense_p50,
        "certified_real_fetch": {
            "lat_ms": {"mean": sum(lat) / len(lat), "p50": pct(lat, 0.50),
                       "p95": pct(lat, 0.95), "p99": pct(lat, 0.99)},
            "fetched_frac": {"mean": sum(frac) / len(frac), "p50": pct(frac, 0.50),
                             "p95": pct(frac, 0.95), "p99": pct(frac, 0.99)},
            "slabs_mean": sum(nslabs) / len(nslabs),
        },
    }
    lm = res["certified_real_fetch"]["lat_ms"]
    fm = res["certified_real_fetch"]["fetched_frac"]
    res["speedup_vs_dense"] = {
        "at_mean": dense_p50 / lm["mean"],
        "at_p95": dense_p50 / lm["p95"],
    }
    print(f"[dense head] p50={dense_p50:.3f} ms", flush=True)
    print(f"[certified REAL fetch] lat ms mean={lm['mean']:.3f} p50={lm['p50']:.3f} "
          f"p95={lm['p95']:.3f} p99={lm['p99']:.3f}", flush=True)
    print(f"  fetched frac mean={fm['mean']*100:.1f}% p50={fm['p50']*100:.1f}% "
          f"p95={fm['p95']*100:.1f}% p99={fm['p99']*100:.1f}%", flush=True)
    print(f"  head-only speedup vs dense: mean {dense_p50/lm['mean']:.2f}x  "
          f"p95 {dense_p50/lm['p95']:.2f}x", flush=True)

    # batched union: per-token opened sets, union across the batch, one gather GEMV
    batched = {}
    for bs in [2, 4, 8, 16]:
        if bs > n:
            break
        Hb = H[:bs]
        # compute each token's opened slab set (host greedy, GPU bound)
        union = set()
        per_tok = []
        for i in range(bs):
            _, _, _ = (None, None, None)
            hf = Hb[i].float()
            q = hf @ B
            rho = torch.sqrt((hf * hf).sum() - (q * q).sum()).clamp_min(0)
            U = shadow_upper_bound(aq, scale, delta, q, float(rho), aq_err)
            U_b = U.view(C, S).amax(1)
            order = U_b.argsort(descending=True)
            ell = torch.tensor(float("-inf"), device=dev)
            opened = []
            for k in range(C):
                bb = int(order[k])
                if ell > U_b[bb]:
                    break
                lo = bb * S
                m = (W_U[lo:lo + S].float() @ hf).max()
                ell = torch.maximum(ell, m)
                opened.append(bb)
            per_tok.append(len(opened))
            union.update(opened)
        union_frac = len(union) * S / V
        per_tok_frac = sum(per_tok) * S / (bs * V)
        batched[bs] = {
            "per_token_frac_mean": per_tok_frac,
            "union_frac": union_frac,
            "union_over_pertoken": union_frac / max(per_tok_frac, 1e-9),
        }
        print(f"  batch {bs:>2}: per-token fetch {per_tok_frac*100:.1f}%  "
              f"UNION {union_frac*100:.1f}%  (union/pertok {union_frac/max(per_tok_frac,1e-9):.2f})",
              flush=True)
    res["batched_union"] = batched

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
