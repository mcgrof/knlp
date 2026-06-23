#!/usr/bin/env python3
"""Batched certified-decode bench (Codex-reviewed H100 design).

Sweeps batch size and times the dense head GEMM vs the certified decode that
produces all B argmaxes, measuring BOTH per-token requested fetch AND the actual
batched-UNION fetch (different tokens open different slabs -> the batch gather
reads their union, which erodes the byte advantage as B grows). Reports
match_fp32 (vs the fp32 dense argmax) and match_deployed (vs the deployed-dtype
dense argmax) separately, since the dense fallback runs in the deployed dtype.

Certified batch path: stage-1 batched shadow bounds U_b[B,C] (fused bf16-WMMA
kernel) -> per token take top-K slabs by bound -> UNION across the batch -> one
fused gather-GEMM over the union rows -> per token restrict to its own opened
slabs for the incumbent and certify m_b > Ub_sorted_b[K]; uncertified tokens
fall back to the dense argmax (lossless). This is the fixed-budget one-shot,
batched.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import shadow_slab_bounds, fused_gather_gemm


def evt_time(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"mean": sum(ts) / len(ts), "p50": statistics.median(ts),
            "p99": ts[int(0.99 * len(ts)) - 1]}


@torch.no_grad()
def certified_batch(Hb, B, aq, scale, delta, W_U, WUt, S, aq_err, aq_l2,
                    k_frac, arangeS, gemm="fused"):
    """Return (ids[B], requested_rows, union_rows). One fused union GEMM."""
    dev = Hb.device
    Bn, d = Hb.shape
    V = W_U.shape[0]
    C = V // S
    Hf = Hb.float()
    Q = Hf @ B                                   # [B,r]
    resid = Hf - Q @ B.t()
    rho = resid.norm(dim=1)                      # [B] exact residual
    Ub = shadow_slab_bounds(aq, scale, delta, Q, rho, aq_err, S, aq_l2=aq_l2)  # [B,C]
    Ub_sorted, order = Ub.sort(dim=1, descending=True)
    K = max(1, min(C, int(-(-k_frac * C // 1))))
    top = order[:, :K]                           # [B,K] slab ids per token
    union = torch.unique(top)                    # [U] slabs to fetch
    urows = (union.unsqueeze(1) * S + arangeS.unsqueeze(0)).reshape(-1)  # [U*S]
    if gemm == "fused":                          # no materialization (tensor-core)
        logits_u = fused_gather_gemm(W_U, urows, Hb)   # [B, U*S] bf16-precise
    else:                                        # torch: materialize + cast (slow)
        logits_u = Hf @ W_U[urows].float().t()   # [B, U*S]
    # map each token's opened slabs to columns in the union block (fully batched)
    slab_pos = torch.full((C,), 0, device=dev, dtype=torch.long)
    slab_pos[union] = torch.arange(union.numel(), device=dev)
    cols = (slab_pos[top].unsqueeze(2) * S + arangeS.view(1, 1, S)).reshape(Bn, K * S)
    vals = logits_u.gather(1, cols)              # [B, K*S] each token's opened logits
    m, jj = vals.max(1)                          # [B]
    slab_in_top = jj // S
    row_in_slab = jj % S
    chosen = top.gather(1, slab_in_top.unsqueeze(1)).squeeze(1)  # [B]
    ids = chosen * S + row_in_slab
    if K < C:
        cert = m > Ub_sorted[:, K]
    else:
        cert = torch.ones(Bn, dtype=torch.bool, device=dev)
    if (~cert).any():                            # dense fallback (deployed dtype)
        dl = Hb[~cert].to(WUt.dtype) @ WUt
        ids = ids.clone()
        ids[~cert] = dl.argmax(1)
    requested = K * S                            # per-token requested rows
    union_rows = int(union.numel()) * S
    return ids, requested, union_rows, int(cert.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--k-frac", type=float, default=0.06)
    ap.add_argument("--gemm", default="fused", choices=["fused", "torch"],
                    help="fused = Triton gather-GEMM (no copy); torch = materialize")
    ap.add_argument("--n-pools", type=int, default=64, help="distinct batches timed")
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
    W_U = torch.load(A / "W_U.pt").to(dev).contiguous()
    gt = torch.load(A / "gt.pt").to(dev)               # fp32 dense argmax
    aq_err = meta["aq_err_norm"]
    WUt = W_U.t().contiguous()
    aq_l2 = aq.float().pow(2).sum(1).sqrt().contiguous()
    arangeS = torch.arange(S, device=dev)
    batches = [int(x) for x in args.batches.split(",")]
    nH = H.shape[0]
    print(f"V={V} d={d} S={S} C={C} dtype={W_U.dtype} batches={batches}", flush=True)

    res = {"model": meta["model"], "S": S, "C": C, "k_frac": args.k_frac,
           "wu_dtype": str(W_U.dtype), "rows": []}
    for Bn in batches:
        pools = [H[(i * Bn) % nH:(i * Bn) % nH + Bn] for i in range(args.n_pools)]
        pools = [p for p in pools if p.shape[0] == Bn]
        # dense deployed latency (one GEMM, amortized W read)
        t_dense = evt_time(lambda: pools[0].to(W_U.dtype) @ WUt)
        # certified batch latency
        t_cert = evt_time(lambda: certified_batch(
            pools[0], B, aq, scale, delta, W_U, WUt, S, aq_err, aq_l2,
            args.k_frac, arangeS)[0])
        # correctness + fetch over all pools
        req = uni = okf = okd = ncert = tot = 0
        for p in pools:
            ids, rq, ur, nc = certified_batch(
                p, B, aq, scale, delta, W_U, WUt, S, aq_err, aq_l2,
                args.k_frac, arangeS, gemm=args.gemm)
            # fp32 + deployed dense argmax for this pool (per-token ground truth)
            gtf = (p.float() @ W_U.float().t()).argmax(1)
            gtd = (p.to(W_U.dtype) @ WUt).argmax(1)
            okf += int((ids == gtf).sum()); okd += int((ids == gtd).sum())
            req += rq; uni += ur; ncert += nc; tot += p.shape[0]
        np = len(pools)
        row = {
            "batch": Bn, "dense_ms": t_dense, "cert_ms": t_cert,
            "speedup_mean": t_dense["mean"] / t_cert["mean"],
            "speedup_p50": t_dense["p50"] / t_cert["p50"],
            "requested_fetch": req / np / V,          # per-token (= K*S/V)
            "union_fetch": uni / np / V,              # per-batch union
            "cert_rate": ncert / tot,
            "match_fp32": okf / tot, "match_deployed": okd / tot,
        }
        res["rows"].append(row)
        print(f"[B={Bn:>2}] dense {t_dense['mean']:.3f} cert {t_cert['mean']:.3f} "
              f"({row['speedup_mean']:.2f}x) req_fetch {100*row['requested_fetch']:.1f}% "
              f"union_fetch {100*row['union_fetch']:.1f}% cert {row['cert_rate']:.3f} "
              f"m_fp32 {row['match_fp32']:.3f} m_dep {row['match_deployed']:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
