#!/usr/bin/env python3
"""Bench the CUDA-graph batch-1 certified decode vs eager one-shot vs dense."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import certified_decode_oneshot
from cert_graph import CertDecodeGraphB1


def t_evt(fn, warmup=20, iters=100):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=400)
    ap.add_argument("--k-frac", type=float, default=0.06)
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
    gt = torch.load(A / "gt.pt").to(dev)
    aq_err = meta["aq_err_norm"]
    WUt = W_U.t().contiguous()
    K = max(1, int(args.k_frac * C))
    n = min(args.n_tokens, H.shape[0])
    print(f"V={V} d={d} S={S} C={C} K={K} tokens={n} dtype={W_U.dtype}", flush=True)

    # dense baseline
    h0 = H[0].to(W_U.dtype)
    dense = t_evt(lambda: h0 @ WUt)

    # eager one-shot (fused gather)
    eager = t_evt(lambda: certified_decode_oneshot(
        H[0], B, aq, scale, delta, W_U, S, aq_err, WUt, k_frac=args.k_frac, fused=True))

    # build the graph
    g = None
    graph_ms = None
    try:
        g = CertDecodeGraphB1(B, aq, scale, delta, W_U, S, aq_err, K, dev).capture()
        graph_ms = t_evt(lambda: (g.h_buf.copy_(H[0]), g.graph.replay()))
    except Exception as ex:
        print(f"[graph capture FAILED] {type(ex).__name__}: {ex}", flush=True)

    # correctness over n tokens (graph end-to-end incl fallback)
    okf = okd = ncert = 0
    if g is not None:
        for i in range(n):
            idd, cert = g(H[i])
            ncert += int(cert)
            okf += int(idd == int(gt[i]))
            gtd = int((H[i].to(W_U.dtype) @ WUt).argmax())
            okd += int(idd == gtd)

    res = {"model": meta["model"], "S": S, "C": C, "K": K, "n": n,
           "dense_ms": dense, "eager_ms": eager, "graph_ms": graph_ms,
           "graph_cert_rate": ncert / n if g is not None else None,
           "graph_match_fp32": okf / n if g is not None else None,
           "graph_match_deployed": okd / n if g is not None else None}
    print(f"[dense]  mean={dense['mean']:.3f} p50={dense['p50']:.3f}", flush=True)
    print(f"[eager]  mean={eager['mean']:.3f} p50={eager['p50']:.3f}  "
          f"dense {dense['p50']/eager['mean']:.2f}x", flush=True)
    if graph_ms is not None:
        print(f"[graph]  mean={graph_ms['mean']:.3f} p50={graph_ms['p50']:.3f}  "
              f"dense {dense['mean']/graph_ms['mean']:.2f}x  (eager "
              f"{eager['mean']/graph_ms['mean']:.2f}x)  cert={ncert/n:.3f} "
              f"m_fp32={okf/n:.3f} m_dep={okd/n:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
