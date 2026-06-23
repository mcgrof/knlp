#!/usr/bin/env python3
"""Bench the batched shared-shortlist certified decode (eager correctness + graph)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from cert_graph_batch import CertDecodeGraphBatch


def t_evt(fn, warmup=15, iters=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return {"mean": sum(ts) / len(ts), "p50": statistics.median(ts), "p99": ts[int(0.99 * len(ts)) - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--r", type=int, default=640)
    ap.add_argument("--lfrac", type=float, default=0.22, help="shortlist size = lfrac*C")
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--n-pools", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from predictor_baseline import load_split, capture
    from shadow_bound_replay import build_basis, shadow_precompute
    from gen_artifact import quantize_cols_codes

    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.to(dev).eval()
    head = model.get_output_embeddings()
    W_U = head.weight.detach().contiguous()
    V, d = W_U.shape
    C = args.clusters
    S = V // C
    ids = load_split(tok, "val", args.seqs, 512)
    H, _, _ = capture(model, head, ids, dev, pos_cap=args.positions)
    del model
    torch.cuda.empty_cache()
    Bb = build_basis("hidden_pca", H, W_U, args.r, dev)
    a_full, delta = shadow_precompute(W_U, Bb, dev)
    codes, scale, _ = quantize_cols_codes(a_full, 8)        # int8 [V,r], scale [r]
    codes = codes.to(dev); scale = scale.to(dev)
    dequant = codes.float() * scale.unsqueeze(0)
    aq_err_row = (a_full - dequant).norm(dim=1).contiguous()  # per-row quant err
    aq_l2 = codes.float().pow(2).sum(1).sqrt().contiguous()
    WUt = W_U.t().contiguous()
    H = H.to(dev)
    N = H.shape[0]
    L = max(8, int(math.ceil(args.lfrac * C)))
    batches = [int(x) for x in args.batches.split(",")]
    print(f"[{args.model}] V={V} d={d} C={C} S={S} r={args.r} L={L} ({100*L/C:.0f}%)", flush=True)

    res = {"model": args.model, "r": args.r, "C": C, "S": S, "L": L, "rows": []}
    for Bn in batches:
        pools = [H[(i * Bn) % N: (i * Bn) % N + Bn] for i in range(args.n_pools)]
        pools = [p for p in pools if p.shape[0] == Bn]
        g = CertDecodeGraphBatch(Bb, codes, scale, delta, aq_err_row, aq_l2,
                                 W_U, S, L, Bn, dev)
        # correctness (eager) + fallback rate
        false_cert = okf = tot = ncert = 0
        for p in pools:
            ids_b, cert_b = g.run_eager(p)
            gt = (p.float() @ WUt.float()).argmax(1)
            false_cert += int(((ids_b != gt) & cert_b).sum())
            okf += int((ids_b == gt).sum())
            ncert += int(cert_b.sum())
            tot += p.shape[0]
        dense = t_evt(lambda: pools[0].to(W_U.dtype) @ WUt)
        graph_ms = None
        try:
            g.capture()
            graph_ms = t_evt(lambda: (g.H_buf.copy_(pools[0]), g.graph.replay()))
        except Exception as ex:
            print(f"  [graph capture FAILED B={Bn}] {type(ex).__name__}: {ex}", flush=True)
        eager_ms = t_evt(lambda: g.run_eager(pools[0]))
        row = {"batch": Bn, "L_frac": L / C, "dense_ms": dense, "eager_ms": eager_ms,
               "graph_ms": graph_ms, "cert_rate": ncert / tot, "match": okf / tot,
               "false_cert": false_cert}
        if graph_ms:
            row["graph_speedup_mean"] = dense["mean"] / graph_ms["mean"]
        res["rows"].append(row)
        gs = f"graph {graph_ms['mean']:.3f} ({dense['mean']/graph_ms['mean']:.2f}x)" if graph_ms else "graph FAIL"
        print(f"[B={Bn:>2}] dense {dense['mean']:.3f} eager {eager_ms['mean']:.3f} {gs} "
              f"cert {ncert/tot:.3f} match {okf/tot:.3f} false_cert {false_cert}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
