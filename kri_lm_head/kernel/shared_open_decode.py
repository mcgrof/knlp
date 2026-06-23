#!/usr/bin/env python3
"""Shared-open adaptive all-certify batched LM-head decode (ChatGPT-Pro v2 design).

Eager reference (correctness first; CUDA-graph static-ladder version is the next
step). One opened slab set for the whole batch; when a slab opens it is computed
for ALL B tokens via the slab-major max+argmax kernel (no union materialization,
direct idblock slab addressing). Per-row aq_err. NO dense fallback in the common
path -- adaptive expansion until every token certifies (strict >). Lossless: the
returned argmax is the exact dense argmax; we verify false_cert == 0.

Stage 0: q = H@B, rho = ||H - qBᵀ|| (explicit residual).
Stage 1: Ub[B,C] = slab-max of  q·dequant(aq) + rho·delta + ||q||·aq_err_up[v].
Round r: urgency G[s] = max over active b of (Ub[b,s] if Ub[b,s] >= m_b else -inf);
         open top-L unopened by G; slab_maxarg over the NEW slabs for all B tokens;
         update incumbent m_b/argmax; certify m_b > max unopened Ub[b,*].
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from certdecode_kernel import slab_maxarg

LADDER = [4, 4, 8, 8, 8, 16, 16, 32, 32, 64, 256]


@torch.no_grad()
def shared_open_decode(Hb, Bb, a_q, delta, aq_err_row, W_U, S, ladder=LADDER):
    dev = Hb.device
    Bn, d = Hb.shape
    V = W_U.shape[0]
    C = V // S
    NEG = torch.tensor(-1e30, device=dev)
    Hf = Hb.float()
    q = Hf @ Bb                                    # [B,r]
    rho = (Hf - q @ Bb.t()).norm(dim=1)            # [B] explicit residual
    qn = q.norm(dim=1)
    # Ub[B,C] (full [B,V] bound then slab-max; B<=64 so ~39 MB at most)
    U = q @ a_q.t() + rho[:, None] * delta[None, :] + qn[:, None] * aq_err_row[None, :]
    Ub = U.view(Bn, C, S).amax(2)                  # [B,C]
    opened = torch.zeros(C, dtype=torch.bool, device=dev)
    m_b = torch.full((Bn,), -1e30, device=dev)
    best_id = torch.zeros(Bn, dtype=torch.long, device=dev)
    cert = torch.zeros(Bn, dtype=torch.bool, device=dev)
    arangeC = torch.arange(C, device=dev)
    for L in ladder:
        if bool(cert.all()):
            break
        active = ~cert
        cand = torch.where((Ub >= m_b[:, None]) & active[:, None], Ub,
                           Ub.new_full((), -1e30))
        G = cand.max(0).values
        G = torch.where(opened, G.new_full((), -1e30), G)
        navail = int((~opened).sum())
        nopen = min(L, navail)
        if nopen <= 0:
            break
        top = G.topk(nopen).indices.to(torch.int32)    # [nopen] new slabs
        opened[top.long()] = True
        maxv, argr = slab_maxarg(W_U, top, Hb, S)       # [nopen,B], [nopen,B]
        rbest, rslab = maxv.max(0)                       # [B] over new slabs
        rrow = argr.gather(0, rslab[None, :]).squeeze(0).long()
        rid = top[rslab].long() * S + rrow               # [B] vocab id
        upd = rbest > m_b
        m_b = torch.where(upd, rbest, m_b)
        best_id = torch.where(upd, rid, best_id)
        rem = torch.where(opened[None, :], Ub.new_full((), -1e30), Ub).max(1).values
        cert = m_b > rem
    return best_id, cert, int(opened.sum())


def t_evt(fn, warmup=10, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return {"mean": sum(ts) / len(ts), "p50": statistics.median(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--r", type=int, default=640)
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--seqs", type=int, default=24)
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--n-pools", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from predictor_baseline import load_split, capture
    from shadow_bound_replay import build_basis, shadow_precompute, quantize_cols

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
    a_q, _ = quantize_cols(a_full.clone(), 8)
    aq_err_row = (a_full - a_q).norm(dim=1)
    WUt = W_U.t().contiguous()
    H = H.to(dev)
    N = H.shape[0]
    batches = [int(x) for x in args.batches.split(",")]
    print(f"[{args.model}] V={V} d={d} C={C} S={S} r={args.r} N={N}", flush=True)

    res = {"model": args.model, "r": args.r, "C": C, "S": S, "rows": []}
    for Bn in batches:
        pools = [H[(i * Bn) % N: (i * Bn) % N + Bn] for i in range(args.n_pools)]
        pools = [p for p in pools if p.shape[0] == Bn]
        # warm
        for _ in range(3):
            shared_open_decode(pools[0], Bb, a_q, delta, aq_err_row, W_U, S)
        dense = t_evt(lambda: pools[0].to(W_U.dtype) @ WUt)
        cert_d = t_evt(lambda: shared_open_decode(pools[0], Bb, a_q, delta,
                                                  aq_err_row, W_U, S)[0])
        # correctness over all pools
        false_cert = okf = tot = uni = ncert = 0
        for p in pools:
            ids_b, cert_b, nop = shared_open_decode(p, Bb, a_q, delta, aq_err_row, W_U, S)
            gt = (p.float() @ WUt.float()).argmax(1)
            false_cert += int(((ids_b != gt) & cert_b).sum())
            okf += int((ids_b == gt).sum())
            ncert += int(cert_b.sum())
            uni += nop
            tot += p.shape[0]
        np_ = len(pools)
        row = {"batch": Bn, "dense_ms": dense, "cert_ms": cert_d,
               "speedup_mean": dense["mean"] / cert_d["mean"],
               "speedup_p50": dense["p50"] / cert_d["p50"],
               "union_frac": uni / np_ / C, "match": okf / tot,
               "cert_rate": ncert / tot, "false_cert": false_cert}
        res["rows"].append(row)
        print(f"[B={Bn:>2}] dense {dense['mean']:.3f} cert {cert_d['mean']:.3f} "
              f"({row['speedup_mean']:.2f}x) union {100*row['union_frac']:.1f}% "
              f"match {row['match']:.3f} false_cert {false_cert}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
