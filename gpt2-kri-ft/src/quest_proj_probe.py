#!/usr/bin/env python3
"""Offline projection-vs-Quest-box page-selection probe (the cheap decisive test).

Question: can a low-rank QUERY-PCA projection rank KV pages better than Quest's
coordinate min/max bounding box, given that KEYS are high-rank (our K16/V8
finding -- Qwen keys can't go below FP16)? Codex: certified/lossless ~DOA, but
the bound residual is a product ||q_perp||*||k_perp||, so it survives iff the
QUERY residual is tiny; an approximate projected score (query-PCA basis) is ~35%
to beat the box. This measures both, offline, no decode loop.

Per (layer, kv-head), for NIAH samples (the answer page is known):
  s_true(page) = max_{k in page} q.k                      (exact oracle)
  s_box(page)  = sum_d (q_d>0 ? q_d*kmax_d : q_d*kmin_d)   (Quest min/max box)
  s_proj(page) = max_{k in page} (B^T q).(B^T k)           (query-PCA projection)
B is built per (layer,kv-head) from CALIBRATION decode queries (held-out from
eval). Decode query = the last prompt-position post-RoPE query (drives the answer
token), GQA-aggregated per kv-head. Metric: does the answer page rank in top-K?
recall@K for box vs proj vs oracle. Also reports the query residual
||q - BB^T q|| / ||q|| (does the asymmetry revive the certified bound).

Usage:
  python3 quest_proj_probe.py --model Qwen/Qwen2.5-1.5B --length 2048 \
      --calib 20 --eval 30 --rank 32 --out OUT/probe.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from niah_task import build_context, load_filler_sentences
from niah_evict_perhead import answer_block_indices, get_keys
from diag_perhead_oracle import capture_q


def collect(model, tok, sents, rng, length, needles, n, bs, device,
            store_keys=True):
    """For n samples: per-layer (kl [Hkv,T,D] or None, q_dec [Hkv,D] last-pos
    GQA-mean, ans_blocks set). Calibration only needs the queries, so pass
    store_keys=False to keep the basis-fit set cheap (lets rank exceed n_calib)."""
    out = []
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    group = Hq // Hkv
    for _ in range(n):
        text, spans, needles_l, (qk, qv) = build_context(
            tok, length, needles, sents, rng
        )
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if not ans or max(ans) >= NB:
            continue
        qs, past = capture_q(model, ids_t, [T - 1], device)  # last-pos query
        keys = get_keys(past)
        layers = []
        for li in range(len(keys)):
            kl = keys[li][0].float().cpu() if store_keys else None
            q = qs[li][:, 0, :].view(Hkv, group, -1).mean(1).cpu()  # [Hkv,D]
            layers.append((kl, q))
        out.append({"layers": layers, "ans": set(ans), "T": T, "NB": NB})
        del past
        torch.cuda.empty_cache()
    return out, Hkv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--budgets", default="8,16,32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    bs = args.block_size
    r = args.rank
    budgets = [int(x) for x in args.budgets.split(",")]
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    calib, Hkv = collect(
        model, tok, sents, rng, args.length, args.needles, args.calib, bs, device,
        store_keys=False,
    )
    ev, _ = collect(
        model, tok, sents, rng, args.length, args.needles, args.eval, bs, device
    )
    nL = len(calib[0]["layers"])

    # query-PCA basis B per (layer, kv-head) from calibration last-pos queries
    # (rank is capped at n_calib, so --calib must exceed --rank to test it)
    B = [[None] * Hkv for _ in range(nL)]
    for li in range(nL):
        for hh in range(Hkv):
            Q = torch.stack([s["layers"][li][1][hh] for s in calib], 0).to(device)
            Q = Q - Q.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(Q, full_matrices=False)
            B[li][hh] = Vh[: min(r, Vh.shape[0])].T.contiguous()  # [D,r']

    # metrics
    hit = {m: {K: [0, 0] for K in budgets} for m in ("box", "proj", "oracle")}
    qres = [0.0, 0]  # query residual ||q-BB^Tq||/||q||

    for s in ev:
        T, NB, ans = s["T"], s["NB"], s["ans"]
        idx = torch.arange(T, device=device) // bs
        for li in range(nL):
            kl, qf = s["layers"][li]  # [Hkv,T,D], [Hkv,D]
            for hh in range(Hkv):
                q = qf[hh].to(device)  # [D]
                k = kl[hh].to(device)  # [T,D]
                Bm = B[li][hh]  # [D,r]
                # exact per-token logits, page max
                lg = k @ q  # [T]
                s_true = torch.full((NB,), -1e30, device=device)
                s_true.scatter_reduce_(0, idx, lg, reduce="amax", include_self=True)
                # Quest box
                kmin = torch.full((NB, k.shape[1]), 1e30, device=device)
                kmax = torch.full((NB, k.shape[1]), -1e30, device=device)
                kmin.scatter_reduce_(
                    0,
                    idx.unsqueeze(1).expand_as(k),
                    k,
                    reduce="amin",
                    include_self=True,
                )
                kmax.scatter_reduce_(
                    0,
                    idx.unsqueeze(1).expand_as(k),
                    k,
                    reduce="amax",
                    include_self=True,
                )
                s_box = torch.where(q > 0, q * kmax, q * kmin).sum(-1)  # [NB]
                # projected
                qp = q @ Bm  # [r]
                kp = k @ Bm  # [T,r]
                lgp = kp @ qp  # [T]
                s_proj = torch.full((NB,), -1e30, device=device)
                s_proj.scatter_reduce_(0, idx, lgp, reduce="amax", include_self=True)
                # query residual
                qhat = (q @ Bm) @ Bm.T
                qres[0] += (torch.norm(q - qhat) / (torch.norm(q) + 1e-9)).item()
                qres[1] += 1
                for K in budgets:
                    for name, sc in (
                        ("box", s_box),
                        ("proj", s_proj),
                        ("oracle", s_true),
                    ):
                        top = set(sc.topk(min(K, NB)).indices.tolist())
                        hit[name][K][0] += int(bool(ans & top))
                        hit[name][K][1] += 1

    res = {
        "model": args.model,
        "rank": r,
        "calib": len(calib),
        "eval": len(ev),
        "Hkv": Hkv,
        "nL": nL,
        "query_residual_mean": qres[0] / max(1, qres[1]),
        "answer_page_recall": {},
    }
    print(
        f"[{args.model}] rank={r}  query_residual="
        f"{res['query_residual_mean']:.3f}  (tiny -> certified may survive)"
    )
    print(f"{'budget':8s} {'box':>8s} {'proj':>8s} {'oracle':>8s}")
    for K in budgets:
        row = {}
        for m in ("box", "proj", "oracle"):
            h, n = hit[m][K]
            row[m] = h / max(1, n)
        res["answer_page_recall"][f"K{K}"] = row
        print(f"K{K:<6d} {row['box']:8.3f} {row['proj']:8.3f} {row['oracle']:8.3f}")
    win = all(
        res["answer_page_recall"][f"K{K}"]["proj"]
        > res["answer_page_recall"][f"K{K}"]["box"] + 0.02
        for K in budgets
    )
    print(
        "VERDICT:",
        (
            "projection BEATS box -> pursue"
            if win
            else "projection ~<= box -> kill projection-Quest, use vanilla Quest"
        ),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
