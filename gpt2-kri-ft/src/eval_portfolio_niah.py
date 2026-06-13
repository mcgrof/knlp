#!/usr/bin/env python3
"""Phase-A grid, retrieval side (Gate R): per-head NIAH answer-block retention
for the protected-portfolio compositions.

Reuses the A-minus diagnostic machinery (true post-RoPE Q probes, exact
attention mass, per-head centroids) and composes kept sets with
portfolio.compose for every pf_m*_h* combo at K in {8,16,32,64}. Reference
lists (R_lastk = legacy rel_only, R_recentq) are reported from the same run so
Gate R (portfolio >= 0.9x rel_only) is computed in-protocol.

Usage:
  python3 eval_portfolio_niah.py --model HuggingFaceTB/SmolLM2-360M \
      --length 2048 --eval 20 --seed 0 --out OUT/pf_niah.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from diag_perhead_oracle import (
    block_stats,
    capture_q,
    get_values,
    mass_trueq,
)
from niah_evict_perhead import answer_block_indices, get_keys
from niah_task import build_context, load_filler_sentences
from portfolio import compose, qualified_pool, spec_n

BUDGETS = (8, 16, 32, 64)
M_SPECS = ("0", "1", "f125", "f250", "f500")
H_SPECS = ("0", "1", "f125", "f250")
REFS = ("R_lastk", "R_recentq", "D_resid")


def order_desc(score):  # [H,NB] -> [H,NB] orderings
    return score.argsort(dim=-1, descending=True)


def resid_sequence(cent, probe, pool_mask, steps):
    """Vectorised residual pick sequence restricted to pool. [H, steps]."""
    Hkv, NB, D = cent.shape
    q_resid = probe.clone()
    chosen = torch.zeros(Hkv, NB, dtype=torch.bool, device=cent.device)
    seq = []
    for _ in range(min(steps, NB)):
        sc = torch.einsum("hd,hnd->hn", F.normalize(q_resid, dim=-1), cent)
        sc = sc.masked_fill(~pool_mask | chosen, float("-inf"))
        pick = sc.argmax(dim=-1)
        seq.append(pick)
        chosen.scatter_(1, pick.unsqueeze(1), True)
        kc = torch.gather(cent, 1, pick.view(Hkv, 1, 1).expand(Hkv, 1, D)).squeeze(1)
        q_resid = q_resid - (q_resid * kc).sum(-1, keepdim=True) * kc
    return torch.stack(seq, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--mass-probes", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    Hq = model.config.num_attention_heads
    Hkv = getattr(model.config, "num_key_value_heads", Hq)
    group = Hq // Hkv
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)
    bs = args.block_size

    combos = [f"pf_m{m}_h{h}" for m in M_SPECS for h in H_SPECS]
    hits = {c: {K: [0, 0] for K in BUDGETS} for c in combos}
    for r in REFS:
        hits[r] = {K: [0, 0] for K in BUDGETS}
    n_used = 0

    for ei in range(args.eval):
        text, spans, needles, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        ids, ans = answer_block_indices(tok, text, qv, spans, bs, 10**9)
        ids_t = torch.tensor(ids)[:-1]
        T = ids_t.shape[0]
        NB = (T + bs - 1) // bs
        if not ans or max(ans) >= NB:
            continue
        n_used += 1
        ans_set = set(ans)

        recent = list(range(max(0, T - args.recent_window), T))
        probes = sorted(
            set(int(x) for x in torch.linspace(0, T - 2, args.mass_probes).tolist())
        )
        sel_pos = sorted(set(recent + probes + [T - 1]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        probe_sel = [(pos_of[p], p) for p in probes]

        qs, past = capture_q(model, ids_t, sel_pos, device)
        keys, vals = get_keys(past), get_values(past)

        for li in range(len(keys)):
            kl, vl = keys[li][0], vals[li][0]
            cent, vn, idx, khn = block_stats(kl, vl, bs)
            ql = qs[li]
            p_lastk = F.normalize(khn[:, -1, :], dim=-1)
            ridx = torch.tensor([pos_of[p] for p in recent], device=device)
            q_rec = ql[:, ridx, :].view(Hkv, group, len(recent), -1)
            p_rec = F.normalize(F.normalize(q_rec, dim=-1).mean(dim=(1, 2)), dim=-1)

            r1o = order_desc(torch.einsum("hd,hnd->hn", p_rec, cent))
            r2o = order_desc(torch.einsum("hd,hnd->hn", p_lastk, cent))
            m_true = mass_trueq(ql, probe_sel, khn, kl, idx, NB, group)
            ho = order_desc(m_true)
            Kmax = max(BUDGETS)
            pool_mask = torch.zeros(Hkv, NB, dtype=torch.bool, device=device)
            pool_mask.scatter_(1, r1o[:, : 2 * Kmax], True)
            pool_mask.scatter_(1, r2o[:, : 2 * Kmax], True)
            d_seq = resid_sequence(cent, p_lastk, pool_mask, Kmax)

            r1l = r1o.tolist()
            r2l = r2o.tolist()
            hl = ho.tolist()
            dl = d_seq.tolist()
            for hh in range(Hkv):
                pools = {K: qualified_pool(r1l[hh], r2l[hh], K) for K in BUDGETS}
                for K in BUDGETS:
                    # references
                    for r, order in (
                        ("R_lastk", r2l[hh]),
                        ("R_recentq", r1l[hh]),
                        ("D_resid", dl[hh]),
                    ):
                        hits[r][K][0] += int(bool(ans_set & set(order[:K])))
                        hits[r][K][1] += 1
                    for m in M_SPECS:
                        m_n = spec_n(m, K)
                        for h in H_SPECS:
                            h_n = spec_n(h, K)
                            c = f"pf_m{m}_h{h}"
                            if m_n + h_n > int(0.75 * K) + 1:
                                hits[c][K][1] += 0
                                continue
                            kept = compose(
                                r1l[hh],
                                r2l[hh],
                                hl[hh],
                                dl[hh],
                                K,
                                m_n,
                                h_n,
                                pool=pools[K],
                            )
                            hits[c][K][0] += int(bool(ans_set & set(kept)))
                            hits[c][K][1] += 1

        del past, qs
        torch.cuda.empty_cache()
        print(f"  sample {ei + 1}/{args.eval} done (NB={NB})", flush=True)

    out = {
        "model": args.model,
        "length": args.length,
        "eval_used": n_used,
        "seed": args.seed,
        "retention": {},
    }
    for name, d in hits.items():
        out["retention"][name] = {
            f"K{K}": (v[0] / v[1] if v[1] else None) for K, v in d.items()
        }
    print(f"\n[{args.model}] portfolio NIAH retention (vs refs)")
    for name in REFS + tuple(combos):
        r = out["retention"][name]
        cells = " ".join(
            f"{r[f'K{K}']:.2f}" if r[f"K{K}"] is not None else "  - " for K in BUDGETS
        )
        print(f"  {name:18s} {cells}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
