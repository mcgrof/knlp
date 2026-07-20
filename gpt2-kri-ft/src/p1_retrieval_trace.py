#!/usr/bin/env python3
"""P1 — retrieval-failure mechanism trace (lattice/KRI plan, regroup-20260720).

Question: does HARD orthogonal explain-away (`residual_rel`, α=1) kill multi-needle
retrieval by removing the shared query/address direction after the first needle is
picked, so the remaining needles' block-ranks balloon? Compare against plain recent-Q
(α=0), which never deflates the query.

Design: multi-needle NIAH (all needle "access code" sentences share a similar key
address). For each (layer, kv-head) we run the greedy selector with a trace, tracking
EACH planted needle block's rank under the current residual query at every step. The
decisive signature (doc): a needle findable at step 0 (low rank under recent-Q) whose
rank explodes right after a DIFFERENT needle is selected under α=1.

Uses the unified selector (`selectors.leaky_residual`, trace mode) so α=0 is provably
plain recent-Q and α=1 is the exact-span MGS residual. Recent-Q probe = the Phase-D
winner (normalize → mean over GQA group + recent window → renormalize).

Env/args: --model, --length, --needles, --eval, --block-size, --recent-window,
--budget (K), --seed, --out. Free on the W7900; no training, torch.no_grad throughout.
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
from diag_perhead_oracle import capture_q, block_stats, get_values
from kv_selectors import leaky_residual


def needle_value_blocks(tok, text, values, bs, max_tokens=10**9):
    """Block index of the LAST ' is {value}.' occurrence for each needle value.
    Mirrors answer_block_indices' value-locating logic, per needle."""
    enc = tok(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    T = len(ids) - 1  # prefill holds out the last token (matches harness convention)
    out = {}
    for v in values:
        needle = f" is {v}."
        ci = text.rfind(needle)
        if ci < 0:
            ci = text.rfind(v)
            if ci < 0:
                continue
            lo, hi = ci, ci + len(v)
        else:
            lo, hi = ci + 4, ci + 4 + len(v)  # the value chars within " is {v}."
        blks = set()
        for ti, (a, b) in enumerate(offs):
            if ti >= T:
                break
            if a < hi and b > lo:
                blks.add(ti // bs)
        if blks:
            out[v] = sorted(blks)
    return out, T


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--needles", type=int, default=4)
    ap.add_argument("--eval", type=int, default=30)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--recent-window", type=int, default=16)
    ap.add_argument("--budget", type=int, default=16, help="K blocks selected")
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
    nL = model.config.num_hidden_layers
    bs = args.block_size
    sents = load_filler_sentences(args.seed)
    rng = random.Random(args.seed)

    # Accumulators. We bucket every (layer, kv-head, needle) rank observation by
    # greedy step and by whether ANOTHER needle has already been picked, separately
    # for α=0 (recent-Q, no deflation) and α=1 (hard residual). The decisive contrast
    # is the rank of a not-yet-picked needle BEFORE vs AFTER a sibling needle is
    # selected, under α=1.
    stats = {
        a: {
            "rank_step0": [],  # needle rank at step 0 (pure relevance) — the baseline
            "rank_before_sibling": [],  # not-yet-picked needle, no sibling picked yet
            "rank_after_sibling": [],  # not-yet-picked needle, >=1 sibling picked
            "qcos_after_sibling": [],  # residual-query cos to q0 at those steps
            "any_recall": [],  # per sample: frac of needles whose block ∈ final K
            "all_recall": [],  # per sample: 1 if ALL needle blocks ∈ final K
        }
        for a in (0.0, 1.0)
    }
    n_used = 0
    examples = []  # a few raw per-head traces for the writeup

    for ei in range(args.eval):
        text, spans, needles, (qk, qv) = build_context(
            tok, args.length, args.needles, sents, rng
        )
        values = [v for (_k, v) in needles]
        nvb, T = needle_value_blocks(tok, text, values, bs)
        if len(nvb) < 2:
            continue
        NB = (T + bs - 1) // bs
        # flatten needle blocks (unique), guard in-range
        needle_blocks = sorted({b for blks in nvb.values() for b in blks if b < NB})
        if len(needle_blocks) < 2:
            continue
        ans_t = torch.tensor(needle_blocks, device=device)

        ids_full = tok(text, add_special_tokens=False)["input_ids"]
        ids_t = torch.tensor(ids_full)[: T + 1]  # keep T+1 so [:-1] gives T
        past = model(ids_t[:-1].unsqueeze(0).to(device), use_cache=True).past_key_values
        keys, vals = get_keys(past), get_values(past)

        # recent-Q probe positions
        recent = list(range(max(0, T - args.recent_window), T))
        sel_pos = sorted(set(recent + [T - 1]))
        pos_of = {p: i for i, p in enumerate(sel_pos)}
        qs, _ = capture_q(model, ids_t[:-1], sel_pos, device)
        ridx = torch.tensor([pos_of[p] for p in recent], device=device)

        n_used += 1
        for a in (0.0, 1.0):
            # per-sample recall accumulators over layers*heads
            recall_hits, recall_all, recall_n = 0, 0, 0
            for li in range(nL):
                cent, _vn, _idx, _khn = block_stats(keys[li][0], vals[li][0], bs)
                ql = qs[li]  # [Hq, P, D]
                q_rec = ql[:, ridx, :].view(Hkv, group, len(recent), -1)
                p_rec = F.normalize(
                    F.normalize(q_rec, dim=-1).mean(dim=(1, 2)), dim=-1
                )  # [Hkv, D] — Phase-D probe
                tr = []
                picks = leaky_residual(
                    p_rec, cent, args.budget, alpha=a, trace=tr, ans=ans_t
                )  # [Hkv, K]
                # needle index positions within ans_t
                A = ans_t.shape[0]
                # recall: needle block ∈ final selected set (per head)
                sel = picks  # [Hkv, K]
                for hh in range(Hkv):
                    sset = set(sel[hh].tolist())
                    hit = sum(1 for b in needle_blocks if b in sset)
                    recall_hits += hit
                    recall_all += int(hit == len(needle_blocks))
                    recall_n += 1
                # walk the trace: for each step, each head, each needle
                for entry in tr:
                    s = entry["step"]
                    ar = entry["ans_rank"]  # [Hkv][A]
                    ca = entry["chosen_ans"]  # [Hkv][A]
                    qcos = entry["q_cos"]  # [Hkv]
                    for hh in range(Hkv):
                        n_chosen_sib_total = sum(ca[hh])
                        for ai in range(A):
                            if ca[hh][ai]:
                                continue  # this needle already picked — skip
                            r = ar[hh][ai]
                            if s == 0:
                                stats[a]["rank_step0"].append(r)
                            # siblings = other needles already chosen
                            sib = n_chosen_sib_total  # this needle not chosen, so all chosen are siblings
                            if sib == 0:
                                stats[a]["rank_before_sibling"].append(r)
                            else:
                                stats[a]["rank_after_sibling"].append(r)
                                stats[a]["qcos_after_sibling"].append(qcos[hh])
            stats[a]["any_recall"].append(recall_hits / max(1, recall_n) / len(needle_blocks))
            stats[a]["all_recall"].append(recall_all / max(1, recall_n))
        # save one example trace (mid layer, first head) per a for the writeup
        if len(examples) < 3:
            examples.append(
                {"sample": ei, "n_needle_blocks": len(needle_blocks), "T": T, "NB": NB}
            )
        print(f"  [{n_used}/{args.eval}] needles={len(needle_blocks)} T={T}", flush=True)

    def summ(xs):
        if not xs:
            return None
        t = torch.tensor(xs, dtype=torch.float)
        return {
            "n": len(xs),
            "mean": round(t.mean().item(), 3),
            "median": round(t.median().item(), 3),
            "p90": round(t.quantile(0.9).item(), 3),
        }

    out = {
        "model": args.model,
        "length": args.length,
        "needles": args.needles,
        "budget": args.budget,
        "block_size": bs,
        "n_used": n_used,
        "Hq": Hq,
        "Hkv": Hkv,
        "nL": nL,
        "alpha": {},
    }
    for a in (0.0, 1.0):
        out["alpha"][str(a)] = {
            "rank_step0": summ(stats[a]["rank_step0"]),
            "rank_before_sibling": summ(stats[a]["rank_before_sibling"]),
            "rank_after_sibling": summ(stats[a]["rank_after_sibling"]),
            "qcos_after_sibling": summ(stats[a]["qcos_after_sibling"]),
            "any_recall_mean": round(
                sum(stats[a]["any_recall"]) / max(1, len(stats[a]["any_recall"])), 4
            ),
            "all_recall_mean": round(
                sum(stats[a]["all_recall"]) / max(1, len(stats[a]["all_recall"])), 4
            ),
        }
    # the P1 gate signal: how much a not-yet-picked needle's rank inflates AFTER a
    # sibling is picked, under α=1 (hard) vs α=0 (recent-Q, should be ~flat).
    def infl(a):
        b = summ(stats[a]["rank_before_sibling"])
        af = summ(stats[a]["rank_after_sibling"])
        if not b or not af:
            return None
        return round(af["median"] - b["median"], 3)

    out["gate"] = {
        "rank_inflation_after_sibling_alpha1": infl(1.0),
        "rank_inflation_after_sibling_alpha0": infl(0.0),
        "verdict_hint": (
            "hard explain-away collapses retrieval if alpha1 inflation >> alpha0"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["alpha"], indent=2))
    print("GATE:", json.dumps(out["gate"]))
    print(f"P1_TRACE_DONE {args.out}")


if __name__ == "__main__":
    main()
