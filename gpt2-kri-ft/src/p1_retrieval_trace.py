#!/usr/bin/env python3
"""Retrieval-failure mechanism trace (lattice/KRI plan, regroup-20260720).

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

    # Clean PAIRED metric (avoids the survivorship confound of pooling by step).
    # For each (layer, kv-head, needle) we take r0 = its rank at step 0 (pure
    # recent-Q relevance, identical for α=0/α=1), find the step the FIRST sibling
    # needle is picked, and record delta = rank_after_that_sibling − r0 for the
    # needle if it is still unpicked. Conditioned on FINDABILITY (r0 ≤ K, i.e.
    # recent-Q alone would have kept it). Under α=0 a sibling pick just removes one
    # block, so delta ≈ 0; under α=1 hard explain-away removes the shared address
    # direction and, if that is the mechanism, a findable needle's rank balloons
    # (delta ≫ 0). The α=1-vs-α=0 gap on FINDABLE needles is the gate.
    FIND = args.budget
    stats = {
        a: {
            "r0_all": [],  # step-0 rank of every needle (findability distribution)
            "delta_findable": [],  # paired rank change after first sibling, findable
            "delta_unfindable": [],  # same, control (needle not findable at step 0)
            "qcos_at_after": [],  # residual cos to q0 at the after-sibling step (findable)
            "recall_findable": [],  # per findable needle: block ∈ final selected set?
            "any_recall": [],  # per sample: frac of needle blocks ∈ final K (mean over heads)
            "all_recall": [],  # per sample: frac of heads keeping ALL needle blocks
        }
        for a in (0.0, 1.0)
    }
    n_used = 0

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
                A = ans_t.shape[0]
                for hh in range(Hkv):
                    sset = set(picks[hh].tolist())
                    hit = sum(1 for b in needle_blocks if b in sset)
                    recall_hits += hit
                    recall_all += int(hit == len(needle_blocks))
                    recall_n += 1
                    # per-needle trajectory for this head
                    r0 = [tr[0]["ans_rank"][hh][ai] for ai in range(A)]
                    pick_step = [None] * A
                    for s, entry in enumerate(tr):
                        pk = entry["pick"][hh]
                        for ai in range(A):
                            if pick_step[ai] is None and pk == needle_blocks[ai]:
                                pick_step[ai] = s
                    for ai in range(A):
                        stats[a]["r0_all"].append(r0[ai])
                        findable = r0[ai] <= FIND
                        if findable:
                            stats[a]["recall_findable"].append(
                                int(needle_blocks[ai] in sset)
                            )
                        sib = [
                            pick_step[aj]
                            for aj in range(A)
                            if aj != ai and pick_step[aj] is not None
                        ]
                        if not sib:
                            continue
                        s_sib = min(sib)
                        # needle ai still unpicked when the first sibling was chosen
                        if pick_step[ai] is not None and pick_step[ai] <= s_sib:
                            continue
                        s_after = s_sib + 1
                        if s_after >= len(tr):
                            continue
                        delta = tr[s_after]["ans_rank"][hh][ai] - r0[ai]
                        bucket = "delta_findable" if findable else "delta_unfindable"
                        stats[a][bucket].append(delta)
                        if findable:
                            stats[a]["qcos_at_after"].append(tr[s_after]["q_cos"][hh])
            stats[a]["any_recall"].append(
                recall_hits / max(1, recall_n) / len(needle_blocks)
            )
            stats[a]["all_recall"].append(recall_all / max(1, recall_n))
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
    def meanf(xs):
        return round(sum(xs) / max(1, len(xs)), 4) if xs else None

    for a in (0.0, 1.0):
        out["alpha"][str(a)] = {
            "r0_all": summ(stats[a]["r0_all"]),
            "delta_findable": summ(stats[a]["delta_findable"]),
            "delta_unfindable": summ(stats[a]["delta_unfindable"]),
            "qcos_at_after": summ(stats[a]["qcos_at_after"]),
            "recall_findable_mean": meanf(stats[a]["recall_findable"]),
            "any_recall_mean": meanf(stats[a]["any_recall"]),
            "all_recall_mean": meanf(stats[a]["all_recall"]),
        }
    # gate: on FINDABLE needles (recent-Q would keep them), how much a needle's
    # rank inflates after the first sibling is picked, α=1 (hard) vs α=0 (flat).
    d1 = summ(stats[1.0]["delta_findable"])
    d0 = summ(stats[0.0]["delta_findable"])
    r1 = meanf(stats[1.0]["recall_findable"])
    r0m = meanf(stats[0.0]["recall_findable"])
    recall_drop = round((r0m or 0) - (r1 or 0), 4)
    out["gate"] = {
        # PRIMARY: on needles recent-Q would keep (rank ≤ K at step 0), how much
        # does hard explain-away drop them from the final selected set?
        "findable_recall_alpha0_recentQ": r0m,
        "findable_recall_alpha1_hardresidual": r1,
        "findable_recall_drop": recall_drop,
        # SECONDARY: per-step rank inflation after the first sibling pick.
        "findable_rank_inflation_alpha1_median": d1["median"] if d1 else None,
        "findable_rank_inflation_alpha0_median": d0["median"] if d0 else None,
        "mechanism_confirmed": bool(recall_drop >= 0.15),
        "verdict": (
            "MECHANISM: hard orthogonal explain-away removes findable needles from "
            "the kept set (recall_drop >= 0.15) -> proceed with leaky/relevance-floor "
            "blend. Else: stop blaming orthogonality, inspect probe/centroid/GQA."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["alpha"], indent=2))
    print("GATE:", json.dumps(out["gate"]))
    print(f"RETRIEVAL_TRACE_DONE {args.out}")


if __name__ == "__main__":
    main()
