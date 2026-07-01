# SPDX-License-Identifier: GPL-2.0
"""KRI-TierKV falsification / kill-test (Priority 2).

The KRI-TierKV milestone-1 emulation showed KRI-D-sum tying FIFO on
attention-mass recall -- "FIFO with paperwork." This run asks the sharper
question the cmcp review demanded: at a fixed slow-tier budget, does the
query-INDEPENDENT KRI-D-sum block score pick the block that holds the answer
better than FIFO or random, and does any deployable query-AWARE score do
materially better?

Setup mirrors the P1 replay but at block granularity. One long shared prefix is
split into fixed 128-token blocks. Several facts are planted, each in its own
OLD-MIDDLE block (never the protected first block, never the recent window), one
question per fact. The slow tier keeps the protected block 0, the last R recent
blocks, and K old blocks chosen by the method under test. Success = the block
holding the queried fact is kept AND the model answers it (exact code, mask +
pinned positions, the construction the drift module validated).

Scorers:
  fifo_old     -- keep the most recent old blocks (recency). Query-independent.
  random_old   -- keep a random subset (mean over seeds). Query-independent.
  kri_d_sum    -- ||sum_t K_t|| + ||sum_t V_t|| per block, block-local, summed
                  over layers/kv-heads. The method on trial. Query-independent.
  k_norm       -- mean per-token key norm per block. Query-independent.
  v_norm       -- mean per-token value norm per block. Query-independent.
  quest_lite   -- attention mass from the last observation-window query tokens
                  to each block. Query-AWARE, deployable (SnapKV/Quest-shaped).
  oracle       -- attention mass from the generated ANSWER tokens to each block.
                  The ceiling; not deployable (needs the answer).

Decision rule (cmcp): KILL KRI-D-sum if it fails to beat FIFO by >= 5 points
absolute accuracy OR fails to close >= 30% of the FIFO-to-oracle gap. Keep
KRI-TierKV only if a deployable (query-aware) score closes >= 50% of that gap
and improves exact accuracy at fixed slow-tier budget.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

NEEDLES = [
    ("the north tower vault", "MAGENTA-4471"),
    ("the harbor cafe wifi", "CRIMSON-8823"),
    ("Dr. Okafor's office", "COBALT-2019"),
    ("the west gate keypad", "AMBER-7734"),
    ("the archive room lock", "VIOLET-5561"),
    ("the rooftop antenna", "TEAL-3390"),
    ("the loading dock badge", "SCARLET-6612"),
    ("the server rack panel", "INDIGO-9045"),
    ("the marina fuel pump", "PEWTER-3308"),
    ("the greenhouse sensor", "OCHRE-7192"),
    ("the transit depot gate", "SIENNA-4055"),
    ("the clinic supply cabinet", "AZURE-6621"),
]


def build_context(tok, needles, target_tokens, mid_mul=48):
    """Facts spread across the OLD MIDDLE, one question each."""
    filler = "Routine log entry with no salient facts, status nominal. "
    facts = [f"IMPORTANT: the access code for {s} is {c}. " for s, c in needles]
    front = filler * 22
    tail = filler * 36
    mid_chunk = filler * mid_mul
    parts = [front]
    for f in facts:
        parts.append(f)
        parts.append(mid_chunk)
    parts.append(tail)
    text = "".join(parts)
    ids = tok(text, return_tensors="pt").input_ids[:, :target_tokens]
    text = tok.decode(ids[0], skip_special_tokens=True)
    qa = [(f"What is the access code for {s}?", c) for s, c in needles]
    return text, qa


def _norm(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())


def _found(ans, expected):
    return _norm(expected) in _norm(ans)


def block_ranges(prefix_len, block_size):
    return [
        (lo, min(lo + block_size, prefix_len))
        for lo in range(0, prefix_len, block_size)
    ]


def needle_block(tok, ctx_ids, code, blocks):
    """Which block holds the fact code (token-subsequence match)."""
    ids = ctx_ids[0].tolist()
    for variant in (" " + code, code):
        pat = tok(variant, add_special_tokens=False).input_ids
        for i in range(len(ids) - len(pat) + 1):
            if ids[i : i + len(pat)] == pat:
                for b, (lo, hi) in enumerate(blocks):
                    if lo <= i < hi:
                        return b
    return -1


def magnitude_scores(cache, blocks):
    """Query-independent block scores from the prefill KV cache."""
    if hasattr(cache, "layers"):
        layers = [(l.keys, l.values) for l in cache.layers]
    else:
        layers = list(cache)
    nb = len(blocks)
    kri = torch.zeros(nb)
    knorm = torch.zeros(nb)
    vnorm = torch.zeros(nb)
    for k, v in layers:
        k = k[0].float()  # [kv_heads, T, d]
        v = v[0].float()
        for b, (lo, hi) in enumerate(blocks):
            kb = k[:, lo:hi, :]
            vb = v[:, lo:hi, :]
            # KRI-D-sum: norm of the block-summed K and V, over heads
            kri[b] += (
                kb.sum(dim=1).norm(dim=-1).sum() + vb.sum(dim=1).norm(dim=-1).sum()
            ).item()
            knorm[b] += kb.norm(dim=-1).mean().item()
            vnorm[b] += vb.norm(dim=-1).mean().item()
    return {"kri_d_sum": kri, "k_norm": knorm, "v_norm": vnorm}


def attention_block_mass(attentions, blocks, prefix_len, row_lo, row_hi):
    """Sum of attention from rows [row_lo,row_hi) onto each prefix block,
    max over heads, summed over layers. Used for quest_lite and oracle."""
    nb = len(blocks)
    score = torch.zeros(nb)
    for aw in attentions:
        a = aw[0, :, row_lo:row_hi, :prefix_len].float()  # [h, rows, prefix]
        col = a.mean(dim=1)  # [h, prefix]
        col = col.max(dim=0).values  # [prefix]
        for b, (lo, hi) in enumerate(blocks):
            score[b] = max(score[b].item(), col[lo:hi].sum().item())
    return score


def answer(model, tok, ctx_ids, kept_tokens, q_text, device, gen=40):
    prefix_len = ctx_ids.shape[1]
    q_ids = tok(
        "\n\nQuestion: " + q_text + "\nAnswer with only the code:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    pmask = torch.zeros(prefix_len, dtype=torch.long, device=device)
    if kept_tokens is None:
        pmask[:] = 1
    else:
        pmask[torch.tensor(sorted(kept_tokens), device=device)] = 1
    with torch.no_grad():
        base = model(ctx_ids, use_cache=True)
    cache = base.past_key_values
    q_len = q_ids.shape[1]
    attn = torch.cat([pmask, torch.ones(q_len, dtype=torch.long, device=device)])
    pos = torch.arange(prefix_len, prefix_len + q_len, device=device).unsqueeze(0)
    gen_ids = []
    with torch.no_grad():
        out = model(
            q_ids,
            past_key_values=cache,
            attention_mask=attn.unsqueeze(0),
            position_ids=pos,
            use_cache=True,
        )
        for step in range(gen):
            nxt = int(out.logits[0, -1].argmax())
            gen_ids.append(nxt)
            attn = torch.cat([attn, torch.ones(1, dtype=torch.long, device=device)])
            p = torch.tensor([[prefix_len + q_len + step]], device=device)
            out = model(
                torch.tensor([[nxt]], device=device),
                past_key_values=out.past_key_values,
                attention_mask=attn.unsqueeze(0),
                position_ids=p,
                use_cache=True,
            )
    return tok.decode(gen_ids)


def kept_token_set(kept_blocks, blocks):
    s = set()
    for b in kept_blocks:
        lo, hi = blocks[b]
        s |= set(range(lo, hi))
    return s


def choose_blocks(scores, old_blocks, k_old, protected, recent):
    """Top-k old blocks by score, plus the always-kept protected + recent."""
    ranked = sorted(old_blocks, key=lambda b: float(scores[b]), reverse=True)
    return set(protected) | set(recent) | set(ranked[:k_old])


def eval_prefix(
    model,
    tok,
    device,
    needles,
    context_len,
    block_size,
    k_old,
    recent_n,
    seeds,
    obs,
    pidx,
):
    ctx_text, qa = build_context(tok, needles, context_len)
    ctx_ids = tok(ctx_text, return_tensors="pt").input_ids.to(device)
    prefix_len = ctx_ids.shape[1]
    blocks = block_ranges(prefix_len, block_size)
    nb = len(blocks)
    protected = [0]
    recent = list(range(max(0, nb - recent_n), nb))
    # old = candidate blocks, excluding protected and recent
    old_blocks = [b for b in range(nb) if b not in protected and b not in recent]

    mag = magnitude_scores(_prefill(model, ctx_ids), blocks)

    # recency score for fifo (higher index = more recent among old)
    recency = torch.tensor([float(b) for b in range(nb)])

    rows = []
    for i, (q, expected) in enumerate(qa):
        nblk = needle_block(tok, ctx_ids, expected, blocks)
        # query-aware attention (last obs query rows -> blocks)
        q_ids = tok(
            "\n\nQuestion: " + q + "\nAnswer with only the code:",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        # generate the answer first (needed for the oracle rows)
        a_full = answer(model, tok, ctx_ids, None, q, device)
        ans_ids = tok(
            a_full, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)

        # query-aware attention (last obs query rows -> blocks). Free the
        # big [H,T,T] attention tensor before allocating the oracle's, so
        # only one such tensor is ever resident (output_attentions is ~26GB
        # at 4K; two at once OOMs an 80GB card).
        full = torch.cat([ctx_ids, q_ids], dim=1)
        with torch.no_grad():
            o = model(full, output_attentions=True, use_cache=False)
        quest = attention_block_mass(
            o.attentions,
            blocks,
            prefix_len,
            prefix_len + q_ids.shape[1] - obs,
            prefix_len + q_ids.shape[1],
        )
        del o, full
        torch.cuda.empty_cache()

        # oracle: attention from the generated answer rows
        fao = torch.cat([ctx_ids, q_ids, ans_ids], dim=1)
        with torch.no_grad():
            oo = model(fao, output_attentions=True, use_cache=False)
        arow_lo = prefix_len + q_ids.shape[1]
        oracle = attention_block_mass(
            oo.attentions, blocks, prefix_len, arow_lo, fao.shape[1]
        )
        del oo, fao
        torch.cuda.empty_cache()

        methods = {}
        methods["fifo_old"] = choose_blocks(
            recency, old_blocks, k_old, protected, recent
        )
        methods["kri_d_sum"] = choose_blocks(
            mag["kri_d_sum"], old_blocks, k_old, protected, recent
        )
        methods["k_norm"] = choose_blocks(
            mag["k_norm"], old_blocks, k_old, protected, recent
        )
        methods["v_norm"] = choose_blocks(
            mag["v_norm"], old_blocks, k_old, protected, recent
        )
        methods["quest_lite"] = choose_blocks(
            quest, old_blocks, k_old, protected, recent
        )
        methods["oracle"] = choose_blocks(oracle, old_blocks, k_old, protected, recent)

        row = {
            "prefix": pidx,
            "needle": i,
            "code": expected,
            "needle_block": nblk,
            "n_blocks": nb,
            "full": _found(a_full, expected),
        }
        for name, kept in methods.items():
            hit = nblk in kept
            ans = answer(model, tok, ctx_ids, kept_token_set(kept, blocks), q, device)
            row[name + "_blockhit"] = hit
            row[name + "_acc"] = _found(ans, expected)
        # random_old: mean over seeds
        rhits, raccs = [], []
        for sd in seeds:
            rng = random.Random(sd * 1000 + pidx * 100 + i)
            pick = set(rng.sample(old_blocks, min(k_old, len(old_blocks))))
            kept = set(protected) | set(recent) | pick
            rhits.append(nblk in kept)
            raccs.append(
                _found(
                    answer(
                        model, tok, ctx_ids, kept_token_set(kept, blocks), q, device
                    ),
                    expected,
                )
            )
        row["random_old_blockhit"] = sum(rhits) / len(rhits)
        row["random_old_acc"] = sum(raccs) / len(raccs)
        rows.append(row)
        print(
            f"  [p{pidx} q{i}] blk={nblk}/{nb} full={row['full']} "
            f"kri={row['kri_d_sum_acc']} fifo={row['fifo_old_acc']} "
            f"quest={row['quest_lite_acc']} oracle={row['oracle_acc']}",
            flush=True,
        )
    return rows, nb


def _prefill(model, ctx_ids):
    with torch.no_grad():
        return model(ctx_ids, use_cache=True).past_key_values


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )

    rng = random.Random(args.seed)
    seeds = list(range(args.n_random_seeds))
    all_rows = []
    nb = 0
    for p in range(args.repeats):
        pool = NEEDLES[:]
        rng.shuffle(pool)
        needles = pool[: args.n_needles]
        print(f"prefix {p}: {[s for s, _ in needles]}", flush=True)
        rows, nb = eval_prefix(
            model,
            tok,
            device,
            needles,
            args.context_len,
            args.block_size,
            args.k_old,
            args.recent_n,
            seeds,
            args.obs,
            p,
        )
        all_rows.extend(rows)

    strong = [r for r in all_rows if r["full"]]
    methods = [
        "fifo_old",
        "random_old",
        "kri_d_sum",
        "k_norm",
        "v_norm",
        "quest_lite",
        "oracle",
    ]
    n = len(strong) if strong else 1
    acc = {m: sum(r[m + "_acc"] for r in strong) / n for m in methods}
    blockhit = {m: sum(r[m + "_blockhit"] for r in strong) / n for m in methods}

    fifo, oracle, kri = acc["fifo_old"], acc["oracle"], acc["kri_d_sum"]
    quest = acc["quest_lite"]
    gap = oracle - fifo
    kri_close = (kri - fifo) / gap if gap > 1e-9 else 0.0
    quest_close = (quest - fifo) / gap if gap > 1e-9 else 0.0
    verdict = {
        "fifo_to_oracle_gap": round(gap, 3),
        "kri_minus_fifo": round(kri - fifo, 3),
        "kri_closes_gap_frac": round(kri_close, 3),
        "quest_minus_fifo": round(quest - fifo, 3),
        "quest_closes_gap_frac": round(quest_close, 3),
        "kri_killed": (kri - fifo) < 0.05 or kri_close < 0.30,
        "kri_tierkv_survives_via_queryaware": quest_close >= 0.50 and quest > fifo,
    }

    result = {
        "model": args.model,
        "context_len": args.context_len,
        "block_size": args.block_size,
        "n_blocks": nb,
        "k_old": args.k_old,
        "recent_n": args.recent_n,
        "repeats": args.repeats,
        "n_needles_per_prefix": args.n_needles,
        "n_queries_total": len(all_rows),
        "n_strong": len(strong),
        "accuracy_strong": acc,
        "block_hit_strong": blockhit,
        "verdict": verdict,
        "per_row": all_rows,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "kri_tierkv_killtest.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nstrong: {len(strong)}/{len(all_rows)}", flush=True)
    print("ACC", json.dumps({k: round(v, 3) for k, v in acc.items()}), flush=True)
    print(
        "BLOCKHIT",
        json.dumps({k: round(v, 3) for k, v in blockhit.items()}),
        flush=True,
    )
    print("VERDICT", json.dumps(verdict), flush=True)
    print("wrote", args.output_dir, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--context-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--k-old", type=int, default=3)
    ap.add_argument("--recent-n", type=int, default=3)
    ap.add_argument("--n-needles", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--n-random-seeds", type=int, default=4)
    ap.add_argument("--obs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
