#!/usr/bin/env python3
"""Does content-based KV importance prune the WHOLE prefix, or only the filler?

Motivation. A tiered-serving evaluation reports that content-based KV scorers
(KVZap-family) evict the prefix and so destroy prefix-cache reuse, while
position-based scorers preserve it. The supporting evidence is a regional MEAN
importance ("first 3 blocks" score low). A mean cannot distinguish two very
different behaviors:
  - BLANKET pruning: the whole prefix scores low, important tokens included
    (genuinely prefix-hostile -> reuse really is destroyed).
  - SELECTIVE pruning: the scorer keeps the IMPORTANT prefix tokens (an
    instruction, a fact the answer needs) and drops only the filler (the reuse
    damage then depends on whether the reused span is the filler or the fact).

This probe answers that directly. We plant a known-important fact near the START
of the context (inside the prefix region), pad with filler, and score every token
by peak received-attention (per key, the max attention it gets from any query,
mean over heads then layers -- a transparent content-based importance in the same
family as KVzip/H2O). We then compare the importance of the planted prefix fact
against the filler tokens around it.

Two arms, because the answer hinges on whether a query is present:
  - query-AGNOSTIC (default): score the context alone, no question. This is the
    KVzip reconstruction regime -- importance must be intrinsic to the content.
  - query-AWARE (--query-aware): append a question that asks for the planted
    code. This is the H2O / decode-time regime -- the live query can light up the
    fact it needs.

Output: importance percentiles (fact vs prefix filler) and, at a range of
keep-budgets, the fair per-token retention of the fact vs prefix filler. If the
fact is kept while filler is dropped -> SELECTIVE (the paper's "evicts the prefix"
overstates; it evicts prefix FILLER). If the fact is dropped with the filler ->
BLANKET (the paper's claim holds even for the important prefix token).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

FILLER = [
    "The weather over the northern plains stayed mild through the afternoon.",
    "Train schedules were revised after the spring timetable came into effect.",
    "A small market opened near the old harbor selling fruit and bread.",
    "The committee postponed the meeting until the following Tuesday morning.",
    "Several boats returned to port before the evening tide rolled in.",
    "Volunteers repainted the community hall over the long weekend.",
    "The library extended its hours during the examination period.",
    "Cyclists gathered at the square before the morning ride began.",
]


def build_prompt(tok, item, code, length, rng, fact_pos="prefix"):
    """Plant 'The <item> access code is <code>.' then pad with filler to ~length
    tokens. Returns the CONTEXT-only ids (no question) + the token-index span of
    the code. The optional query is appended by the caller so the prefix-region
    token indices are identical across the query-agnostic and query-aware arms."""
    fact = f"The {item} access code is {code}. "
    pieces = []
    # a little lead-in filler so the fact is INSIDE the prefix, not token 0
    lead = " ".join(rng.sample(FILLER, 2)) + " "
    head = lead + fact if fact_pos == "prefix" else lead
    pieces.append(head)
    # pad
    body = []
    while True:
        body.append(rng.choice(FILLER))
        if len(tok(" ".join(pieces + body)).input_ids) > length - 40:
            break
    pieces.append(" ".join(body))
    if fact_pos != "prefix":
        pieces.append(fact)
    text = " ".join(pieces)
    ids = tok(text, return_tensors="pt").input_ids[0]
    # locate the code token span by re-encoding the code substring in context
    code_ids = tok(f" {code}", add_special_tokens=False).input_ids
    # find the first occurrence of code_ids as a contiguous run
    span = None
    L = len(code_ids)
    idl = ids.tolist()
    for i in range(len(idl) - L):
        if idl[i : i + L] == code_ids:
            span = (i, i + L)
            break
    return ids, span, len(tok(head).input_ids)


@torch.no_grad()
def importance(model, ids, device):
    """Transparent proxy for KVzip's query-agnostic reconstruction importance:
    per key, the PEAK attention it receives from any query (amax over queries,
    like KVzip's `attn_weights.amax`), averaged over heads then over layers. Max
    (not sum) is the point -- sum is dominated by attention-sink position mass;
    max captures whether a key is strongly attended by SOME query, i.e. content
    relevance. Computed on the context only (no question) for query-agnosticism."""
    out = model(
        ids.unsqueeze(0).to(device),
        output_attentions=True,
        use_cache=False,
    )
    T = ids.shape[0]
    acc = torch.zeros(T, dtype=torch.float32, device=device)
    nl = 0
    for a in out.attentions:  # [1, H, T(query), T(key)]
        peak = a[0].float().amax(dim=1)  # max over queries -> [H, key]
        acc += peak.mean(dim=0)  # mean over heads -> [key]; accumulate layers
        nl += 1
    del out
    torch.cuda.empty_cache()
    return (acc / max(1, nl)).cpu()  # mean over layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--length", type=int, default=1536)
    ap.add_argument(
        "--prefix-frac", type=float, default=0.25, help="prefix = first frac of tokens"
    )
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument(
        "--keep", default="0.05,0.10,0.20,0.30,0.50", help="keep-top fractions"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--query-aware",
        action="store_true",
        help="append a question that asks for the planted code, so scoring is "
        "query-AWARE (H2O/decode-time regime). Default off = query-agnostic "
        "(KVzip reconstruction regime), scoring the context only.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager"
    ).to(device)
    model.eval()
    # clamp the context to the model's positional limit (leave room for the
    # appended query); gpt2 caps at 1024, so 1536 would index past the table.
    max_pos = (
        getattr(model.config, "max_position_embeddings", args.length) or args.length
    )
    length = min(args.length, max_pos - 48)
    rng = random.Random(args.seed)
    keeps = [float(x) for x in args.keep.split(",")]
    items = ["vault", "server", "gateway", "archive", "reactor", "console"]

    # accumulators
    fact_pct = []  # importance percentile of the planted fact (0..1, 1=most important)
    prefix_filler_pct = []  # mean percentile of prefix filler tokens
    # fact_kept_any: kept if ANY fact token survives (whole-fact recoverable).
    # fact_kept_mean: fraction of fact tokens surviving (fair vs filler's mean).
    fact_kept_any = {k: [0, 0] for k in keeps}
    fact_kept_mean = {k: [0, 0] for k in keeps}
    prefix_filler_kept = {k: [0, 0] for k in keeps}  # avg prefix-filler retention
    n_used = 0
    for i in range(args.n):
        item = rng.choice(items)
        code = str(rng.randint(1000, 9999))
        ctx_ids, span, head_len = build_prompt(tok, item, code, length, rng)
        if span is None:
            continue
        ctx_len = ctx_ids.shape[0]
        pref_end = int(args.prefix_frac * ctx_len)
        if span[0] >= pref_end:  # fact must land in the prefix region
            continue
        # query-aware: append a question referencing the planted fact. The prefix
        # region indices are unchanged (question is appended at the end).
        if args.query_aware:
            q = f" Question: what is the {item} access code? Answer:"
            q_ids = torch.tensor(
                tok(q, add_special_tokens=False).input_ids, dtype=ctx_ids.dtype
            )
            ids = torch.cat([ctx_ids, q_ids])
        else:
            ids = ctx_ids
        T = ids.shape[0]
        imp = importance(model, ids, device)  # [T]
        # rank -> percentile (1 = highest importance), over the prefix-eligible
        # scoring universe (the context tokens), so the query tokens do not shift
        # the percentile scale between arms.
        ctx_imp = imp[:ctx_len]
        order = ctx_imp.argsort()  # ascending
        rank = torch.empty(ctx_len)
        rank[order] = torch.arange(ctx_len, dtype=torch.float32)
        pct = rank / (ctx_len - 1)  # 0..1
        n_used += 1
        # planted fact percentile (mean over its tokens)
        fact_pct.append(pct[span[0] : span[1]].mean().item())
        # prefix filler = prefix tokens excluding the fact span and excluding sinks
        pref_mask = torch.zeros(ctx_len, dtype=torch.bool)
        pref_mask[4:pref_end] = True  # skip first 4 (attention sinks)
        pref_mask[span[0] : span[1]] = False
        prefix_filler_pct.append(pct[pref_mask].mean().item())
        # retention at keep-top-k (global threshold over the context universe)
        for k in keeps:
            thr = torch.quantile(ctx_imp, 1.0 - k)
            fspan = ctx_imp[span[0] : span[1]]
            fact_kept_any[k][0] += int(bool((fspan >= thr).any()))
            fact_kept_any[k][1] += 1
            fact_kept_mean[k][0] += float((fspan >= thr).float().mean())
            fact_kept_mean[k][1] += 1
            pf = ctx_imp[pref_mask]
            prefix_filler_kept[k][0] += float((pf >= thr).float().mean())
            prefix_filler_kept[k][1] += 1

    res = {
        "model": args.model,
        "length": length,
        "prefix_frac": args.prefix_frac,
        "query_aware": bool(args.query_aware),
        "n_used": n_used,
        "fact_importance_pct_mean": sum(fact_pct) / max(1, len(fact_pct)),
        "prefix_filler_importance_pct_mean": sum(prefix_filler_pct)
        / max(1, len(prefix_filler_pct)),
        "keep_budgets": {},
    }
    arm = "query-AWARE" if args.query_aware else "query-agnostic"
    print(
        f"\n[{args.model}] n={n_used}  {arm}  prefix=first "
        f"{args.prefix_frac:.0%} of {length} tok"
    )
    print(
        f"  importance percentile (1.0=most important):  "
        f"planted fact {res['fact_importance_pct_mean']:.2f}  vs  "
        f"prefix filler {res['prefix_filler_importance_pct_mean']:.2f}"
    )
    print(
        f"  {'keep%':>6s}  {'fact(any)':>10s}  {'fact(mean)':>11s}  "
        f"{'filler(mean)':>12s}"
    )
    for k in keeps:
        fa = fact_kept_any[k][0] / max(1, fact_kept_any[k][1])
        fm = fact_kept_mean[k][0] / max(1, fact_kept_mean[k][1])
        pf = prefix_filler_kept[k][0] / max(1, prefix_filler_kept[k][1])
        res["keep_budgets"][f"{k}"] = {
            "fact_kept_any": fa,
            "fact_kept_mean": fm,
            "prefix_filler_kept": pf,
        }
        print(f"  {k:>6.0%}  {fa:>10.2f}  {fm:>11.2f}  {pf:>12.2f}")
    print(
        "\n  reading (use fact-mean vs filler-mean, the fair per-token compare):\n"
        "    fact-mean >> filler-mean  => SELECTIVE (protects the important token).\n"
        "    fact-mean ~= filler-mean  => BLANKET  (treats the fact like filler)."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
