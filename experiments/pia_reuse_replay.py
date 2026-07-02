# SPDX-License-Identifier: GPL-2.0
"""PIA same-prefix, many-query cache-reuse replay (Priority 1).

The drift run left a hole: it compressed query-independently, so it never
exercised the query-dependence that makes an adaptive method dangerous for
prefix sharing. This experiment closes it. For one long shared prefix with
several planted facts, it builds each method's compressed artifact and then asks
what happens when that artifact is REUSED across different queries under a
prefix-only key -- exactly what a prefix cache does.

Two methods stand in for the two contract classes:
  query_aware  -- selects prefix tokens by how much the query attends to them
                  (SnapKV-shaped): the artifact depends on the query.
  positional   -- sink + recent window (StreamingLLM-shaped): query-independent.

Modes:
  full                  -- uncompressed prefix (dense baseline; keep only prefixes
                           the model answers correctly at full KV).
  honest_extended_key   -- select using THIS query, answer THIS query. The method
                           deployed correctly, with the query in the key.
  unsafe_prefix_reuse   -- select using the FIRST query, reuse that artifact for
                           every query. The prefix-only-key deployment.

The decisive result: for query_aware, honest is high but unsafe collapses (the
artifact kept query 0's facts, not query k's); for positional, honest == unsafe
(the artifact never depended on the query). Kept-set Jaccard across queries is
low for query_aware (the harness's DANGEROUS signal) and 1.0 for positional.

Selection is applied as an attention mask over prefix positions with the query's
RoPE phase pinned to the original offset, the same construction the drift module
validated to be exact in the no-drop case.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

# distinct subjects so the facts do not interfere with each other; a strong
# dense baseline needs the model to disambiguate which code goes with which
# fact, which near-identical "building N" phrasing defeats.
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
    ("the observatory dome", "UMBER-9847"),
    ("the foundry east kiln", "SABLE-2276"),
    ("the reservoir valve", "OLIVE-5130"),
    ("the printworks press", "CORAL-8409"),
]


def build_context(tok, needles, target_tokens, mid_mul=22):
    """Distinct facts planted across the OLD MIDDLE of a long filler prefix,
    one question per needle. The needles sit between ~15% and ~65% of the
    prefix -- never in the sink or the recent window -- so an 8% budget cannot
    retain them by position alone, which is what separates a query-aware
    method (keeps the fact the query points at) from a positional one.
    `needles` is a list of (subject, code); vary it per prefix for repeats.
    Returns (context_text, list[(question, answer)]).
    """
    filler = "Routine log entry with no salient facts, status nominal. "
    facts = [f"IMPORTANT: the access code for {s} is {c}. " for s, c in needles]
    # front padding (becomes the sink region), then needles interleaved with
    # filler across the middle, then a long recent tail of pure filler.
    front = filler * 25
    tail = filler * 40
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


def query_aware_kept(model, tok, ctx_ids, q_text, keep_tokens, device, obs=10):
    """Top prefix tokens by query attention, SnapKV-shaped.

    A plain mean over all layers/heads is dominated by the attention sink
    (token 0) and the local window, so the mid-context needle never makes the
    budget. SnapKV instead pools attention from a small OBSERVATION WINDOW (the
    last few query tokens, which carry the question's content) and keeps what
    any retrieval head points at. We approximate that with a max over heads and
    layers of the observation-window attention -- one strong retrieval head is
    enough to save a token, which is what surfaces the needle.
    """
    q_ids = tok(
        "\n\nQuestion: " + q_text + "\nAnswer with only the code:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    full = torch.cat([ctx_ids, q_ids], dim=1)
    prefix_len = ctx_ids.shape[1]
    with torch.no_grad():
        out = model(full, output_attentions=True, use_cache=False)
    score = torch.zeros(prefix_len, device=device)
    for aw in out.attentions:
        # observation window = last `obs` query rows; max over heads, mean over
        # the window rows, then max-accumulate across layers.
        a = aw[0, :, prefix_len:, :prefix_len].float()  # [h, q_len, prefix]
        win = a[:, -obs:, :].mean(dim=1)  # [h, prefix]
        score = torch.maximum(score, win.max(dim=0).values)
    keep = torch.topk(score, min(keep_tokens, prefix_len)).indices
    return set(keep.tolist())


def qk_aware_kept(model, tok, ctx_ids, q_text, keep_tokens, device, obs=10):
    """Same query-aware selection as query_aware_kept, but scored with the cheap
    q.k path (pia_qk_selector) instead of output_attentions -- so it runs at
    16-32K where the [layers, heads, T, T] attention tensor will not fit. The
    parity harness (pia_qk_parity.py) confirms it tracks query_aware_kept."""
    from pia_qk_selector import qk_top_tokens

    q_ids = tok(
        "\n\nQuestion: " + q_text + "\nAnswer with only the code:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    full = torch.cat([ctx_ids, q_ids], dim=1)
    return qk_top_tokens(model, full, ctx_ids.shape[1], keep_tokens, obs, device)


def positional_kept(prefix_len, keep_tokens, sink=128, recent=384):
    """Sink + recent window (StreamingLLM-shaped), query-independent."""
    kept = set(range(min(sink, prefix_len)))
    kept |= set(range(max(0, prefix_len - recent), prefix_len))
    # pad toward the budget with evenly spaced middle tokens (still query-free)
    remaining = keep_tokens - len(kept)
    if remaining > 0:
        mid = list(range(sink, prefix_len - recent))
        if mid:
            step = max(1, len(mid) // remaining)
            kept |= set(mid[::step][:remaining])
    return kept


def answer(model, tok, ctx_ids, kept_prefix, q_text, device, gen=40):
    """Greedy-generate an answer attending only to kept prefix positions."""
    prefix_len = ctx_ids.shape[1]
    q_ids = tok(
        "\n\nQuestion: " + q_text + "\nAnswer with only the code:",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    pmask = torch.zeros(prefix_len, dtype=torch.long, device=device)
    if kept_prefix is None:
        pmask[:] = 1
    else:
        idx = torch.tensor(sorted(kept_prefix), device=device)
        pmask[idx] = 1

    # prefill the full prefix once (dense cache), then answer with the mask
    with torch.no_grad():
        base = model(ctx_ids, use_cache=True)
    cache = base.past_key_values
    q_len = q_ids.shape[1]
    attn = torch.cat([pmask, torch.ones(q_len, dtype=torch.long, device=device)])
    pos = torch.arange(prefix_len, prefix_len + q_len, device=device).unsqueeze(0)
    ids = q_ids
    gen_ids = []
    with torch.no_grad():
        out = model(
            ids,
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


def jaccard(sets):
    keys = list(sets)
    if len(keys) < 2:
        return 1.0
    inter = set.intersection(*[sets[k] for k in keys])
    union = set.union(*[sets[k] for k in keys])
    return len(inter) / max(1, len(union))


def _norm(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())


def _found(ans, expected):
    # BPE splits "MAGENTA-4471" into "MAGENTA 447..."; compare on the
    # alphanumeric-only form so the separator and spacing don't matter.
    return _norm(expected) in _norm(ans)


def eval_prefix(
    model, tok, device, needles, context_len, keep_ratio, pidx, selector="attn"
):
    """One shared prefix, one query per needle, four modes. Returns per-needle
    rows plus the query-aware kept-set Jaccard across queries. `selector` picks
    the query-aware scorer: "attn" (output_attentions, <=4K) or "qk" (cheap q.k
    path, scales to 16-32K)."""
    ctx_text, qa = build_context(tok, needles, context_len)
    ctx_ids = tok(ctx_text, return_tensors="pt").input_ids.to(device)
    prefix_len = ctx_ids.shape[1]
    keep_tokens = max(1, int(prefix_len * keep_ratio))

    select = qk_aware_kept if selector == "qk" else query_aware_kept
    qa_kept = {
        i: select(model, tok, ctx_ids, q, keep_tokens, device)
        for i, (q, _) in enumerate(qa)
    }
    pos_kept = positional_kept(prefix_len, keep_tokens)

    per = []
    for i, (q, expected) in enumerate(qa):
        a_full = answer(model, tok, ctx_ids, None, q, device)
        a_qh = answer(model, tok, ctx_ids, qa_kept[i], q, device)
        a_qu = answer(model, tok, ctx_ids, qa_kept[0], q, device)  # reuse query 0
        a_ph = answer(model, tok, ctx_ids, pos_kept, q, device)
        row = {
            "prefix": pidx,
            "needle": i,
            "answer_code": expected,
            "full": _found(a_full, expected),
            "qaware_honest": _found(a_qh, expected),
            "qaware_unsafe": _found(a_qu, expected),  # reuse query 0's kept set
            "positional_honest": _found(a_ph, expected),
            "positional_unsafe": _found(a_ph, expected),  # query-independent: same
        }
        per.append(row)
        print(
            f"  [p{pidx} q{i}] full={row['full']} q_honest={row['qaware_honest']} "
            f"q_unsafe={row['qaware_unsafe']} pos={row['positional_honest']}",
            flush=True,
        )
    return per, prefix_len, keep_tokens, jaccard(qa_kept)


def run(args):
    import random

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    # the qk selector recomputes its own scores in a pre-hook on the layer
    # inputs, so it does not need eager; sdpa avoids materializing the full
    # [heads, T, T] score matrix, which is what lets the forward reach 16-32K.
    # the attn selector (output_attentions) requires eager.
    impl = "eager" if args.selector == "attn" else "sdpa"
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=impl
        )
        .to(device)
        .eval()
    )

    rng = random.Random(args.seed)
    all_rows, jaccards = [], []
    prefix_len = keep_tokens = 0
    for p in range(args.repeats):
        # each prefix draws a fresh subset+order of the needle pool so the
        # aggregate isn't a property of one arrangement.
        pool = NEEDLES[:]
        rng.shuffle(pool)
        needles = pool[: args.n_needles]
        print(f"prefix {p}: {[s for s, _ in needles]}", flush=True)
        per, prefix_len, keep_tokens, jac = eval_prefix(
            model,
            tok,
            device,
            needles,
            args.context_len,
            args.keep_ratio,
            p,
            selector=args.selector,
        )
        all_rows.extend(per)
        jaccards.append(jac)

    modes = [
        "full",
        "qaware_honest",
        "qaware_unsafe",
        "positional_honest",
        "positional_unsafe",
    ]
    strong = [r for r in all_rows if r["full"]]
    acc_all = {m: sum(r[m] for r in all_rows) / len(all_rows) for m in modes}
    acc_strong = (
        {m: sum(r[m] for r in strong) / len(strong) for m in modes}
        if strong
        else {m: 0.0 for m in modes}
    )
    result = {
        "model": args.model,
        "selector": args.selector,
        "context_len": prefix_len,
        "keep_ratio": args.keep_ratio,
        "keep_tokens": keep_tokens,
        "repeats": args.repeats,
        "n_needles_per_prefix": args.n_needles,
        "n_queries_total": len(all_rows),
        "n_strong": len(strong),
        "accuracy_all_needles": acc_all,
        "accuracy_strong_needles": acc_strong,
        "kept_jaccard_query_aware_mean": sum(jaccards) / len(jaccards),
        "kept_jaccard_positional": 1.0,
        "per_needle": all_rows,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "reuse_replay.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nstrong needles: {len(strong)}/{len(all_rows)}", flush=True)
    print(
        "ACC(strong)",
        json.dumps({k: round(v, 3) for k, v in acc_strong.items()}),
        flush=True,
    )
    print(
        f"jaccard qaware(mean)={result['kept_jaccard_query_aware_mean']:.3f} "
        f"positional=1.0",
        flush=True,
    )
    print("wrote", args.output_dir, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--context-len", type=int, default=4096)
    ap.add_argument("--n-needles", type=int, default=8)
    ap.add_argument("--keep-ratio", type=float, default=0.08)
    ap.add_argument(
        "--selector",
        choices=["attn", "qk"],
        default="attn",
        help="query-aware scorer: attn=output_attentions (<=4K), "
        "qk=cheap q.k path (scales to 16-32K)",
    )
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
