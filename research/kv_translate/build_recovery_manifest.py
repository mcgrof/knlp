# SPDX-License-Identifier: GPL-2.0
"""Freeze the 4,096-token documents the recovery diagnostic will be scored on.

The plan asks for sixteen documents with their planted facts spread evenly
across four depth quartiles, and it is specific that appending filler to the
old short documents does not populate those quartiles: a fact planted inside
the first 512 tokens is in the first eighth of a 4,096-token context no matter
how much text follows it. So each context is rebuilt at full length and the
fact is placed to land in its assigned quartile, with the depth measured in
tokens of the final context rather than in sentences of the source.

Three things are fixed before anything is scored. Which documents are used,
which quartile each is assigned, and where its fact sits. None of the three
may be chosen from results, and the filler is drawn from the training role so
that no document being evaluated becomes another's background.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.tasks import (
    _code,
    _sentences,
    wikitext_documents,
)  # noqa: E402

QUARTILES = ((0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00))
FORBIDDEN_ROLES = ("confirm128", "reserve384")


def token_depth(tok, prompt_ids, code):
    """Where the planted code sits, as a fraction of the final context.

    Measured by decoding prefixes rather than by sentence index, because the
    quartile the plan asks about is a property of the context the receiver
    holds, not of the text it was built from.
    """
    lo, hi = 0, len(prompt_ids)
    text_all = tok.decode(prompt_ids, skip_special_tokens=True)
    if code not in text_all:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if code in tok.decode(prompt_ids[:mid], skip_special_tokens=True):
            hi = mid
        else:
            lo = mid + 1
    return lo / len(prompt_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="the exposed 64-document manifest")
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--filler-role", default="train308")
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.filler_role not in FORBIDDEN_ROLES, "filler must not be a holdout"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)
    gold = json.load(open(args.gold))
    man = json.load(open(args.manifests))
    exposed = [d["doc"] for d in gold["documents"]]

    # dev64 records the placeholder split "dev"; the documents themselves came
    # from the default training split, the same pool the filler is drawn from.
    # build_manifests excludes the dev titles from that pool, so the two are
    # disjoint by construction -- asserted below rather than assumed.
    split = "train[:3%]"
    bodies = dict(wikitext_documents(split=split, max_docs=100000))
    fill_titles = [e["doc"] for e in man["roles"][args.filler_role]]
    fill_split = man["roles"][args.filler_role][0]["split"]
    fill_bodies = (
        bodies
        if fill_split == split
        else dict(wikitext_documents(split=fill_split, max_docs=100000))
    )
    overlap = sorted(set(fill_titles) & set(exposed))
    if overlap:
        raise SystemExit(
            f"BLOCKED: {len(overlap)} filler documents are also evaluated "
            f"documents, first {overlap[:3]}. A document under evaluation "
            "cannot be another's background."
        )
    filler = [fill_bodies[t] for t in fill_titles if t in fill_bodies]
    assert filler, "no filler text available"
    # One deterministic filler stream, shared by every document, so a context's
    # background is a property of the build and not of which document it is.
    filler_text = "\n\n".join(filler)
    filler_ids = tok(filler_text, return_tensors="pt").input_ids[0]

    # Deterministic selection: the exposed pool in its manifest order, every
    # (64/16)th document. No sampling, nothing chosen by outcome.
    stride = max(1, len(exposed) // args.docs)
    chosen = [exposed[i * stride] for i in range(args.docs)]
    chosen = [t for t in chosen if t in bodies][: args.docs]
    if len(chosen) < args.docs:
        raise SystemExit(f"only {len(chosen)} of {args.docs} documents recoverable")

    rng = np.random.default_rng(args.seed)
    out_docs, problems = [], []
    for i, title in enumerate(chosen):
        q = i % 4  # four documents per quartile, assigned by position
        q_lo, q_hi = QUARTILES[q]
        code = _code(f"{args.seed}:{title}")
        planted = f" The archive registration code for {title} is {code}. "

        # Aim the plant at the middle of its quartile, in tokens, then build
        # the context around it out of the document and the shared filler.
        target_frac = (q_lo + q_hi) / 2.0
        target_tok = int(target_frac * args.ctx)

        body_ids = tok(bodies[title], return_tensors="pt").input_ids[0]
        plant_ids = tok(planted, return_tensors="pt").input_ids[0]
        # Lead: the document's own text, topped up from the filler stream so
        # the fact lands where it was assigned rather than wherever the
        # document happens to end.
        need_lead = max(0, target_tok)
        lead = body_ids[:need_lead]
        if lead.shape[0] < need_lead:
            short = need_lead - lead.shape[0]
            off = int(rng.integers(0, max(1, filler_ids.shape[0] - short - 1)))
            lead = torch.cat([lead, filler_ids[off : off + short]])
        rest_need = args.ctx - lead.shape[0] - plant_ids.shape[0]
        tail = body_ids[need_lead : need_lead + rest_need]
        if tail.shape[0] < rest_need:
            short = rest_need - tail.shape[0]
            off = int(rng.integers(0, max(1, filler_ids.shape[0] - short - 1)))
            tail = torch.cat([tail, filler_ids[off : off + short]])
        prompt_ids = torch.cat([lead, plant_ids, tail])[: args.ctx]
        if prompt_ids.shape[0] != args.ctx:
            problems.append(f"{title}: built {prompt_ids.shape[0]} tokens")
            continue

        text = tok.decode(prompt_ids, skip_special_tokens=True)
        if code not in text:
            problems.append(f"{title}: plant fell outside the built window")
            continue
        depth = token_depth(tok, prompt_ids.tolist(), code)
        if depth is None or not (q_lo <= depth < q_hi):
            problems.append(f"{title}: depth {depth} outside quartile {q}")
            continue

        # Cloze rebuilt from the final context, so its answer is a span the
        # receiver's cache actually holds at this length.
        cand = [
            s
            for s in _sentences(text)
            if len(s.split()) >= 10 and code not in s and title not in s
        ]
        if not cand:
            problems.append(f"{title}: no cloze span in the built context")
            continue
        s = cand[int(rng.integers(0, len(cand)))]
        w = s.split()
        cut = max(5, len(w) // 2)
        stem, cl_gold = " ".join(w[:cut]), " ".join(w[cut : cut + 5])
        if not cl_gold:
            problems.append(f"{title}: empty cloze answer")
            continue

        pid = prompt_ids.tolist()
        out_docs.append(
            {
                "doc": title,
                "quartile": q,
                "quartile_range": [q_lo, q_hi],
                "plant_depth_tokens": int(depth * args.ctx),
                "plant_depth_frac": depth,
                "prompt_len": args.ctx,
                "prompt_ids": pid,
                "prompt_ids_sha256": hashlib.sha256(
                    np.asarray(pid, dtype=np.int64).tobytes()
                ).hexdigest(),
                "retrieval_gold": code,
                "retrieval_query": (
                    f"\n\nQuestion: What is the archive registration code for "
                    f"{title}?\nAnswer with the code only.\nAnswer:"
                ),
                "cloze_gold": cl_gold,
                "cloze_query": (
                    "\n\nComplete this sentence from the passage above, exactly "
                    f"as written.\n{stem}"
                ),
                "cloze_stem_words": cut,
                "free_query": (
                    "\n\nContinue the passage above in the same style.\n\nContinuation:"
                ),
            }
        )

    counts = {q: sum(1 for d in out_docs if d["quartile"] == q) for q in range(4)}
    rec = {
        "contract": "recovery_manifest_v1",
        "ctx": args.ctx,
        "docs_requested": args.docs,
        "docs_built": len(out_docs),
        "per_quartile": counts,
        "selection": (
            "the exposed 64-document pool in manifest order, every fourth "
            "document; quartile assigned by position. Neither depends on any "
            "measured outcome."
        ),
        "filler_role": args.filler_role,
        "filler_note": (
            "one deterministic stream built from the training role, shared by "
            "every context, so no evaluated document is another's background"
        ),
        "holdouts_untouched": list(FORBIDDEN_ROLES),
        "seed": args.seed,
        "target": args.target,
        "problems": problems,
        "documents": out_docs,
    }
    rec["joint_sha256"] = hashlib.sha256(
        "".join(d["prompt_ids_sha256"] for d in out_docs).encode()
    ).hexdigest()
    json.dump(rec, open(args.out, "w"), indent=2)

    print(f"built {len(out_docs)} of {args.docs} at {args.ctx} tokens")
    print(f"  per quartile: {counts}")
    for d in out_docs:
        print(f"    q{d['quartile']} depth {d['plant_depth_frac']:.3f} {d['doc'][:44]}")
    if problems:
        print(f"  problems: {len(problems)}: {problems[:3]}")
    print(f"  joint {rec['joint_sha256'][:16]}")
    return 0 if len(out_docs) == args.docs else 1


if __name__ == "__main__":
    raise SystemExit(main())
