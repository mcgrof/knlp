# SPDX-License-Identifier: GPL-2.0
"""Assign every document one role, and record which, before any training.

The correction was previously trained on token windows cut from a
concatenated corpus. Windows do not respect article boundaries, so two of them
can come from one article and a disjointness assertion over window indices
does not establish that the training and evaluation data are independent.
Worse for the experiment now planned, enlarging the window count in the old
runner also redraws the calibration set and refits the affine map, so a
comparison between two window counts varies several things at once and cannot
measure what more correction data is worth.

This builds the roles as whole documents instead, hashes them, and refuses to
let one article hold two roles. The unit is the article; a different planted
code or a different window into the same article is the same document and is
not independent evidence.

Roles, all disjoint by article:

    train77      the repaired anchor's training set
    val19        held back to check checkpoints; no learning-rate search
    train154     a superset of train77, for the first data rung
    train308     a superset of train154, for the conditional second rung
    dev64        the existing task-development documents, fixed
    confirm128   untouched until one frozen candidate is confirmed
    reserve384   the extension, opened only if the first look is inconclusive

The confirmation and reserve sets are drawn from a different slice of the
corpus than the training and development sets, so that a candidate selected on
development is not confirmed on its neighbours.

The old WikiText-2 test corpus is excluded wholesale from every new role. Its
article provenance was never recorded, so any claim that a new document is
independent of it would be an assumption rather than a check.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.tasks import wikitext_documents  # noqa: E402


def doc_hash(title: str, body: str) -> str:
    h = hashlib.sha256()
    h.update(title.encode())
    h.update(b"\0")
    h.update(" ".join(body.split()).encode())
    return h.hexdigest()[:32]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-manifest", required=True, help="the fixed 64 dev documents")
    ap.add_argument(
        "--train-split",
        default="train[:3%]",
        help="corpus slice for training and validation roles",
    )
    ap.add_argument(
        "--confirm-split",
        default="train[40%:44%]",
        help="a different slice, for confirmation and reserve",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)
    need = args.ctx + args.cont_len

    # Every article of the corpus the old windows were cut from, excluded by
    # content rather than by name, since that corpus was never recorded at
    # article granularity.
    legacy = " ".join(
        " ".join(
            t
            for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
            if t.strip()
        ).split()
    )

    dev = json.load(open(args.dev_manifest))
    dev_titles = {d["doc"] for d in dev["documents"]}

    def usable(split, exclude_titles, want):
        """Articles long enough to carry one training window, in corpus order."""
        out, seen = [], set()
        for title, body in wikitext_documents(split=split, max_docs=100000):
            if title in seen or title in exclude_titles:
                continue
            seen.add(title)
            flat = " ".join(body.split())
            probe = flat[400:560]
            if len(probe) < 120 or probe in legacy or " ".join(title.split()) in legacy:
                continue
            ids = tok(body, return_tensors="pt").input_ids[0]
            if ids.shape[0] < need:
                continue
            out.append(
                {
                    "doc": title,
                    "sha256": doc_hash(title, body),
                    "n_tokens": int(ids.shape[0]),
                    "window": [0, need],
                    "split": split,
                }
            )
            if len(out) >= want:
                break
        return out

    # Training and validation come from the same slice the development set was
    # drawn from, with the development titles removed first so no article can
    # be both trained on and scored on.
    pool = usable(args.train_split, dev_titles, 308 + 19 + 40)
    if len(pool) < 308 + 19:
        raise SystemExit(
            f"INVALID: only {len(pool)} usable training-side documents, need 327"
        )
    train308 = pool[:308]
    val19 = pool[308 : 308 + 19]
    roles = {
        "train77": train308[:77],
        "val19": val19,
        "train154": train308[:154],
        "train308": train308,
        "dev64": [
            {"doc": d["doc"], "sha256": None, "window": [0, args.ctx], "split": "dev"}
            for d in dev["documents"]
        ],
    }

    held = usable(args.confirm_split, dev_titles, 128 + 384)
    if len(held) < 128 + 384:
        raise SystemExit(
            f"INVALID: only {len(held)} usable held-out documents, need 512"
        )
    roles["confirm128"] = held[:128]
    roles["reserve384"] = held[128 : 128 + 384]

    # No article in two roles. Nesting is deliberate and declared; everything
    # else must be disjoint.
    # Declared as unordered pairs: the loop below walks roles alphabetically,
    # and "train154" sorts before "train77", so an ordered declaration silently
    # fails to match the pair it was written for.
    nested = {
        frozenset(("train77", "train154")),
        frozenset(("train77", "train308")),
        frozenset(("train154", "train308")),
    }
    titles = {k: {d["doc"] for d in v} for k, v in roles.items()}
    clashes = []
    for a in sorted(titles):
        for b in sorted(titles):
            if a >= b or frozenset((a, b)) in nested:
                continue
            both = titles[a] & titles[b]
            if both:
                clashes.append(
                    {"roles": [a, b], "n": len(both), "examples": sorted(both)[:3]}
                )
    if clashes:
        raise SystemExit(
            "INVALID: articles hold more than one role: " + json.dumps(clashes)
        )

    for outer, inner in (("train154", "train77"), ("train308", "train154")):
        if not titles[inner] <= titles[outer]:
            raise SystemExit(f"INVALID: {outer} does not contain {inner}")

    def rhash(v):
        h = hashlib.sha256()
        for d in v:
            h.update(d["doc"].encode())
            h.update(b"\0")
        return h.hexdigest()[:32]

    out = {
        "config": vars(args),
        "tokenizer": args.target,
        "dataset": "wikitext/wikitext-103-raw-v1",
        "legacy_corpus_excluded": "wikitext-2-raw-v1 test, by content",
        "role_sizes": {k: len(v) for k, v in roles.items()},
        "role_hashes": {k: rhash(v) for k, v in roles.items()},
        "nested_declared": sorted("/".join(sorted(p)) for p in nested),
        "roles": roles,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("%-12s %6s  %s" % ("role", "n", "hash"))
    for k in (
        "train77",
        "val19",
        "train154",
        "train308",
        "dev64",
        "confirm128",
        "reserve384",
    ):
        print("%-12s %6d  %s" % (k, len(roles[k]), out["role_hashes"][k]))
    print()
    print(f"no article holds two roles; nesting verified; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
