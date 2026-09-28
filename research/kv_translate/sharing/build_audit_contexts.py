#!/usr/bin/env python3
"""Short document contexts, to tell a loss at length from a loss of the task.

The mappers were fitted at 1,024 tokens and scored on short multiple-choice
prompts. If they answer the same kind of question correctly when the prefix
is 1,024 tokens and not when it is 4,096 or more, the loss is one of length.
If they fail at 1,024 too, the loss belongs to the task: answering from a
passage by generating, which a likelihood score over fixed choices did not
test.

These contexts are a diagnostic. They are built like the workload's document
contexts, from questions and passages the workload did not use, and they
enter no gate and no estimate of the workload's quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import build_workload as bw
from . import contexts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--documents", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="20260928")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--count", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)

    def encode(text):
        return tok.encode(text, add_special_tokens=False)

    questions = []
    for path in args.documents:
        with open(path) as f:
            questions += [json.loads(x) for x in f if x.strip()]
    questions = [q for q in questions if q.get("answerable", True)]
    pool = bw.DocumentPool(questions, encode, args.seed)
    # replay the workload's own builds, so what it used stays used
    for length, count in bw.DEVELOPMENT.items():
        for i in range(count):
            if pool.build(length, f"dev-doc-{length}-{i:02d}") is None:
                print("the workload's own contexts could not be replayed")
                return 1
    os.makedirs(args.out, exist_ok=True)
    head = len(encode(contexts.DOCUMENT_HEAD))
    records, tries = [], 0
    while len(records) < args.count and tries < 40 * args.count:
        tries += 1
        cid = f"audit-doc-{args.length}-{len(records):02d}"
        ctx = pool.build(args.length, f"audit-{tries}")
        if ctx is None or not bw.plain(ctx["prefix"], head):
            continue
        records.append(
            bw.write_context(
                args.out, cid, f"audit-group-{len(records):05d}", "document",
                args.length, ctx, "musique", {"diagnostic": True},
            )
        )  # fmt: skip
    if len(records) < args.count:
        print(f"only {len(records)} contexts could be built")
        return 1
    with open(os.path.join(args.out, "records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"contexts": len(records), "tries": tries, "length": args.length}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
