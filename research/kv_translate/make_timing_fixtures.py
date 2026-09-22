# SPDX-License-Identifier: GPL-2.0
"""Freeze the token sequences the latency diagnostic will time.

Timing fixtures are built once on the CPU and shipped as token ids, so the
measured card never downloads a corpus, never tokenises, and never sees a
sampling decision. Every length uses a prefix of the same sequence, which is
what makes 512, 2048 and 4096 comparable: they differ in how much of one
document stream is present, not in which text it is.

The text comes from the training role. Held-out roles are not opened here --
these inputs support timing and nothing else, and reading a confirmation or
reserve document to time a matrix multiply would spend a held-out set on a
question that does not need it.

Env: needs the corpus and tokenizer available; run on the CPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.tasks import wikitext_documents  # noqa: E402

FORBIDDEN_ROLES = ("confirm128", "reserve384")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--role", default="train308", help="training role to draw from")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-fixtures", type=int, default=3)
    ap.add_argument("--length", type=int, default=4096, help="tokens per fixture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.role not in FORBIDDEN_ROLES, f"{args.role} is held out"

    from transformers import AutoTokenizer

    man = json.load(open(args.manifests))
    assert args.role in man["roles"], f"{args.role} not in manifest"
    wanted = [e["doc"] for e in man["roles"][args.role]]
    split = man["roles"][args.role][0]["split"]

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    by_title = dict(wikitext_documents(split=split, max_docs=100000))
    missing = [d for d in wanted if d not in by_title]
    assert not missing, f"{len(missing)} manifest documents not found in {split}"

    # Deterministic: manifest order, whole documents, concatenated with a
    # single blank line, cut at exactly `length` tokens. Recorded rather than
    # described, so the same sequence can be rebuilt without this script.
    fixtures, cursor = [], 0
    for fi in range(args.n_fixtures):
        used, ids = [], []
        while len(ids) < args.length:
            if cursor >= len(wanted):
                raise SystemExit(
                    f"ran out of {args.role} documents building fixture {fi}"
                )
            title = wanted[cursor]
            cursor += 1
            piece = tok(by_title[title] + "\n\n", add_special_tokens=False).input_ids
            ids.extend(piece)
            used.append({"doc": title, "tokens_contributed": len(piece)})
        ids = ids[: args.length]
        t = torch.tensor(ids, dtype=torch.long)
        fixtures.append(
            {
                "index": fi,
                "ids": t,
                "n_tokens": len(ids),
                "documents": used,
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        )

    payload = {
        "contract": "timing_fixtures_v1",
        "recipe": (
            "documents of the named role, in manifest order, each followed by "
            "one blank line, tokenised without special tokens, concatenated "
            "and cut at the exact token count; fixtures consume the role in "
            "order without reuse"
        ),
        "role": args.role,
        "split": split,
        "dataset": man["dataset"],
        "tokenizer": args.tokenizer,
        "length": args.length,
        "n_fixtures": args.n_fixtures,
        "forbidden_roles_untouched": list(FORBIDDEN_ROLES),
        "fixtures": fixtures,
    }
    payload["joint_sha256"] = hashlib.sha256(
        "".join(f["sha256"] for f in fixtures).encode()
    ).hexdigest()
    torch.save(payload, args.out)

    for f in fixtures:
        print(
            f"fixture {f['index']}: {f['n_tokens']} tokens, "
            f"{len(f['documents'])} documents, {f['sha256'][:16]}"
        )
    print(f"joint {payload['joint_sha256'][:16]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
