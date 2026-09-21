# SPDX-License-Identifier: GPL-2.0
"""Reconstruct the gold answers a finished task run was scored against.

The run that produced the rows wrote what each arm said and whether the scorer
called it correct, but not what the right answer was. That is enough to
reproduce the verdict and not enough to check it, which is the wrong way round:
an audit of the scorer then has nothing to score against, and any doubt about
the scoring rule cannot be settled without re-running the model.

Everything needed is recoverable because the suite is deterministic. The
documents come from a fixed corpus filtered by a fixed rule, the planted code
is a hash of the seed and the title, and the prompt is the first `ctx` tokens
of the assembled text. Regenerating them and checking that the document set
matches the one the run recorded is itself the test that the reconstruction is
faithful; if a single title differs, nothing here should be trusted.

Writes a manifest carrying, per document, the planted code, the depth it was
planted at, the cloze answer and its stem, a hash of the prompt token ids, and
the prompt length. That is what a boundary-aware rescore needs and what the
original run should have written.

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.kv_translate.tasks import build_items, wikitext_documents  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--expect",
        default="",
        help="smoke.json whose document list the reconstruction must match",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.target)

    # Identical to the run's own filter, including the whitespace normalisation
    # on both sides. Reproducing the filter matters as much as reproducing the
    # corpus: a different rejection rule yields a different document set and
    # the manifest would then describe a suite that was never scored.
    fit_corpus = " ".join(
        " ".join(
            t
            for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
            if t.strip()
        ).split()
    )
    docs, dropped, too_short = [], 0, 0
    for title, body in wikitext_documents():
        flat = " ".join(body.split())
        probes = [
            flat[i : i + 160] for i in (400, len(flat) // 2, max(0, len(flat) - 600))
        ]
        probes = [q for q in probes if len(q) >= 120]
        if not probes:
            too_short += 1
            continue
        if (
            any(q in fit_corpus for q in probes)
            or " ".join(title.split()) in fit_corpus
        ):
            dropped += 1
            continue
        docs.append((title, body))
        if len(docs) >= args.docs * 3:
            break

    items = build_items(docs, tok, args.ctx, seed=args.seed)
    order = []
    for it in items:
        if it.doc_id not in order:
            order.append(it.doc_id)
    keep = set(order[: args.docs])
    items = [it for it in items if it.doc_id in keep]

    manifest = {}
    for it in items:
        rec = manifest.setdefault(it.doc_id, {"doc": it.doc_id})
        pid = it.meta.get("prompt_ids")
        if pid is not None:
            rec["prompt_len"] = len(pid)
            rec["prompt_ids_sha256"] = hashlib.sha256(
                json.dumps(pid).encode()
            ).hexdigest()
        if it.kind == "retrieval":
            rec["retrieval_gold"] = it.answer
            rec["retrieval_query"] = it.query
            rec["plant_depth_frac"] = it.meta.get("depth_frac")
        elif it.kind == "cloze":
            rec["cloze_gold"] = it.answer
            rec["cloze_query"] = it.query
            rec["cloze_stem_words"] = it.meta.get("stem_words")
    docs_out = [manifest[d] for d in order[: args.docs]]

    check = {
        "n_documents": len(docs_out),
        "candidates_rejected_as_seen": dropped,
        "candidates_too_short": too_short,
        "all_have_retrieval_gold": all("retrieval_gold" in r for r in docs_out),
        "all_have_cloze_gold": all("cloze_gold" in r for r in docs_out),
        "gold_codes_unique": len({r["retrieval_gold"] for r in docs_out})
        == len(docs_out),
        "gold_code_lengths": sorted({len(r["retrieval_gold"]) for r in docs_out}),
    }
    if args.expect:
        want = json.load(open(args.expect)).get("documents", [])
        got = [r["doc"] for r in docs_out]
        check["expected_documents"] = len(want)
        check["document_sets_match"] = sorted(want) == sorted(got)
        check["missing_from_reconstruction"] = sorted(set(want) - set(got))[:5]
        check["extra_in_reconstruction"] = sorted(set(got) - set(want))[:5]

    out = {
        "config": vars(args),
        "check": check,
        "documents": docs_out,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(check, indent=2))
    print(f"wrote {args.out}")
    return 0 if check.get("document_sets_match", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
