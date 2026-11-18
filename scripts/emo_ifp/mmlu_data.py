#!/usr/bin/env python3
"""Build MMLU category calibration, selector-training and test sets (CPU).

MMLU (Massive Multitask Language Understanding, cais/mmlu on the Hugging
Face Hub) has 57 subjects. EMO evaluates selective expert use on coarse
categories of those subjects, following the grouping of the MMLU authors
(their categories.py), so this script builds one data set per category.

Prompts use the same ranked-choice format as arc_data.py: five fixed
demonstrations, here the dev-split items of the request's own subject,
then

    Question: <question>
    Answer:<space><choice text>

Items come from the validation and test splits of the category's
subjects. Items whose context plus longest choice exceeds --max-context
tokens are dropped (the passage-based history subjects are dropped this
way) and so are repeated questions. A seeded permutation of the rest
gives, in order, the calibration items (EMO pool selection; texts are the
prompt plus the correct answer), the selector-training items and the
test items, so the three sets share no question.

Usage:
    mmlu_data.py TOKENIZER_DIR OUT_ROOT --domain health --domain psychology
        [--n-cal 8] [--n-train 128] [--n-test 128] [--max-context 2048]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_data import fmt_q  # noqa
from common import write_json  # noqa

DATASET = "cais/mmlu"
N_SHOTS = 5
LABELS = ["A", "B", "C", "D"]

# Subject to category, from the MMLU authors' categories.py, for the
# categories this project uses.
CATEGORIES = {
    "health": [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "medical_genetics",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
    "psychology": ["high_school_psychology", "professional_psychology"],
    "history": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
    ],
    "politics": [
        "high_school_government_and_politics",
        "public_relations",
        "security_studies",
        "us_foreign_policy",
    ],
}


def build_prefix(shots):
    return (
        "\n\n".join(
            f"{fmt_q(s['question'])} {s['choices'][s['answer']]}" for s in shots
        )
        + "\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_dir")
    ap.add_argument("out_root")
    ap.add_argument("--domain", action="append", choices=sorted(CATEGORIES))
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa
    rev = args.revision or HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, "all", revision=rev)

    for dom in args.domain:
        subjects = CATEGORIES[dom]
        prefix = {}
        for subj in subjects:
            shots = [ex for ex in ds["dev"] if ex["subject"] == subj]
            assert len(shots) == N_SHOTS, (subj, len(shots))
            prefix[subj] = build_prefix(shots)
        # A demonstration must not reappear as a request.
        seen = {ex["question"].strip() for ex in ds["dev"] if ex["subject"] in prefix}
        pool, dropped = [], Counter()
        for split in ("validation", "test"):
            for i, ex in enumerate(ds[split]):
                subj = ex["subject"]
                if subj not in prefix:
                    continue
                q = ex["question"].strip()
                if q in seen:
                    dropped[f"{subj}:repeat"] += 1
                    continue
                ctx = prefix[subj] + fmt_q(ex["question"])
                choices = [" " + c for c in ex["choices"]]
                if ntok(ctx) + max(ntok(c) for c in choices) > args.max_context:
                    dropped[f"{subj}:long"] += 1
                    continue
                seen.add(q)
                pool.append(
                    {
                        "id": f"{subj}/{split}/{i}",
                        "subject": subj,
                        "context": ctx,
                        "choices": choices,
                        "labels": LABELS[: len(choices)],
                        "gold": int(ex["answer"]),
                    }
                )
        random.Random(args.seed).shuffle(pool)
        need = args.n_cal + args.n_train + args.n_test
        assert len(pool) >= need, f"{dom}: {len(pool)} items, need {need}"
        cal = pool[: args.n_cal]
        train = pool[args.n_cal : args.n_cal + args.n_train]
        test = pool[args.n_cal + args.n_train : need]

        out = os.path.join(args.out_root, f"mmlu-{dom}")
        os.makedirs(out, exist_ok=True)
        files = {
            "calibration.json": {
                "items": cal,
                "texts": [c["context"] + c["choices"][c["gold"]] for c in cal],
            },
            "train.json": {"items": train},
            "test.json": {"items": test},
        }
        hashes = {}
        for fn, obj in files.items():
            p = os.path.join(out, fn)
            write_json(p, obj)
            hashes[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        manifest = {
            "dataset": DATASET,
            "config": "all",
            "revision": rev,
            "category": dom,
            "subjects": subjects,
            "seed": args.seed,
            "format": "ranked choice: the subject's 5 dev demonstrations, "
            "'Question: q\\nAnswer:' context, ' choice' continuation",
            "max_context_tokens": args.max_context,
            "candidates_after_filters": len(pool),
            "dropped": dict(sorted(dropped.items())),
            "subject_counts": {
                name: dict(Counter(it["subject"] for it in part))
                for name, part in (
                    ("calibration", cal),
                    ("train", train),
                    ("test", test),
                )
            },
            "calibration_ids": [c["id"] for c in cal],
            "train_ids": [c["id"] for c in train],
            "test_ids": [c["id"] for c in test],
            "file_sha256": hashes,
        }
        write_json(os.path.join(out, "data-manifest.json"), manifest)
        print(
            dom,
            len(pool),
            "candidates; dropped",
            dict(dropped),
            "test subjects",
            manifest["subject_counts"]["test"],
            flush=True,
        )


if __name__ == "__main__":
    main()
