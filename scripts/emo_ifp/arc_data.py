#!/usr/bin/env python3
"""Build the ARC-Challenge calibration and screening sets (CPU).

ARC-Challenge comes from the AI2 Reasoning Challenge dataset
(allenai/ai2_arc on the Hugging Face Hub). Prompts use a ranked-choice
format: five fixed demonstrations from the train split, then

    Question: <question>
    Answer:<space><choice text>

Calibration examples come from the validation split and screening examples
from the test split, so the two sets share no question. Each set is a seeded
permutation prefix. Calibration texts are the prompt plus the correct
answer, matching how EMO computes router statistics for expert selection.

Usage:
    arc_data.py OUT_DIR [--revision REV] [--n-cal 8] [--n-screen 64]
"""

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_json  # noqa

DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Challenge"
N_SHOTS = 5


def fmt_q(q):
    return f"Question: {q}\nAnswer:"


def build_prefix(shots):
    parts = []
    for ex in shots:
        ans = ex["choices"]["text"][ex["choices"]["label"].index(ex["answerKey"])]
        parts.append(f"{fmt_q(ex['question'])} {ans}")
    return "\n\n".join(parts) + "\n\n"


def to_item(ex, prefix):
    labels = ex["choices"]["label"]
    return {
        "id": ex["id"],
        "context": prefix + fmt_q(ex["question"]),
        "choices": [" " + t for t in ex["choices"]["text"]],
        "labels": labels,
        "gold": labels.index(ex["answerKey"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--n-screen", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    rev = args.revision or HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, CONFIG, revision=rev)

    rng = random.Random(args.seed)
    tr = list(range(len(ds["train"])))
    rng.shuffle(tr)
    shots = [ds["train"][i] for i in tr[:N_SHOTS]]
    prefix = build_prefix(shots)

    def take(split, n):
        idx = list(range(len(ds[split])))
        random.Random(args.seed).shuffle(idx)
        out = []
        for i in idx:
            ex = ds[split][i]
            # Keep four- and five-choice items with a resolvable answer key.
            if ex["answerKey"] not in ex["choices"]["label"]:
                continue
            out.append(to_item(ex, prefix))
            if len(out) == n:
                break
        return out

    cal = take("validation", args.n_cal)
    screen = take("test", args.n_screen)
    shot_q = {s["question"] for s in shots}
    cal_q = {c["context"] for c in cal}
    scr_q = {c["context"] for c in screen}
    assert not (cal_q & scr_q), "calibration and screen overlap"
    for s in cal + screen:
        assert not any(q in s["context"].split(prefix, 1)[1] for q in shot_q)

    cal_texts = [c["context"] + c["choices"][c["gold"]] for c in cal]
    os.makedirs(args.out_dir, exist_ok=True)
    files = {
        "calibration.json": {"items": cal, "texts": cal_texts},
        "screen.json": {"items": screen},
    }
    hashes = {}
    for fn, obj in files.items():
        p = os.path.join(args.out_dir, fn)
        write_json(p, obj)
        hashes[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "revision": rev,
        "seed": args.seed,
        "format": "ranked choice: 5 train demonstrations, 'Question: q\\nAnswer:' "
        "context, ' choice' continuation",
        "shot_ids": [s["id"] for s in shots],
        "calibration": {"split": "validation", "ids": [c["id"] for c in cal]},
        "screen": {"split": "test", "ids": [c["id"] for c in screen]},
        "file_sha256": hashes,
        "split_sizes": {k: len(v) for k, v in ds.items()},
    }
    write_json(os.path.join(args.out_dir, "data-manifest.json"), manifest)
    print(json.dumps(manifest, indent=1)[:1500])


if __name__ == "__main__":
    main()
