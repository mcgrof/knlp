#!/usr/bin/env python3
"""Build GSM8K calibration texts and request prompts (CPU).

GSM8K is the grade-school math word problem dataset (openai/gsm8k on the
Hugging Face Hub, config "main"). It gives a second domain, unlike
ARC-Challenge's science questions, for building a second expert pool.
Prompts use five fixed demonstrations from the train split, then

    Question: <question>
    Answer:<space><reference solution>

Calibration examples come from the rest of the train split and request
prompts from the test split, so the two sets share no question. Each set
is a seeded permutation prefix. Calibration texts are the prompt plus the
reference solution, as in arc_data.py. The request file has the same
``items[].context`` layout as arc_data.py's screen file, so
layer_offload.build_requests can read either.

Usage:
    gsm8k_data.py OUT_DIR [--revision REV] [--n-cal 8] [--n-requests 64]
"""

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_json  # noqa

DATASET = "openai/gsm8k"
CONFIG = "main"
N_SHOTS = 5


def fmt_q(q):
    return f"Question: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--n-requests", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import HfApi

    rev = args.revision or HfApi().dataset_info(DATASET).sha
    ds = load_dataset(DATASET, CONFIG, revision=rev)

    tr = list(range(len(ds["train"])))
    random.Random(args.seed).shuffle(tr)
    shots = [ds["train"][i] for i in tr[:N_SHOTS]]
    prefix = (
        "\n\n".join(f"{fmt_q(s['question'])} {s['answer']}" for s in shots) + "\n\n"
    )

    def item(split, i):
        ex = ds[split][i]
        return {
            "id": f"{split}-{i}",
            "context": prefix + fmt_q(ex["question"]),
            "answer": " " + ex["answer"],
        }

    cal = [item("train", i) for i in tr[N_SHOTS : N_SHOTS + args.n_cal]]
    te = list(range(len(ds["test"])))
    random.Random(args.seed).shuffle(te)
    reqs = [item("test", i) for i in te[: args.n_requests]]
    shot_q = {s["question"] for s in shots}
    for c in cal + reqs:
        assert not any(q in c["context"].split(prefix, 1)[1] for q in shot_q)
    assert not ({c["context"] for c in cal} & {c["context"] for c in reqs})

    os.makedirs(args.out_dir, exist_ok=True)
    files = {
        "calibration.json": {
            "items": cal,
            "texts": [c["context"] + c["answer"] for c in cal],
        },
        "requests.json": {"items": reqs},
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
        "format": "5 train demonstrations, 'Question: q\\nAnswer:' context, "
        "' solution' continuation for calibration",
        "shot_ids": [f"train-{i}" for i in tr[:N_SHOTS]],
        "calibration": {"split": "train", "ids": [c["id"] for c in cal]},
        "requests": {"split": "test", "ids": [c["id"] for c in reqs]},
        "file_sha256": hashes,
        "split_sizes": {k: len(v) for k, v in ds.items()},
    }
    write_json(os.path.join(args.out_dir, "data-manifest.json"), manifest)
    print(json.dumps(manifest, indent=1)[:1500])


if __name__ == "__main__":
    main()
