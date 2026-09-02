#!/usr/bin/env python3
"""Split one patient's self-study parquet into a train and a held-out
validation parquet, deterministically.

The cartridge trainer's only dense signal is the top-20 distillation loss,
and on the training rows that loss measures memorization once the run
has seen every conversation many times over (80 epochs). Holding a fixed
slice of conversations out of training gives a loss the run cannot
memorize, which is the instrument the meta-initialization comparisons
decide on. The slice is chosen by a seeded shuffle, not by taking the
tail of the file, because the synthesis pipeline wrote rows in prompt
order and the tail would over-represent the end of the document.

The synthesis pipeline also produces duplicates: rows that share a
prompt with another row and rows that are byte-identical to another row
(about 4% of patient_02). A split by row index would put copies of
training text into the held-out slice, so the split is by prompt: the
unique prompts are shuffled and whole prompt groups go to the val side
until it holds at least VAL_N rows, so no val prompt ever appears in
training. The duplicate counts go into the split record.

Every arm of one comparison must train on the same train file and be
scored on the same val file, so the split is a function of (parquet,
VAL_N, SEED) only, and the chosen row indices are written next to the
outputs for the record.

Usage:
    PARQUET=patient_02.parquet VAL_N=256 SEED=0 OUT_DIR=split/ \\
        python3 cas_split_val.py

Writes OUT_DIR/<stem>_train.parquet, OUT_DIR/<stem>_val.parquet and
OUT_DIR/<stem>_split.json (indices, counts, seed).
"""

import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PARQUET = os.environ["PARQUET"]
VAL_N = int(os.environ.get("VAL_N", "256"))
SEED = int(os.environ.get("SEED", "0"))
OUT_DIR = Path(os.environ.get("OUT_DIR", os.path.dirname(PARQUET) or "."))


def row_keys(table):
    """(prompt, answer) per row: the first user turn and the first reply.
    Reads only the text fields; decoding the per-token logprobs that sit
    next to them would take minutes per file."""
    msgs = table.column("messages")
    flat = pc.list_flatten(msgs)
    rows = pc.list_parent_indices(msgs).to_pylist()
    roles = pc.struct_field(flat, "role").to_pylist()
    texts = pc.struct_field(flat, "content").to_pylist()
    prompt, answer = {}, {}
    for r, role, text in zip(rows, roles, texts):
        if role == "user":
            prompt.setdefault(r, text)
        elif role == "assistant":
            answer.setdefault(r, text)
    return [(prompt.get(i, ""), answer.get(i, "")) for i in range(table.num_rows)]


def main():
    table = pq.read_table(PARQUET)
    n = table.num_rows
    assert 0 < VAL_N < n, f"VAL_N={VAL_N} must be inside (0, {n})"
    keys = row_keys(table)
    groups = {}
    for i, (prompt, _) in enumerate(keys):
        groups.setdefault(prompt, []).append(i)
    prompts = sorted(groups)
    random.Random(SEED).shuffle(prompts)
    val_idx = []
    for prompt in prompts:
        if len(val_idx) >= VAL_N:
            break
        val_idx.extend(groups[prompt])
    val_set = set(val_idx)
    val_idx = sorted(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]
    val_prompts = {keys[i][0] for i in val_idx}
    assert not any(keys[i][0] in val_prompts for i in train_idx), "prompt leak"
    dup = {
        "unique_prompts": len(groups),
        "unique_pairs": len(set(keys)),
        "rows_sharing_a_prompt": n - len(groups),
        "exact_duplicate_rows": n - len(set(keys)),
    }

    stem = Path(PARQUET).stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / f"{stem}_train.parquet"
    val_path = OUT_DIR / f"{stem}_val.parquet"
    pq.write_table(table.take(pa.array(train_idx)), train_path)
    pq.write_table(table.take(pa.array(val_idx)), val_path)

    meta = {
        "source": os.path.abspath(PARQUET),
        "n_rows": n,
        "val_n_requested": VAL_N,
        "val_n": len(val_idx),
        "train_n": len(train_idx),
        "seed": SEED,
        "split_by": "prompt",
        "duplicates": dup,
        "val_indices": val_idx,
        "train": str(train_path),
        "val": str(val_path),
    }
    with open(OUT_DIR / f"{stem}_split.json", "w") as f:
        json.dump(meta, f)
    print(
        f"SPLIT_DONE {stem}: {n} rows ({dup['unique_prompts']} prompts, "
        f"{dup['exact_duplicate_rows']} exact duplicates) -> train "
        f"{len(train_idx)} ({train_path}), val {len(val_idx)} ({val_path}), "
        f"seed {SEED}, no prompt shared across the split",
        flush=True,
    )


if __name__ == "__main__":
    main()
