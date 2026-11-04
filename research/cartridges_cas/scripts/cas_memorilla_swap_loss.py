#!/usr/bin/env python3
"""Measure how much a Memorilla reader uses the patient record.

The training objective (answer-token loss on self-study conversations)
falls both when the reader learns facts from the record and when it
only learns the answer style. This script separates the two: it scores
the validation conversations of a training config once with each
conversation's own patient collection and once with another patient's
collection, for each reader checkpoint given. The gap between the two
losses is the part of the loss that the record explains. If the gap
grows from checkpoint to checkpoint, training is still extracting
record information; if it stays near zero, more of the same training
mostly teaches style.

Env:
    MEM_SRC   checkout of the Memorilla source
    DATA_ROOT data root holding data/<config> and embeddings/ (the
              root the training run used)
    CONFIG    training config (default longhealth_ss4_c256)
    CKPTS     space-separated name=path pairs of reader checkpoints
    SWAP      space-separated a:b collection pairs (default
              1:3 3:5 5:6 6:1, a cycle over the trained patients)
    OUT_JSON  result path
    DECODER   default Qwen/Qwen3-8B
    ENCODER   default Qwen/Qwen3-Embedding-4B
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ["MEM_SRC"])
from memorilla.data import (
    COLLECTION_COLUMN,
    MemoryCollator,
    MemoryDataset,
)  # noqa: E402
from memorilla.embeddings import load_split_with_embeddings  # noqa: E402
from memorilla.llm import MemoryLLM  # noqa: E402
from memorilla.memory import MemoryModule  # noqa: E402

CONFIG = os.environ.get("CONFIG", "longhealth_ss4_c256")
DECODER = os.environ.get("DECODER", "Qwen/Qwen3-8B")
ENCODER = os.environ.get("ENCODER", "Qwen/Qwen3-Embedding-4B")
SPLIT = "validation"
COLUMNS = ["collection_id", "question", "answer"]


class SwappedDataset(MemoryDataset):
    """A dataset whose rows read another collection's documents."""

    def __init__(self, parts, swap):
        super().__init__(parts)
        self.swap = swap

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item[COLLECTION_COLUMN] = self.swap[item[COLLECTION_COLUMN]]
        return item


def losses(llm, dataset, collator):
    """Per-row mean answer loss and answer token count, batch size 1."""
    out = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            batch = {k: v.cuda() for k, v in collator([item]).items()}
            loss = llm(**batch).loss.float().item()
            ntok = int((batch["labels"] != -100).sum().item()) - 1
            out.append((dataset[i][COLLECTION_COLUMN], loss, ntok))
    return out


def summary(rows):
    by = {}
    for cid, loss, ntok in rows:
        by.setdefault(cid, []).append((loss, ntok))
    per = {str(c): sum(loss for loss, _ in v) / len(v) for c, v in sorted(by.items())}
    return dict(
        mean_row_loss=sum(r[1] for r in rows) / len(rows),
        token_weighted_loss=sum(r[1] * r[2] for r in rows) / sum(r[2] for r in rows),
        per_collection_row_loss=per,
    )


def main():
    root = os.environ["DATA_ROOT"]
    dataset, store = load_split_with_embeddings(CONFIG, SPLIT, root, COLUMNS, ENCODER)
    parts = [(CONFIG, SPLIT, dataset)]
    swap = dict(
        tuple(int(x) for x in p.split(":"))
        for p in os.environ.get("SWAP", "1:3 3:5 5:6 6:1").split()
    )
    own = MemoryDataset(parts)
    other = SwappedDataset(parts, swap)
    ckpts = [p.split("=", 1) for p in os.environ["CKPTS"].split()]

    memory = MemoryModule.from_pretrained(ckpts[0][1])
    llm = MemoryLLM(DECODER, memory, attn_implementation="sdpa").cuda().eval()
    collator = MemoryCollator(
        llm.tokenizer, {(CONFIG, SPLIT): store}, memory.config.num_memories, 2048
    )
    result = dict(config=CONFIG, split=SPLIT, rows=len(own), swap=swap, ckpts={})
    for name, path in ckpts:
        llm.memory = MemoryModule.from_pretrained(path).to(torch.bfloat16).cuda().eval()
        entry = {}
        for label, ds in (("own", own), ("swapped", other)):
            t0 = time.time()
            entry[label] = summary(losses(llm, ds, collator))
            entry[label]["seconds"] = round(time.time() - t0, 1)
        entry["gap_row"] = (
            entry["swapped"]["mean_row_loss"] - entry["own"]["mean_row_loss"]
        )
        entry["gap_token"] = (
            entry["swapped"]["token_weighted_loss"]
            - entry["own"]["token_weighted_loss"]
        )
        result["ckpts"][name] = entry
        print(
            f"SWAPLOSS,{name},own={entry['own']['mean_row_loss']:.4f},"
            f"swapped={entry['swapped']['mean_row_loss']:.4f},"
            f"gap={entry['gap_row']:.4f},gap_tok={entry['gap_token']:.4f}",
            flush=True,
        )
    with open(os.environ["OUT_JSON"], "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
