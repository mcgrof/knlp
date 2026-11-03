#!/usr/bin/env python3
"""Check the Cartridges library's packing against the unpacked targets.

Builds the library's own TrainDataset for one self-study parquet, once
per packing mode, and collates every pack exactly as training does.
Each pack's teacher targets are then matched against the same
conversations converted one at a time with nothing packed or cut (the
reference). A target is identified by (conversation, position in that
conversation, top-k slot), so the check compares identities and order,
not counts.

Reported per mode: packs, conversations placed, conversations whose
targets are complete, partial or absent, targets lost, targets seen
more than once, empty-target packs, packs over the length limit, and
whether every retained target points at the same token, top-k id and
log-probability as the reference, at a position whose input token is
the reference's input token. In pad mode every conversation no longer
than PACKED_LEN must be complete, and the check exits nonzero if not.

Env: DATA_PARQUET, PACKED_LEN (8192), MODES (pad,truncate), SEED (42),
LIMIT (rows, default all), CARTRIDGES_DIR, MODEL.
"""

import os
import sys

sys.path.insert(0, os.environ.get("CARTRIDGES_DIR", "cartridges"))

import torch
from transformers import AutoTokenizer

from cartridges.datasets import DataSource, TrainDataset

DATA = os.environ["DATA_PARQUET"]
S = int(os.environ.get("PACKED_LEN", "8192"))
MODES = os.environ.get("MODES", "pad,truncate").split(",")
SEED = int(os.environ.get("SEED", "42"))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")


def check(mode, tok):
    cfg = TrainDataset.Config(
        data_sources=[DataSource(path=DATA, type="local", limit=LIMIT)],
        top_k_logits=20,
        packed_seq_length=S,
        packing_mode=mode,
    )
    ds = cfg.instantiate(tokenizer=tok, seed=SEED)
    ref = ds.elements  # one element per conversation, never cut
    ntgt = [len(e.topk_token_idxs) for e in ref]
    seen = [0] * len(ref)
    placed = [0] * len(ref)
    bad_match = empty = over = 0
    for b, elem_ids in enumerate(ds.batches):
        batch = ds.collate([ds.elements[i] for i in elem_ids])
        if len(batch.input_ids) > S:
            over += 1
        if len(batch.topk_token_idxs) == 0:
            empty += 1
        # the collated targets are the per-element targets concatenated
        # in element order, shifted by each element's start, then cut
        off, cursor = 0, 0
        for i in elem_ids:
            placed[i] += 1
            e = ref[i]
            m = len(e.topk_token_idxs)
            want = e.topk_token_idxs + off
            got = batch.topk_token_idxs[cursor : cursor + m]
            n = len(got)
            if n and not (
                torch.equal(got, want[:n])
                and torch.equal(
                    batch.topk_token_ids[cursor : cursor + n], e.topk_token_ids[:n]
                )
                and torch.equal(
                    batch.topk_logprobs[cursor : cursor + n], e.topk_logprobs[:n]
                )
                and torch.equal(
                    batch.input_ids[got], e.input_ids[e.topk_token_idxs[:n]]
                )
            ):
                bad_match += 1
            seen[i] += n
            cursor += n
            off += len(e.input_ids)
        if cursor != len(batch.topk_token_idxs):
            bad_match += 1
    complete = sum(1 for i in range(len(ref)) if seen[i] == ntgt[i])
    partial = sum(1 for i in range(len(ref)) if 0 < seen[i] < ntgt[i])
    absent = sum(1 for i in range(len(ref)) if seen[i] == 0)
    dup = sum(1 for i in range(len(ref)) if placed[i] > 1)
    oversize = sum(1 for e in ref if len(e.input_ids) > S)
    lost = sum(ntgt) - sum(seen)
    print(
        f"PACKCHECK mode={mode} len={S} packs={len(ds.batches)} "
        f"conversations={len(ref)} placed_once={sum(1 for p in placed if p == 1)} "
        f"placed_twice_or_more={dup} complete={complete} partial={partial} "
        f"absent={absent} oversize={oversize} targets={sum(ntgt)} lost={lost} "
        f"empty_packs={empty} over_length={over} mismatches={bad_match}",
        flush=True,
    )
    ok = dup == 0 and bad_match == 0 and over == 0 and empty == 0
    if mode == "pad":
        ok = ok and partial + absent <= oversize and min(placed) == 1
    return ok


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    results = {m: check(m, tok) for m in MODES}
    print(
        "PACKCHECK_RESULT "
        + " ".join(f"{m}={'PASS' if v else 'FAIL'}" for m, v in results.items())
    )
    sys.exit(0 if results.get("pad", True) else 1)


if __name__ == "__main__":
    main()
