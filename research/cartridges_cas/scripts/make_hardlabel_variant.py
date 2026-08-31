#!/usr/bin/env python3
"""Reduce every target row to the teacher's chosen token alone.

The duplicate-target defect happened to answer a question nobody had
asked: it collapsed rows from 3.66 entries to 1.09 unique ones, which
is very nearly a hard label, and the cartridges trained on it were
indistinguishable on document specificity from cartridges trained on
the full distribution.  That is suggestive but confounded -- the
damaged rows also double-count the chosen token, so its weight is not
the weight a clean hard label would carry.

This builds the unconfounded arm.  Each row keeps its argmax token and
nothing else, at probability one, so the row's total coefficient mass
matches the clean row's ~0.998 and the only thing removed is the
teacher's distribution over alternatives.  Train on this against the
clean parquet and the comparison asks exactly one question: what does
distilling the teacher's top-k buy over imitating its argmax?

A row that is already a single entry is unchanged, so the transform is
idempotent.

Env: IN_PARQUET, OUT_PARQUET.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

IN_PARQUET = os.environ["IN_PARQUET"]
OUT_PARQUET = os.environ["OUT_PARQUET"]


def hard_label(token_idx, token_id, logprobs):
    """Keep only each row's most likely token, at probability one."""
    token_idx = np.asarray(token_idx)
    token_id = np.asarray(token_id)
    logprobs = np.asarray(logprobs, dtype=np.float64)
    out_idx, out_id, out_lp = [], [], []
    stats = dict(rows=0, dropped_alternatives=0, already_single=0)

    for r in np.unique(token_idx):
        m = token_idx == r
        ids, lps = token_id[m], logprobs[m]
        if ids.size == 0:
            continue
        stats["rows"] += 1
        if ids.size == 1:
            stats["already_single"] += 1
        stats["dropped_alternatives"] += ids.size - 1
        top = int(np.argmax(lps))
        out_idx.append(int(r))
        out_id.append(int(ids[top]))
        # probability one: the arm differs from clean in the shape of the
        # target, not in how much total weight the row carries
        out_lp.append(0.0)
    return out_idx, out_id, out_lp, stats


def main():
    t = pq.read_table(IN_PARQUET)
    rows = t.to_pylist()
    agg = dict(rows=0, dropped_alternatives=0, already_single=0, messages=0)
    for rec in rows:
        for msg in rec.get("messages") or []:
            tl = msg.get("top_logprobs")
            if not tl or not tl.get("token_idx"):
                continue
            idx, tid, lp, st = hard_label(
                tl["token_idx"], tl["token_id"], tl["logprobs"]
            )
            tl["token_idx"], tl["token_id"], tl["logprobs"] = idx, tid, lp
            if tl.get("shape") is not None:
                tl["shape"] = [len(idx), 1]
            agg["messages"] += 1
            for k in ("rows", "dropped_alternatives", "already_single"):
                agg[k] += st[k]
    out = pa.Table.from_pylist(rows, schema=t.schema)
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, OUT_PARQUET)
    agg["entries_per_row_after"] = 1.0
    print(json.dumps(agg, indent=1))
    print(f"HARDLABEL_VARIANT_DONE {OUT_PARQUET}")


if __name__ == "__main__":
    main()
