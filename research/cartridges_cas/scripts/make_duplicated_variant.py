#!/usr/bin/env python3
"""Re-introduce the duplicate-target defect into a clean dataset.

To measure what the bug cost, comparing a freshly synthesized dataset
against the old one confounds two things: the row construction changed,
but so did every question and answer, because synthesis cannot be
replayed.  A fair comparison needs the same content under both
constructions.

So take a correctly synthesized parquet and derive its damaged twin,
reproducing exactly what the old generator did: prepend the sampled
token to its own top-k row, so the trainer counts it twice, and let the
cumulative-mass truncation see the doubled mass and stop early,
discarding the alternatives it would otherwise have kept.  Both effects
are reproduced, because the second is the one no read-time transform
can undo -- alternatives dropped at synthesis are simply absent.

Train on both and the only difference is the defect.

Env: IN_PARQUET, OUT_PARQUET, MASS_THRESHOLD (default 0.998).
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

IN_PARQUET = os.environ["IN_PARQUET"]
OUT_PARQUET = os.environ["OUT_PARQUET"]
MASS_THRESHOLD = float(os.environ.get("MASS_THRESHOLD", "0.998"))


def damage(token_idx, token_id, logprobs, shape):
    """Rebuild the flat rows the way the defective generator did."""
    token_idx = np.asarray(token_idx)
    token_id = np.asarray(token_id)
    logprobs = np.asarray(logprobs, dtype=np.float64)
    out_idx, out_id, out_lp = [], [], []
    stats = dict(rows=0, duplicated=0, dropped_alternatives=0)

    for r in np.unique(token_idx):
        m = token_idx == r
        ids, lps = token_id[m], logprobs[m]
        if ids.size == 0:
            continue
        stats["rows"] += 1
        # the chosen token is the row's top-1 under greedy synthesis
        top = int(np.argmax(lps))
        # prepend it to its own row: this is the defect
        ids = np.concatenate([[ids[top]], ids])
        lps = np.concatenate([[lps[top]], lps])
        stats["duplicated"] += 1
        # the truncation then sees the doubled mass and stops early
        probs = np.exp(lps)
        keep = len(ids)
        cum = np.cumsum(probs)
        crossed = np.nonzero(cum >= MASS_THRESHOLD)[0]
        if crossed.size:
            keep = int(crossed[0]) + 1
        stats["dropped_alternatives"] += len(ids) - keep
        out_idx.extend([int(r)] * keep)
        out_id.extend(int(x) for x in ids[:keep])
        out_lp.extend(float(x) for x in lps[:keep])
    return out_idx, out_id, out_lp, stats


def main():
    t = pq.read_table(IN_PARQUET)
    rows = t.to_pylist()
    agg = dict(rows=0, duplicated=0, dropped_alternatives=0, messages=0)
    for rec in rows:
        for msg in rec.get("messages") or []:
            tl = msg.get("top_logprobs")
            if not tl or not tl.get("token_idx"):
                continue
            idx, tid, lp, st = damage(
                tl["token_idx"], tl["token_id"], tl["logprobs"], tl.get("shape")
            )
            tl["token_idx"], tl["token_id"], tl["logprobs"] = idx, tid, lp
            if tl.get("shape") is not None:
                tl["shape"] = [len(idx), 1]
            agg["messages"] += 1
            for k in ("rows", "duplicated", "dropped_alternatives"):
                agg[k] += st[k]
    out = pa.Table.from_pylist(rows, schema=t.schema)
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, OUT_PARQUET)
    agg["duplicate_row_fraction"] = agg["duplicated"] / max(agg["rows"], 1)
    print(json.dumps(agg, indent=1))
    print(f"DUPLICATED_VARIANT_DONE {OUT_PARQUET}")


if __name__ == "__main__":
    main()
