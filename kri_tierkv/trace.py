# SPDX-License-Identifier: GPL-2.0
"""JSONL trace + emulation metrics for KRI-TierKV.

The emulation records, per probe/step, which blocks each policy would keep,
fetch, and skip, and (when attention weights are available) how much attention
mass the kept set captures. From that we can answer the only question milestone 1
needs to answer: does KRI-D-sum pick the blocks that actually carry attention
mass, better than FIFO or recent-only?
"""

from __future__ import annotations

import json
import os


class TraceWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w")

    def write(self, record: dict):
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def attention_mass_recall(selected_ids, mass_scores) -> float:
    """Fraction of total attention mass that the selected blocks capture."""
    sel = set(selected_ids)
    total = sum(mass_scores)
    if total <= 0:
        return 0.0
    return sum(mass_scores[b] for b in sel if b < len(mass_scores)) / total


def high_mass_false_negative_rate(selected_ids, mass_scores, quantile=0.9) -> float:
    """Fraction of high-mass blocks (top decile by attention mass) that the
    selection dropped. A policy that skips blocks the model actually attends to
    is losing information regardless of its byte savings.
    """
    if not mass_scores:
        return 0.0
    # rank only blocks that actually carry mass -- a zero-mass block is never a
    # "high-mass" block even if the cut would otherwise reach it.
    positive = [b for b in range(len(mass_scores)) if mass_scores[b] > 0]
    ordered = sorted(positive, key=lambda b: mass_scores[b], reverse=True)
    cut = max(1, int(len(mass_scores) * (1.0 - quantile)))
    high = set(ordered[:cut])
    if not high:
        return 0.0
    sel = set(selected_ids)
    missed = sum(1 for b in high if b not in sel)
    return missed / len(high)


def bytes_moved_per_token(
    selected_ids, index, n_layers, n_kv_heads, head_dim, k_bits, v_bits
) -> float:
    """Estimated slow-tier bytes fetched per decode token = bytes of the fetched
    slow blocks. Fast-tier blocks are resident and not counted as movement.
    """
    from .block_index import FAST

    per_block = index.bytes_per_block(n_layers, n_kv_heads, head_dim, k_bits, v_bits)
    fast = {b.block_id for b in index.blocks if b.tier == FAST}
    fetched_slow = [b for b in selected_ids if b not in fast]
    return len(fetched_slow) * per_block
