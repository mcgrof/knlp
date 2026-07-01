# SPDX-License-Identifier: GPL-2.0
"""Eviction and retrieval policies over the block index.

Eviction decides which slow, non-protected blocks to drop when the fast tier is
over budget. Retrieval decides which blocks a given decode step actually reads:
the recent fast window plus, depending on policy, a top-K set of slow blocks.

All of this is pure Python over the BlockIndex + a per-block score vector, so it
runs in the emulation with no attention changes. The scores come from
scoring.py (KRI-D-sum, recency, fifo, or the oracle attention mass).
"""

from __future__ import annotations

from typing import Optional

from .block_index import BlockIndex, FAST, SLOW
from .config import EvictionPolicy, RetrievalPolicy, TierKVConfig


def evict(
    index: BlockIndex,
    cfg: TierKVConfig,
    scores: list,
    decode_block: Optional[int] = None,
) -> list:
    """Return the block ids to evict so the slow tier fits slow_topk_blocks.

    Keep the slow_topk_blocks highest-scoring evictable blocks; evict the rest.
    FIFO/recency use positional scores; kri_d_sum uses the KRI-D-sum vector.
    Protected blocks are never evicted.
    """
    evictable = index.evictable_ids(decode_block)
    budget = cfg.slow_topk_blocks
    if len(evictable) <= budget:
        return []
    if cfg.eviction_policy == EvictionPolicy.FIFO:
        # oldest first out -> keep the newest `budget`
        ranked = sorted(evictable)  # ascending block_id == oldest first
        return ranked[: len(evictable) - budget]
    if cfg.eviction_policy == EvictionPolicy.RECENCY:
        ranked = sorted(evictable, key=lambda b: b)  # keep newest
        return ranked[: len(evictable) - budget]
    # KRI_D_SUM: keep highest score, evict lowest
    ranked = sorted(evictable, key=lambda b: scores[b])  # ascending score
    return ranked[: len(evictable) - budget]


def apply_eviction(index: BlockIndex, evicted_ids, step: int):
    """Mark evicted blocks (emulation: record the decision, do not free memory)."""
    ev = set(evicted_ids)
    for blk in index.blocks:
        if blk.block_id in ev and blk.evicted_at_step is None:
            blk.evicted_at_step = step


def retrieve(
    index: BlockIndex,
    cfg: TierKVConfig,
    scores: list,
    decode_block: Optional[int] = None,
) -> list:
    """Return the sorted block ids this decode step reads under the policy."""
    n = index.num_blocks
    fast = set(index.fast_ids())
    protected = index.protected_ids(decode_block)

    if cfg.retrieval_policy == RetrievalPolicy.DENSE_REFERENCE:
        return list(range(n))
    if cfg.retrieval_policy == RetrievalPolicy.RECENT_ONLY:
        return sorted(fast | protected)

    # KRI_TOPK / ORACLE_TOPK: fast + protected + top-K slow by score.
    slow = [b for b in index.slow_ids() if b not in protected]
    topk = sorted(slow, key=lambda b: scores[b], reverse=True)[: cfg.slow_topk_blocks]
    for b in topk:
        index.blocks[b].fetched_count += 1
    return sorted(fast | protected | set(topk))


def selected_mask(index: BlockIndex, selected_ids) -> list:
    """Boolean per-block mask for the selection (for sparse attention later)."""
    sel = set(selected_ids)
    return [b.block_id in sel for b in index.blocks]
