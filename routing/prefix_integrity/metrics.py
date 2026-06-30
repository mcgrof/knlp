# SPDX-License-Identifier: GPL-2.0
"""Prefix-integrity metrics.

These quantify what a compression/routing/offload algorithm did to a logical
prefix, in cache-contract terms rather than accuracy terms. Everything here is
pure Python over BlockManifest objects and digest lists — no torch, no GPU — so
the manifest-only path (MVP 1, 2, 4) runs anywhere.

Let B be all logical blocks in the shared prefix and w_b a reuse weight for
block b (uniform by default, or hot-block weights from a trace). A block is
intact only if it reloads at its original position with original geometry; a
partial block is not intact for vanilla block-hash prefix caching.
"""

from __future__ import annotations

import math
from typing import Optional

from .datatypes import BlockManifest, BlockStatus


def _weights(num_blocks: int, weights: Optional[list]) -> list:
    if weights is None:
        return [1.0] * num_blocks
    if len(weights) != num_blocks:
        raise ValueError(f"weights length {len(weights)} != num_blocks {num_blocks}")
    return [float(w) for w in weights]


def prefix_reuse_efficiency(m: BlockManifest, weights: Optional[list] = None) -> float:
    """PRE = sum_b w_b*intact_b / sum_b w_b.

    The headline number: what weighted fraction of the prefix is still a
    reusable cache object. Partial and missing blocks both count as zero.
    """
    w = _weights(m.num_blocks, weights)
    num = sum(w[b] for b in range(m.num_blocks) if m.status[b] == BlockStatus.INTACT)
    den = sum(w)
    return num / den if den else 0.0


def hot_prefix_reuse_efficiency(m: BlockManifest, hot_weights: list) -> float:
    """HPRE: PRE under observed/synthetic hot-block weights. An algorithm can

    have a fine uniform PRE while dropping exactly the blocks that get reused
    most — HPRE is what catches that.
    """
    return prefix_reuse_efficiency(m, hot_weights)


def anchor_survival(m: BlockManifest, anchor_blocks: int = 1) -> float:
    """Fraction of the first `anchor_blocks` blocks that are intact. Anchor /
    system-prompt blocks are the ones a prefix-cache-safe policy must never
    drop or partial.
    """
    a = max(0, min(anchor_blocks, m.num_blocks))
    if a == 0:
        return 1.0
    return sum(1 for b in range(a) if m.status[b] == BlockStatus.INTACT) / a


def recent_survival(m: BlockManifest, recent_blocks: int = 1) -> float:
    """Fraction of the last `recent_blocks` blocks that are intact."""
    r = max(0, min(recent_blocks, m.num_blocks))
    if r == 0:
        return 1.0
    lo = m.num_blocks - r
    return (
        sum(1 for b in range(lo, m.num_blocks) if m.status[b] == BlockStatus.INTACT) / r
    )


def contiguous_prefix_survival(m: BlockManifest) -> int:
    """Largest m such that blocks [0, m) are all intact. Vanilla prefix caching
    reuses the longest intact run from the start; a hole at block 1 makes
    everything after it un-shareable even if individually intact.
    """
    n = 0
    for b in range(m.num_blocks):
        if m.status[b] == BlockStatus.INTACT:
            n += 1
        else:
            break
    return n


def partial_block_rate(m: BlockManifest) -> float:
    """number_partial_blocks / num_blocks. For a vanilla block-based offload /
    prefix-cache path, any partial block is a contract violation.
    """
    if m.num_blocks == 0:
        return 0.0
    return m.count(BlockStatus.PARTIAL) / m.num_blocks


def block_survival_rows(m: BlockManifest, weights: Optional[list] = None) -> list:
    """Per-block rows for block_survival.csv."""
    w = _weights(m.num_blocks, weights)
    sel = set(m.selected)
    return [
        {
            "block": b,
            "status": m.status[b].value,
            "weight": w[b],
            "selected": b in sel,
        }
        for b in range(m.num_blocks)
    ]


# --- Stability (determinism / cache-key) metrics -------------------------


def manifest_stability(selected_sets: list) -> int:
    """Number of distinct selected-block sets produced for the same prefix_hash
    under repeated runs. 1 == deterministic selection. >1 across *re-runs of the
    same query* means non-determinism; >1 across *different queries* means the
    selection is query-dependent (routing), which is fine only with query_hash
    in the cache key.
    """
    return len({frozenset(int(b) for b in s) for s in selected_sets})


def artifact_stability(digests: list) -> int:
    """Number of distinct artifact digests for the same prefix_hash + config.
    >1 under a fixed config and fixed query is a determinism failure; >1 across
    queries means the stored bytes are query-dependent.
    """
    return len(set(digests))


# --- Storage / offload behavior ------------------------------------------


def contiguous_read_ranges(selected: list) -> int:
    """Number of contiguous block ranges a retrieve would read. Many small
    ranges = random-read amplification on an offload store.
    """
    s = sorted(set(int(b) for b in selected))
    if not s:
        return 0
    ranges = 1
    for prev, cur in zip(s, s[1:]):
        if cur != prev + 1:
            ranges += 1
    return ranges


def read_amplification(selected: list) -> float:
    """Read ranges per selected block — 1.0 is one perfectly contiguous read,
    higher means the selection is scattered across the offload store.
    """
    s = sorted(set(int(b) for b in selected))
    if not s:
        return 0.0
    return contiguous_read_ranges(s) / len(s)


def bytes_per_block(
    identity_kv_shape: tuple, num_blocks: int, dtype_bytes: int
) -> float:
    """Bytes one logical block occupies, from total kv element count / blocks."""
    if num_blocks == 0:
        return 0.0
    elems = 1
    for d in identity_kv_shape:
        elems *= int(d)
    return elems * dtype_bytes / num_blocks


def compression_stats(cr_values: list) -> dict:
    """CR mean + coefficient of variation across request replays. A high CV
    means the compression ratio swings request-to-request, which is exactly the
    regime where a leaderboard mean hides prefix damage.
    """
    if not cr_values:
        return {"cr_mean": 0.0, "cr_cv": 0.0, "cr_n": 0}
    n = len(cr_values)
    mean = sum(cr_values) / n
    if mean == 0 or n < 2:
        return {"cr_mean": mean, "cr_cv": 0.0, "cr_n": n}
    var = sum((x - mean) ** 2 for x in cr_values) / (n - 1)
    return {"cr_mean": mean, "cr_cv": math.sqrt(var) / mean, "cr_n": n}


def manifest_metrics(
    m: BlockManifest,
    anchor_blocks: int = 1,
    recent_blocks: int = 2,
    weights: Optional[list] = None,
    hot_weights: Optional[list] = None,
) -> dict:
    """All single-manifest metrics in one dict (MVP 1)."""
    out = {
        "num_blocks": m.num_blocks,
        "block_size": m.block_size,
        "n_intact": m.count(BlockStatus.INTACT),
        "n_partial": m.count(BlockStatus.PARTIAL),
        "n_missing": m.count(BlockStatus.MISSING),
        "n_selected": len(set(m.selected)),
        "pre": prefix_reuse_efficiency(m, weights),
        "anchor_survival": anchor_survival(m, anchor_blocks),
        "recent_survival": recent_survival(m, recent_blocks),
        "contiguous_prefix_survival": contiguous_prefix_survival(m),
        "partial_block_rate": partial_block_rate(m),
        "read_ranges": contiguous_read_ranges(m.selected),
        "read_amplification": read_amplification(m.selected),
    }
    if hot_weights is not None:
        out["hpre"] = hot_prefix_reuse_efficiency(m, hot_weights)
    return out
