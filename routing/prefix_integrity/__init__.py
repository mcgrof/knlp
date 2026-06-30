# SPDX-License-Identifier: GPL-2.0
"""Prefix Integrity Analysis (PIA).

A preflight harness for KV-cache compression, pruning, routing, quantization,
or offload algorithms. It uses Cartridges as deterministic stand-ins for
prefix-cached / offloaded KV blocks and asks whether a logical prefix cache
entry stays reloadable, reusable, position-compatible, and semantically close
after the algorithm touches it.

This is a cache-contract evaluation, not a leaderboard. An algorithm can hold
next-token accuracy on one prompt and still be dangerous for prefix sharing,
because two requests with the same prefix no longer agree on what the prefix
object is. PIA is meant to catch that class before anything is tested inside
LMCache or a distributed KV store.

Public surface:
    load_identity / load_tensors        -- read a cartridge as a prefix object
    BlockManifest                       -- per-block survival of one prefix
    manifest_metrics, ...               -- cache-contract metrics
    InvariantInput, evaluate            -- hard gates -> status/classification
    run_validate                        -- end-to-end harness (CPU MVP)
"""

from .datatypes import (
    BlockManifest,
    BlockStatus,
    CandidateArtifact,
    Classification,
    Mode,
    PrefixIdentity,
    PrefixIntegrityResult,
    Status,
)
from .metrics import (
    manifest_metrics,
    prefix_reuse_efficiency,
    anchor_survival,
    recent_survival,
    contiguous_prefix_survival,
    partial_block_rate,
    manifest_stability,
    artifact_stability,
    contiguous_read_ranges,
    compression_stats,
)
from .invariants import InvariantInput, Thresholds, evaluate

__all__ = [
    "BlockManifest",
    "BlockStatus",
    "CandidateArtifact",
    "Classification",
    "Mode",
    "PrefixIdentity",
    "PrefixIntegrityResult",
    "Status",
    "manifest_metrics",
    "prefix_reuse_efficiency",
    "anchor_survival",
    "recent_survival",
    "contiguous_prefix_survival",
    "partial_block_rate",
    "manifest_stability",
    "artifact_stability",
    "contiguous_read_ranges",
    "compression_stats",
    "InvariantInput",
    "Thresholds",
    "evaluate",
]
