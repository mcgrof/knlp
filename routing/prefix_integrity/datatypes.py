# SPDX-License-Identifier: GPL-2.0
"""Data model for Prefix Integrity Analysis (PIA).

PIA treats a Cartridge (a fixed precomputed KV cache with stable block
boundaries) as a deterministic stand-in for a prefix-cached / offloaded KV
object. A candidate KV-cache compression, pruning, routing, quantization, or
offload algorithm is run against that cartridge, and the harness asks one
question: after the algorithm touches it, is the logical prefix still the same
reloadable, reusable, position-compatible cache object?

That is a cache-contract question, not an accuracy question. These dataclasses
are the vocabulary the rest of the harness speaks in. None of them import torch
so the manifest/metrics/invariant path runs on CPU with no heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BlockStatus(str, Enum):
    """Integrity of one logical prefix block after the algorithm ran.

    A block is INTACT only if it can be reloaded into its original position
    with its original shape/layout. PARTIAL means some-but-not-all of its
    tokens survived: semantically maybe usable by a custom kernel, but NOT
    reusable by ordinary block-hash prefix caching without new metadata. This
    is the distinction many papers quietly launder into a benchmark win.
    """

    INTACT = "intact"
    PARTIAL = "partial"
    MISSING = "missing"


class Mode(str, Enum):
    """How the candidate algorithm relates to the prefix."""

    SELECTOR = "selector"  # chooses which blocks survive / are loaded
    CODEC = "codec"  # transforms KV tensors, every logical block stays present
    SELECTOR_CODEC = "selector+codec"  # both


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Classification(str, Enum):
    """Plain-language verdict on where the algorithm is safe to deploy."""

    SAFE_FOR_PREFIX_OFFLOAD = "SAFE_FOR_PREFIX_OFFLOAD"
    SAFE_ONLY_WITH_EXTENDED_CACHE_KEY = "SAFE_ONLY_WITH_EXTENDED_CACHE_KEY"
    SAFE_ONLY_WITH_CUSTOM_CONNECTOR = "SAFE_ONLY_WITH_CUSTOM_CONNECTOR"
    ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE = "ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE"
    DANGEROUS_FOR_PREFIX_SHARING = "DANGEROUS_FOR_PREFIX_SHARING"


@dataclass
class PrefixIdentity:
    """Everything that defines *which* logical prefix object we are protecting.

    Two requests that agree on these fields must agree on the prefix KV object.
    rope_config_hash matters because re-phasing K vectors silently changes the
    object even when the bytes "look" compatible.
    """

    model_id: str
    cartridge_id: str
    prefix_hash: str
    block_size: int
    num_blocks: int
    dtype: str
    kv_shape: tuple
    model_revision: Optional[str] = None
    tokenizer_id: Optional[str] = None
    rope_config_hash: Optional[str] = None


@dataclass
class BlockManifest:
    """Per-block survival of one logical prefix after the algorithm ran.

    status[b] is the integrity of block b. selected is the ordered list of
    block ids the algorithm chose to keep / load (used for storage geometry:
    contiguous read ranges, bytes moved). For a pure codec every block is
    selected and INTACT or, if the transform is in-place lossy, still INTACT
    (shape preserved) — codec damage shows up in semantic drift, not here.
    """

    num_blocks: int
    status: list  # list[BlockStatus], length == num_blocks
    selected: list  # list[int], block ids kept/loaded (subset of range)
    block_size: int = 1

    def __post_init__(self):
        if len(self.status) != self.num_blocks:
            raise ValueError(
                f"status length {len(self.status)} != num_blocks {self.num_blocks}"
            )

    def count(self, st: BlockStatus) -> int:
        return sum(1 for s in self.status if s == st)

    @classmethod
    def from_selected(cls, num_blocks: int, selected, block_size: int = 1):
        """Selector that keeps a set of whole blocks: kept=INTACT, rest=MISSING."""
        sel = sorted(set(int(b) for b in selected))
        status = [
            BlockStatus.INTACT if b in set(sel) else BlockStatus.MISSING
            for b in range(num_blocks)
        ]
        return cls(
            num_blocks=num_blocks, status=status, selected=sel, block_size=block_size
        )

    @classmethod
    def from_token_mask(cls, num_blocks: int, block_size: int, kept_tokens):
        """Collapse a token-level keep mask to block-level integrity.

        block intact  = all tokens in block retained
        block partial = some but not all retained
        block missing = none retained
        """
        kept = set(int(t) for t in kept_tokens)
        status = []
        selected = []
        for b in range(num_blocks):
            lo, hi = b * block_size, (b + 1) * block_size
            present = sum(1 for t in range(lo, hi) if t in kept)
            if present == block_size:
                status.append(BlockStatus.INTACT)
                selected.append(b)
            elif present == 0:
                status.append(BlockStatus.MISSING)
            else:
                status.append(BlockStatus.PARTIAL)
                selected.append(b)  # partial block still incurs a read
        return cls(
            num_blocks=num_blocks,
            status=status,
            selected=selected,
            block_size=block_size,
        )


@dataclass
class CandidateArtifact:
    """A concrete thing the algorithm produced for one prefix: a manifest, a
    transformed cartridge, a prior, or some combination. artifact_digest is the
    content hash used for determinism/cache-key checks. cache_key_fields is the
    set of variables the algorithm claims its artifact depends on — the harness
    checks that claim against observed instability.
    """

    algorithm_id: str
    config_hash: str
    mode: str  # one of Mode values
    artifact_digest: str
    prior_path: Optional[str] = None
    cartridge_path: Optional[str] = None
    manifest_path: Optional[str] = None
    cache_key_fields: dict = field(default_factory=dict)


@dataclass
class PrefixIntegrityResult:
    status: str  # Status value
    classification: str  # Classification value
    danger_score: float
    metrics: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
