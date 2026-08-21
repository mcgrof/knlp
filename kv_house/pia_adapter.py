# SPDX-License-Identifier: GPL-2.0
"""Prefix Integrity Analysis adapter for the KV-House codec.

KV-House keeps every block (it is a codec, not a selector), changes
the stored layout (transformed + quantized modes), and therefore
needs a codec-aware restore path — the same honest resting place as
the KIVI exemplar: SAFE_ONLY_WITH_CUSTOM_CONNECTOR. What PIA must
verify beyond that label is the sealed-block contract: identical
manifests and artifact digests for the same prefix regardless of
query or repeat, and a cache key that includes the codec
configuration so a transform or allocation change can never alias a
stale artifact.

Load with:

    python -m routing.prefix_integrity.cli validate \\
        --algorithm kv_house.pia_adapter:make \\
        --algorithm-config '{"transform": "dc_householder", "block_size": 16}'
"""

from __future__ import annotations

import hashlib

from routing.prefix_integrity import BlockManifest, CandidateArtifact, Mode

from kv_house.prefix_codec import CodecConfig


class KVHouseAdapter:
    """Whole-block, query-independent, layout-changing codec."""

    name = "kv_house"
    mode = Mode.CODEC.value
    policy = "prefix_cache"
    granularity = "block"
    shape_preserved = False
    has_custom_restore_path = True

    def __init__(self, transform="dc_householder", block_size=16, quant_alloc=None):
        alloc = tuple(quant_alloc) if quant_alloc else (8,) * block_size
        self.config = CodecConfig(
            block_size=block_size, transform=transform, quant_alloc=alloc
        )
        self.cache_key_fields = ("prefix_hash", "kv_house_config")

    def select_blocks(self, request, num_blocks, budget_k):
        return list(range(num_blocks))

    def manifest(self, request, num_blocks, budget_k, block_size=1):
        return BlockManifest.from_selected(
            num_blocks, range(num_blocks), block_size=block_size
        )

    def artifact(self, request, num_blocks, budget_k, config_hash=""):
        digest = hashlib.sha256(
            f"{self.name}|{self.config.cache_key()}|{num_blocks}".encode()
        ).hexdigest()[:16]
        return CandidateArtifact(
            algorithm_id=self.name,
            config_hash=config_hash or self.config.cache_key(),
            mode=self.mode,
            artifact_digest=digest,
            cache_key_fields={k: "1" for k in self.cache_key_fields},
        )


def make(**kwargs):
    return KVHouseAdapter(**kwargs)
