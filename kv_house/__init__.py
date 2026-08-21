# SPDX-License-Identifier: GPL-2.0
"""KV-House: prefix-stable sequence-axis transform coding of sealed
KV-cache blocks.

Existing KV transform coders act on the feature/channel axis and
treat tokens as independent; eviction methods drop tokens outright.
KV-House asks whether the ignored axis — cross-token (temporal)
correlation inside a fixed block of B consecutive tokens — carries
enough structure to be concentrated by an orthogonal transform
(Householder reflectors, DCT, Haar, or a KLT oracle) and exploited
with heterogeneous per-mode quantization, while every logical token
is retained and sealed blocks stay immutable, query-independent,
and randomly accessible (ordinary prefix-cache objects).

Falsification targets and honesty controls are explicit: the KLT
oracle is the ceiling, a random Householder is the placebo, and
CacheGen-style anchor-delta is the prior-art baseline that must be
beaten at equal bytes for the idea to matter. The structural
precedent in this repo is ``symkv`` (head-axis orthogonal bases);
KV-House swaps the transformed axis to the sequence dimension.
"""

from __future__ import annotations

from kv_house.config import KVHouseConfig
from kv_house.prefix_codec import BlockArtifact, CodecConfig, SealedBlockCodec
from kv_house.transforms import TRANSFORMS, make_transform

__all__ = [
    "KVHouseConfig",
    "BlockArtifact",
    "CodecConfig",
    "SealedBlockCodec",
    "TRANSFORMS",
    "make_transform",
]
