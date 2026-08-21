# SPDX-License-Identifier: GPL-2.0
"""Experiment configuration for KV-House milestone runs.

One frozen dataclass so a run's identity is printable, hashable,
and impossible to mutate mid-experiment. Block sizes follow the
milestone plan (8/16/32 first; 4 and 64 in the wider sweep) and
must stay compatible with serving/cache page geometry — vLLM's
default block size is 16 tokens, which is why 16 sits in the
center of the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MILESTONE_BLOCK_SIZES = (8, 16, 32)
FULL_BLOCK_SIZES = (4, 8, 16, 32, 64)

MILESTONE_TRANSFORMS = (
    "identity",
    "anchor_delta",
    "dc_householder",
    "random_householder",
    "dct2",
    "haar",
    "hadamard",
    "pca_oracle",
    "corpus_pca",
    "poweriter_householder",
)


@dataclass(frozen=True)
class KVHouseConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    block_sizes: tuple = MILESTONE_BLOCK_SIZES
    transforms: tuple = MILESTONE_TRANSFORMS
    contexts: tuple = (512, 1024, 2048)
    seed: int = 0
    max_layers: int = 0  # 0 means all layers

    def __post_init__(self):
        for b in self.block_sizes:
            assert b >= 2 and (b & (b - 1)) == 0, "block sizes are powers of two"
        for t in self.transforms:
            assert t in MILESTONE_TRANSFORMS, f"unknown transform {t}"
