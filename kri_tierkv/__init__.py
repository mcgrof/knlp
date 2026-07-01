# SPDX-License-Identifier: GPL-2.0
"""KRI-TierKV: KRI-guided tiered KV-cache eviction and sparse retrieval.

Inspired by "TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference"
(given as arXiv:2604.19769), which partitions the KV cache into a fast HBM tier
and a slow DRAM tier with heterogeneous precision and overlaps slow-tier access
with attention work. KRI-TierKV is NOT a TTKV clone -- it keeps the temporal
tiering framing but replaces the tiering policy with KRI-D-sum block selection
and adds asymmetric fake quantization.

Note the TTKV arXiv id (2604.19769, April 2026) could not be verified from the
authoring assistant's training data; confirm it before publishing (see
docs/kri-tierkv.md).

The cache is split into fixed-size token blocks. Recent blocks are always fast
tier; old blocks are slow tier. KRI-D-sum scores old blocks so the low-value
ones can be evicted and the high-value ones retrieved. Sparse attention uses the
recent fast window plus the top-K selected slow blocks. Quantization is layered
on only after sparse selection works, V first (K16/V8 is the safe default).

This first version is an emulation: it runs normal dense inference and records
what each policy would keep, fetch, and skip, so we can measure whether KRI-D-sum
picks the blocks that actually carry attention mass before building any real
offload path. No custom CUDA, no real DRAM/SSD paging. On A100, FP8 is treated as
fake/storage quantization only -- native FP8 is a Hopper/H100 feature.
"""

from .config import TierKVConfig, EvictionPolicy, RetrievalPolicy, QuantPolicy
from .block_index import Block, BlockIndex

__all__ = [
    "TierKVConfig",
    "EvictionPolicy",
    "RetrievalPolicy",
    "QuantPolicy",
    "Block",
    "BlockIndex",
]
