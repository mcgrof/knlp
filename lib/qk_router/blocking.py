"""KV cache block definitions and block map construction."""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVBlock:
    """A single routed KV cache block."""

    block_id: int
    request_id: int
    token_start: int
    token_end: int
    num_tokens: int
    bytes_per_layer: int
    total_bytes: int
    num_layers: int
    tier: int = 1  # 0=hot/GPU, 1=warm/tmpfs, 2=cold/SFS
    reuse_score: float = 0.0
    estimated_fetch_latency_us: float = 0.0
    storage_path: Optional[str] = None


def compute_block_bytes(
    num_kv_heads: int, head_dim: int, block_size: int, dtype_bytes: int = 2
):
    """Compute bytes for one KV block per layer.

    Each block stores K and V tensors:
      K: [block_size, num_kv_heads, head_dim]
      V: [block_size, num_kv_heads, head_dim]
    """
    per_tensor = block_size * num_kv_heads * head_dim * dtype_bytes
    return per_tensor * 2  # K + V


def build_block_map(
    prefix_length: int,
    block_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    request_id: int = 0,
    dtype_bytes: int = 2,
) -> list[KVBlock]:
    """Build a deterministic block map for a single request prefix."""
    bytes_per_layer = compute_block_bytes(
        num_kv_heads, head_dim, block_size, dtype_bytes
    )
    total_bytes = bytes_per_layer * num_layers
    num_blocks = prefix_length // block_size
    blocks = []
    for i in range(num_blocks):
        t_start = i * block_size
        t_end = t_start + block_size
        blocks.append(
            KVBlock(
                block_id=i,
                request_id=request_id,
                token_start=t_start,
                token_end=t_end,
                num_tokens=block_size,
                bytes_per_layer=bytes_per_layer,
                total_bytes=total_bytes,
                num_layers=num_layers,
            )
        )
    return blocks


def assign_tiers(
    blocks: list[KVBlock],
    tier1_budget: int,
    tier2_budget: int,
    hot_blocks: int = 0,
):
    """Assign tier labels to blocks.

    hot_blocks: last N blocks kept GPU-resident (decode tail).
    tier1_budget: blocks on tmpfs.
    tier2_budget: blocks on SFS.
    Remaining blocks that don't fit get tier 2 (cold).
    """
    n = len(blocks)
    for i, b in enumerate(blocks):
        if i >= n - hot_blocks:
            b.tier = 0  # GPU hot
        elif i < tier1_budget:
            b.tier = 1  # tmpfs warm
        else:
            b.tier = 2  # SFS cold
