# Fused Routed Attention Kernel

Triton kernel for per-head block-selective paged attention during decode.

## Problem

Standard paged attention uses a single block table shared across all KV heads.
With routing, each KV head attends to a *different* subset of KV cache blocks
(selected by a routing prior). The naive implementation requires N separate
attention kernel launches (one per KV head), creating ~25-60x overhead vs dense.

## Solution

A fused Triton kernel that accepts **per-head block tables** and processes all
heads in a single launch. The kernel grid is `(batch, group_size, n_kv_heads)`,
so each program handles one (batch, query_head, kv_head) triple and iterates
only over that head's selected blocks.

## Interface

```python
from routing.fused_routed_attention import fused_routed_decode, select_top_k_blocks

# Build block tables from routing prior
block_tables, block_counts = select_top_k_blocks(routing_prior, k=4)

# Run fused decode
output = fused_routed_decode(
    q,              # [batch, n_heads, head_dim]
    k_cache,        # [max_blocks, block_size, n_kv_heads, head_dim]
    v_cache,        # [max_blocks, block_size, n_kv_heads, head_dim]
    block_tables,   # [batch, n_kv_heads, max_selected_blocks]
    block_counts,   # [batch, n_kv_heads]
)
```

## KV cache layout

```
k_cache[block_idx, token_offset, kv_head, dim]
```

- `block_idx`: physical block ID in the paged cache
- `token_offset`: position within block (0..block_size-1)
- `kv_head`: KV head index
- `dim`: head dimension

This matches vLLM's internal paged KV layout.

## GQA support

With grouped-query attention (e.g., 32 query heads, 8 KV heads), all query
heads in a group share the same KV head's block selection. The kernel handles
this via the `group_size` grid dimension.

## GPU tuning

| Parameter | A100 | H100 |
|-----------|------|------|
| BLOCK_T | 32 | 64 |
| num_warps | 4 | 8 |
| BLOCK_D | 128 | 128 |
| num_stages | 2 | 3 |

A100 has smaller SMs (108 SMs, 4 warp schedulers each). H100 has larger SMs
(132 SMs) with more registers and wider memory buses. The H100 path benefits
from larger tile sizes and more pipeline stages.

## Running tests

```bash
python -m pytest routing/tests/test_fused_routed_attention.py -v
```

## Running benchmarks

```bash
python -m routing.benchmark
```
