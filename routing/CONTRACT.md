# Routing contract (2026-04-03)

Canonical routing metadata contract for the fused routed decode kernel.

## Tier 0 — static format contract

These are compile-time or configuration-time conventions.

| Element | Shape | Dtype | Rule |
|---------|-------|-------|------|
| q | `[batch, n_heads, head_dim]` | fp16/bf16 | Single decode-step query |
| k_cache | `[max_blocks, block_size, n_kv_heads, head_dim]` | fp16/bf16 | Paged block layout |
| v_cache | `[max_blocks, block_size, n_kv_heads, head_dim]` | fp16/bf16 | Same layout as K |
| block_tables | `[batch, n_kv_heads, max_selected_blocks]` | int64 | Physical block indices |
| block_counts | `[batch, n_kv_heads]` | int32 | Actual block count per head |
| scale | float | — | `1/sqrt(head_dim)` by convention |
| GQA | — | — | `group_size = n_heads // n_kv_heads` |

**Grid:** `(batch, group_size, n_kv_heads)` — one program per (batch, query-head-in-group, KV-head).

**Block iteration:** `for block_idx in range(max_selected_blocks)` with masking. `max_selected_blocks` is a compile-time constant.

## Tier 1 — request/cache-owned structural metadata

These are computed once per request/prefix and reused for all decode steps.

| Element | Shape | Producer | Lifetime |
|---------|-------|----------|----------|
| routing_prior | `[n_layers, n_kv_heads, n_blocks]` | Prior extractor (prefill) | Per-request |
| block_affinities | `[n_layers, n_q_heads, n_blocks]` | `extract_prefill_priors()` | Per-request |
| K-summaries | `[num_blocks, num_layers, summary_dim]` | `build_k_summaries()` | Per-prefix |

**Rule:** Tier 1 metadata is not updated during decode.

## Tier 2 — phase-local derived execution metadata

These are derived from Tier 1 and consumed by the kernel.

| Element | Shape | Producer | Lifetime |
|---------|-------|----------|----------|
| block_tables | `[batch, n_kv_heads, K]` | `select_top_k_blocks()` | Per-request (computed once) |
| block_counts | `[batch, n_kv_heads]` | Same | Per-request (computed once) |

**Rule:** Tier 2 is derived once and reused. No per-step rebuild.

## Source producers

Any routing source must produce Tier 1 metadata that maps into the same Tier 2 derived format:

| Source | Current status |
|--------|---------------|
| Prefill-derived offline priors | Live (`extract_prefill_priors`) |
| K-summary centroids | Live (`build_k_summaries`) |
| Cartridge sidecars | Planned (vLLM integration) |
| Retrieval/RAG-derived priors | Future |

## Canonical vs derived boundary

```
Source → [Tier 1: routing_prior] → select_top_k_blocks() → [Tier 2: block_tables, block_counts] → kernel
```

The kernel does not know or care where the routing prior came from. It consumes `block_tables` and `block_counts` directly.

## Entry points

```python
from routing.fused_routed_attention import fused_routed_decode, select_top_k_blocks

# Base kernel (A100 defaults)
output = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)

# Autotuned kernel (H100 recommended)
output = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts, autotune=True)
```
