# vLLM Integration Gap Audit: Fused Routed Decode Kernel

**Date:** 2026-03-31
**Target vLLM version:** 0.18.1 (V1 engine architecture)
**Kernel:** `fused_routed_attention` from `staging/fused_routed_attention.py`
**Purpose:** Identify exact code points, data structure gaps, and engineering blockers for hooking the per-head-block-table routing kernel into vLLM's decode attention path. This document does NOT implement the integration.

---

## 1. Decode Attention Forward Path

### Architecture overview

vLLM 0.18.x uses the **V1 engine** exclusively. All attention code lives under `vllm/v1/attention/`. The legacy `vllm/attention/` tree from V0 is removed.

### Key files and classes

| File | Class / Function | Role |
|------|-----------------|------|
| `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.forward()` | Primary NVIDIA decode attention. Calls `flash_attn_varlen_func` with paged KV cache. |
| `vllm/v1/attention/backends/flashinfer.py` | `FlashInferImpl.forward()` | Alternative decode backend. Uses `BatchDecodeWithPagedKVCacheWrapper`. |
| `vllm/v1/attention/backends/registry.py` | `AttentionBackendEnum` | Enum-based registry. Each backend maps to a fully-qualified class path. Supports `CUSTOM` slot with `register_backend()`. |
| `vllm/v1/attention/backend.py` | `AttentionBackend` (ABC), `AttentionImpl` (ABC) | Defines the interface every backend must implement: `get_kv_cache_shape()`, `get_impl_cls()`, `get_builder_cls()`, and `forward()`. |
| `vllm/model_executor/layers/attention/attention.py` | `Attention` (nn.Module) | The model-level layer. Selects backend via `get_attn_backend()`, instantiates `impl`, calls `impl.forward(query, key, value, kv_cache, attn_metadata, output)`. |
| `vllm/v1/worker/gpu/block_table.py` | `BlockTables` | Manages block table tensors on GPU, computes slot mappings, gathers block tables for the forward pass. |

### FlashAttention decode call chain (the hot path)

```
Attention.forward()
  -> FlashAttentionImpl.forward(query, key, value, kv_cache, attn_metadata, output)
     -> kv_cache.unbind(0) to get key_cache, value_cache
     -> flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,            # full paged cache
            v=value_cache,          # full paged cache
            cu_seqlens_q=...,
            seqused_k=seq_lens,     # per-sequence KV length
            block_table=block_table, # [batch, max_num_blocks] int32
            ...
        )
```

During decode (single token per request), `max_query_len == 1`, `cu_seqlens_q` is `[0, 1, 2, ..., batch]`, and `block_table` tells Flash Attention which physical blocks to read for each sequence.

### FlashInfer decode call chain

FlashInfer wraps `BatchDecodeWithPagedKVCacheWrapper` which takes a pre-built paged KV cache index. The block tables are baked into the wrapper during `begin_forward()`, called from `FlashInferMetadataBuilder.build()`. The wrapper internally handles page table indirection.

---

## 2. Current Block Table Format

### Shape and dtype

```
block_table: torch.Tensor  # shape [batch, max_num_blocks], dtype int32
```

Each row maps a **sequence** to its list of **physical block indices**. The mapping is **uniform across all KV heads** -- every head reads the same set of blocks for a given sequence.

### KV cache shape (Flash Attention path)

```python
# From FlashAttentionBackend.get_kv_cache_shape():
(2, num_blocks, block_size, num_kv_heads, head_size)
#  ^                         ^
#  K/V split                 all heads share same blocks
```

The `2` dimension separates K and V. `num_blocks` is the total physical block count across all sequences. Each block holds `block_size` tokens (typically 16) for **all** KV heads together.

### BlockTables class (worker side)

In `vllm/v1/worker/gpu/block_table.py`, the `BlockTables` class manages:

```python
self.block_tables: list[StagedWriteTensor]
    # num_kv_cache_groups x [max_num_reqs, max_num_blocks]  int32
```

The `num_kv_cache_groups` dimension exists to support hybrid models (e.g., attention + mamba layers with different block sizes), but it is **NOT a per-KV-head dimension**. Typical transformer models have `num_kv_cache_groups == 1`.

### Routing kernel's block table format (the gap)

```
block_tables: [batch, n_kv_heads, max_selected_blocks]  int32
block_counts: [batch, n_kv_heads]                        int32
```

Each KV head has its **own** subset of selected blocks, determined by a routing prior. This is fundamentally different from vLLM's format where all heads share the same block list.

### Summary of the gap

| Dimension | vLLM current | Routing kernel needs |
|-----------|-------------|---------------------|
| Block table shape | `[batch, max_blocks]` | `[batch, n_kv_heads, max_selected_blocks]` |
| Block selection | Same for all heads | Per-KV-head (routing prior selects top-K blocks per head) |
| Block count | Implicit from `seq_lens` | Explicit `[batch, n_kv_heads]` int32 (varies per head) |
| KV cache layout | `[2, total_blocks, block_size, n_kv_heads, head_dim]` | `[total_blocks, block_size, n_kv_heads, head_dim]` (K and V separate tensors) |

---

## 3. Injection Point: Where to Hook the Routing Kernel

### Option A: New Attention Backend (RECOMMENDED)

Register a new backend via the `AttentionBackendEnum.CUSTOM` slot or add a new enum entry:

```python
# In registry.py or via register_backend():
@register_backend(AttentionBackendEnum.CUSTOM)
class RoutedAttentionBackend(AttentionBackend):
    ...
```

This gives you:
- A custom `RoutedAttentionImpl.forward()` that calls `fused_routed_attention` instead of `flash_attn_varlen_func`.
- A custom `RoutedAttentionMetadata` dataclass with the per-head block tables and block counts.
- A custom `RoutedAttentionMetadataBuilder` that constructs the routing metadata from the scheduler output.
- A custom `get_kv_cache_shape()` that can return separate K/V tensors or adapt the layout.

**Why this is best:** The backend interface is the designed extension point. It avoids monkey-patching and keeps the change isolated.

### Option B: Wrapper Around Existing Decode

Subclass `FlashAttentionImpl` and override `forward()` to:
1. Run the routing prior to select per-head blocks.
2. Reshape the block tables from `[batch, max_blocks]` to `[batch, n_kv_heads, max_selected_blocks]`.
3. Call `fused_routed_attention` instead of `flash_attn_varlen_func`.

**Downside:** Still requires custom metadata. The wrapper approach is fragile across vLLM version bumps since `FlashAttentionImpl.forward()` changes frequently.

### Option C: Layer-Level Hook (NOT recommended)

Override `Attention.forward()` at the model layer level. This bypasses the backend system entirely and requires forking the model definition. Not viable for upstream contribution.

### Verdict

**Option A** (new backend) is the cleanest path. vLLM's `CUSTOM` backend slot and `register_backend()` decorator exist precisely for this use case.

---

## 4. Data Structure Changes Required

### 4.1 Block Table Extension

The central structural change: block tables must grow a **per-KV-head** dimension.

**Current:** `BlockTables` stores `[max_num_reqs, max_num_blocks]` per KV cache group.

**Required:** A new `RoutedBlockTables` that stores:
- `routed_block_tables: [max_num_reqs, n_kv_heads, max_selected_blocks]` int32
- `routed_block_counts: [max_num_reqs, n_kv_heads]` int32

This tensor must be populated each decode step by the routing prior, which selects which physical blocks each KV head should attend to.

### 4.2 Attention Metadata Extension

A new `RoutedAttentionMetadata` dataclass:

```python
@dataclass
class RoutedAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor

    # Standard vLLM block table (for KV cache writes / slot mapping)
    block_table: torch.Tensor       # [batch, max_blocks]
    slot_mapping: torch.Tensor

    # Routing-specific
    routed_block_tables: torch.Tensor  # [batch, n_kv_heads, max_selected_blocks]
    routed_block_counts: torch.Tensor  # [batch, n_kv_heads]
```

The standard `block_table` and `slot_mapping` are still needed for **writing** new KV entries to the cache (the `reshape_and_cache` step). Only the **read** path during attention uses the routed block tables.

### 4.3 KV Cache Layout Adaptation

The routing kernel expects K and V as **separate** tensors:
```
k_cache: [total_blocks, block_size, n_kv_heads, head_dim]
v_cache: [total_blocks, block_size, n_kv_heads, head_dim]
```

vLLM Flash Attention stores them stacked:
```
kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
```

This is a trivial `kv_cache.unbind(0)` -- FlashAttentionImpl already does this in `forward()`. The shapes match after unbinding.

### 4.4 Routing Prior Interface

A new component is needed to produce `routed_block_tables` and `routed_block_counts` each decode step. This must:
- Take the current query state (or a compressed representation).
- Look up per-head block importance scores (precomputed during prefill or maintained incrementally).
- Select top-K blocks per KV head.
- Write the result into the `routed_block_tables` tensor.

This does NOT exist anywhere in vLLM today and must be built as part of the integration.

---

## 5. Blocking Engineering Issues

### BLOCKER 1: vLLM's Block Manager Has No Per-Head Block Concept

**Severity: High**

The `KVCacheManager`, `BlockPool`, and `KVCacheBlock` all treat blocks as **per-sequence** resources. A block is allocated to a sequence, and all KV heads in that sequence share it. The block table is `[batch, max_blocks]`.

The routing kernel does not need per-head block **allocation** (all heads still write to the same physical blocks during prefill/append). It needs per-head block **selection** at decode time. This is an important distinction:

- **Write path:** Unchanged. New tokens are written to the same blocks for all heads (standard PagedAttention append).
- **Read path (decode attention):** Each KV head reads only its selected subset of blocks.

**Implication:** The block manager itself does not need modification. The per-head block selection is a **read-time routing decision**, not an allocation decision. This significantly reduces the integration surface.

### BLOCKER 2: No Routing Prior Pipeline Exists

**Severity: High**

vLLM has no concept of "which blocks matter more for which head." The routing prior must be computed somehow. Options:
- **Offline:** Precompute importance scores per block per head during prefill, store alongside the KV cache.
- **Online:** Compute importance scores from the current query at each decode step (adds latency).
- **Hybrid:** Precompute during prefill, refine incrementally during decode.

None of these exist in vLLM today. The routing prior pipeline must be designed and built.

### BLOCKER 3: Metadata Builder Must Populate Routed Block Tables

**Severity: Medium**

The `AttentionMetadataBuilder` (e.g., `FlashAttentionMetadataBuilder`) constructs metadata from `CommonAttentionMetadata` provided by the model runner. The model runner gets block tables from `BlockTables.gather_block_tables()`.

A new `RoutedAttentionMetadataBuilder` must:
1. Receive the standard block table from the model runner.
2. Run the routing prior to produce per-head block selection.
3. Pack `routed_block_tables` and `routed_block_counts` into the metadata.

This computation happens on the **critical path** of each decode step, so it must be fast (ideally a single Triton kernel for the top-K selection).

### BLOCKER 4: CUDA Graph Compatibility

**Severity: Medium**

vLLM uses CUDA graphs for decode to eliminate kernel launch overhead. The routed block tables change every decode step (the routing prior may select different blocks as the query evolves). This means:
- `routed_block_tables` must be pre-allocated at graph capture time with maximum dimensions.
- The routing prior kernel must write into fixed-address tensors.
- `routed_block_counts` must similarly be pre-allocated.

This follows the same pattern as vLLM's existing `input_block_tables` (pre-allocated, overwritten each step), so it is feasible but requires care.

### BLOCKER 5: Prefill Path Is Unaffected (But Must Coexist)

**Severity: Low**

The routing kernel targets **decode only** (single-token queries). Prefill uses dense attention over all tokens. The backend must handle both:
- **Prefill:** Fall through to standard `flash_attn_varlen_func` or equivalent.
- **Decode:** Use `fused_routed_attention` with per-head block tables.

This is straightforward since `FlashAttentionImpl.forward()` already distinguishes prefill vs decode via `attn_metadata.max_query_len`.

### BLOCKER 6: GQA Head Mapping Must Be Consistent

**Severity: Low**

The routing kernel maps Q heads to KV heads via integer division (`kv_head = pid_qhead // gqa_ratio`). vLLM's attention backends use the same GQA convention. No conflict expected, but the GQA ratio must be passed correctly through the backend interface.

### NON-BLOCKER: KV Cache Write Path

The `reshape_and_cache` operation (writing new K/V into the paged cache) is handled **separately** from the attention forward pass in vLLM V1. The `FlashAttentionBackend.forward_includes_kv_cache_update = False` flag confirms this. The routing kernel does not touch the write path, so no changes are needed there.

---

## 6. Recommended Integration Plan (Summary)

| Step | Component | Effort | Dependency |
|------|-----------|--------|------------|
| 1 | Define `RoutedAttentionMetadata` dataclass | Low | None |
| 2 | Implement `RoutedAttentionBackend` + `RoutedAttentionImpl` as a CUSTOM backend | Medium | Step 1 |
| 3 | Implement routing prior interface (block importance scoring) | High | Domain research |
| 4 | Implement `RoutedAttentionMetadataBuilder` with top-K block selection | Medium | Steps 1, 3 |
| 5 | Pre-allocate routed block table tensors for CUDA graph compat | Medium | Step 2 |
| 6 | End-to-end serving test with model-derived priors | High | Steps 1-5 |

### Minimum viable prototype (bypasses blockers 2-4)

For an initial proof-of-concept without a real routing prior:
1. Register a CUSTOM backend.
2. In `forward()`, take the standard block table, expand it to `[batch, n_kv_heads, max_blocks]` (same blocks for all heads, no actual routing).
3. Call `fused_routed_attention` with the expanded block table.
4. Verify output matches dense FlashAttention output.

This proves the integration plumbing without solving the routing prior problem.

---

## 7. Key Source File References (vLLM v0.18.1)

- `vllm/v1/attention/backends/flash_attn.py` -- FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata, FlashAttentionMetadataBuilder
- `vllm/v1/attention/backends/flashinfer.py` -- FlashInferBackend, FlashInferImpl, FlashInferMetadata
- `vllm/v1/attention/backends/registry.py` -- AttentionBackendEnum, register_backend()
- `vllm/v1/attention/backend.py` -- AttentionBackend (ABC), AttentionImpl (ABC)
- `vllm/v1/worker/gpu/block_table.py` -- BlockTables class, gather/slot_mapping Triton kernels
- `vllm/v1/core/kv_cache_manager.py` -- KVCacheManager, KVCacheBlocks
- `vllm/v1/core/block_pool.py` -- BlockPool, block hash caching
- `vllm/model_executor/layers/attention/attention.py` -- Attention nn.Module (backend dispatch)
