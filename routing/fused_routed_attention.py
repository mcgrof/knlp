# SPDX-License-Identifier: GPL-2.0
"""Triton fused routed attention kernel.

Implements single-token decode attention where each KV head attends to a
different subset of KV cache blocks, selected by a routing prior. This
replaces the Python per-head loop (N separate flash_attn calls) with a
single fused kernel launch.

Interface:
    fused_routed_decode(
        q,                  # [batch, n_heads, head_dim]
        k_cache,            # [max_blocks, block_size, n_kv_heads, head_dim]
        v_cache,            # [max_blocks, block_size, n_kv_heads, head_dim]
        block_tables,       # [batch, n_kv_heads, max_selected_blocks]
        block_counts,       # [batch, n_kv_heads]  — actual blocks per head
        scale,              # float — 1/sqrt(head_dim)
    ) -> output             # [batch, n_heads, head_dim]

Block layout:
    k_cache / v_cache are stored in paged format: each block holds
    `block_size` tokens for all KV heads. block_tables[b, h, i] gives
    the physical block index for batch b, KV head h, i-th selected block.

GQA support:
    n_heads can be a multiple of n_kv_heads (grouped-query attention).
    All query heads in a group share the same KV head's block selection.

GPU tuning:
    - A100: BLOCK_SIZE_K=64, num_warps=4
    - H100: BLOCK_SIZE_K=128, num_warps=8 (wider SMs, more registers)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_routed_decode_kernel(
    # Pointers
    Q,          # [batch, n_heads, head_dim]
    K_cache,    # [max_blocks, block_size, n_kv_heads, head_dim]
    V_cache,    # [max_blocks, block_size, n_kv_heads, head_dim]
    Block_tables,  # [batch, n_kv_heads, max_selected_blocks]
    Block_counts,  # [batch, n_kv_heads]
    Output,     # [batch, n_heads, head_dim]
    # Scalars
    scale: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_selected_blocks: tl.constexpr,
    # Strides for Q: [batch, n_heads, head_dim]
    stride_qb, stride_qh, stride_qd,
    # Strides for K_cache: [max_blocks, block_size, n_kv_heads, head_dim]
    stride_kb, stride_kt, stride_kh, stride_kd,
    # Strides for V_cache: [max_blocks, block_size, n_kv_heads, head_dim]
    stride_vb, stride_vt, stride_vh, stride_vd,
    # Strides for Block_tables: [batch, n_kv_heads, max_selected_blocks]
    stride_btb, stride_bth, stride_bts,
    # Strides for Block_counts: [batch, n_kv_heads]
    stride_bcb, stride_bch,
    # Strides for Output: [batch, n_heads, head_dim]
    stride_ob, stride_oh, stride_od,
    # Block dimensions
    BLOCK_D: tl.constexpr,       # head_dim tile (usually == head_dim)
    BLOCK_T: tl.constexpr,       # tokens per inner iteration
):
    """Fused decode attention with per-KV-head block selection.

    Grid: (batch, n_query_heads_in_group, n_kv_heads)
    Each program computes attention for one (batch, query_head, kv_head) triple.
    """
    pid_batch = tl.program_id(0)
    pid_qh_in_group = tl.program_id(1)
    pid_kv_head = tl.program_id(2)

    # Map query head index
    group_size = n_heads // n_kv_heads
    query_head = pid_kv_head * group_size + pid_qh_in_group

    # Load number of selected blocks for this (batch, kv_head)
    n_blocks = tl.load(
        Block_counts + pid_batch * stride_bcb + pid_kv_head * stride_bch
    )

    # Load query vector: q[batch, query_head, :]
    d_offsets = tl.arange(0, BLOCK_D)
    q_ptrs = Q + pid_batch * stride_qb + query_head * stride_qh + d_offsets * stride_qd
    q_vec = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

    # Online softmax accumulators
    m_prev = float('-inf') + tl.zeros([1], dtype=tl.float32)  # running max (scalar-like)
    l_prev = tl.zeros([1], dtype=tl.float32)  # running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # weighted value accumulator

    # Iterate over selected blocks (no break — use masking for inactive blocks)
    for block_idx in range(max_selected_blocks):
        # Use block 0 as a dummy when block_idx >= n_blocks.
        # The scores will be masked to -inf so they contribute nothing.
        active = block_idx < n_blocks

        # Load physical block ID from block table (use 0 for inactive)
        phys_block = tl.load(
            Block_tables + pid_batch * stride_btb + pid_kv_head * stride_bth + block_idx * stride_bts
        )

        # Process tokens within this block in tiles of BLOCK_T
        for t_start in range(0, block_size, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            t_mask = (t_offsets < block_size) & active

            # Load K[phys_block, t_offsets, kv_head, :] -> [BLOCK_T, BLOCK_D]
            k_ptrs = (K_cache
                      + phys_block * stride_kb
                      + t_offsets[:, None] * stride_kt
                      + pid_kv_head * stride_kh
                      + d_offsets[None, :] * stride_kd)
            k_tile = tl.load(k_ptrs,
                             mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                             other=0.0).to(tl.float32)

            # Compute attention scores: q @ k^T -> [BLOCK_T]
            scores = tl.sum(q_vec[None, :] * k_tile, axis=1) * scale  # [BLOCK_T]
            scores = tl.where(t_mask, scores, float('-inf'))

            # Online softmax update
            m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
            # Correction factor for previous accumulator
            correction = tl.exp(m_prev - m_new)
            # New exponentials
            p = tl.exp(scores - m_new)
            l_new = l_prev * correction + tl.sum(p, axis=0)

            # Load V[phys_block, t_offsets, kv_head, :] -> [BLOCK_T, BLOCK_D]
            v_ptrs = (V_cache
                      + phys_block * stride_vb
                      + t_offsets[:, None] * stride_vt
                      + pid_kv_head * stride_vh
                      + d_offsets[None, :] * stride_vd)
            v_tile = tl.load(v_ptrs,
                             mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                             other=0.0).to(tl.float32)

            # Update accumulator: rescale old + add new
            acc = acc * correction + tl.sum(p[:, None] * v_tile, axis=0)

            m_prev = m_new
            l_prev = l_new

    # Normalize
    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    out = acc / safe_l

    # Store output
    out_ptrs = Output + pid_batch * stride_ob + query_head * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, out.to(Output.dtype.element_ty), mask=d_offsets < head_dim)


def fused_routed_decode(
    q: torch.Tensor,           # [batch, n_heads, head_dim]
    k_cache: torch.Tensor,     # [max_blocks, block_size, n_kv_heads, head_dim]
    v_cache: torch.Tensor,     # [max_blocks, block_size, n_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch, n_kv_heads, max_selected_blocks]
    block_counts: torch.Tensor,  # [batch, n_kv_heads]
    scale: float = None,
) -> torch.Tensor:
    """Fused routed decode attention.

    Each KV head attends only to its selected blocks (given by block_tables).
    All query heads in a GQA group share the same KV head's block selection.

    Args:
        q: Query vectors for decode step [batch, n_heads, head_dim]
        k_cache: Paged KV cache keys [max_blocks, block_size, n_kv_heads, head_dim]
        v_cache: Paged KV cache values [max_blocks, block_size, n_kv_heads, head_dim]
        block_tables: Per-head selected block indices [batch, n_kv_heads, max_selected_blocks]
        block_counts: Number of valid blocks per head [batch, n_kv_heads]
        scale: Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [batch, n_heads, head_dim]
    """
    batch, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_cache.shape
    max_selected_blocks = block_tables.shape[2]
    group_size = n_heads // n_kv_heads

    if scale is None:
        scale = head_dim ** -0.5

    output = torch.empty_like(q)

    # Tuning: A100 vs H100
    # A100: smaller SMs, prefer BLOCK_T=32
    # H100: larger SMs, can handle BLOCK_T=64
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_T = min(32, block_size)  # tokens per inner tile
    num_warps = 4 if BLOCK_D <= 128 else 8

    grid = (batch, group_size, n_kv_heads)

    _fused_routed_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, block_counts, output,
        scale=scale,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_selected_blocks=max_selected_blocks,
        # Q strides
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        # K strides
        stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        # V strides
        stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        # Block table strides
        stride_btb=block_tables.stride(0), stride_bth=block_tables.stride(1),
        stride_bts=block_tables.stride(2),
        # Block count strides
        stride_bcb=block_counts.stride(0), stride_bch=block_counts.stride(1),
        # Output strides
        stride_ob=output.stride(0), stride_oh=output.stride(1), stride_od=output.stride(2),
        # Tile sizes
        BLOCK_D=BLOCK_D,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
        num_stages=2,
    )

    return output


def reference_routed_decode(
    q: torch.Tensor,           # [batch, n_heads, head_dim]
    k_cache: torch.Tensor,     # [max_blocks, block_size, n_kv_heads, head_dim]
    v_cache: torch.Tensor,     # [max_blocks, block_size, n_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch, n_kv_heads, max_selected_blocks]
    block_counts: torch.Tensor,  # [batch, n_kv_heads]
    scale: float = None,
) -> torch.Tensor:
    """Reference implementation using Python loop + PyTorch ops.

    Equivalent to the per-head flash_attn loop but using vanilla PyTorch
    for correctness verification. Not optimized for speed.
    """
    batch, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_cache.shape
    group_size = n_heads // n_kv_heads

    if scale is None:
        scale = head_dim ** -0.5

    output = torch.zeros_like(q, dtype=torch.float32)

    for b in range(batch):
        for kv_h in range(n_kv_heads):
            n_blk = block_counts[b, kv_h].item()
            if n_blk == 0:
                continue

            # Gather selected KV blocks for this head
            selected_blocks = block_tables[b, kv_h, :n_blk]  # [n_blk]
            # k_selected: [n_blk * block_size, head_dim]
            k_selected = k_cache[selected_blocks, :, kv_h, :]  # [n_blk, block_size, head_dim]
            v_selected = v_cache[selected_blocks, :, kv_h, :]  # [n_blk, block_size, head_dim]
            k_flat = k_selected.reshape(-1, head_dim).float()  # [total_tokens, head_dim]
            v_flat = v_selected.reshape(-1, head_dim).float()

            for qh_offset in range(group_size):
                qh = kv_h * group_size + qh_offset
                q_vec = q[b, qh, :].float()  # [head_dim]

                # Attention scores
                scores = (k_flat @ q_vec) * scale  # [total_tokens]
                weights = torch.softmax(scores, dim=0)  # [total_tokens]
                out_vec = weights @ v_flat  # [head_dim]
                output[b, qh, :] = out_vec

    return output.to(q.dtype)


def select_top_k_blocks(
    routing_prior: torch.Tensor,  # [n_layers, n_kv_heads, n_blocks]
    k: int,
) -> tuple:
    """Select top-K blocks per KV head from a routing prior.

    Args:
        routing_prior: Block affinity scores [n_layers, n_kv_heads, n_blocks]
        k: Number of blocks to select per head

    Returns:
        block_tables: [n_layers, n_kv_heads, k]
        block_counts: [n_layers, n_kv_heads] (all == k)
    """
    n_layers, n_kv_heads, n_blocks = routing_prior.shape
    actual_k = min(k, n_blocks)
    block_tables = torch.topk(routing_prior, actual_k, dim=-1).indices  # [n_layers, n_kv_heads, k]
    block_counts = torch.full((n_layers, n_kv_heads), actual_k, dtype=torch.int32)
    return block_tables, block_counts
