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
        autotune,           # bool — use autotuned kernel variant
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

Autotune:
    Pass autotune=True to use the @triton.autotune variant, which sweeps
    BLOCK_T in {32, 64, 128}, num_warps in {4, 8}, num_stages in {2, 3, 4}.
    The autotune key is (head_dim, block_size, max_selected_blocks).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_routed_decode_kernel(
    # Pointers
    Q,  # [batch, n_heads, head_dim]
    K_cache,  # [max_blocks, block_size, n_kv_heads, head_dim]
    V_cache,  # [max_blocks, block_size, n_kv_heads, head_dim]
    Block_tables,  # [batch, n_kv_heads, max_selected_blocks]
    Block_counts,  # [batch, n_kv_heads]
    Output,  # [batch, n_heads, head_dim]
    # Scalars
    scale: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_selected_blocks: tl.constexpr,
    # Strides for Q: [batch, n_heads, head_dim]
    stride_qb,
    stride_qh,
    stride_qd,
    # Strides for K_cache: [max_blocks, block_size, n_kv_heads, head_dim]
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    # Strides for V_cache: [max_blocks, block_size, n_kv_heads, head_dim]
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    # Strides for Block_tables: [batch, n_kv_heads, max_selected_blocks]
    stride_btb,
    stride_bth,
    stride_bts,
    # Strides for Block_counts: [batch, n_kv_heads]
    stride_bcb,
    stride_bch,
    # Strides for Output: [batch, n_heads, head_dim]
    stride_ob,
    stride_oh,
    stride_od,
    # Block dimensions
    BLOCK_D: tl.constexpr,  # head_dim tile (usually == head_dim)
    BLOCK_T: tl.constexpr,  # tokens per inner iteration
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
    n_blocks = tl.load(Block_counts + pid_batch * stride_bcb + pid_kv_head * stride_bch)

    # Load query vector: q[batch, query_head, :]
    d_offsets = tl.arange(0, BLOCK_D)
    q_ptrs = Q + pid_batch * stride_qb + query_head * stride_qh + d_offsets * stride_qd
    q_vec = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

    # Online softmax accumulators
    m_prev = float("-inf") + tl.zeros(
        [1], dtype=tl.float32
    )  # running max (scalar-like)
    l_prev = tl.zeros([1], dtype=tl.float32)  # running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # weighted value accumulator

    # Iterate over selected blocks (no break — use masking for inactive blocks)
    for block_idx in range(max_selected_blocks):
        # Use block 0 as a dummy when block_idx >= n_blocks.
        # The scores will be masked to -inf so they contribute nothing.
        active = block_idx < n_blocks

        # Load physical block ID from block table (use 0 for inactive)
        phys_block = tl.load(
            Block_tables
            + pid_batch * stride_btb
            + pid_kv_head * stride_bth
            + block_idx * stride_bts
        )

        # Process tokens within this block in tiles of BLOCK_T
        for t_start in range(0, block_size, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            t_mask = (t_offsets < block_size) & active

            # Load K[phys_block, t_offsets, kv_head, :] -> [BLOCK_T, BLOCK_D]
            k_ptrs = (
                K_cache
                + phys_block * stride_kb
                + t_offsets[:, None] * stride_kt
                + pid_kv_head * stride_kh
                + d_offsets[None, :] * stride_kd
            )
            k_tile = tl.load(
                k_ptrs,
                mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)

            # Compute attention scores: q @ k^T -> [BLOCK_T]
            scores = tl.sum(q_vec[None, :] * k_tile, axis=1) * scale  # [BLOCK_T]
            scores = tl.where(t_mask, scores, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
            # Correction factor for previous accumulator
            correction = tl.exp(m_prev - m_new)
            # New exponentials
            p = tl.exp(scores - m_new)
            l_new = l_prev * correction + tl.sum(p, axis=0)

            # Load V[phys_block, t_offsets, kv_head, :] -> [BLOCK_T, BLOCK_D]
            v_ptrs = (
                V_cache
                + phys_block * stride_vb
                + t_offsets[:, None] * stride_vt
                + pid_kv_head * stride_vh
                + d_offsets[None, :] * stride_vd
            )
            v_tile = tl.load(
                v_ptrs,
                mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)

            # Update accumulator: rescale old + add new
            acc = acc * correction + tl.sum(p[:, None] * v_tile, axis=0)

            m_prev = m_new
            l_prev = l_new

    # Normalize
    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    out = acc / safe_l

    # Store output
    out_ptrs = (
        Output + pid_batch * stride_ob + query_head * stride_oh + d_offsets * stride_od
    )
    tl.store(out_ptrs, out.to(Output.dtype.element_ty), mask=d_offsets < head_dim)


# ---------------------------------------------------------------------------
# Autotuned kernel variant
# ---------------------------------------------------------------------------


def _get_autotune_configs():
    """Bounded autotune config space for H100 optimization.

    Targets real routing operating points:
    - BS=128 K=8: 1024 selected tokens per head
    - BS=256 K=8: 2048 selected tokens per head

    Sweep:
    - BLOCK_T in {32, 64, 128, 256} (token tile size)
    - num_warps in {4, 8}
    - num_stages in {2, 3, 4}

    P1 finding (2026-04-03): BLOCK_T=256 dominates for BS=256 operating
    points, giving 35-36% kernel speedup over the old max of BLOCK_T=128.
    For BS=128, BLOCK_T=128 remains optimal (processes entire block in one
    tile, eliminating inner-loop overhead).
    """
    configs = []
    for block_t in [32, 64, 128, 256]:
        for nw in [4, 8]:
            for ns in [2, 3, 4]:
                configs.append(
                    triton.Config(
                        {"BLOCK_T": block_t},
                        num_warps=nw,
                        num_stages=ns,
                    )
                )
    return configs


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["head_dim", "block_size", "max_selected_blocks"],
)
@triton.jit
def _fused_routed_decode_autotune_kernel(
    # Pointers
    Q,
    K_cache,
    V_cache,
    Block_tables,
    Block_counts,
    Output,
    # Scalars
    scale: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_selected_blocks: tl.constexpr,
    # Strides for Q
    stride_qb,
    stride_qh,
    stride_qd,
    # Strides for K_cache
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    # Strides for V_cache
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    # Strides for Block_tables
    stride_btb,
    stride_bth,
    stride_bts,
    # Strides for Block_counts
    stride_bcb,
    stride_bch,
    # Strides for Output
    stride_ob,
    stride_oh,
    stride_od,
    # Tile sizes
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Fused decode attention with per-KV-head block selection (autotuned).

    Identical kernel body to _fused_routed_decode_kernel; the only difference
    is the @triton.autotune decorator that sweeps BLOCK_T / num_warps /
    num_stages automatically.

    Grid: (batch, n_query_heads_in_group, n_kv_heads)
    """
    pid_batch = tl.program_id(0)
    pid_qh_in_group = tl.program_id(1)
    pid_kv_head = tl.program_id(2)

    group_size = n_heads // n_kv_heads
    query_head = pid_kv_head * group_size + pid_qh_in_group

    n_blocks = tl.load(Block_counts + pid_batch * stride_bcb + pid_kv_head * stride_bch)

    d_offsets = tl.arange(0, BLOCK_D)
    q_ptrs = Q + pid_batch * stride_qb + query_head * stride_qh + d_offsets * stride_qd
    q_vec = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

    m_prev = float("-inf") + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for block_idx in range(max_selected_blocks):
        active = block_idx < n_blocks
        phys_block = tl.load(
            Block_tables
            + pid_batch * stride_btb
            + pid_kv_head * stride_bth
            + block_idx * stride_bts
        )

        for t_start in range(0, block_size, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            t_mask = (t_offsets < block_size) & active

            k_ptrs = (
                K_cache
                + phys_block * stride_kb
                + t_offsets[:, None] * stride_kt
                + pid_kv_head * stride_kh
                + d_offsets[None, :] * stride_kd
            )
            k_tile = tl.load(
                k_ptrs,
                mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q_vec[None, :] * k_tile, axis=1) * scale
            scores = tl.where(t_mask, scores, float("-inf"))

            m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
            correction = tl.exp(m_prev - m_new)
            p = tl.exp(scores - m_new)
            l_new = l_prev * correction + tl.sum(p, axis=0)

            v_ptrs = (
                V_cache
                + phys_block * stride_vb
                + t_offsets[:, None] * stride_vt
                + pid_kv_head * stride_vh
                + d_offsets[None, :] * stride_vd
            )
            v_tile = tl.load(
                v_ptrs,
                mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)

            acc = acc * correction + tl.sum(p[:, None] * v_tile, axis=0)
            m_prev = m_new
            l_prev = l_new

    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    out = acc / safe_l

    out_ptrs = (
        Output + pid_batch * stride_ob + query_head * stride_oh + d_offsets * stride_od
    )
    tl.store(out_ptrs, out.to(Output.dtype.element_ty), mask=d_offsets < head_dim)


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------


def _launch_kernel(
    kernel,
    q,
    k_cache,
    v_cache,
    block_tables,
    block_counts,
    output,
    scale,
    **extra_kwargs
):
    """Shared launch logic for both kernel variants."""
    batch, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_cache.shape
    max_selected_blocks = block_tables.shape[2]
    group_size = n_heads // n_kv_heads
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (batch, group_size, n_kv_heads)

    kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        block_counts,
        output,
        scale=scale,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_selected_blocks=max_selected_blocks,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k_cache.stride(0),
        stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2),
        stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0),
        stride_vt=v_cache.stride(1),
        stride_vh=v_cache.stride(2),
        stride_vd=v_cache.stride(3),
        stride_btb=block_tables.stride(0),
        stride_bth=block_tables.stride(1),
        stride_bts=block_tables.stride(2),
        stride_bcb=block_counts.stride(0),
        stride_bch=block_counts.stride(1),
        stride_ob=output.stride(0),
        stride_oh=output.stride(1),
        stride_od=output.stride(2),
        BLOCK_D=BLOCK_D,
        **extra_kwargs,
    )


def fused_routed_decode(
    q: torch.Tensor,  # [batch, n_heads, head_dim]
    k_cache: torch.Tensor,  # [max_blocks, block_size, n_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_blocks, block_size, n_kv_heads, head_dim]
    block_tables: torch.Tensor,  # [batch, n_kv_heads, max_selected_blocks]
    block_counts: torch.Tensor,  # [batch, n_kv_heads]
    scale: float = None,
    autotune: bool = False,
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
        autotune: If True, use the @triton.autotune kernel variant that
            sweeps BLOCK_T/num_warps/num_stages. Recommended for H100.

    Returns:
        Output tensor [batch, n_heads, head_dim]
    """
    batch, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_cache.shape

    if scale is None:
        scale = head_dim**-0.5

    output = torch.empty_like(q)

    if autotune:
        _launch_kernel(
            _fused_routed_decode_autotune_kernel,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_counts,
            output,
            scale,
        )
    else:
        BLOCK_D = triton.next_power_of_2(head_dim)
        # P1 finding: BLOCK_T=block_size (process entire block in one tile)
        # eliminates inner-loop iteration overhead.  For block_size <= 128
        # this is a strict win; for block_size=256 it is still beneficial
        # on H100 (wider SMs, more register file).
        BLOCK_T = min(block_size, 256)
        num_warps = 4 if block_size <= 128 else 8
        _launch_kernel(
            _fused_routed_decode_kernel,
            q,
            k_cache,
            v_cache,
            block_tables,
            block_counts,
            output,
            scale,
            BLOCK_T=BLOCK_T,
            num_warps=num_warps,
            num_stages=3,
        )

    return output


def reference_routed_decode(
    q: torch.Tensor,  # [batch, n_heads, head_dim]
    k_cache: torch.Tensor,  # [max_blocks, block_size, n_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [max_blocks, block_size, n_kv_heads, head_dim]
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
        scale = head_dim**-0.5

    output = torch.zeros_like(q, dtype=torch.float32)

    for b in range(batch):
        for kv_h in range(n_kv_heads):
            n_blk = block_counts[b, kv_h].item()
            if n_blk == 0:
                continue

            # Gather selected KV blocks for this head
            selected_blocks = block_tables[b, kv_h, :n_blk]  # [n_blk]
            # k_selected: [n_blk * block_size, head_dim]
            k_selected = k_cache[
                selected_blocks, :, kv_h, :
            ]  # [n_blk, block_size, head_dim]
            v_selected = v_cache[
                selected_blocks, :, kv_h, :
            ]  # [n_blk, block_size, head_dim]
            k_flat = k_selected.reshape(
                -1, head_dim
            ).float()  # [total_tokens, head_dim]
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
    block_tables = torch.topk(
        routing_prior, actual_k, dim=-1
    ).indices  # [n_layers, n_kv_heads, k]
    block_counts = torch.full((n_layers, n_kv_heads), actual_k, dtype=torch.int32)
    return block_tables, block_counts
