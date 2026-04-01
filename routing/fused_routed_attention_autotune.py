# SPDX-License-Identifier: GPL-2.0
"""Autotuned fused routed attention kernel for H100 optimization.

Extends the landed fused_routed_attention.py with:
- @triton.autotune over BLOCK_T, num_warps, num_stages
- Bounded config space targeting real operating points (BS=128/256 K=8)
- Preserves the same interface as the original kernel

This file is the Step 2 (autotune sweep) artifact from the H100
optimization plan (routing-h100-kernel-optimization-plan-final-20260401.md).
"""

import torch
import triton
import triton.language as tl

from routing.fused_routed_attention import (
    reference_routed_decode,
    select_top_k_blocks,
)


def _get_autotune_configs():
    """Bounded autotune config space for H100 optimization.

    Targets real routing operating points:
    - BS=128 K=8: 1024 selected tokens per head
    - BS=256 K=8: 2048 selected tokens per head

    Sweep:
    - BLOCK_T in {32, 64, 128} (token tile size)
    - num_warps in {4, 8}
    - num_stages in {2, 3, 4}
    """
    configs = []
    for block_t in [32, 64, 128]:
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
    stride_qb, stride_qh, stride_qd,
    # Strides for K_cache
    stride_kb, stride_kt, stride_kh, stride_kd,
    # Strides for V_cache
    stride_vb, stride_vt, stride_vh, stride_vd,
    # Strides for Block_tables
    stride_btb, stride_bth, stride_bts,
    # Strides for Block_counts
    stride_bcb, stride_bch,
    # Strides for Output
    stride_ob, stride_oh, stride_od,
    # Tile sizes
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Fused decode attention with per-KV-head block selection (autotuned).

    Grid: (batch, n_query_heads_in_group, n_kv_heads)
    """
    pid_batch = tl.program_id(0)
    pid_qh_in_group = tl.program_id(1)
    pid_kv_head = tl.program_id(2)

    group_size = n_heads // n_kv_heads
    query_head = pid_kv_head * group_size + pid_qh_in_group

    n_blocks = tl.load(
        Block_counts + pid_batch * stride_bcb + pid_kv_head * stride_bch
    )

    d_offsets = tl.arange(0, BLOCK_D)
    q_ptrs = Q + pid_batch * stride_qb + query_head * stride_qh + d_offsets * stride_qd
    q_vec = tl.load(q_ptrs, mask=d_offsets < head_dim, other=0.0).to(tl.float32)

    m_prev = float('-inf') + tl.zeros([1], dtype=tl.float32)
    l_prev = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for block_idx in range(max_selected_blocks):
        active = block_idx < n_blocks
        phys_block = tl.load(
            Block_tables + pid_batch * stride_btb + pid_kv_head * stride_bth + block_idx * stride_bts
        )

        for t_start in range(0, block_size, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            t_mask = (t_offsets < block_size) & active

            k_ptrs = (K_cache
                      + phys_block * stride_kb
                      + t_offsets[:, None] * stride_kt
                      + pid_kv_head * stride_kh
                      + d_offsets[None, :] * stride_kd)
            k_tile = tl.load(k_ptrs,
                             mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                             other=0.0).to(tl.float32)

            scores = tl.sum(q_vec[None, :] * k_tile, axis=1) * scale
            scores = tl.where(t_mask, scores, float('-inf'))

            m_new = tl.maximum(m_prev, tl.max(scores, axis=0))
            correction = tl.exp(m_prev - m_new)
            p = tl.exp(scores - m_new)
            l_new = l_prev * correction + tl.sum(p, axis=0)

            v_ptrs = (V_cache
                      + phys_block * stride_vb
                      + t_offsets[:, None] * stride_vt
                      + pid_kv_head * stride_vh
                      + d_offsets[None, :] * stride_vd)
            v_tile = tl.load(v_ptrs,
                             mask=t_mask[:, None] & (d_offsets[None, :] < head_dim),
                             other=0.0).to(tl.float32)

            acc = acc * correction + tl.sum(p[:, None] * v_tile, axis=0)
            m_prev = m_new
            l_prev = l_new

    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    out = acc / safe_l

    out_ptrs = Output + pid_batch * stride_ob + query_head * stride_oh + d_offsets * stride_od
    tl.store(out_ptrs, out.to(Output.dtype.element_ty), mask=d_offsets < head_dim)


def fused_routed_decode_autotune(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_counts: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """Fused routed decode with Triton autotune.

    Same interface as fused_routed_decode but uses @triton.autotune
    to sweep BLOCK_T, num_warps, num_stages.
    """
    batch, n_heads, head_dim = q.shape
    _, block_size, n_kv_heads, _ = k_cache.shape
    max_selected_blocks = block_tables.shape[2]
    group_size = n_heads // n_kv_heads

    if scale is None:
        scale = head_dim ** -0.5

    output = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (batch, group_size, n_kv_heads)

    _fused_routed_decode_autotune_kernel[grid](
        q, k_cache, v_cache, block_tables, block_counts, output,
        scale=scale,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_selected_blocks=max_selected_blocks,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        stride_kb=k_cache.stride(0), stride_kt=k_cache.stride(1),
        stride_kh=k_cache.stride(2), stride_kd=k_cache.stride(3),
        stride_vb=v_cache.stride(0), stride_vt=v_cache.stride(1),
        stride_vh=v_cache.stride(2), stride_vd=v_cache.stride(3),
        stride_btb=block_tables.stride(0), stride_bth=block_tables.stride(1),
        stride_bts=block_tables.stride(2),
        stride_bcb=block_counts.stride(0), stride_bch=block_counts.stride(1),
        stride_ob=output.stride(0), stride_oh=output.stride(1), stride_od=output.stride(2),
        BLOCK_D=BLOCK_D,
    )

    return output
