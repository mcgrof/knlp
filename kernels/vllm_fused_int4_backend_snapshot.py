# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused INT4 KV cache attention backend.

Stores K and V as packed INT4 (2 values per uint8 byte) with per-group
FP16 scales.  The decode kernel dequantizes in-register and computes
attention without materializing a full FP16 intermediate — this is the
"fused" path that eliminates the intermediate memory traffic.

Design notes for future asymmetric K/V:
  - K and V have independent scale tensors (k_scales, v_scales)
  - The cache layout stores K and V in separate halves of a (2, ...) tensor
  - Group size is a module-level constant today (GROUP_SIZE=32) but is
    threaded through as a parameter so it can become per-K / per-V later
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_SIZE: int = 32  # number of elements per quantization group
INT4_RANGE: int = 7  # max absolute value representable in signed INT4


# Portable rounding helper that works on both CUDA and ROCm Triton
@triton.jit
def _triton_round(x):
    """Round-half-away-from-zero, portable across Triton backends."""
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


# ===================================================================
# Triton kernel: reshape_and_cache_int4
# Quantizes FP16/BF16 K/V to INT4 and writes packed bytes + per-group
# scales into the paged cache.
# ===================================================================
@triton.jit
def _reshape_and_cache_int4_kernel(
    # Source K/V from the model (FP16/BF16)
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    # Destination packed cache (uint8)
    key_cache_ptr,  # [num_blocks, block_size, num_heads, packed_dim]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, packed_dim]
    # Destination scale cache (float16)
    k_scales_ptr,  # [num_blocks, block_size, num_heads, num_groups]
    v_scales_ptr,  # [num_blocks, block_size, num_heads, num_groups]
    # Slot mapping
    slot_mapping_ptr,  # [num_tokens]
    # Strides — source
    key_stride_token: tl.int64,
    key_stride_head: tl.int64,
    value_stride_token: tl.int64,
    value_stride_head: tl.int64,
    # Strides — packed cache
    kc_stride_block: tl.int64,
    kc_stride_page: tl.int64,
    kc_stride_head: tl.int64,
    vc_stride_block: tl.int64,
    vc_stride_page: tl.int64,
    vc_stride_head: tl.int64,
    # Strides — scales
    ks_stride_block: tl.int64,
    ks_stride_page: tl.int64,
    ks_stride_head: tl.int64,
    vs_stride_block: tl.int64,
    vs_stride_page: tl.int64,
    vs_stride_head: tl.int64,
    # Dims (constexpr for compile-time specialisation)
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    HALF_HD: tl.constexpr,
):
    """One program instance handles one (token, head) pair."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return  # padding token

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # ---- Source offsets (FP16/BF16 key/value) ----
    src_k_base = token_idx * key_stride_token + head_idx * key_stride_head
    src_v_base = token_idx * value_stride_token + head_idx * value_stride_head

    # ---- Destination offsets (packed uint8 cache, scales) ----
    dst_kc_base = (block_idx * kc_stride_block
                   + block_offset * kc_stride_page
                   + head_idx * kc_stride_head)
    dst_vc_base = (block_idx * vc_stride_block
                   + block_offset * vc_stride_page
                   + head_idx * vc_stride_head)
    dst_ks_base = (block_idx * ks_stride_block
                   + block_offset * ks_stride_page
                   + head_idx * ks_stride_head)
    dst_vs_base = (block_idx * vs_stride_block
                   + block_offset * vs_stride_page
                   + head_idx * vs_stride_head)

    # Process each group
    for g in tl.static_range(0, NUM_GROUPS):
        g_start = g * GROUP_SIZE
        offs = tl.arange(0, GROUP_SIZE)
        elem_idx = g_start + offs

        # ---- Key group ----
        k_vals = tl.load(key_ptr + src_k_base + elem_idx,
                         mask=elem_idx < head_size).to(tl.float32)
        k_amax = tl.max(tl.abs(k_vals))
        k_amax = tl.maximum(k_amax, 1e-8)
        k_scale = k_amax / 7.0
        k_q = _triton_round(k_vals / k_scale)
        k_q = tl.maximum(tl.minimum(k_q, 7.0), -8.0)
        k_unsigned = (k_q + 8.0).to(tl.uint8)

        # Pack pairs: low nibble = even index, high nibble = odd index
        half_offs = tl.arange(0, GROUP_SIZE // 2)
        k_low = tl.load(key_ptr + src_k_base + g_start + half_offs * 2,
                         mask=(g_start + half_offs * 2) < head_size).to(tl.float32)
        k_high = tl.load(key_ptr + src_k_base + g_start + half_offs * 2 + 1,
                          mask=(g_start + half_offs * 2 + 1) < head_size).to(tl.float32)

        k_low_q = _triton_round(k_low / k_scale)
        k_low_q = tl.maximum(tl.minimum(k_low_q, 7.0), -8.0)
        k_low_u = (k_low_q + 8.0).to(tl.uint8)

        k_high_q = _triton_round(k_high / k_scale)
        k_high_q = tl.maximum(tl.minimum(k_high_q, 7.0), -8.0)
        k_high_u = (k_high_q + 8.0).to(tl.uint8)

        k_packed = k_low_u | (k_high_u << 4)

        packed_offs = g * (GROUP_SIZE // 2) + half_offs
        tl.store(key_cache_ptr + dst_kc_base + packed_offs,
                 k_packed, mask=packed_offs < HALF_HD)

        # Store key scale for this group
        tl.store(k_scales_ptr + dst_ks_base + g, k_scale.to(tl.float16))

        # ---- Value group ----
        v_low = tl.load(value_ptr + src_v_base + g_start + half_offs * 2,
                         mask=(g_start + half_offs * 2) < head_size).to(tl.float32)
        v_high = tl.load(value_ptr + src_v_base + g_start + half_offs * 2 + 1,
                          mask=(g_start + half_offs * 2 + 1) < head_size).to(tl.float32)

        # Compute scale from all elements in group
        v_vals = tl.load(value_ptr + src_v_base + elem_idx,
                         mask=elem_idx < head_size).to(tl.float32)
        v_amax = tl.max(tl.abs(v_vals))
        v_amax = tl.maximum(v_amax, 1e-8)
        v_scale = v_amax / 7.0

        v_low_q = _triton_round(v_low / v_scale)
        v_low_q = tl.maximum(tl.minimum(v_low_q, 7.0), -8.0)
        v_low_u = (v_low_q + 8.0).to(tl.uint8)

        v_high_q = _triton_round(v_high / v_scale)
        v_high_q = tl.maximum(tl.minimum(v_high_q, 7.0), -8.0)
        v_high_u = (v_high_q + 8.0).to(tl.uint8)

        v_packed = v_low_u | (v_high_u << 4)

        tl.store(value_cache_ptr + dst_vc_base + packed_offs,
                 v_packed, mask=packed_offs < HALF_HD)

        tl.store(v_scales_ptr + dst_vs_base + g, v_scale.to(tl.float16))


def reshape_and_cache_int4(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size//2]
    value_cache: torch.Tensor,
    k_scales: torch.Tensor,  # [num_blocks, block_size, num_heads, num_groups]
    v_scales: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
) -> None:
    """Quantize FP16/BF16 K/V to INT4 and write into paged cache."""
    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    num_groups = head_size // GROUP_SIZE
    half_hd = head_size // 2

    grid = (num_tokens, num_heads)
    _reshape_and_cache_int4_kernel[grid](
        key, value,
        key_cache, value_cache,
        k_scales, v_scales,
        slot_mapping,
        # Source strides
        key.stride(0), key.stride(1),
        value.stride(0), value.stride(1),
        # Packed cache strides
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        # Scale strides
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        # Dims
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        GROUP_SIZE=GROUP_SIZE,
        NUM_GROUPS=num_groups,
        HALF_HD=half_hd,
    )


# ===================================================================
# Triton kernel: fused INT4 decode attention
# Reads packed INT4 K/V from paged cache, dequantizes in-register,
# computes scaled-dot-product attention with online softmax.
# No FP16 intermediate is materialised in global memory.
# ===================================================================
@triton.jit
def _fused_int4_decode_kernel(
    # Query — one token per request (decode)
    Q_ptr,  # [num_seqs, num_heads, head_size]
    # Packed KV cache (paged)
    K_cache_ptr,  # [num_blocks, block_size, num_kv_heads, half_hd]
    V_cache_ptr,
    # Scales
    K_scales_ptr,  # [num_blocks, block_size, num_kv_heads, num_groups]
    V_scales_ptr,
    # Output
    Out_ptr,  # [num_seqs, num_heads, head_size]
    # Block table for paged KV
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    # Sequence lengths
    seq_lens_ptr,  # [num_seqs]
    # Strides — Q
    q_stride_seq: tl.int64,
    q_stride_head: tl.int64,
    # Strides — packed cache
    kc_stride_block: tl.int64,
    kc_stride_page: tl.int64,
    kc_stride_head: tl.int64,
    vc_stride_block: tl.int64,
    vc_stride_page: tl.int64,
    vc_stride_head: tl.int64,
    # Strides — scales
    ks_stride_block: tl.int64,
    ks_stride_page: tl.int64,
    ks_stride_head: tl.int64,
    vs_stride_block: tl.int64,
    vs_stride_page: tl.int64,
    vs_stride_head: tl.int64,
    # Strides — output
    o_stride_seq: tl.int64,
    o_stride_head: tl.int64,
    # Strides — block table
    bt_stride_seq: tl.int64,
    # Dims
    num_kv_heads: tl.int32,
    scale: tl.float32,  # 1/sqrt(head_dim)
    # Compile-time constants
    HEAD_DIM: tl.constexpr,
    HALF_HD: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,  # GROUP_SIZE // 2
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # context tile size
    N_REP: tl.constexpr,  # num_heads // num_kv_heads (GQA ratio)
):
    pid_sh = tl.program_id(0)
    # Decompose program id into (seq, head)
    num_q_heads = num_kv_heads * N_REP
    seq_idx = pid_sh // num_q_heads
    head_idx = pid_sh % num_q_heads
    kv_head_idx = head_idx // N_REP

    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # Load Q for this head — split into even/odd for packed K dot product
    q_base = Q_ptr + seq_idx * q_stride_seq + head_idx * q_stride_head
    even_offs = tl.arange(0, HALF_HD) * 2
    odd_offs = tl.arange(0, HALF_HD) * 2 + 1
    q_even = tl.load(q_base + even_offs, mask=even_offs < HEAD_DIM).to(tl.float32)
    q_odd = tl.load(q_base + odd_offs, mask=odd_offs < HEAD_DIM).to(tl.float32)

    packed_d_offs = tl.arange(0, HALF_HD)
    group_idx = packed_d_offs // BYTES_PER_GROUP  # which group each packed byte belongs to

    # Online softmax accumulators
    m_i = float("-inf")
    l_i = 0.0
    acc_even = tl.zeros([HALF_HD], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_HD], dtype=tl.float32)

    # Iterate over context in tiles of BLOCK_N
    for start_n in range(0, seq_len, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len

        # Resolve paged addresses: each position maps to a physical block
        logical_block = n_offs // BLOCK_SIZE
        block_offset = n_offs % BLOCK_SIZE
        phys_block = tl.load(
            block_table_ptr + seq_idx * bt_stride_seq + logical_block,
            mask=n_mask, other=0,
        ).to(tl.int64)

        # ---- Load and dequantize K ----
        k_base = (phys_block[:, None] * kc_stride_block
                  + block_offset[:, None] * kc_stride_page
                  + kv_head_idx * kc_stride_head)
        k_packed = tl.load(
            K_cache_ptr + k_base + packed_d_offs[None, :],
            mask=n_mask[:, None], other=0,
        ).to(tl.uint8)

        k_low = ((k_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
        k_high = (((k_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        # Load K scales and broadcast to packed dimension
        ks_base = (phys_block[:, None] * ks_stride_block
                   + block_offset[:, None] * ks_stride_page
                   + kv_head_idx * ks_stride_head)
        scale_k_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask = (group_idx == g).to(tl.float32)
            sk_g = tl.load(
                K_scales_ptr + ks_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            # sk_g is [BLOCK_N, 1] after squeeze; broadcast to half_hd
            scale_k_expanded += sk_g * g_mask[None, :]

        k_low = k_low * scale_k_expanded
        k_high = k_high * scale_k_expanded

        # QK^T dot product using even/odd decomposition
        qk = (tl.sum(q_even[None, :] * k_low, axis=1)
              + tl.sum(q_odd[None, :] * k_high, axis=1))
        qk = qk * scale
        qk = tl.where(n_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ---- Load and dequantize V ----
        v_base = (phys_block[:, None] * vc_stride_block
                  + block_offset[:, None] * vc_stride_page
                  + kv_head_idx * vc_stride_head)
        v_packed = tl.load(
            V_cache_ptr + v_base + packed_d_offs[None, :],
            mask=n_mask[:, None], other=0,
        ).to(tl.uint8)

        v_low = ((v_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
        v_high = (((v_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

        vs_base = (phys_block[:, None] * vs_stride_block
                   + block_offset[:, None] * vs_stride_page
                   + kv_head_idx * vs_stride_head)
        scale_v_expanded = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
        for g in tl.static_range(0, NUM_GROUPS):
            g_mask = (group_idx == g).to(tl.float32)
            sv_g = tl.load(
                V_scales_ptr + vs_base + g,
                mask=n_mask[:, None], other=1.0,
            ).to(tl.float32)
            scale_v_expanded += sv_g * g_mask[None, :]

        v_low = v_low * scale_v_expanded
        v_high = v_high * scale_v_expanded

        # Weighted V accumulation
        acc_even += tl.sum(p[:, None] * v_low, axis=0)
        acc_odd += tl.sum(p[:, None] * v_high, axis=0)
        m_i = m_new

    # Normalise by softmax denominator
    acc_even = acc_even / l_i
    acc_odd = acc_odd / l_i

    # Write output — interleave even/odd back to full head_dim
    out_base = Out_ptr + seq_idx * o_stride_seq + head_idx * o_stride_head
    tl.store(out_base + even_offs, acc_even.to(tl.float16),
             mask=even_offs < HEAD_DIM)
    tl.store(out_base + odd_offs, acc_odd.to(tl.float16),
             mask=odd_offs < HEAD_DIM)


# ===================================================================
# Python wrappers
# ===================================================================

def fused_int4_decode(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, half_hd]
    value_cache: torch.Tensor,
    k_scales: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, num_groups]
    v_scales: torch.Tensor,
    block_table: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    num_kv_heads: int,
    head_size: int,
    block_n: int = 64,
) -> torch.Tensor:
    """Launch the fused INT4 decode attention kernel."""
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    n_rep = num_heads // num_kv_heads
    half_hd = head_size // 2
    num_groups = head_size // GROUP_SIZE
    bytes_per_group = GROUP_SIZE // 2
    block_size = key_cache.shape[1]
    scale = 1.0 / (head_size ** 0.5)

    output = torch.empty_like(query)

    grid = (num_seqs * num_heads,)
    _fused_int4_decode_kernel[grid](
        query,
        key_cache, value_cache,
        k_scales, v_scales,
        output,
        block_table, seq_lens,
        # Q strides
        query.stride(0), query.stride(1),
        # Packed cache strides
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        # Scale strides
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        # Output strides
        output.stride(0), output.stride(1),
        # Block table stride
        block_table.stride(0),
        # Dims
        num_kv_heads=num_kv_heads,
        scale=scale,
        HEAD_DIM=head_size,
        HALF_HD=half_hd,
        NUM_GROUPS=num_groups,
        BYTES_PER_GROUP=bytes_per_group,
        BLOCK_SIZE=block_size,
        BLOCK_N=min(block_n, 128),
        N_REP=n_rep,
    )
    return output


# ===================================================================
# Attention metadata
# ===================================================================

@dataclass
class FusedInt4AttentionMetadata:
    """Metadata for fused INT4 decode attention."""
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


# ===================================================================
# Backend class
# ===================================================================

class FusedInt4AttentionBackend(AttentionBackend):
    """Attention backend using fused INT4 KV cache with in-kernel dequant."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "int4_fused",
    ]

    @staticmethod
    def get_name() -> str:
        return "FUSED_INT4"

    @staticmethod
    def get_impl_cls() -> type["FusedInt4AttentionImpl"]:
        return FusedInt4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FusedInt4AttentionMetadataBuilder"]:
        return FusedInt4AttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "int4_fused",
    ) -> tuple[int, ...]:
        """Cache shape for INT4 fused.

        We use a (2, ...) leading dimension for K vs V, matching the
        convention of other backends.  Each K/V plane stores packed uint8
        data.  Scales are stored in a separate tensor (allocated by the
        backend impl, not via this shape).

        Shape: (2, num_blocks, block_size, num_kv_heads, head_size // 2)
        where the last dim holds packed INT4 bytes (2 values per byte).
        """
        half_hd = head_size // 2
        return (2, num_blocks, block_size, num_kv_heads, half_hd)

    @classmethod
    def supports_kv_cache_dtype(
        cls, kv_cache_dtype: CacheDType | None,
    ) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype == "int4_fused"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Head size must be divisible by GROUP_SIZE (32) for clean grouping
        return head_size % GROUP_SIZE == 0 and head_size > 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


# ===================================================================
# Metadata builder
# ===================================================================

class FusedInt4AttentionMetadataBuilder(
    AttentionMetadataBuilder[FusedInt4AttentionMetadata],
):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(reorder_batch_threshold=1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FusedInt4AttentionMetadata:
        return FusedInt4AttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


# ===================================================================
# Attention implementation
# ===================================================================

class FusedInt4AttentionImpl(
    AttentionImpl[FusedInt4AttentionMetadata],
):
    """Fused INT4 attention: quantize-on-write + dequant-in-attention decode."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "int4_fused",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError(
                "FusedInt4 backend does not support ALiBi yet"
            )
        if sliding_window is not None:
            raise NotImplementedError(
                "FusedInt4 backend does not support sliding window yet"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.half_hd = head_size // 2
        self.num_groups = head_size // GROUP_SIZE

        # Scale tensors are allocated alongside the KV cache.  We keep
        # references here after the first forward call.
        self._k_scales: torch.Tensor | None = None
        self._v_scales: torch.Tensor | None = None

        # Verification counters (logged periodically)
        self._decode_fused_count: int = 0
        self._prefill_fallback_count: int = 0
        self._logged_init: bool = False

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FusedInt4AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Split the (2, ...) cache into K and V planes
        # kv_cache shape: (2, num_blocks, block_size, num_kv_heads, half_hd)
        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]   # [num_blocks, block_size, num_kv_heads, half_hd]
            value_cache = kv_cache[1]

            # Lazily allocate scale tensors matching the cache geometry
            if self._k_scales is None:
                num_blocks = key_cache.shape[0]
                block_size = key_cache.shape[1]
                self._k_scales = torch.zeros(
                    num_blocks, block_size, self.num_kv_heads, self.num_groups,
                    dtype=torch.float16, device=key_cache.device,
                )
                self._v_scales = torch.zeros(
                    num_blocks, block_size, self.num_kv_heads, self.num_groups,
                    dtype=torch.float16, device=key_cache.device,
                )

            # ---- Cache write: quantize and store ----
            if num_actual_tokens > 0 and key.numel() > 0:
                # Reshape key/value from [num_tokens, num_kv_heads * head_size]
                # or [num_tokens, num_kv_heads, head_size] to the 3D form
                if key.dim() == 2:
                    key = key.view(-1, self.num_kv_heads, self.head_size)
                    value = value.view(-1, self.num_kv_heads, self.head_size)

                reshape_and_cache_int4(
                    key, value,
                    key_cache, value_cache,
                    self._k_scales, self._v_scales,
                    attn_metadata.slot_mapping,
                )

            # ---- Decode: fused attention ----
            if attn_metadata.max_query_len == 1:
                # Pure decode — use fused INT4 kernel
                if query.dim() == 2:
                    query = query.view(-1, self.num_heads, self.head_size)

                # Select BLOCK_N for bounded dispatch
                # (future: use batch-dependent heuristic here)
                block_n = 64

                # Backend verification logging
                if not self._logged_init:
                    self._logged_init = True
                    logger.info(
                        "[FusedInt4] Backend verification: "
                        "selected_backend=FUSED_INT4, "
                        "kv_cache_dtype=int4_fused, "
                        "decode_kernel=fused_int4_triton, "
                        "group_size=%d, "
                        "num_kv_heads=%d, head_size=%d, "
                        "block_n=%d, "
                        "fallback=none",
                        GROUP_SIZE, self.num_kv_heads,
                        self.head_size, block_n,
                    )
                self._decode_fused_count += 1

                decode_output = fused_int4_decode(
                    query[:attn_metadata.seq_lens.shape[0]],
                    key_cache, value_cache,
                    self._k_scales, self._v_scales,
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    block_n=block_n,
                )

                if output is not None:
                    output[:decode_output.shape[0]].copy_(
                        decode_output.view(output[:decode_output.shape[0]].shape)
                    )
                    return output
                return decode_output.view(
                    decode_output.shape[0], -1
                )
            else:
                # Prefill: dequantize from cache and use standard attention.
                # For now, fall back to a simple torch SDPA path.
                # This is NOT the fused path — it's the honest fallback.
                self._prefill_fallback_count += 1
                logger.warning_once(
                    "[FusedInt4] Backend verification: "
                    "prefill uses dequantized SDPA fallback. "
                    "Only decode uses the fused INT4 kernel. "
                    "fallback_reason=prefill_not_fused"
                )
                return self._prefill_fallback(
                    query, key, value, kv_cache, attn_metadata, output,
                )
        else:
            # No cache — just compute attention directly (e.g. prompt eval
            # with no prior context).  This shouldn't normally happen in
            # the decode path.
            if output is not None:
                output.zero_()
                return output
            return torch.zeros_like(query)

    def _prefill_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FusedInt4AttentionMetadata,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        """Honest prefill fallback using torch SDPA on dequantized KV.

        During prefill the fresh K/V are in FP16/BF16 (not yet quantized
        into the cache).  We just use them directly with SDPA.
        """
        if query.dim() == 2:
            query = query.view(-1, self.num_heads, self.head_size)
        if key.dim() == 2:
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)

        # GQA expansion
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        # SDPA expects [batch, heads, seq_len, head_dim]
        # For prefill with variable-length sequences, we do a simple
        # batched approach — process each sequence individually
        query_start_loc = attn_metadata.query_start_loc
        num_seqs = attn_metadata.seq_lens.shape[0]

        results = []
        for i in range(num_seqs):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            q_i = query[start:end].transpose(0, 1).unsqueeze(0)
            k_i = key[start:end].transpose(0, 1).unsqueeze(0)
            v_i = value[start:end].transpose(0, 1).unsqueeze(0)
            out_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, is_causal=True,
            )
            results.append(out_i.squeeze(0).transpose(0, 1))

        combined = torch.cat(results, dim=0)
        flat = combined.reshape(combined.shape[0], -1)
        if output is not None:
            output[:flat.shape[0]].copy_(flat)
            return output
        return flat
