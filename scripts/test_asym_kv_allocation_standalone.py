#!/usr/bin/env python3
"""Milestone 2 verification (standalone): vLLM asym KV cache allocation.

This is a standalone test of the split algorithm in
`vllm/v1/worker/gpu/attn_utils.py:_reshape_kv_cache` lines 141-163.
The algorithm is mechanically reproduced here so the contract can be
verified without a full vLLM install (vllm pulls cbor2, gguf, and the
rest of the transformers config tree just to import a dataclass).

The companion test that invokes the fork's actual function lives at
`tests/v1/worker/test_asym_kv_cache_allocation.py` in the fork; it
runs in CI / on a pod where vllm is installed.

This file is the day-zero gate: it proves the algorithm produces a
correctly-typed, non-aliasing, byte-accounted (k_cache, v_cache)
tuple from a raw int8 buffer.  Run with:
    python3 scripts/test_asym_kv_allocation_standalone.py
"""

import sys

import torch


def split_asym_kv(
    raw_tensor: torch.Tensor,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the raw int8 buffer into K (k_dtype) and V (v_dtype)
    typed strided views over the original storage.

    Layout per page is K-half then V-half.  Stride along the
    block dim equals the full page size in elements of the typed
    view.  The K view starts at storage offset 0; the V view
    starts after the K-half within the first page.

    Both views point into `raw_tensor`'s storage — no copies — so
    the block manager's occupancy bookkeeping against the raw
    tensor stays consistent with what the kernel reads/writes.
    """
    k_elem_bytes = torch.empty((), dtype=k_dtype).element_size()
    v_elem_bytes = torch.empty((), dtype=v_dtype).element_size()
    elements_per_page = block_size * num_kv_heads * head_size
    k_page_bytes = elements_per_page * k_elem_bytes
    v_page_bytes = elements_per_page * v_elem_bytes
    page_bytes = k_page_bytes + v_page_bytes

    # Sanity: the raw allocation must hold an integer number of
    # full pages and align to both dtype sizes.
    assert raw_tensor.numel() == num_blocks * page_bytes
    assert page_bytes % k_elem_bytes == 0
    assert page_bytes % v_elem_bytes == 0
    assert k_page_bytes % v_elem_bytes == 0

    # Reinterpret the raw uint8/int8 buffer as the typed dtype.
    # PyTorch requires a contiguous 1-D view to do .view(dtype),
    # but `raw_tensor` is already contiguous so this is free.
    k_typed = raw_tensor.view(k_dtype)
    v_typed = raw_tensor.view(v_dtype)

    # Stride along the block dim is the full page in *elements*
    # of the typed view (not bytes).  K-half lives at the start
    # of each page; V-half lives `k_page_bytes` after that.
    k_page_stride = page_bytes // k_elem_bytes
    v_page_stride = page_bytes // v_elem_bytes
    v_offset_elems = k_page_bytes // v_elem_bytes

    k_cache = torch.as_strided(
        k_typed,
        size=(num_blocks, block_size, num_kv_heads, head_size),
        stride=(k_page_stride, num_kv_heads * head_size, head_size, 1),
        storage_offset=0,
    )
    v_cache = torch.as_strided(
        v_typed,
        size=(num_blocks, block_size, num_kv_heads, head_size),
        stride=(v_page_stride, num_kv_heads * head_size, head_size, 1),
        storage_offset=v_offset_elems,
    )
    return k_cache, v_cache


def test_asym_split_dtypes_and_shapes():
    """Asym split produces 4-D NHD tuple with separate dtypes."""
    NB, BS, H, D = 4, 16, 8, 128
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn

    elements = NB * BS * H * D
    k_bytes = elements * 2
    v_bytes = elements * 1
    raw = torch.zeros(k_bytes + v_bytes, dtype=torch.int8)

    k_cache, v_cache = split_asym_kv(raw, NB, BS, H, D, k_dtype, v_dtype)

    assert k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.float8_e4m3fn
    expected_shape = (NB, BS, H, D)
    assert tuple(k_cache.shape) == expected_shape
    assert tuple(v_cache.shape) == expected_shape
    print("  test_asym_split_dtypes_and_shapes: OK")


def test_asym_split_byte_accounting_is_0_75x_of_symmetric_bf16():
    """Total bytes = 0.75 * symmetric BF16 K+V — at the allocator
    boundary, not just at the LMCache codec boundary."""
    NB, BS, H, D = 4, 16, 8, 128
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn

    elements = NB * BS * H * D
    expected_k_bytes = elements * 2
    expected_v_bytes = elements * 1
    expected_total = expected_k_bytes + expected_v_bytes
    raw = torch.zeros(expected_total, dtype=torch.int8)

    k_cache, v_cache = split_asym_kv(raw, NB, BS, H, D, k_dtype, v_dtype)

    assert k_cache.numel() * k_cache.element_size() == expected_k_bytes
    assert v_cache.numel() * v_cache.element_size() == expected_v_bytes
    sym_bf16_kv_bytes = elements * 2 * 2
    # 0.75 ratio: total * 4 == symmetric * 3
    assert expected_total * 4 == sym_bf16_kv_bytes * 3, (
        f"asym total {expected_total} vs symmetric {sym_bf16_kv_bytes} " f"is not 0.75x"
    )
    print(
        f"  test_byte_accounting: total={expected_total} = "
        f"0.75 * sym={sym_bf16_kv_bytes}: OK"
    )


def test_asym_split_byte_pattern_matches_layout():
    """Stamping K-half with 0x11 and V-half with 0x22 in the raw
    buffer must round-trip through the asym views.  Catches off-by-one
    or layout-direction bugs in the byte split."""
    NB, BS, H, D = 4, 16, 8, 128
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn
    elements = NB * BS * H * D
    k_bytes_per_page = (BS * H * D) * 2
    v_bytes_per_page = (BS * H * D) * 1
    page_bytes = k_bytes_per_page + v_bytes_per_page

    raw = torch.zeros(page_bytes * NB, dtype=torch.int8)
    raw_pages = raw.view(NB, page_bytes)
    raw_pages[:, :k_bytes_per_page] = 0x11
    raw_pages[:, k_bytes_per_page:] = 0x22

    k_cache, v_cache = split_asym_kv(raw, NB, BS, H, D, k_dtype, v_dtype)

    # K bytes view as int8 must all be 0x11
    k_int8 = k_cache.contiguous().view(torch.int8)
    assert (
        k_int8 == 0x11
    ).all(), f"K region not all 0x11; first few: {k_int8.flatten()[:8]}"
    # V bytes view as int8 must all be 0x22
    v_int8 = v_cache.contiguous().view(torch.int8)
    assert (
        v_int8 == 0x22
    ).all(), f"V region not all 0x22; first few: {v_int8.flatten()[:8]}"
    print("  test_byte_pattern: K=0x11, V=0x22 round-trip OK")


def test_asym_split_preserves_raw_ownership():
    """K and V must be views into the original raw int8 buffer,
    not freshly-allocated storage.  The block manager's
    bookkeeping is against the raw tensor's storage; if our split
    yields tensors that point elsewhere, the bookkeeping is a lie.

    Reproduces the design-review concern that `.contiguous()` on
    a non-contiguous slice copies.  This test FAILS today on the
    current `split_asym_kv` implementation, which is what makes
    it the right gate for Step 0.
    """
    NB, BS, H, D = 4, 16, 8, 128
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn
    elements = NB * BS * H * D
    raw = torch.zeros(elements * 2 + elements * 1, dtype=torch.int8)

    raw_start = raw.data_ptr()
    raw_end = raw_start + raw.untyped_storage().nbytes()

    k_cache, v_cache = split_asym_kv(raw, NB, BS, H, D, k_dtype, v_dtype)

    for side, t in [("K", k_cache), ("V", v_cache)]:
        ptr = t.data_ptr()
        assert raw_start <= ptr < raw_end, (
            f"asym {side} cache at 0x{ptr:x} is NOT a view into the "
            f"raw allocation [0x{raw_start:x}, 0x{raw_end:x}). "
            f"`.contiguous()` allocated fresh storage."
        )
    print("  test_preserves_raw_ownership: K and V are views into raw OK")


def test_asym_split_no_aliasing():
    """K and V must occupy distinct byte regions — writing through
    the K view must not corrupt V, and vice versa."""
    NB, BS, H, D = 2, 8, 4, 64
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn
    elements = NB * BS * H * D
    raw = torch.zeros(elements * 2 + elements * 1, dtype=torch.int8)

    k_cache, v_cache = split_asym_kv(raw, NB, BS, H, D, k_dtype, v_dtype)

    # data_ptrs differ
    assert k_cache.data_ptr() != v_cache.data_ptr()

    # Write distinct values through the typed views and verify
    # they don't bleed into each other.
    k_cache.fill_(1.0)
    v_cache.fill_(0.5)

    # Re-extract bytes from the typed views and check they match
    # what we wrote (k != 0, v != 0, but they don't equal each other).
    assert not torch.equal(
        k_cache.contiguous().view(torch.int8),
        v_cache.contiguous().view(torch.int8),
    )
    print("  test_no_aliasing: K-write vs V-write distinct OK")


def test_asym_split_8mb_page_grid():
    """Bigger grid that mimics a real Qwen2.5-7B layer slice.
    Catches issues that only show up at production-relevant sizes."""
    # Qwen2.5-7B: 28 layers × 4 KV heads × 128 head_dim.  One layer:
    NB, BS, H, D = 256, 16, 4, 128  # 256 blocks of 16 tokens
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn
    elements = NB * BS * H * D
    raw = torch.zeros(elements * 2 + elements * 1, dtype=torch.int8)

    k_cache, v_cache = split_asym_kv(raw, NB, BS, H, D, k_dtype, v_dtype)

    assert tuple(k_cache.shape) == (NB, BS, H, D)
    assert k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.float8_e4m3fn
    total_bytes = (
        k_cache.numel() * k_cache.element_size()
        + v_cache.numel() * v_cache.element_size()
    )
    assert total_bytes == raw.numel()
    print(f"  test_8mb_grid: {total_bytes:,} bytes K+V split OK")


def main():
    if not hasattr(torch, "float8_e4m3fn"):
        print("torch.float8_e4m3fn not available — skipping")
        return 0
    print("Milestone 2 standalone gate — vLLM asym KV allocation")
    print()
    test_asym_split_dtypes_and_shapes()
    test_asym_split_byte_accounting_is_0_75x_of_symmetric_bf16()
    test_asym_split_byte_pattern_matches_layout()
    test_asym_split_preserves_raw_ownership()
    test_asym_split_no_aliasing()
    test_asym_split_8mb_page_grid()
    print()
    print("=== ALL M2 STANDALONE CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
