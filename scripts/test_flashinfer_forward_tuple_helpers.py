#!/usr/bin/env python3
"""Step 1 gate: vLLM FlashInfer forward() tuple-handling helpers.

These are pure-function unit tests for the three helpers added to
`vllm/v1/attention/backends/flashinfer.py`:
  - `_is_asym_paged_kv_cache(kv_cache)` — tuple sniff
  - `_derive_4d_stride_order_from_5d(...)` — drop kv_side dim, renumber
  - `_prepare_flashinfer_paged_kv_cache(kv_cache)` — symmetric or asym-aware permute

The functions themselves don't import vllm at runtime, so we
mechanically copy their contracts here and exercise them on
fabricated tensors.  The fork-side test that calls the actual
helpers lives in
`tests/v1/attention/backends/test_flashinfer_asym_forward.py`
and runs in CI / on a pod where vllm is installed.

Runs on CPU.  No GPU, no FlashInfer.
"""

import sys
from typing import Tuple

import torch


def _is_asym_paged_kv_cache(kv_cache):
    return (
        isinstance(kv_cache, tuple)
        and len(kv_cache) == 2
        and isinstance(kv_cache[0], torch.Tensor)
        and isinstance(kv_cache[1], torch.Tensor)
    )


def _derive_4d_stride_order_from_5d(stride_order_5d):
    assert len(stride_order_5d) == 5
    return tuple(
        dim - 1 if dim > 1 else dim
        for dim in stride_order_5d
        if dim != 1
    )


def _prepare_flashinfer_paged_kv_cache(kv_cache, stride_order_5d):
    """Slightly different from the in-tree version: takes the stride
    order as an explicit arg so we don't need to import the vllm
    backend module.  The dispatch logic is otherwise identical."""
    if _is_asym_paged_kv_cache(kv_cache):
        k_cache, v_cache = kv_cache
        assert k_cache.ndim == 4
        assert v_cache.ndim == 4
        assert k_cache.shape == v_cache.shape
        stride_order_4d = _derive_4d_stride_order_from_5d(stride_order_5d)
        return (
            k_cache.permute(*stride_order_4d),
            v_cache.permute(*stride_order_4d),
        )
    assert isinstance(kv_cache, torch.Tensor)
    assert kv_cache.ndim == 5
    return kv_cache.permute(*stride_order_5d)


def test_is_asym_sniff_recognises_tuple():
    a = torch.zeros(2, 3, 4, 5, dtype=torch.bfloat16)
    b = torch.zeros(2, 3, 4, 5, dtype=torch.float8_e4m3fn)
    assert _is_asym_paged_kv_cache((a, b)) is True
    print("  test_sniff_tuple: OK")


def test_is_asym_sniff_rejects_tensor():
    t = torch.zeros(2, 2, 16, 8, 64, dtype=torch.bfloat16)
    assert _is_asym_paged_kv_cache(t) is False
    print("  test_sniff_tensor: OK")


def test_is_asym_sniff_rejects_3_tuple():
    a = torch.zeros(2, 3, 4, 5)
    assert _is_asym_paged_kv_cache((a, a, a)) is False
    print("  test_sniff_3tuple: OK")


def test_is_asym_sniff_rejects_tuple_of_non_tensors():
    a = torch.zeros(2, 3, 4, 5)
    assert _is_asym_paged_kv_cache((a, "not a tensor")) is False
    assert _is_asym_paged_kv_cache((1, 2)) is False
    print("  test_sniff_non_tensor_tuple: OK")


def test_derive_4d_from_5d_NHD():
    """NHD 5-D order is identity (0,1,2,3,4).  Drop dim==1, renumber:
    (0,) + (2-1, 3-1, 4-1) = (0, 1, 2, 3) — identity 4-D."""
    assert _derive_4d_stride_order_from_5d((0, 1, 2, 3, 4)) == (0, 1, 2, 3)
    print("  test_derive_NHD: OK")


def test_derive_4d_from_5d_HND():
    """HND 5-D order swaps block_size and num_kv_heads: (0,1,3,2,4).
    Drop dim==1, renumber: (0,) + (3-1, 2-1, 4-1) = (0, 2, 1, 3)."""
    assert _derive_4d_stride_order_from_5d((0, 1, 3, 2, 4)) == (0, 2, 1, 3)
    print("  test_derive_HND: OK")


def test_prepare_symmetric_5d_NHD():
    """Symmetric 5-D NHD: identity permute, returns same shape."""
    t = torch.zeros(4, 2, 16, 8, 128, dtype=torch.bfloat16)
    out = _prepare_flashinfer_paged_kv_cache(t, (0, 1, 2, 3, 4))
    assert isinstance(out, torch.Tensor)
    assert out.shape == t.shape
    print(f"  test_prepare_sym_NHD: shape={tuple(out.shape)} OK")


def test_prepare_symmetric_5d_HND():
    """Symmetric 5-D HND: swap dims 2 and 3."""
    t = torch.zeros(4, 2, 16, 8, 128, dtype=torch.bfloat16)
    out = _prepare_flashinfer_paged_kv_cache(t, (0, 1, 3, 2, 4))
    assert isinstance(out, torch.Tensor)
    # After permute (0,1,3,2,4): [4, 2, 8, 16, 128]
    assert tuple(out.shape) == (4, 2, 8, 16, 128)
    print(f"  test_prepare_sym_HND: shape={tuple(out.shape)} OK")


def test_prepare_asym_4d_NHD():
    """Asymmetric tuple, NHD: identity 4-D permute on each side."""
    k = torch.zeros(4, 16, 8, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 16, 8, 128, dtype=torch.float8_e4m3fn)
    out = _prepare_flashinfer_paged_kv_cache((k, v), (0, 1, 2, 3, 4))
    assert isinstance(out, tuple) and len(out) == 2
    k_out, v_out = out
    assert k_out.shape == k.shape
    assert v_out.shape == v.shape
    assert k_out.dtype == torch.bfloat16
    assert v_out.dtype == torch.float8_e4m3fn
    print(f"  test_prepare_asym_NHD: K/V shapes preserved OK")


def test_prepare_asym_4d_HND():
    """Asymmetric tuple, HND: swap block_size and num_kv_heads dims."""
    k = torch.zeros(4, 16, 8, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 16, 8, 128, dtype=torch.float8_e4m3fn)
    out = _prepare_flashinfer_paged_kv_cache((k, v), (0, 1, 3, 2, 4))
    assert isinstance(out, tuple) and len(out) == 2
    k_out, v_out = out
    # 4-D HND permute is (0, 2, 1, 3): [4, 8, 16, 128]
    assert tuple(k_out.shape) == (4, 8, 16, 128)
    assert tuple(v_out.shape) == (4, 8, 16, 128)
    assert k_out.dtype == torch.bfloat16
    assert v_out.dtype == torch.float8_e4m3fn
    print(f"  test_prepare_asym_HND: K/V dims swapped OK")


def test_prepare_asym_views_not_copies():
    """Permute returns a view, not a copy.  K/V must still alias the
    same storage as the input tensors so writes through the cache
    writer are visible to the kernel."""
    k = torch.zeros(4, 16, 8, 128, dtype=torch.bfloat16)
    v = torch.zeros(4, 16, 8, 128, dtype=torch.float8_e4m3fn)
    out = _prepare_flashinfer_paged_kv_cache((k, v), (0, 1, 3, 2, 4))
    k_out, v_out = out
    assert k_out.data_ptr() == k.data_ptr()
    assert v_out.data_ptr() == v.data_ptr()
    print("  test_prepare_views_not_copies: OK")


def main():
    if not hasattr(torch, "float8_e4m3fn"):
        print("torch.float8_e4m3fn not available — skipping")
        return 0
    print("Step 1 standalone gate — FlashInfer forward() tuple helpers")
    print()
    test_is_asym_sniff_recognises_tuple()
    test_is_asym_sniff_rejects_tensor()
    test_is_asym_sniff_rejects_3_tuple()
    test_is_asym_sniff_rejects_tuple_of_non_tensors()
    test_derive_4d_from_5d_NHD()
    test_derive_4d_from_5d_HND()
    test_prepare_symmetric_5d_NHD()
    test_prepare_symmetric_5d_HND()
    test_prepare_asym_4d_NHD()
    test_prepare_asym_4d_HND()
    test_prepare_asym_views_not_copies()
    print()
    print("=== ALL STEP 1 HELPER CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
