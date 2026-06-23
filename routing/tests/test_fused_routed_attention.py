# SPDX-License-Identifier: GPL-2.0
"""Correctness tests for fused routed attention kernel.

Tests:
1. Numerical agreement with reference implementation across configs
2. GQA support (group_size > 1)
3. Variable block counts per head
4. Edge cases: single block, all blocks
5. Gradient-free (decode-only, no backward needed)
"""

import pytest
import torch

from routing.fused_routed_attention import (
    fused_routed_decode,
    reference_routed_decode,
    select_top_k_blocks,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_test_data(
    batch=1,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    n_total_blocks=64,
    block_size=16,
    max_k=4,
    dtype=torch.float16,
):
    """Create synthetic test data."""
    q = torch.randn(batch, n_heads, head_dim, device=DEVICE, dtype=dtype)
    k_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    # Random block selection per head
    block_tables = torch.zeros(batch, n_kv_heads, max_k, dtype=torch.int64, device=DEVICE)
    block_counts = torch.zeros(batch, n_kv_heads, dtype=torch.int32, device=DEVICE)

    for b in range(batch):
        for h in range(n_kv_heads):
            n_sel = max_k
            perm = torch.randperm(n_total_blocks, device=DEVICE)[:n_sel]
            block_tables[b, h, :n_sel] = perm
            block_counts[b, h] = n_sel

    return q, k_cache, v_cache, block_tables, block_counts


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("n_kv_heads,n_heads", [(8, 32), (4, 32), (32, 32)])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("max_k", [1, 2, 4, 8])
def test_correctness_vs_reference(head_dim, n_kv_heads, n_heads, block_size, max_k):
    """Fused kernel output matches reference within fp16 tolerance."""
    q, k_cache, v_cache, block_tables, block_counts = make_test_data(
        batch=1, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        n_total_blocks=64, block_size=block_size, max_k=max_k,
    )

    out_fused = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    out_ref = reference_routed_decode(q, k_cache, v_cache, block_tables, block_counts)

    # fp16 tolerance: ~1e-2 relative, ~1e-3 absolute
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=2e-2, rtol=5e-2)


def test_batch_dimension():
    """Kernel handles batch > 1."""
    q, k_cache, v_cache, block_tables, block_counts = make_test_data(batch=4, max_k=4)
    out_fused = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    out_ref = reference_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=2e-2, rtol=5e-2)


def test_variable_block_counts():
    """Different heads can have different numbers of selected blocks."""
    batch, n_heads, n_kv_heads, head_dim = 1, 32, 8, 128
    n_total_blocks, block_size, max_k = 64, 16, 8

    q = torch.randn(batch, n_heads, head_dim, device=DEVICE, dtype=torch.float16)
    k_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=DEVICE, dtype=torch.float16)
    v_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=DEVICE, dtype=torch.float16)

    block_tables = torch.zeros(batch, n_kv_heads, max_k, dtype=torch.int64, device=DEVICE)
    block_counts = torch.zeros(batch, n_kv_heads, dtype=torch.int32, device=DEVICE)

    # Give each head a different count
    for h in range(n_kv_heads):
        n_sel = h + 1  # head 0 gets 1 block, head 7 gets 8
        perm = torch.randperm(n_total_blocks, device=DEVICE)[:n_sel]
        block_tables[0, h, :n_sel] = perm
        block_counts[0, h] = n_sel

    out_fused = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    out_ref = reference_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=2e-2, rtol=5e-2)


def test_single_block():
    """K=1: each head attends to exactly one block."""
    q, k_cache, v_cache, block_tables, block_counts = make_test_data(max_k=1)
    out_fused = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    out_ref = reference_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=2e-2, rtol=5e-2)


def test_all_blocks():
    """K=all: equivalent to dense attention over all blocks."""
    n_total_blocks = 16
    q, k_cache, v_cache, block_tables, block_counts = make_test_data(
        n_total_blocks=n_total_blocks, max_k=n_total_blocks,
    )
    # All heads see all blocks
    for h in range(block_tables.shape[1]):
        block_tables[0, h, :] = torch.arange(n_total_blocks, device=DEVICE)
        block_counts[0, h] = n_total_blocks

    out_fused = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    out_ref = reference_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=2e-2, rtol=5e-2)


def test_select_top_k_blocks():
    """Block selection from routing prior."""
    n_layers, n_kv_heads, n_blocks = 32, 8, 256
    routing_prior = torch.randn(n_layers, n_kv_heads, n_blocks)

    for k in [1, 2, 4, 8, 16]:
        tables, counts = select_top_k_blocks(routing_prior, k)
        assert tables.shape == (n_layers, n_kv_heads, min(k, n_blocks))
        assert (counts == min(k, n_blocks)).all()

        # Verify selections match torch.topk
        for l in range(n_layers):
            for h in range(n_kv_heads):
                expected = torch.topk(routing_prior[l, h], min(k, n_blocks)).indices
                assert set(tables[l, h].tolist()) == set(expected.tolist())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtypes(dtype):
    """Kernel handles fp16 and bf16."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported")

    q, k_cache, v_cache, block_tables, block_counts = make_test_data(dtype=dtype)
    out_fused = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    out_ref = reference_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
    torch.testing.assert_close(out_fused.float(), out_ref.float(), atol=5e-2, rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
