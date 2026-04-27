#!/usr/bin/env python3
"""Milestone 3 verification: asymmetric KV cache *writer* correctness.

M2 proved the allocator splits the raw buffer into a typed
(k_cache, v_cache) tuple with the right dtypes and byte layout.
That is necessary but not sufficient.  M3 has to prove the
*writer* preserves K losslessly and quantizes only V.

This is a standalone test of the write contract.  It does not run
a model or FlashInfer attention.  It exercises the same algorithm
the asym writer is supposed to implement: given fresh BF16 K and
V tensors plus a slot mapping, populate a tuple cache so that:

  - k_cache stores K bit-exact at the requested slots
  - v_cache stores V quantized to FP8 e4m3 with v_scale
  - K bytes are not overwritten when writing V (or vice versa)
  - dequantized V at the slots matches the original V within
    FP8 e4m3 noise

The fork's commit 84d8633a4 ("flashinfer: split cache write for
asymmetric V FP8") implements this on the GPU side.  This test is
the contract gate; the GPU writer is "implemented" but not yet
"verified."  Software has made a career out of humiliating that
distinction.

Runs on CPU.  No GPU, no FlashInfer, no vLLM.
"""

import sys

import torch


# Modest grid; enough to exercise per-slot mapping without
# turning the test into a benchmark.
NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128


def _alloc_asym_cache(
    num_blocks,
    block_size,
    num_kv_heads,
    head_size,
    k_dtype,
    v_dtype,
):
    """Allocate the same (k_cache, v_cache) tuple shape that the
    M2-verified allocator produces.  Each tensor is its own
    storage so writes don't alias."""
    shape = (num_blocks, block_size, num_kv_heads, head_size)
    k_cache = torch.zeros(shape, dtype=k_dtype)
    v_cache = torch.zeros(shape, dtype=v_dtype)
    return k_cache, v_cache


def asym_writer(
    key,
    value,
    k_cache,
    v_cache,
    slot_mapping,
    v_scale,
):
    """Mechanical replica of what the asymmetric cache writer
    should do.  K is stored bit-exact, V is quantized via v_scale.

    key:           (n_tokens, num_kv_heads, head_size) at K dtype
    value:         (n_tokens, num_kv_heads, head_size) at K dtype
    k_cache:       (num_blocks, block_size, num_kv_heads, head_size)
                   at K dtype
    v_cache:       (num_blocks, block_size, num_kv_heads, head_size)
                   at V dtype (FP8)
    slot_mapping:  (n_tokens,) int64; -1 for tokens to skip
    v_scale:       float; v / v_scale fits FP8 e4m3 range
    """
    n_tokens = key.shape[0]
    block_size = k_cache.shape[1]
    for t in range(n_tokens):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        block = slot // block_size
        offset = slot % block_size
        # K: bit-exact write at K dtype
        k_cache[block, offset] = key[t]
        # V: quantize to FP8 with scale
        v_cache[block, offset] = (value[t] / v_scale).to(v_cache.dtype)


def test_writer_preserves_k_bit_exact():
    """K must be bit-exact at the slots we wrote, and zero
    elsewhere.  Catches a writer that accidentally quantizes K."""
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB,
        BS,
        H,
        D,
        torch.bfloat16,
        torch.float8_e4m3fn,
    )

    n_tokens = 24
    g = torch.Generator()
    g.manual_seed(0)
    key = torch.randn(n_tokens, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(n_tokens, H, D, dtype=torch.bfloat16, generator=g)

    # Map each token to a unique slot in the first 2 blocks
    slots = torch.arange(n_tokens, dtype=torch.int64)
    v_amax = value.abs().amax().clamp_min(1e-6)
    v_scale = (v_amax / 448.0).item()

    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    # K at each slot should equal key bit-exact
    for t in range(n_tokens):
        slot = int(slots[t])
        block, off = slot // BS, slot % BS
        assert torch.equal(
            k_cache[block, off], key[t]
        ), f"K not bit-exact at token {t}, slot {slot}"

    # K outside written slots should be untouched (still zero)
    written_blocks = set(int(s) // BS for s in slots)
    for b in range(NB):
        for off in range(BS):
            if b * BS + off in slots:
                continue
            assert (
                k_cache[b, off] == 0
            ).all(), f"K modified at unwritten slot block={b} off={off}"

    print("  test_writer_preserves_k_bit_exact: OK")


def test_writer_v_dequantizes_within_fp8_noise():
    """V at each slot, dequantized via v_scale, must match the
    original V within FP8 e4m3 noise.  Catches a writer that
    skips quantization or uses the wrong scale."""
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB,
        BS,
        H,
        D,
        torch.bfloat16,
        torch.float8_e4m3fn,
    )

    n_tokens = 24
    g = torch.Generator()
    g.manual_seed(1)
    key = torch.randn(n_tokens, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(n_tokens, H, D, dtype=torch.bfloat16, generator=g)
    slots = torch.arange(n_tokens, dtype=torch.int64)
    v_amax = value.abs().amax().clamp_min(1e-6)
    v_scale = (v_amax / 448.0).item()

    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    # Dequantize V and compare
    for t in range(n_tokens):
        slot = int(slots[t])
        block, off = slot // BS, slot % BS
        v_dq = v_cache[block, off].to(torch.float32) * v_scale
        v_orig = value[t].to(torch.float32)
        rel = (v_dq - v_orig).abs() / (v_orig.abs() + 1e-6)
        assert (
            rel.median().item() < 0.075
        ), f"V dequant relerr too high at token {t}: {rel.median()}"

    print("  test_writer_v_dequantizes_within_fp8_noise: OK")


def test_writer_k_and_v_do_not_alias():
    """Writing V must not corrupt K bytes, and vice versa.  Crucial
    for the asym writer because the cache backing storage might be
    one raw allocation sliced into two views."""
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB,
        BS,
        H,
        D,
        torch.bfloat16,
        torch.float8_e4m3fn,
    )

    n_tokens = 24
    g = torch.Generator()
    g.manual_seed(2)
    key = torch.randn(n_tokens, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(n_tokens, H, D, dtype=torch.bfloat16, generator=g)
    slots = torch.arange(n_tokens, dtype=torch.int64)
    v_amax = value.abs().amax().clamp_min(1e-6)
    v_scale = (v_amax / 448.0).item()

    # Snapshot K and V cache before write
    k_before = k_cache.clone()
    v_before = v_cache.clone()
    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    # Touched slots
    touched = {int(s) for s in slots}
    # K and V regions outside the slots must be unchanged.  We
    # iterate explicitly to avoid any tensor-view ambiguity.
    for b in range(NB):
        for off in range(BS):
            slot = b * BS + off
            if slot in touched:
                continue
            assert torch.equal(
                k_cache[b, off], k_before[b, off]
            ), f"K corrupted at unwritten slot {slot}"
            # FP8 equality check via int8 byte view
            assert torch.equal(
                v_cache[b, off].view(torch.int8),
                v_before[b, off].view(torch.int8),
            ), f"V corrupted at unwritten slot {slot}"

    print("  test_writer_k_and_v_do_not_alias: OK")


def test_writer_handles_skip_slots():
    """slot_mapping == -1 must skip the token.  Catches off-by-one
    on the masking logic."""
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB,
        BS,
        H,
        D,
        torch.bfloat16,
        torch.float8_e4m3fn,
    )

    g = torch.Generator()
    g.manual_seed(3)
    key = torch.randn(4, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(4, H, D, dtype=torch.bfloat16, generator=g)
    # First and third tokens are masked off; second and fourth are
    # written to slots 5 and 9.
    slots = torch.tensor([-1, 5, -1, 9], dtype=torch.int64)
    v_scale = 1.0

    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    # Slots 5 and 9 must have the right K
    assert torch.equal(k_cache[0, 5], key[1])
    assert torch.equal(k_cache[0, 9], key[3])
    # All other slots untouched (still zero)
    for off in range(BS):
        if off in (5, 9):
            continue
        assert (k_cache[0, off] == 0).all(), f"K modified at unwritten slot {off}"
    print("  test_writer_handles_skip_slots: OK")


def main():
    if not hasattr(torch, "float8_e4m3fn"):
        print("torch.float8_e4m3fn not available — skipping")
        return 0
    print("Milestone 3 standalone gate — asym KV writer correctness")
    print()
    test_writer_preserves_k_bit_exact()
    test_writer_v_dequantizes_within_fp8_noise()
    test_writer_k_and_v_do_not_alias()
    test_writer_handles_skip_slots()
    print()
    print("=== ALL M3 STANDALONE WRITER CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
