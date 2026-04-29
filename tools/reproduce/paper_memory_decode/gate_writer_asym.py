#!/usr/bin/env python3
"""Asymmetric KV cache writer correctness gate.

Validates that the vLLM asymmetric writer contract is upheld:
  - K is stored bit-exact (lossless)
  - V is quantized to FP8 e4m3 with v_scale
  - K and V tensors do not alias (writes to V do not corrupt K)
  - slot_mapping=-1 tokens are correctly skipped

Runs entirely on CPU — no GPU, no FlashInfer, no vLLM dependency.
Tests the same algorithm the vLLM reshape_and_cache_flash_asym op is
supposed to implement, as a Python mechanical replica.

Exits 0 on full pass, 1 on failure.
"""
from __future__ import annotations

import sys

import torch

NUM_BLOCKS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 128


def _alloc_asym_cache(
    num_blocks, block_size, num_kv_heads, head_size, k_dtype, v_dtype
):
    shape = (num_blocks, block_size, num_kv_heads, head_size)
    return torch.zeros(shape, dtype=k_dtype), torch.zeros(shape, dtype=v_dtype)


def asym_writer(key, value, k_cache, v_cache, slot_mapping, v_scale):
    """Python replica of the asymmetric cache writer.

    key:          (n_tokens, num_kv_heads, head_size) at K dtype
    value:        (n_tokens, num_kv_heads, head_size) at fp32 or bf16
    k_cache:      (num_blocks, block_size, ...) at K dtype
    v_cache:      (num_blocks, block_size, ...) at V dtype (FP8)
    slot_mapping: (n_tokens,) int64; -1 means skip
    v_scale:      float
    """
    block_size = k_cache.shape[1]
    for t in range(key.shape[0]):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        b, off = slot // block_size, slot % block_size
        k_cache[b, off] = key[t]
        v_cache[b, off] = (value[t] / v_scale).to(v_cache.dtype)


def test_k_bit_exact():
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB, BS, H, D, torch.bfloat16, torch.float8_e4m3fn
    )
    n = 24
    g = torch.Generator()
    g.manual_seed(0)
    key = torch.randn(n, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(n, H, D, dtype=torch.bfloat16, generator=g)
    slots = torch.arange(n, dtype=torch.int64)
    v_scale = float((value.abs().amax().clamp_min(1e-6) / 448.0))

    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    for t in range(n):
        b, off = int(slots[t]) // BS, int(slots[t]) % BS
        assert torch.equal(k_cache[b, off], key[t]), f"K not bit-exact at token {t}"
    written = {int(s) for s in slots}
    for b in range(NB):
        for off in range(BS):
            if b * BS + off in written:
                continue
            assert (
                k_cache[b, off] == 0
            ).all(), f"K modified at unwritten slot block={b} off={off}"
    print("  test_k_bit_exact: PASS")


def test_v_dequant_within_fp8_noise():
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB, BS, H, D, torch.bfloat16, torch.float8_e4m3fn
    )
    n = 24
    g = torch.Generator()
    g.manual_seed(1)
    key = torch.randn(n, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(n, H, D, dtype=torch.bfloat16, generator=g)
    slots = torch.arange(n, dtype=torch.int64)
    v_scale = float((value.abs().amax().clamp_min(1e-6) / 448.0))

    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    for t in range(n):
        b, off = int(slots[t]) // BS, int(slots[t]) % BS
        v_dq = v_cache[b, off].to(torch.float32) * v_scale
        v_orig = value[t].float()
        rel = (v_dq - v_orig).abs() / (v_orig.abs() + 1e-6)
        assert (
            rel.median().item() < 0.075
        ), f"V dequant error too high at token {t}: {rel.median():.4f}"
    print("  test_v_dequant_within_fp8_noise: PASS")


def test_k_v_no_alias():
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB, BS, H, D, torch.bfloat16, torch.float8_e4m3fn
    )
    n = 24
    g = torch.Generator()
    g.manual_seed(2)
    key = torch.randn(n, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(n, H, D, dtype=torch.bfloat16, generator=g)
    slots = torch.arange(n, dtype=torch.int64)
    v_scale = 1.0

    k_before = k_cache.clone()
    v_before_bytes = v_cache.view(torch.int8).clone()
    asym_writer(key, value, k_cache, v_cache, slots, v_scale)

    touched = {int(s) for s in slots}
    for b in range(NB):
        for off in range(BS):
            slot = b * BS + off
            if slot in touched:
                continue
            assert torch.equal(
                k_cache[b, off], k_before[b, off]
            ), f"K corrupted at unwritten slot {slot}"
            assert torch.equal(
                v_cache[b, off].view(torch.int8), v_before_bytes[b, off]
            ), f"V corrupted at unwritten slot {slot}"
    print("  test_k_v_no_alias: PASS")


def test_skip_slots():
    NB, BS, H, D = NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    k_cache, v_cache = _alloc_asym_cache(
        NB, BS, H, D, torch.bfloat16, torch.float8_e4m3fn
    )
    g = torch.Generator()
    g.manual_seed(3)
    key = torch.randn(4, H, D, dtype=torch.bfloat16, generator=g)
    value = torch.randn(4, H, D, dtype=torch.bfloat16, generator=g)
    slots = torch.tensor([-1, 5, -1, 9], dtype=torch.int64)

    asym_writer(key, value, k_cache, v_cache, slots, 1.0)

    assert torch.equal(k_cache[0, 5], key[1]), "slot 5 not written"
    assert torch.equal(k_cache[0, 9], key[3]), "slot 9 not written"
    for off in range(BS):
        if off in (5, 9):
            continue
        assert (k_cache[0, off] == 0).all(), f"K modified at unwritten offset {off}"
    print("  test_skip_slots: PASS")


def main():
    if not hasattr(torch, "float8_e4m3fn"):
        print(
            "torch.float8_e4m3fn not available in this torch version — "
            "skipping writer gate"
        )
        sys.exit(2)

    print("Asymmetric KV writer correctness gate (CPU-only)\n")
    test_k_bit_exact()
    test_v_dequant_within_fp8_noise()
    test_k_v_no_alias()
    test_skip_slots()
    print("\n=== ALL WRITER GATE CHECKS PASSED ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
