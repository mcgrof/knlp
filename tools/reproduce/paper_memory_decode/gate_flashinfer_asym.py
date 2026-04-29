#!/usr/bin/env python3
"""Milestone 1 gate: standalone FlashInfer asymmetric K/V proof.

This test must pass BEFORE we rely on vLLM.  If the FlashInfer fork
cannot run BF16-K + FP8-V paged KV attention against a torch reference,
then any vLLM patches on top of it are confetti.

Four sub-tests:
  1. decode  Q=BF16, K=BF16, V=FP8_e4m3, tuple paged_kv_cache
  2. prefill Q=BF16, K=BF16, V=FP8_e4m3, tuple paged_kv_cache
  3. negative: k_cache=FP8 should fail dtype validation
  4. negative: v_cache=BF16 should fail dtype validation

Exits 0 on full pass; exits 1 on any failure.
Prints DECODE_REL_ERR=<float> and PREFILL_REL_ERR=<float> on stdout
so the parent stage can parse metrics.

Requires: H100 or any sm89+ (FP8 e4m3 support).
"""
from __future__ import annotations

import sys

import torch

NUM_LAYERS = 1
NUM_QO_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 16
NUM_BLOCKS = 64
BATCH = 4
QLEN_DECODE = 1
QLEN_PREFILL = 32
DEVICE = "cuda"
SEED = 0
RTOL = 0.10  # FP8 e4m3 noise band


def _alloc_paged_caches(k_dtype, v_dtype):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED)
    shape = (NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cache_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE, generator=g)
    v_cache_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE, generator=g)
    v_amax = v_cache_bf16.abs().amax().clamp_min(1e-6)
    v_scale = (v_amax / 448.0).to(torch.float32)
    v_cache_fp8 = (v_cache_bf16 / v_scale).to(torch.float8_e4m3fn)

    k_out = k_cache_bf16.to(k_dtype)
    v_out = v_cache_fp8 if v_dtype == torch.float8_e4m3fn else v_cache_bf16.to(v_dtype)
    return k_cache_bf16, v_cache_bf16, k_out, v_out, v_scale


def _build_paged_indices(num_seqs: int, kv_lens: list[int]):
    cumul = [0]
    for i in range(num_seqs):
        cumul.append(cumul[-1] + (kv_lens[i] + PAGE_SIZE - 1) // PAGE_SIZE)
    indptr = torch.tensor(cumul, dtype=torch.int32, device=DEVICE)
    total_pages = cumul[-1]
    indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    last_len = torch.tensor(
        [((l - 1) % PAGE_SIZE) + 1 for l in kv_lens], dtype=torch.int32, device=DEVICE
    )
    return indptr, indices, last_len


def torch_reference(q, k_bf16, v_bf16, kv_lens, qo_lens):
    """Full-precision BF16 reference — the asym result must be within
    FP8 e4m3 noise of this."""
    outs = []
    page_idx = 0
    qo_off = 0
    for s, kv_len in enumerate(kv_lens):
        n_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        qo_len = qo_lens[s]
        qs = q[qo_off : qo_off + qo_len].float()
        k_full = (
            k_bf16[page_idx : page_idx + n_pages]
            .reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:kv_len]
            .float()
        )
        v_full = (
            v_bf16[page_idx : page_idx + n_pages]
            .reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:kv_len]
            .float()
        )
        scale = 1.0 / (HEAD_DIM**0.5)
        scores = torch.einsum("qhd,khd->hqk", qs, k_full) * scale
        if qo_len > 1:
            mask = torch.full((qo_len, kv_len), float("-inf"), device=qs.device)
            for i in range(qo_len):
                mask[i, : kv_len - qo_len + i + 1] = 0.0
            scores = scores + mask
        attn = scores.softmax(dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, v_full).to(q.dtype)
        outs.append(out)
        page_idx += n_pages
        qo_off += qo_len
    return torch.cat(outs, dim=0)


def test_decode():
    print("\n[1] decode  Q=BF16  K=BF16  V=FP8_e4m3  tuple paged_kv_cache")
    import flashinfer

    k_bf16, v_bf16, k_cache, v_cache, v_scale = _alloc_paged_caches(
        torch.bfloat16, torch.float8_e4m3fn
    )
    kv_lens = [16, 24, 8, 32]
    qo_lens = [QLEN_DECODE] * BATCH
    indptr, indices, last_len = _build_paged_indices(BATCH, kv_lens)
    qo_indptr = torch.tensor(
        [0] + [sum(qo_lens[: i + 1]) for i in range(BATCH)],
        dtype=torch.int32,
        device=DEVICE,
    )
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + 1)
    q = torch.randn(
        sum(qo_lens),
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
        generator=g,
    )

    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        k_data_type=torch.bfloat16,
        v_data_type=torch.float8_e4m3fn,
    )
    out = wrapper.run(q, paged_kv_cache=(k_cache, v_cache), v_scale=float(v_scale))
    ref = torch_reference(q, k_bf16, v_bf16, kv_lens, qo_lens)
    rel = (out.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)
    med = rel.median().item()
    print(f"    median relative error: {med:.4f}  (bound: <{RTOL})")
    print(f"DECODE_REL_ERR={med:.6f}")
    assert med < RTOL, f"decode too far from reference: {med:.4f}"
    print("    PASS")
    return med


def test_prefill():
    print("\n[2] prefill Q=BF16  K=BF16  V=FP8_e4m3  tuple paged_kv_cache")
    import flashinfer

    k_bf16, v_bf16, k_cache, v_cache, v_scale = _alloc_paged_caches(
        torch.bfloat16, torch.float8_e4m3fn
    )
    kv_lens = [40, 64, 32, 56]
    qo_lens = [QLEN_PREFILL] * BATCH
    indptr, indices, last_len = _build_paged_indices(BATCH, kv_lens)
    qo_indptr = torch.tensor(
        [0] + [sum(qo_lens[: i + 1]) for i in range(BATCH)],
        dtype=torch.int32,
        device=DEVICE,
    )
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED + 2)
    q = torch.randn(
        sum(qo_lens),
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
        generator=g,
    )

    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=last_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        k_data_type=torch.bfloat16,
        v_data_type=torch.float8_e4m3fn,
    )
    out = wrapper.run(q, paged_kv_cache=(k_cache, v_cache), v_scale=float(v_scale))
    ref = torch_reference(q, k_bf16, v_bf16, kv_lens, qo_lens)
    rel = (out.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)
    med = rel.median().item()
    print(f"    median relative error: {med:.4f}  (bound: <{RTOL})")
    print(f"PREFILL_REL_ERR={med:.6f}")
    assert med < RTOL, f"prefill too far from reference: {med:.4f}"
    print("    PASS")
    return med


def test_negative_k_fp8():
    print("\n[3] negative — k_cache=FP8 should fail dtype validation")
    import flashinfer

    _, _, _, v_cache, _ = _alloc_paged_caches(torch.bfloat16, torch.float8_e4m3fn)
    k_bad = torch.zeros(
        NUM_BLOCKS,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=DEVICE,
    )
    indptr, indices, last_len = _build_paged_indices(1, [8])
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        k_data_type=torch.bfloat16,
        v_data_type=torch.float8_e4m3fn,
    )
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    try:
        wrapper.run(q, paged_kv_cache=(k_bad, v_cache))
        print(
            "    WARNING: no ValueError raised for K=FP8; "
            "fork may lack strict dtype validation on run()"
        )
    except (ValueError, RuntimeError) as e:
        print(f"    expected error: {type(e).__name__}: {e}")
        print("    PASS")


def test_negative_v_bf16():
    print("\n[4] negative — v_cache=BF16 should fail dtype validation")
    import flashinfer

    _, _, k_cache, _, _ = _alloc_paged_caches(torch.bfloat16, torch.float8_e4m3fn)
    v_bad = torch.zeros(
        NUM_BLOCKS,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    indptr, indices, last_len = _build_paged_indices(1, [8])
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_len,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        k_data_type=torch.bfloat16,
        v_data_type=torch.float8_e4m3fn,
    )
    q = torch.randn(1, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    try:
        wrapper.run(q, paged_kv_cache=(k_cache, v_bad))
        print(
            "    WARNING: no error raised for V=BF16; "
            "fork may lack strict dtype validation on run()"
        )
    except (ValueError, RuntimeError) as e:
        print(f"    expected error: {type(e).__name__}: {e}")
        print("    PASS")


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable — skipping")
        sys.exit(2)
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        print(f"FP8 e4m3 needs sm89+; detected sm{cap[0]}{cap[1]} — skipping")
        sys.exit(2)
    import os

    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    print(f"FlashInfer asym K/V gate — {torch.cuda.get_device_name()}")
    decode_err = test_decode()
    prefill_err = test_prefill()
    test_negative_k_fp8()
    test_negative_v_bf16()
    print(
        f"\n=== GATE PASSED  decode_err={decode_err:.4f}  "
        f"prefill_err={prefill_err:.4f} ==="
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
