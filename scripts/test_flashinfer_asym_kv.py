#!/usr/bin/env python3
"""Milestone 1 gate: standalone FlashInfer asymmetric K/V proof.

Per the design review, this test must pass BEFORE we touch vLLM.  If
the FlashInfer fork cannot honestly run BF16-K + FP8-V paged KV
attention against a torch reference, then any vLLM patches sitting on
top of it are confetti.

Five sub-tests:
  1. decode  Q=BF16, K=BF16, V=FP8_e4m3, tuple paged_kv_cache
  2. prefill Q=BF16, K=BF16, V=FP8_e4m3, tuple paged_kv_cache
  3. negative — k_cache intentionally FP8 should fail validation
  4. negative — v_cache intentionally BF16 should fail validation
  5. compare against a torch reference using BF16 K and dequantized V,
     within FP8 e4m3 tolerance

Run on H100 (or any sm89+ GPU with FP8 e4m3 support):
    python3 test_flashinfer_asym_kv.py
"""
import sys

import torch


# Modest config — fast to run, exercises the asym path realistically.
NUM_LAYERS = 1  # one layer is enough to gate the kernel
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
RTOL = 0.10  # FP8 e4m3 noise band; assert median is well under


def _alloc_paged_caches(k_dtype, v_dtype):
    g = torch.Generator(device=DEVICE)
    g.manual_seed(SEED)
    k_shape = (NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    v_shape = (NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k_cache = torch.randn(k_shape, dtype=torch.bfloat16, device=DEVICE, generator=g)
    v_cache_ref = torch.randn(v_shape, dtype=torch.bfloat16, device=DEVICE, generator=g)
    # Quantize V to FP8 e4m3 with a per-tensor scale.
    v_amax = v_cache_ref.abs().amax().clamp_min(1e-6)
    v_scale = (v_amax / 448.0).to(torch.float32)  # FP8 e4m3 max ~ 448
    v_cache_fp8 = (v_cache_ref / v_scale).to(torch.float8_e4m3fn)
    # Final K and V tensors at the requested dtypes.
    k_cache_out = k_cache.to(k_dtype)
    if v_dtype == torch.float8_e4m3fn:
        v_cache_out = v_cache_fp8
    else:
        v_cache_out = v_cache_ref.to(v_dtype)
    return k_cache, v_cache_ref, k_cache_out, v_cache_out, v_scale


def _build_paged_indices(num_seqs, kv_lens):
    paged_kv_indptr = torch.tensor(
        [0]
        + [
            sum(((l + PAGE_SIZE - 1) // PAGE_SIZE) for l in kv_lens[: i + 1])
            for i in range(num_seqs)
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    total_pages = paged_kv_indptr[-1].item()
    paged_kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    paged_kv_last_page_len = torch.tensor(
        [((l - 1) % PAGE_SIZE) + 1 for l in kv_lens],
        dtype=torch.int32,
        device=DEVICE,
    )
    return paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len


def torch_reference_attention(q, k_cache_ref, v_cache_ref, kv_lens, qo_lens):
    """Reference computation in BF16 against the un-quantized V cache.
    The asymmetric kernel result should be close to this within FP8 noise."""
    outs = []
    page_idx = 0
    qo_off = 0
    for s, kv_len in enumerate(kv_lens):
        n_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        qo_len = qo_lens[s]
        qs = q[qo_off : qo_off + qo_len].float()  # [qo_len, H_q, D]
        # Gather K and V pages for this seq, then truncate to kv_len tokens.
        k_pages = k_cache_ref[page_idx : page_idx + n_pages]  # [P, page, H_kv, D]
        v_pages = v_cache_ref[page_idx : page_idx + n_pages]
        k_full = k_pages.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:kv_len].float()
        v_full = v_pages.reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:kv_len].float()
        # For GQA with H_q == H_kv (our config), no head broadcast needed.
        scale = 1.0 / (HEAD_DIM**0.5)
        scores = torch.einsum("qhd,khd->hqk", qs, k_full) * scale
        # Causal masking for prefill: q at position i (counting from
        # kv_len - qo_len) attends to tokens 0..i.
        if qo_len > 1:
            mask = torch.full((qo_len, kv_len), float("-inf"), device=qs.device)
            for i in range(qo_len):
                # The i-th query is at kv-position (kv_len - qo_len + i).
                mask[i, : kv_len - qo_len + i + 1] = 0.0
            scores = scores + mask  # broadcast over heads
        attn = scores.softmax(dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, v_full).to(q.dtype)
        outs.append(out)
        page_idx += n_pages
        qo_off += qo_len
    return torch.cat(outs, dim=0)


def run_test_decode():
    print("\n[1] decode  Q=BF16  K=BF16  V=FP8_e4m3  tuple paged_kv_cache")
    import flashinfer

    k_cache_ref, v_cache_ref, k_cache, v_cache, v_scale = _alloc_paged_caches(
        torch.bfloat16,
        torch.float8_e4m3fn,
    )
    kv_lens = [16, 24, 8, 32]  # variable per seq
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

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    decode.plan(
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
    out = decode.run(
        q,
        paged_kv_cache=(k_cache, v_cache),
        v_scale=v_scale.item() if hasattr(v_scale, "item") else v_scale,
    )
    ref = torch_reference_attention(q, k_cache_ref, v_cache_ref, kv_lens, qo_lens)
    rel = (out.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)
    med = rel.median().item()
    print(
        f"    median relative error vs BF16 reference: {med:.4f}" f"  (bound: <{RTOL})"
    )
    assert med < RTOL, f"decode asym output too far from reference: {med}"
    print("    OK")


def run_test_prefill():
    print("\n[2] prefill Q=BF16  K=BF16  V=FP8_e4m3  tuple paged_kv_cache")
    import flashinfer

    k_cache_ref, v_cache_ref, k_cache, v_cache, v_scale = _alloc_paged_caches(
        torch.bfloat16,
        torch.float8_e4m3fn,
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

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    prefill.plan(
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
    out = prefill.run(
        q,
        paged_kv_cache=(k_cache, v_cache),
        v_scale=v_scale.item() if hasattr(v_scale, "item") else v_scale,
    )
    ref = torch_reference_attention(q, k_cache_ref, v_cache_ref, kv_lens, qo_lens)
    rel = (out.float() - ref.float()).abs() / (ref.float().abs() + 1e-6)
    med = rel.median().item()
    print(
        f"    median relative error vs BF16 reference: {med:.4f}" f"  (bound: <{RTOL})"
    )
    assert med < RTOL, f"prefill asym output too far from reference: {med}"
    print("    OK")


def run_test_negative_k_fp8():
    print("\n[3] negative — k_cache=FP8 should fail validation")
    import flashinfer

    _, _, _, v_cache, _ = _alloc_paged_caches(torch.bfloat16, torch.float8_e4m3fn)
    # K cache built at FP8 instead of BF16.
    k_cache = torch.zeros(
        NUM_BLOCKS,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device=DEVICE,
    )

    workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    indptr, indices, last_len = _build_paged_indices([8])
    decode.plan(
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
        decode.run(q, paged_kv_cache=(k_cache, v_cache))
    except ValueError as e:
        print(f"    expected ValueError: {e}")
        print("    OK")
        return
    raise AssertionError("expected ValueError on K=FP8 vs cached K=BF16")


def run_test_negative_v_bf16():
    print("\n[4] negative — v_cache=BF16 should fail validation")
    import flashinfer

    k_cache_ref, _, k_cache, _, _ = _alloc_paged_caches(
        torch.bfloat16,
        torch.float8_e4m3fn,
    )
    # V at BF16 instead of FP8.
    v_cache = torch.zeros(
        NUM_BLOCKS,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    indptr, indices, last_len = _build_paged_indices([8])
    decode.plan(
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
        decode.run(q, paged_kv_cache=(k_cache, v_cache))
    except ValueError as e:
        print(f"    expected ValueError: {e}")
        print("    OK")
        return
    raise AssertionError("expected ValueError on V=BF16 vs cached V=FP8")


def main():
    if not torch.cuda.is_available():
        print("CUDA required for FP8 e4m3 path")
        sys.exit(1)
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        print(f"FP8 e4m3 needs sm89+; got {cap}")
        sys.exit(1)

    print(f"FlashInfer asym K/V proof on {torch.cuda.get_device_name()}")
    run_test_decode()
    run_test_prefill()
    run_test_negative_k_fp8()
    run_test_negative_v_bf16()
    print("\n=== Milestone 1 GATE PASSED ===")


if __name__ == "__main__":
    main()
