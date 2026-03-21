#!/usr/bin/env python3
"""BPA v35: Cross-Model Fused INT4 Attention Benchmark on H100.

Reproduces BPA fused attention experiments on NVIDIA H100 with an expanded
model suite. Measures hardware limits, runs decode/prefill benchmarks across
multiple architectures, performs roofline analysis, and generates 12 plots.

Steps:
  1. Hardware limits (STREAM BW, GEMM TFLOP/s)
  2. Validation (fused vs reference)
  3. Decode benchmark matrix (6 configs x L x B)
  4. Prefill benchmark matrix
  5. Arithmetic intensity + roofline
  6. Head-dim transition
  7. Bandwidth scaling (R=1,2,4,8)
  8. Generate all 12 plots + report data
"""

import csv
import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import torch
import triton
import triton.language as tl

RESULTS_DIR = "artifacts/v35"
N_WARMUP = 5
N_REPEATS = 5


@dataclass
class ModelConfig:
    name: str
    n_heads: int
    n_kv_heads: int
    head_dim: int
    group_size: int = 32


# Model suite from spec
CONFIGS = [
    # Small
    ModelConfig("qwen25_05b", n_heads=14, n_kv_heads=2, head_dim=64),
    ModelConfig("qwen25_1.8b", n_heads=16, n_kv_heads=2, head_dim=128),
    # Mid-size
    ModelConfig("qwen25_7b", n_heads=28, n_kv_heads=4, head_dim=128),
    ModelConfig("mistral_7b", n_heads=32, n_kv_heads=8, head_dim=128),
    ModelConfig("llama31_8b", n_heads=32, n_kv_heads=8, head_dim=128),
]

# Benchmark matrix from spec
DECODE_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]
DECODE_BATCHES = [1, 2, 4, 8, 16, 32]
PREFILL_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]
PREFILL_BATCHES = [1, 2, 4, 8]


# ============================================================
# INT4 packing / unpacking
# ============================================================
def quantize_and_pack_int4(tensor, group_size=32):
    shape = tensor.shape
    hd = shape[-1]
    ng = hd // group_size
    r = tensor.reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (amax / 7.0).squeeze(-1)
    q = (r / (amax / 7.0)).round().clamp(-8, 7).to(torch.int8)
    q = q.reshape(*shape[:-1], hd)
    q_unsigned = (q + 8).to(torch.uint8)
    low = q_unsigned[..., 0::2]
    high = q_unsigned[..., 1::2]
    packed = low | (high << 4)
    return packed, scales


def dequant_int4(packed, scales, group_size=32):
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    hd = packed.shape[-1] * 2
    out = torch.empty(*packed.shape[:-1], hd, device=packed.device, dtype=torch.float16)
    out[..., 0::2] = low.to(torch.float16)
    out[..., 1::2] = high.to(torch.float16)
    ng = hd // group_size
    out = out.reshape(*packed.shape[:-1], ng, group_size)
    out = out * scales.unsqueeze(-1)
    out = out.reshape(*packed.shape[:-1], hd)
    return out


# ============================================================
# Step 1: Hardware limits
# ============================================================
@triton.jit
def _stream_kernel(
    A_ptr,
    B_ptr,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMENTS
    a = tl.load(A_ptr + offs, mask=mask)
    tl.store(B_ptr + offs, a, mask=mask)


def measure_bandwidth():
    """STREAM-like bandwidth test."""
    N = 256 * 1024 * 1024  # 256M elements = 512 MB
    A = torch.randn(N, device="cuda", dtype=torch.float16)
    B = torch.empty_like(A)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)

    for _ in range(10):
        _stream_kernel[grid](A, B, N, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _stream_kernel[grid](A, B, N, BLOCK_SIZE=BLOCK)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_s = sum(times) / len(times)
    bytes_moved = N * 2 * 2  # read + write, fp16 = 2 bytes
    bw_gbs = bytes_moved / mean_s / 1e9
    del A, B
    torch.cuda.empty_cache()
    return bw_gbs


def measure_fp16_compute():
    """GEMM benchmark for peak FP16 TFLOP/s."""
    M, N, K = 8192, 8192, 8192
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    for _ in range(10):
        torch.mm(A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.mm(A, B)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_s = sum(times) / len(times)
    flops = 2 * M * N * K
    tflops = flops / mean_s / 1e12
    del A, B
    torch.cuda.empty_cache()
    return tflops


# ============================================================
# Fused INT4 decode kernel
# ============================================================
@triton.jit
def _fused_decode_kernel(
    Q_ptr,
    K_packed_ptr,
    V_packed_ptr,
    Scale_k_ptr,
    Scale_v_ptr,
    Out_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kn,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vt,
    stride_vd,
    stride_skb,
    stride_skn,
    stride_skt,
    stride_skg,
    stride_svb,
    stride_svn,
    stride_svt,
    stride_svg,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    T_kv,
    n_kv_heads,
    n_rep: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HD: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    R_FACTOR: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    batch_id = pid_bh // (n_kv_heads * n_rep)
    head_id = pid_bh % (n_kv_heads * n_rep)
    kv_head_id = head_id // n_rep

    even_offs = tl.arange(0, HALF_HD) * 2
    odd_offs = tl.arange(0, HALF_HD) * 2 + 1
    q_base = Q_ptr + batch_id * stride_qb + head_id * stride_qh
    q_even = tl.load(q_base + even_offs * stride_qk).to(tl.float32)
    q_odd = tl.load(q_base + odd_offs * stride_qk).to(tl.float32)

    qk_scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)
    packed_d_offs = tl.arange(0, HALF_HD)
    group_idx = packed_d_offs // BYTES_PER_GROUP
    sk_base = Scale_k_ptr + batch_id * stride_skb + kv_head_id * stride_skn
    sv_base = Scale_v_ptr + batch_id * stride_svb + kv_head_id * stride_svn

    for _r in tl.static_range(0, R_FACTOR):
        m_i = float("-inf")
        l_i = 0.0
        acc_even = tl.zeros([HALF_HD], dtype=tl.float32)
        acc_odd = tl.zeros([HALF_HD], dtype=tl.float32)

        for start_n in range(0, T_kv, BLOCK_N):
            n_offs = start_n + tl.arange(0, BLOCK_N)
            n_mask = n_offs < T_kv
            k_packed = tl.load(
                K_packed_ptr
                + batch_id * stride_kb
                + kv_head_id * stride_kn
                + n_offs[:, None] * stride_kt
                + packed_d_offs[None, :] * stride_kd,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.uint8)
            k_low = ((k_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            k_high = (((k_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)
            scale_k_byte = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
            for g in tl.static_range(0, N_GROUPS):
                g_mask = (group_idx == g).to(tl.float32)
                sk_g = tl.load(
                    sk_base + n_offs * stride_skt + g * stride_skg,
                    mask=n_mask,
                    other=1.0,
                ).to(tl.float32)
                scale_k_byte += sk_g[:, None] * g_mask[None, :]
            k_low = k_low * scale_k_byte
            k_high = k_high * scale_k_byte
            qk = tl.sum(q_even[None, :] * k_low, axis=1) + tl.sum(
                q_odd[None, :] * k_high, axis=1
            )
            qk = qk * qk_scale
            qk = tl.where(n_mask, qk, float("-inf"))
            m_ij = tl.max(qk)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)
            l_i = l_i * alpha + tl.sum(p)
            acc_even = acc_even * alpha
            acc_odd = acc_odd * alpha
            v_packed = tl.load(
                V_packed_ptr
                + batch_id * stride_vb
                + kv_head_id * stride_vn
                + n_offs[:, None] * stride_vt
                + packed_d_offs[None, :] * stride_vd,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.uint8)
            v_low = ((v_packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
            v_high = (((v_packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)
            scale_v_byte = tl.zeros([BLOCK_N, HALF_HD], dtype=tl.float32)
            for g in tl.static_range(0, N_GROUPS):
                g_mask = (group_idx == g).to(tl.float32)
                sv_g = tl.load(
                    sv_base + n_offs * stride_svt + g * stride_svg,
                    mask=n_mask,
                    other=1.0,
                ).to(tl.float32)
                scale_v_byte += sv_g[:, None] * g_mask[None, :]
            v_low = v_low * scale_v_byte
            v_high = v_high * scale_v_byte
            acc_even += tl.sum(p[:, None] * v_low, axis=0)
            acc_odd += tl.sum(p[:, None] * v_high, axis=0)
            m_i = m_new

    acc_even = acc_even / l_i
    acc_odd = acc_odd / l_i
    out_base = Out_ptr + batch_id * stride_ob + head_id * stride_oh
    tl.store(out_base + even_offs * stride_ok, acc_even.to(tl.float16))
    tl.store(out_base + odd_offs * stride_ok, acc_odd.to(tl.float16))


def _launch_decode(Q, K_packed, V_packed, scale_k, scale_v, cfg, block_n, R):
    B_dim, n_heads_q, M, hd = Q.shape
    _, n_kv, T_kv, half_hd = K_packed.shape
    n_rep = n_heads_q // n_kv
    ng = hd // cfg.group_size
    bpg = cfg.group_size // 2
    assert M == 1
    out = torch.empty_like(Q)
    bn = min(block_n, T_kv)
    grid = (B_dim * n_heads_q,)
    _fused_decode_kernel[grid](
        Q,
        K_packed,
        V_packed,
        scale_k,
        scale_v,
        out,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K_packed.stride(0),
        K_packed.stride(1),
        K_packed.stride(2),
        K_packed.stride(3),
        V_packed.stride(0),
        V_packed.stride(1),
        V_packed.stride(2),
        V_packed.stride(3),
        scale_k.stride(0),
        scale_k.stride(1),
        scale_k.stride(2),
        scale_k.stride(3),
        scale_v.stride(0),
        scale_v.stride(1),
        scale_v.stride(2),
        scale_v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        T_kv=T_kv,
        n_kv_heads=n_kv,
        n_rep=n_rep,
        HEAD_DIM=hd,
        HALF_HD=half_hd,
        N_GROUPS=ng,
        BYTES_PER_GROUP=bpg,
        BLOCK_N=bn,
        R_FACTOR=R,
    )
    return out


# ============================================================
# Pipelines
# ============================================================
def pipeline_P0(Q, K_fp16, V_fp16, cfg, R=1):
    n_rep = cfg.n_heads // cfg.n_kv_heads
    K_exp = K_fp16.repeat_interleave(n_rep, dim=1)
    V_exp = V_fp16.repeat_interleave(n_rep, dim=1)
    out = None
    for _ in range(R):
        out = torch.nn.functional.scaled_dot_product_attention(
            Q, K_exp, V_exp, is_causal=False
        )
    return out


def pipeline_P1(Q, K_packed, V_packed, scale_k, scale_v, cfg, R=1):
    out = None
    for _ in range(R):
        K_fp16 = dequant_int4(K_packed, scale_k, cfg.group_size)
        V_fp16 = dequant_int4(V_packed, scale_v, cfg.group_size)
        n_rep = cfg.n_heads // cfg.n_kv_heads
        K_exp = K_fp16.repeat_interleave(n_rep, dim=1)
        V_exp = V_fp16.repeat_interleave(n_rep, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            Q, K_exp, V_exp, is_causal=False
        )
    return out


def pipeline_P2(Q, K_packed, V_packed, scale_k, scale_v, cfg, R=1):
    return _launch_decode(Q, K_packed, V_packed, scale_k, scale_v, cfg, 64, R)


def _select_h100_block_n(Q, K_packed, cfg):
    """Pick a decode tile for Hopper based on the active decode regime."""
    B_dim = Q.shape[0]
    T_kv = K_packed.shape[2]
    if cfg.head_dim <= 64:
        return 128
    if B_dim >= 2:
        return 128
    if T_kv >= 8192:
        return 128
    return 64


def pipeline_P3(Q, K_packed, V_packed, scale_k, scale_v, cfg, R=1):
    bn = _select_h100_block_n(Q, K_packed, cfg)
    return _launch_decode(Q, K_packed, V_packed, scale_k, scale_v, cfg, bn, R)


def pipeline_P5(Q, K_packed, V_packed, scale_k, scale_v, cfg, R=1):
    M = Q.shape[2]
    if M == 1:
        bn = _select_h100_block_n(Q, K_packed, cfg)
        return _launch_decode(Q, K_packed, V_packed, scale_k, scale_v, cfg, bn, R)
    else:
        K_fp16 = dequant_int4(K_packed, scale_k, cfg.group_size)
        V_fp16 = dequant_int4(V_packed, scale_v, cfg.group_size)
        n_rep = cfg.n_heads // cfg.n_kv_heads
        K_exp = K_fp16.repeat_interleave(n_rep, dim=1)
        V_exp = V_fp16.repeat_interleave(n_rep, dim=1)
        out = None
        for _ in range(R):
            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K_exp, V_exp, is_causal=False
            )
        return out


# ============================================================
# Arithmetic intensity
# ============================================================
def compute_arithmetic_intensity(cfg, T_kv, mode="decode"):
    D = cfg.head_dim
    Hq = cfg.n_heads
    Hkv = cfg.n_kv_heads
    gs = cfg.group_size
    ng = D // gs

    M = 1 if mode == "decode" else T_kv

    flops_per_head = 2 * M * T_kv * D + 5 * M * T_kv + 2 * M * T_kv * D
    total_flops = flops_per_head * Hq

    results = {}

    q_bytes = Hq * M * D * 2
    kv_bytes_p0 = 2 * Hkv * T_kv * D * 2
    total_bytes_p0 = q_bytes + kv_bytes_p0
    results["P0"] = {
        "flops": total_flops,
        "bytes": total_bytes_p0,
        "ai": total_flops / total_bytes_p0 if total_bytes_p0 > 0 else 0,
    }

    int4_bytes = 2 * Hkv * T_kv * (D // 2)
    scale_bytes = 2 * Hkv * T_kv * ng * 2
    fp16_write = 2 * Hkv * T_kv * D * 2
    fp16_read = 2 * Hkv * T_kv * D * 2
    gqa_bytes = 2 * Hq * T_kv * D * 2
    total_bytes_p1 = (
        q_bytes + int4_bytes + scale_bytes + fp16_write + fp16_read + gqa_bytes
    )
    results["P1"] = {
        "flops": total_flops,
        "bytes": total_bytes_p1,
        "ai": total_flops / total_bytes_p1 if total_bytes_p1 > 0 else 0,
    }

    fused_bytes = q_bytes + int4_bytes + scale_bytes
    results["P3"] = {
        "flops": total_flops,
        "bytes": fused_bytes,
        "ai": total_flops / fused_bytes if fused_bytes > 0 else 0,
    }
    results["P5"] = results["P3"].copy()

    return results


# ============================================================
# Benchmark helpers
# ============================================================
def benchmark_fn(fn, args, kwargs=None):
    if kwargs is None:
        kwargs = {}
    for _ in range(N_WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def estimate_decode_mem_gb(cfg, T_kv, B_dim):
    """Estimate peak GPU memory for decode benchmark."""
    fp16_kv = B_dim * cfg.n_kv_heads * T_kv * cfg.head_dim * 2 * 2
    int4_kv = B_dim * cfg.n_kv_heads * T_kv * (cfg.head_dim // 2) * 2
    scales = B_dim * cfg.n_kv_heads * T_kv * (cfg.head_dim // cfg.group_size) * 2 * 2
    gqa_expand = B_dim * cfg.n_heads * T_kv * cfg.head_dim * 2 * 2
    return (fp16_kv + int4_kv + scales + gqa_expand) / 1e9


def estimate_prefill_mem_gb(cfg, T_kv, B_dim):
    """Estimate peak GPU memory for prefill benchmark."""
    q_mem = B_dim * cfg.n_heads * T_kv * cfg.head_dim * 2
    kv_mem = B_dim * cfg.n_kv_heads * T_kv * cfg.head_dim * 2 * 2
    gqa_expand = B_dim * cfg.n_heads * T_kv * cfg.head_dim * 2 * 2
    attn_scores = B_dim * cfg.n_heads * T_kv * T_kv * 2
    return (q_mem + kv_mem + gqa_expand + attn_scores) / 1e9


def run_decode_bench(cfg, T_kv, B_dim, R=1, pipelines=None):
    if pipelines is None:
        pipelines = ["P0", "P1", "P2", "P3", "P5"]
    device = "cuda"
    K_fp16 = torch.randn(
        B_dim,
        cfg.n_kv_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    V_fp16 = torch.randn(
        B_dim,
        cfg.n_kv_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    Q = torch.randn(
        B_dim,
        cfg.n_heads,
        1,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    K_packed, scale_k = quantize_and_pack_int4(K_fp16, cfg.group_size)
    V_packed, scale_v = quantize_and_pack_int4(V_fp16, cfg.group_size)

    results = {}
    if "P0" in pipelines:
        results["P0"] = benchmark_fn(pipeline_P0, (Q, K_fp16, V_fp16, cfg), {"R": R})
    if "P1" in pipelines:
        results["P1"] = benchmark_fn(
            pipeline_P1,
            (Q, K_packed, V_packed, scale_k, scale_v, cfg),
            {"R": R},
        )
    if "P2" in pipelines:
        results["P2"] = benchmark_fn(
            pipeline_P2,
            (Q, K_packed, V_packed, scale_k, scale_v, cfg),
            {"R": R},
        )
    if "P3" in pipelines:
        results["P3"] = benchmark_fn(
            pipeline_P3,
            (Q, K_packed, V_packed, scale_k, scale_v, cfg),
            {"R": R},
        )
    if "P5" in pipelines:
        results["P5"] = benchmark_fn(
            pipeline_P5,
            (Q, K_packed, V_packed, scale_k, scale_v, cfg),
            {"R": R},
        )

    del K_fp16, V_fp16, Q, K_packed, V_packed, scale_k, scale_v
    torch.cuda.empty_cache()
    gc.collect()
    return results


def run_prefill_bench(cfg, T_kv, B_dim, pipelines=None):
    if pipelines is None:
        pipelines = ["P0", "P1", "P5"]
    device = "cuda"
    K_fp16 = torch.randn(
        B_dim,
        cfg.n_kv_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    V_fp16 = torch.randn(
        B_dim,
        cfg.n_kv_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    Q = torch.randn(
        B_dim,
        cfg.n_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    K_packed, scale_k = quantize_and_pack_int4(K_fp16, cfg.group_size)
    V_packed, scale_v = quantize_and_pack_int4(V_fp16, cfg.group_size)

    results = {}
    if "P0" in pipelines:
        results["P0"] = benchmark_fn(pipeline_P0, (Q, K_fp16, V_fp16, cfg))
    if "P1" in pipelines:
        results["P1"] = benchmark_fn(
            pipeline_P1, (Q, K_packed, V_packed, scale_k, scale_v, cfg)
        )
    if "P5" in pipelines:
        results["P5"] = benchmark_fn(
            pipeline_P5, (Q, K_packed, V_packed, scale_k, scale_v, cfg)
        )

    del K_fp16, V_fp16, Q, K_packed, V_packed, scale_k, scale_v
    torch.cuda.empty_cache()
    gc.collect()
    return results


def validate_fused(cfg, T_kv=512, B_dim=1):
    """Validate fused kernel output against reference."""
    device = "cuda"
    K_fp16 = torch.randn(
        B_dim,
        cfg.n_kv_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    V_fp16 = torch.randn(
        B_dim,
        cfg.n_kv_heads,
        T_kv,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    Q = torch.randn(
        B_dim,
        cfg.n_heads,
        1,
        cfg.head_dim,
        device=device,
        dtype=torch.float16,
    )
    K_packed, scale_k = quantize_and_pack_int4(K_fp16, cfg.group_size)
    V_packed, scale_v = quantize_and_pack_int4(V_fp16, cfg.group_size)

    # Reference: dequant + SDPA
    K_deq = dequant_int4(K_packed, scale_k, cfg.group_size)
    V_deq = dequant_int4(V_packed, scale_v, cfg.group_size)
    n_rep = cfg.n_heads // cfg.n_kv_heads
    K_exp = K_deq.repeat_interleave(n_rep, dim=1)
    V_exp = V_deq.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, is_causal=False
    )

    # Fused
    fused = pipeline_P3(Q, K_packed, V_packed, scale_k, scale_v, cfg)

    max_err = (fused - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-8)

    del K_fp16, V_fp16, Q, K_packed, V_packed, scale_k, scale_v
    torch.cuda.empty_cache()

    return {
        "config": cfg.name,
        "max_abs_error": max_err,
        "max_rel_error": rel_err,
        "pass": rel_err <= 1e-3,
    }


# ============================================================
# Main
# ============================================================
def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("BPA v35: H100 Cross-Model Fused INT4 Attention Benchmark")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    MEM_LIMIT_GB = 70  # leave headroom on 80GB

    # ========================================
    # Step 1: Hardware limits
    # ========================================
    print("\nStep 1: Measuring hardware limits...")
    bw_gbs = measure_bandwidth()
    print(f"  Sustained bandwidth: {bw_gbs:.1f} GB/s")
    tflops = measure_fp16_compute()
    print(f"  Peak FP16 compute: {tflops:.2f} TFLOP/s")

    hw_limits = {
        "gpu": gpu_name,
        "sustained_bandwidth_GBs": round(bw_gbs, 2),
        "peak_fp16_TFLOPs": round(tflops, 3),
        "ridge_point_flops_per_byte": round(tflops * 1e12 / (bw_gbs * 1e9), 2),
    }
    with open(os.path.join(RESULTS_DIR, "hardware_limits.json"), "w") as f:
        json.dump(hw_limits, f, indent=2)
    print(f"  Ridge point: {hw_limits['ridge_point_flops_per_byte']:.1f}")

    peak_gflops = tflops * 1000
    ridge = hw_limits["ridge_point_flops_per_byte"]

    # ========================================
    # Step 2: Validation
    # ========================================
    print("\nStep 2: Validating fused kernels...")
    val_results = []
    for cfg in CONFIGS:
        r = validate_fused(cfg)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {cfg.name}: rel_err={r['max_rel_error']:.2e} {status}")
        val_results.append(r)

    with open(os.path.join(RESULTS_DIR, "validation.json"), "w") as f:
        json.dump(val_results, f, indent=2)

    # ========================================
    # Step 3: Decode benchmark matrix
    # ========================================
    print("\nStep 3: Decode benchmark matrix...")
    all_decode_rows = []
    for cfg in CONFIGS:
        for T_kv in DECODE_LENGTHS:
            for B_dim in DECODE_BATCHES:
                est = estimate_decode_mem_gb(cfg, T_kv, B_dim)
                if est > MEM_LIMIT_GB:
                    continue
                print(
                    f"  {cfg.name} T={T_kv} B={B_dim} " f"(~{est:.1f}GB)...",
                    end="",
                    flush=True,
                )
                try:
                    lats = run_decode_bench(cfg, T_kv, B_dim)
                    for pn, lat in lats.items():
                        all_decode_rows.append(
                            {
                                "config": cfg.name,
                                "mode": "decode",
                                "T_kv": T_kv,
                                "B": B_dim,
                                "pipeline": pn,
                                "latency_ms": round(lat * 1000, 4),
                                "tokens_per_sec": round(
                                    B_dim / lat if lat > 0 else 0, 1
                                ),
                            }
                        )
                    p0 = lats.get("P0", 1)
                    p5 = lats.get("P5", 1)
                    sp = p0 / p5 if p5 > 0 else 0
                    print(f" P0={p0*1000:.3f}ms P5={p5*1000:.3f}ms ({sp:.1f}x)")
                except torch.cuda.OutOfMemoryError:
                    print(" OOM")
                    torch.cuda.empty_cache()
                    gc.collect()

    # ========================================
    # Step 4: Prefill benchmark matrix
    # ========================================
    print("\nStep 4: Prefill benchmark matrix...")
    all_prefill_rows = []
    for cfg in CONFIGS:
        for T_kv in PREFILL_LENGTHS:
            for B_dim in PREFILL_BATCHES:
                est = estimate_prefill_mem_gb(cfg, T_kv, B_dim)
                if est > MEM_LIMIT_GB:
                    continue
                print(
                    f"  {cfg.name} T={T_kv} B={B_dim} " f"(~{est:.1f}GB)...",
                    end="",
                    flush=True,
                )
                try:
                    lats = run_prefill_bench(cfg, T_kv, B_dim)
                    for pn, lat in lats.items():
                        all_prefill_rows.append(
                            {
                                "config": cfg.name,
                                "mode": "prefill",
                                "T_kv": T_kv,
                                "B": B_dim,
                                "pipeline": pn,
                                "latency_ms": round(lat * 1000, 4),
                            }
                        )
                    p0 = lats.get("P0", 1)
                    p5 = lats.get("P5", 1)
                    sp = p0 / p5 if p5 > 0 else 0
                    print(f" P0={p0*1000:.3f}ms P5={p5*1000:.3f}ms ({sp:.1f}x)")
                except torch.cuda.OutOfMemoryError:
                    print(" OOM")
                    torch.cuda.empty_cache()
                    gc.collect()

    # Save all bench results
    all_rows = all_decode_rows + all_prefill_rows
    csv_path = os.path.join(RESULTS_DIR, "bench_results.csv")
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)
        print(f"  Saved: {csv_path} ({len(all_rows)} rows)")

    # ========================================
    # Step 5: Arithmetic intensity + roofline
    # ========================================
    print("\nStep 5: Arithmetic intensity + roofline...")
    ai_rows = []
    for cfg in CONFIGS:
        for T_kv in [2048, 4096, 8192, 16384]:
            for mode in ["decode", "prefill"]:
                ai = compute_arithmetic_intensity(cfg, T_kv, mode)
                for pn, vals in ai.items():
                    ai_rows.append(
                        {
                            "config": cfg.name,
                            "T_kv": T_kv,
                            "mode": mode,
                            "pipeline": pn,
                            "flops": vals["flops"],
                            "bytes": vals["bytes"],
                            "arithmetic_intensity": round(vals["ai"], 4),
                        }
                    )

    ai_csv = os.path.join(RESULTS_DIR, "pipeline_arithmetic_intensity.csv")
    if ai_rows:
        fields = list(ai_rows[0].keys())
        with open(ai_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in ai_rows:
                writer.writerow(row)

    colors = {
        "P0": "tab:gray",
        "P1": "tab:blue",
        "P2": "tab:orange",
        "P3": "tab:red",
        "P5": "tab:purple",
    }
    labels = {
        "P0": "Dense FP16",
        "P1": "INT4 dequant+SDPA",
        "P2": "Fused INT4 (BN=64)",
        "P3": "Fused INT4+opt",
        "P5": "Unified",
    }

    # Decode roofline (use first 2 configs for clarity)
    plot_cfgs = CONFIGS[:2]
    fig, axes = plt.subplots(1, len(plot_cfgs), figsize=(8 * len(plot_cfgs), 6))
    if len(plot_cfgs) == 1:
        axes = [axes]

    for ci, cfg in enumerate(plot_cfgs):
        ax = axes[ci]
        for T_kv in [2048, 8192, 16384]:
            B_dim = 1
            est = estimate_decode_mem_gb(cfg, T_kv, B_dim)
            if est > MEM_LIMIT_GB:
                continue
            try:
                latencies = run_decode_bench(
                    cfg, T_kv, B_dim, pipelines=["P0", "P1", "P3", "P5"]
                )
                ai = compute_arithmetic_intensity(cfg, T_kv, "decode")
                for pn in ["P0", "P1", "P3", "P5"]:
                    if pn in latencies:
                        flops = ai[pn]["flops"]
                        achieved_gflops = flops / latencies[pn] / 1e9
                        intensity = ai[pn]["ai"]
                        marker = "o" if T_kv == 2048 else ("s" if T_kv == 8192 else "^")
                        ax.scatter(
                            intensity,
                            achieved_gflops,
                            c=colors[pn],
                            marker=marker,
                            s=80,
                            zorder=5,
                        )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        mem_x = [0.01, 0.1, 1, ridge]
        mem_y = [bw_gbs * x for x in mem_x]
        ax.plot(mem_x, mem_y, "k-", linewidth=2, label="Memory roof")
        ax.plot(
            [ridge, 10000],
            [peak_gflops, peak_gflops],
            "k--",
            linewidth=2,
            label="Compute roof",
        )
        ax.axvline(x=ridge, color="gray", linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title(f"{cfg.name} Decode Roofline")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0.01, 10000)

    # Add legend proxies
    for pn in ["P0", "P1", "P3", "P5"]:
        axes[0].scatter([], [], c=colors[pn], label=labels[pn], s=40)
    axes[0].legend(fontsize=7, loc="lower right")
    plt.suptitle("Decode Attention Roofline (H100)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roofline_decode.png"), dpi=200)
    plt.close()

    # Prefill roofline
    fig, axes = plt.subplots(1, len(plot_cfgs), figsize=(8 * len(plot_cfgs), 6))
    if len(plot_cfgs) == 1:
        axes = [axes]

    for ci, cfg in enumerate(plot_cfgs):
        ax = axes[ci]
        for T_kv in [512, 1024, 2048]:
            B_dim = 1
            try:
                latencies = run_prefill_bench(cfg, T_kv, B_dim)
                ai = compute_arithmetic_intensity(cfg, T_kv, "prefill")
                for pn in ["P0", "P1", "P5"]:
                    if pn in latencies:
                        flops = ai[pn]["flops"]
                        achieved_gflops = flops / latencies[pn] / 1e9
                        intensity = ai[pn]["ai"]
                        marker = "o" if T_kv == 512 else ("s" if T_kv == 1024 else "^")
                        ax.scatter(
                            intensity,
                            achieved_gflops,
                            c=colors[pn],
                            marker=marker,
                            s=80,
                            zorder=5,
                        )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        mem_x = [0.01, 0.1, 1, ridge]
        mem_y = [bw_gbs * x for x in mem_x]
        ax.plot(mem_x, mem_y, "k-", linewidth=2, label="Memory roof")
        ax.plot(
            [ridge, 100000],
            [peak_gflops, peak_gflops],
            "k--",
            linewidth=2,
            label="Compute roof",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
        ax.set_ylabel("Performance (GFLOP/s)")
        ax.set_title(f"{cfg.name} Prefill Roofline")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(0.01, 100000)

    plt.suptitle("Prefill Attention Roofline (H100)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roofline_prefill.png"), dpi=200)
    plt.close()

    # ========================================
    # Step 6: Head-dim transition
    # ========================================
    print("\nStep 6: Head-dim transition experiment...")
    head_dims = [32, 64, 128, 256]
    hd_results = {"P0": [], "P3": []}
    T_kv = 8192
    B_dim = 1

    for D in head_dims:
        hd_cfg = ModelConfig(f"hd{D}", n_heads=8, n_kv_heads=2, head_dim=D)
        print(f"  D={D}...", end="", flush=True)
        try:
            latencies = run_decode_bench(hd_cfg, T_kv, B_dim, pipelines=["P0", "P3"])
            hd_results["P0"].append(latencies["P0"] * 1000)
            hd_results["P3"].append(latencies["P3"] * 1000)
            print(
                f" P0={latencies['P0']*1000:.3f}ms " f"P3={latencies['P3']*1000:.3f}ms"
            )
        except Exception as e:
            print(f" ERROR: {e}")
            hd_results["P0"].append(None)
            hd_results["P3"].append(None)

    fig, ax = plt.subplots(figsize=(8, 5))
    valid_D = [d for d, v in zip(head_dims, hd_results["P0"]) if v is not None]
    valid_P0 = [v for v in hd_results["P0"] if v is not None]
    valid_P3 = [v for v in hd_results["P3"] if v is not None]
    ax.plot(valid_D, valid_P0, "o-", label="P0 (Dense FP16)", color="tab:gray")
    ax.plot(valid_D, valid_P3, "s-", label="P3 (Fused INT4)", color="tab:red")
    ax.set_xlabel("Head Dimension D")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Decode Latency vs Head Dimension (H100, T={T_kv}, B={B_dim})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_vs_head_dim.png"), dpi=200)
    plt.close()

    # ========================================
    # Step 7: Bandwidth scaling
    # ========================================
    print("\nStep 7: Bandwidth scaling (R=1,2,4,8)...")
    R_SET = [1, 2, 4, 8]
    bw_data = {pn: [] for pn in ["P0", "P1", "P3", "P5"]}
    bw_cfg = CONFIGS[0]  # qwen25_05b
    T_kv = 8192
    B_dim = 1

    for R in R_SET:
        print(f"  R={R}...", end="", flush=True)
        latencies = run_decode_bench(
            bw_cfg,
            T_kv,
            B_dim,
            R=R,
            pipelines=["P0", "P1", "P3", "P5"],
        )
        for pn in ["P0", "P1", "P3", "P5"]:
            bw_data[pn].append(latencies[pn] * 1000)
        print(f" P0={latencies['P0']*1000:.3f} " f"P3={latencies['P3']*1000:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for pn in ["P0", "P1", "P3", "P5"]:
        ax.plot(R_SET, bw_data[pn], "o-", label=labels[pn], color=colors[pn])
    ax.set_xlabel("Replication Factor R")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Bandwidth Scaling (H100, {bw_cfg.name}, T={T_kv})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(R_SET)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "speedup_vs_replication.png"), dpi=200)
    plt.close()

    # ========================================
    # Step 8: Generate remaining plots
    # ========================================
    print("\nStep 8: Generating plots...")

    # Helper to extract data from decode rows
    def get_decode_data(config_name, pipeline, metric="latency_ms"):
        return [
            (r["T_kv"], r["B"], r[metric])
            for r in all_decode_rows
            if r["config"] == config_name and r["pipeline"] == pipeline
        ]

    # --- latency_vs_context_decode.png ---
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5 * len(CONFIGS), 5))
    if len(CONFIGS) == 1:
        axes = [axes]
    for ci, cfg in enumerate(CONFIGS):
        ax = axes[ci]
        for pn in ["P0", "P3", "P5"]:
            data = get_decode_data(cfg.name, pn)
            # Filter B=1
            pts = sorted([(t, lat) for t, b, lat in data if b == 1])
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o-",
                    label=labels.get(pn, pn),
                    color=colors.get(pn, "k"),
                )
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{cfg.name} (B=1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Decode Latency vs Context Length (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_vs_context_decode.png"), dpi=200)
    plt.close()

    # --- latency_vs_context_prefill.png ---
    fig, axes = plt.subplots(
        1, min(3, len(CONFIGS)), figsize=(5 * min(3, len(CONFIGS)), 5)
    )
    if not isinstance(axes, list) and not hasattr(axes, "__len__"):
        axes = [axes]
    for ci, cfg in enumerate(CONFIGS[:3]):
        ax = axes[ci]
        for pn in ["P0", "P1", "P5"]:
            pts = sorted(
                [
                    (r["T_kv"], r["latency_ms"])
                    for r in all_prefill_rows
                    if r["config"] == cfg.name and r["pipeline"] == pn and r["B"] == 1
                ]
            )
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o-",
                    label=labels.get(pn, pn),
                    color=colors.get(pn, "k"),
                )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{cfg.name} (B=1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Prefill Latency vs Sequence Length (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_vs_context_prefill.png"), dpi=200)
    plt.close()

    # --- speedup_vs_context_decode.png ---
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5 * len(CONFIGS), 5))
    if len(CONFIGS) == 1:
        axes = [axes]
    for ci, cfg in enumerate(CONFIGS):
        ax = axes[ci]
        p0_data = {(t, b): lat for t, b, lat in get_decode_data(cfg.name, "P0")}
        for pn in ["P3", "P5"]:
            data = get_decode_data(cfg.name, pn)
            pts_b1 = sorted(
                [
                    (t, p0_data.get((t, 1), 0) / lat if lat > 0 else 0)
                    for t, b, lat in data
                    if b == 1 and (t, 1) in p0_data
                ]
            )
            if pts_b1:
                ax.plot(
                    [p[0] for p in pts_b1],
                    [p[1] for p in pts_b1],
                    "o-",
                    label=labels.get(pn, pn),
                    color=colors.get(pn, "k"),
                )
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Speedup vs P0")
        ax.set_title(f"{cfg.name} (B=1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Decode Speedup vs Context Length (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "speedup_vs_context_decode.png"), dpi=200)
    plt.close()

    # --- speedup_vs_context_prefill.png ---
    fig, axes = plt.subplots(
        1, min(3, len(CONFIGS)), figsize=(5 * min(3, len(CONFIGS)), 5)
    )
    if not isinstance(axes, list) and not hasattr(axes, "__len__"):
        axes = [axes]
    for ci, cfg in enumerate(CONFIGS[:3]):
        ax = axes[ci]
        p0_data = {
            (r["T_kv"], r["B"]): r["latency_ms"]
            for r in all_prefill_rows
            if r["config"] == cfg.name and r["pipeline"] == "P0"
        }
        for pn in ["P1", "P5"]:
            pts = sorted(
                [
                    (
                        r["T_kv"],
                        (
                            p0_data.get((r["T_kv"], 1), 0) / r["latency_ms"]
                            if r["latency_ms"] > 0
                            else 0
                        ),
                    )
                    for r in all_prefill_rows
                    if r["config"] == cfg.name
                    and r["pipeline"] == pn
                    and r["B"] == 1
                    and (r["T_kv"], 1) in p0_data
                ]
            )
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o-",
                    label=labels.get(pn, pn),
                    color=colors.get(pn, "k"),
                )
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Speedup vs P0")
        ax.set_title(f"{cfg.name} (B=1)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Prefill Speedup vs Sequence Length (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "speedup_vs_context_prefill.png"), dpi=200)
    plt.close()

    # --- throughput_vs_batch_decode.png ---
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5 * len(CONFIGS), 5))
    if len(CONFIGS) == 1:
        axes = [axes]
    for ci, cfg in enumerate(CONFIGS):
        ax = axes[ci]
        for pn in ["P0", "P3", "P5"]:
            data = get_decode_data(cfg.name, pn, "tokens_per_sec")
            # Filter T=8192
            pts = sorted([(b, tps) for t, b, tps in data if t == 8192])
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o-",
                    label=labels.get(pn, pn),
                    color=colors.get(pn, "k"),
                )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Tokens/sec")
        ax.set_title(f"{cfg.name} (T=8192)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Decode Throughput vs Batch Size (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "throughput_vs_batch_decode.png"), dpi=200)
    plt.close()

    # --- throughput_vs_batch_prefill.png ---
    fig, axes = plt.subplots(
        1, min(3, len(CONFIGS)), figsize=(5 * min(3, len(CONFIGS)), 5)
    )
    if not isinstance(axes, list) and not hasattr(axes, "__len__"):
        axes = [axes]
    for ci, cfg in enumerate(CONFIGS[:3]):
        ax = axes[ci]
        for pn in ["P0", "P5"]:
            pts = sorted(
                [
                    (
                        r["B"],
                        r["B"] / (r["latency_ms"] / 1000) if r["latency_ms"] > 0 else 0,
                    )
                    for r in all_prefill_rows
                    if r["config"] == cfg.name
                    and r["pipeline"] == pn
                    and r["T_kv"] == 2048
                ]
            )
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    "o-",
                    label=labels.get(pn, pn),
                    color=colors.get(pn, "k"),
                )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Sequences/sec")
        ax.set_title(f"{cfg.name} (T=2048)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Prefill Throughput vs Batch Size (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "throughput_vs_batch_prefill.png"), dpi=200)
    plt.close()

    # --- KV_bytes_vs_latency_frontier.png ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ci, cfg in enumerate(CONFIGS[:2]):
        ax = axes[ci]
        for pn in ["P0", "P1", "P3", "P5"]:
            data = get_decode_data(cfg.name, pn)
            ai_lookup = {}
            for T_kv in DECODE_LENGTHS:
                ai = compute_arithmetic_intensity(cfg, T_kv, "decode")
                if pn in ai:
                    ai_lookup[T_kv] = ai[pn]["bytes"]
            pts = [
                (ai_lookup.get(t, 0) * b / 1e6, lat)
                for t, b, lat in data
                if t in ai_lookup
            ]
            if pts:
                ax.scatter(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    c=colors.get(pn, "k"),
                    alpha=0.6,
                    s=30,
                    label=labels.get(pn, pn),
                )
        ax.set_xlabel("KV Bytes Read (MB)")
        ax.set_ylabel("Decode Latency (ms)")
        ax.set_title(f"{cfg.name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("KV Bytes vs Decode Latency Frontier (H100)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "KV_bytes_vs_latency_frontier.png"), dpi=200)
    plt.close()

    # --- model_architecture_speedup.png ---
    print("  Generating model architecture speedup plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = []
    speedups_mean = []
    speedups_min = []
    speedups_max = []

    for cfg in CONFIGS:
        p0_data = {
            (r["T_kv"], r["B"]): r["latency_ms"]
            for r in all_decode_rows
            if r["config"] == cfg.name and r["pipeline"] == "P0"
        }
        p5_data = {
            (r["T_kv"], r["B"]): r["latency_ms"]
            for r in all_decode_rows
            if r["config"] == cfg.name and r["pipeline"] == "P5"
        }
        sps = []
        for key in p0_data:
            if key in p5_data and p5_data[key] > 0:
                sps.append(p0_data[key] / p5_data[key])
        if sps:
            model_names.append(cfg.name)
            speedups_mean.append(sum(sps) / len(sps))
            speedups_min.append(min(sps))
            speedups_max.append(max(sps))

    if model_names:
        x = range(len(model_names))
        ax.bar(x, speedups_mean, color="tab:red", alpha=0.7, label="Mean")
        ax.errorbar(
            x,
            speedups_mean,
            yerr=[
                [m - mn for m, mn in zip(speedups_mean, speedups_min)],
                [mx - m for m, mx in zip(speedups_mean, speedups_max)],
            ],
            fmt="none",
            color="black",
            capsize=5,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_ylabel("Decode Speedup (P5/P0)")
        ax.set_title("Cross-Model Decode Speedup (H100)")
        ax.axhline(y=1, color="gray", linestyle="--")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_architecture_speedup.png"), dpi=200)
    plt.close()

    # ========================================
    # Save kernel profile data
    # ========================================
    profile_data = {
        "hardware_limits": hw_limits,
        "validation": val_results,
        "head_dim_results": {
            "head_dims": head_dims,
            "P0_ms": hd_results["P0"],
            "P3_ms": hd_results["P3"],
        },
        "bandwidth_scaling": {
            "R_values": R_SET,
            "config": bw_cfg.name,
            "P0_ms": bw_data["P0"],
            "P1_ms": bw_data["P1"],
            "P3_ms": bw_data["P3"],
            "P5_ms": bw_data["P5"],
        },
        "decode_summary": {},
        "prefill_summary": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Compute per-model decode summary
    for cfg in CONFIGS:
        p0_lats = [
            r["latency_ms"]
            for r in all_decode_rows
            if r["config"] == cfg.name and r["pipeline"] == "P0"
        ]
        p5_lats = [
            r["latency_ms"]
            for r in all_decode_rows
            if r["config"] == cfg.name and r["pipeline"] == "P5"
        ]
        p0_map = {
            (r["T_kv"], r["B"]): r["latency_ms"]
            for r in all_decode_rows
            if r["config"] == cfg.name and r["pipeline"] == "P0"
        }
        p5_map = {
            (r["T_kv"], r["B"]): r["latency_ms"]
            for r in all_decode_rows
            if r["config"] == cfg.name and r["pipeline"] == "P5"
        }
        sps = [p0_map[k] / p5_map[k] for k in p0_map if k in p5_map and p5_map[k] > 0]
        if sps:
            profile_data["decode_summary"][cfg.name] = {
                "speedup_min": round(min(sps), 2),
                "speedup_max": round(max(sps), 2),
                "speedup_mean": round(sum(sps) / len(sps), 2),
                "n_configs": len(sps),
            }

    with open(os.path.join(RESULTS_DIR, "kernel_profile.json"), "w") as f:
        json.dump(profile_data, f, indent=2)

    print(f"\n{'='*60}")
    print("All steps complete.")
    print(f"Artifacts saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")

    # Print summary
    print("\n=== DECODE SPEEDUP SUMMARY (P5 vs P0) ===")
    for name, data in profile_data["decode_summary"].items():
        print(
            f"  {name}: {data['speedup_min']:.1f}x - "
            f"{data['speedup_max']:.1f}x (mean {data['speedup_mean']:.1f}x)"
        )

    print("\n=== BANDWIDTH SCALING (R=8/R=1) ===")
    for pn in ["P0", "P1", "P3", "P5"]:
        ratio = bw_data[pn][-1] / bw_data[pn][0] if bw_data[pn][0] > 0 else 0
        print(f"  {pn}: {ratio:.2f}x")


if __name__ == "__main__":
    main()
