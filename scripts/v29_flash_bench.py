#!/usr/bin/env python3
"""BPA v29: Fused INT4 FlashAttention A/B Test on AMD W7900.

Pipeline A (control): INT4 KV → dequant to FP16 → SDPA
Pipeline B (experimental): INT4 KV → fused Triton attention kernel
    (unpack in registers, no intermediate FP16 KV tensors)

Tests L={2048,4096,8192,16384}, B={1,4,8} on Qwen2.5-0.5B config.
"""

import csv
import gc
import json
import os
import time
from datetime import datetime

import torch
import triton
import triton.language as tl

RESULTS_DIR = "artifacts/v29_flash"

# Model config (Qwen2.5-0.5B)
N_HEADS = 14  # query heads
N_KV_HEADS = 2  # KV heads (GQA)
HEAD_DIM = 64
GROUP_SIZE = 32  # INT4 quantization group size

# Experiment grid
L_SET = [2048, 4096, 8192, 16384]
B_SET = [1, 4, 8]
N_WARMUP = 5
N_REPEATS = 5


# ============================================================
# INT4 packing / unpacking utilities
# ============================================================
def quantize_and_pack_int4(tensor, group_size=GROUP_SIZE):
    """Quantize fp16 tensor to INT4 and pack 2 values per byte.

    Returns (packed_data, scales).
    packed_data: uint8 [B, n_kv, T, hd//2]
    scales: fp16 [B, n_kv, T, ng]
    """
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


def dequant_int4(packed, scales, group_size=GROUP_SIZE):
    """Dequantize packed INT4 back to fp16."""
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
# Pipeline A: dequant_then_attention
# ============================================================
def pipeline_A(Q, K_packed, V_packed, scale_k, scale_v):
    """Dequantize INT4 KV to FP16, then use SDPA."""
    K_fp16 = dequant_int4(K_packed, scale_k)
    V_fp16 = dequant_int4(V_packed, scale_v)

    n_rep = N_HEADS // N_KV_HEADS
    K_exp = K_fp16.repeat_interleave(n_rep, dim=1)
    V_exp = V_fp16.repeat_interleave(n_rep, dim=1)

    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, is_causal=False
    )


# ============================================================
# Pipeline B: Fused Triton INT4 FlashAttention
# ============================================================
# For HEAD_DIM=64 and GROUP_SIZE=32, we have:
# - HALF_HD = 32 (packed bytes per token)
# - N_GROUPS = 2
# - Each packed byte holds 2 INT4 values (even + odd indices)
# - Group 0 covers elements [0:32], Group 1 covers elements [32:64]
# - In packed layout: Group 0 is packed bytes [0:16], Group 1 is [16:32]


@triton.jit
def _unpack_and_scale_int4(packed, scale_g0, scale_g1, HALF_HD: tl.constexpr):
    """Unpack INT4 packed tensor and apply per-group scales.

    packed: [BLOCK_N, HALF_HD] uint8
    scale_g0, scale_g1: [BLOCK_N] fp16 (one scale per group per token)

    Returns full: [BLOCK_N, HEAD_DIM] fp32
    where HEAD_DIM = HALF_HD * 2.

    Layout: packed byte i holds element 2*i (low nibble) and 2*i+1 (high).
    Group 0: elements 0..31 = packed bytes 0..15
    Group 1: elements 32..63 = packed bytes 16..31
    """
    low = ((packed & 0x0F).to(tl.int8) - 8).to(tl.float32)
    high = (((packed >> 4) & 0x0F).to(tl.int8) - 8).to(tl.float32)

    # Build per-element scale: [BLOCK_N, HALF_HD]
    # packed byte j → group 0 if j < HALF_HD//2, else group 1
    # HALF_HD//2 = 16 for HEAD_DIM=64
    d_idx = tl.arange(0, HALF_HD)
    # group_mask: 1.0 for group 1, 0.0 for group 0
    group_mask = (d_idx >= (HALF_HD // 2)).to(tl.float32)  # [HALF_HD]
    # scale_per_byte[n, j] = s0[n]*(1-mask[j]) + s1[n]*mask[j]
    s0 = scale_g0.to(tl.float32)  # [BLOCK_N]
    s1 = scale_g1.to(tl.float32)  # [BLOCK_N]
    scale_per_byte = (
        s0[:, None] * (1.0 - group_mask[None, :]) + s1[:, None] * group_mask[None, :]
    )
    # [BLOCK_N, HALF_HD]

    low = low * scale_per_byte
    high = high * scale_per_byte
    return low, high


@triton.jit
def _fused_int4_attn_kernel(
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
    BLOCK_N: tl.constexpr,
):
    """Fused INT4 FlashAttention for single-token decode.

    Each program handles one (batch, query_head) pair.
    M=1 (single query token). Iterates over KV in tiles of BLOCK_N.

    Avoids all tensor slicing — uses broadcast masks for group
    scales and explicit index arithmetic for even/odd interleaving.
    """
    pid_bh = tl.program_id(0)
    batch_id = pid_bh // (n_kv_heads * n_rep)
    head_id = pid_bh % (n_kv_heads * n_rep)
    kv_head_id = head_id // n_rep

    # Load Q: [HEAD_DIM]
    d_offs = tl.arange(0, HEAD_DIM)
    q_ptrs = (
        Q_ptr
        + batch_id * stride_qb
        + head_id * stride_qh
        + 0 * stride_qm
        + d_offs * stride_qk
    )
    q = tl.load(q_ptrs).to(tl.float32)

    # Precompute Q split into even and odd elements
    # Q layout: q[0], q[1], q[2], ..., q[63]
    # Packed byte j holds elements 2j (low) and 2j+1 (high)
    # So for dot product: sum_j q[2j]*k_low[j] + q[2j+1]*k_high[j]
    even_offs = tl.arange(0, HALF_HD) * 2  # [0,2,4,...,62]
    odd_offs = tl.arange(0, HALF_HD) * 2 + 1  # [1,3,5,...,63]
    q_even_ptrs = (
        Q_ptr
        + batch_id * stride_qb
        + head_id * stride_qh
        + 0 * stride_qm
        + even_offs * stride_qk
    )
    q_odd_ptrs = (
        Q_ptr
        + batch_id * stride_qb
        + head_id * stride_qh
        + 0 * stride_qm
        + odd_offs * stride_qk
    )
    q_even = tl.load(q_even_ptrs).to(tl.float32)  # [HALF_HD]
    q_odd = tl.load(q_odd_ptrs).to(tl.float32)  # [HALF_HD]

    qk_scale = 1.0 / tl.sqrt(HEAD_DIM * 1.0)

    # Online softmax accumulators
    m_i = float("-inf")
    l_i = 0.0
    # Accumulate output in even/odd halves for easier PV scatter
    acc_even = tl.zeros([HALF_HD], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_HD], dtype=tl.float32)

    packed_d_offs = tl.arange(0, HALF_HD)

    for start_n in range(0, T_kv, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < T_kv

        # Load packed K: [BLOCK_N, HALF_HD]
        k_packed_ptrs = (
            K_packed_ptr
            + batch_id * stride_kb
            + kv_head_id * stride_kn
            + n_offs[:, None] * stride_kt
            + packed_d_offs[None, :] * stride_kd
        )
        k_packed = tl.load(k_packed_ptrs, mask=n_mask[:, None], other=0).to(tl.uint8)

        # Load K scales
        sk_ptrs_g0 = (
            Scale_k_ptr
            + batch_id * stride_skb
            + kv_head_id * stride_skn
            + n_offs * stride_skt
            + 0 * stride_skg
        )
        sk_ptrs_g1 = (
            Scale_k_ptr
            + batch_id * stride_skb
            + kv_head_id * stride_skn
            + n_offs * stride_skt
            + 1 * stride_skg
        )
        sk_g0 = tl.load(sk_ptrs_g0, mask=n_mask, other=1.0)
        sk_g1 = tl.load(sk_ptrs_g1, mask=n_mask, other=1.0)

        # Unpack + scale K
        k_low, k_high = _unpack_and_scale_int4(k_packed, sk_g0, sk_g1, HALF_HD)
        # k_low, k_high: [BLOCK_N, HALF_HD] fp32

        # QK^T: qk[n] = sum_j q_even[j]*k_low[n,j] + q_odd[j]*k_high[n,j]
        qk = tl.sum(q_even[None, :] * k_low, axis=1) + tl.sum(
            q_odd[None, :] * k_high, axis=1
        )  # [BLOCK_N]

        qk = qk * qk_scale
        qk = tl.where(n_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)  # [BLOCK_N]

        l_i = l_i * alpha + tl.sum(p)
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # Load and unpack V
        v_packed_ptrs = (
            V_packed_ptr
            + batch_id * stride_vb
            + kv_head_id * stride_vn
            + n_offs[:, None] * stride_vt
            + packed_d_offs[None, :] * stride_vd
        )
        v_packed = tl.load(v_packed_ptrs, mask=n_mask[:, None], other=0).to(tl.uint8)

        sv_ptrs_g0 = (
            Scale_v_ptr
            + batch_id * stride_svb
            + kv_head_id * stride_svn
            + n_offs * stride_svt
            + 0 * stride_svg
        )
        sv_ptrs_g1 = (
            Scale_v_ptr
            + batch_id * stride_svb
            + kv_head_id * stride_svn
            + n_offs * stride_svt
            + 1 * stride_svg
        )
        sv_g0 = tl.load(sv_ptrs_g0, mask=n_mask, other=1.0)
        sv_g1 = tl.load(sv_ptrs_g1, mask=n_mask, other=1.0)

        v_low, v_high = _unpack_and_scale_int4(v_packed, sv_g0, sv_g1, HALF_HD)
        # v_low, v_high: [BLOCK_N, HALF_HD] fp32

        # PV accumulation: acc[2j] += p . v_low[:,j], acc[2j+1] += p . v_high[:,j]
        acc_even += tl.sum(p[:, None] * v_low, axis=0)  # [HALF_HD]
        acc_odd += tl.sum(p[:, None] * v_high, axis=0)  # [HALF_HD]

        m_i = m_new

    # Normalize
    acc_even = acc_even / l_i
    acc_odd = acc_odd / l_i

    # Store output — interleave even and odd back into HEAD_DIM
    even_out_ptrs = (
        Out_ptr
        + batch_id * stride_ob
        + head_id * stride_oh
        + 0 * stride_om
        + even_offs * stride_ok
    )
    odd_out_ptrs = (
        Out_ptr
        + batch_id * stride_ob
        + head_id * stride_oh
        + 0 * stride_om
        + odd_offs * stride_ok
    )
    tl.store(even_out_ptrs, acc_even.to(tl.float16))
    tl.store(odd_out_ptrs, acc_odd.to(tl.float16))


def pipeline_B(Q, K_packed, V_packed, scale_k, scale_v):
    """Fused INT4 attention via Triton kernel."""
    B_dim, n_heads_q, M, hd = Q.shape
    _, n_kv, T_kv, half_hd = K_packed.shape
    n_rep = n_heads_q // n_kv

    assert M == 1, "Fused kernel only supports single-token decode (M=1)"

    out = torch.empty_like(Q)
    BLOCK_N = min(64, T_kv)

    grid = (B_dim * n_heads_q,)

    _fused_int4_attn_kernel[grid](
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
        BLOCK_N=BLOCK_N,
    )
    return out


# ============================================================
# Correctness validation
# ============================================================
def validate_correctness():
    """Verify Pipeline B matches Pipeline A."""
    print("Validating correctness...")
    device = "cuda"

    for B_dim in [1, 2]:
        for T_kv in [64, 256, 1024]:
            K_fp16 = torch.randn(
                B_dim,
                N_KV_HEADS,
                T_kv,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )
            V_fp16 = torch.randn(
                B_dim,
                N_KV_HEADS,
                T_kv,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )
            Q = torch.randn(
                B_dim,
                N_HEADS,
                1,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )

            K_packed, scale_k = quantize_and_pack_int4(K_fp16)
            V_packed, scale_v = quantize_and_pack_int4(V_fp16)

            out_a = pipeline_A(Q, K_packed, V_packed, scale_k, scale_v)
            out_b = pipeline_B(Q, K_packed, V_packed, scale_k, scale_v)
            torch.cuda.synchronize()

            diff = (out_a - out_b).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            # Relative error
            rel_err = (diff / (out_a.abs() + 1e-8)).max().item()
            print(
                f"  B={B_dim} T={T_kv}: "
                f"max_err={max_err:.6f} mean_err={mean_err:.8f} "
                f"rel_err={rel_err:.6f}"
            )

            del K_fp16, V_fp16, Q, K_packed, V_packed, scale_k, scale_v
            del out_a, out_b
            torch.cuda.empty_cache()

    print("  Correctness validation complete.")


# ============================================================
# Memory accounting
# ============================================================
def compute_memory_metrics(B_dim, T_kv):
    """Compute KV memory traffic for both pipelines."""
    hd = HEAD_DIM
    ng = hd // GROUP_SIZE

    dense_bytes_per_tok = 2 * N_KV_HEADS * hd * 2
    int4_packed_per_tok = 2 * N_KV_HEADS * (hd // 2)
    int4_scale_per_tok = 2 * N_KV_HEADS * ng * 2
    int4_total_per_tok = int4_packed_per_tok + int4_scale_per_tok

    # Pipeline A: read packed + write fp16 intermediate + read fp16 for attn
    pa_total = T_kv * (int4_total_per_tok + dense_bytes_per_tok + dense_bytes_per_tok)
    # Pipeline B: read packed only (no intermediate)
    pb_total = T_kv * int4_total_per_tok

    return {
        "B": B_dim,
        "T_kv": T_kv,
        "dense_kv_bytes_per_token": dense_bytes_per_tok,
        "int4_kv_bytes_per_token": int4_total_per_tok,
        "pipeline_a_traffic_bytes": pa_total * B_dim,
        "pipeline_b_traffic_bytes": pb_total * B_dim,
        "traffic_reduction_pct": round((1 - pb_total / pa_total) * 100, 1),
        "int4_cache_bytes_total": T_kv * int4_total_per_tok * B_dim,
        "fp16_cache_bytes_total": T_kv * dense_bytes_per_tok * B_dim,
    }


# ============================================================
# Benchmark
# ============================================================
def benchmark_pipeline(fn, Q, K_packed, V_packed, scale_k, scale_v):
    """Benchmark a pipeline, return timing stats."""
    for _ in range(N_WARMUP):
        _ = fn(Q, K_packed, V_packed, scale_k, scale_v)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn(Q, K_packed, V_packed, scale_k, scale_v)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "mean_ms": round(sum(times) / len(times), 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "std_ms": round(
            (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
            ** 0.5,
            4,
        ),
        "times_ms": [round(t, 4) for t in times],
    }


def run_ab_benchmark():
    """Run the full A/B benchmark."""
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Config: n_heads={N_HEADS}, n_kv={N_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"L_SET={L_SET}, B_SET={B_SET}")
    print(f"Warmup={N_WARMUP}, Repeats={N_REPEATS}\n")

    results = []
    memory_results = []

    for T_kv in L_SET:
        for B_dim in B_SET:
            print(f"  B={B_dim} L={T_kv}...", end="", flush=True)

            K_fp16 = torch.randn(
                B_dim,
                N_KV_HEADS,
                T_kv,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )
            V_fp16 = torch.randn(
                B_dim,
                N_KV_HEADS,
                T_kv,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )
            Q = torch.randn(
                B_dim,
                N_HEADS,
                1,
                HEAD_DIM,
                device=device,
                dtype=torch.float16,
            )

            K_packed, scale_k = quantize_and_pack_int4(K_fp16)
            V_packed, scale_v = quantize_and_pack_int4(V_fp16)
            del K_fp16, V_fp16
            torch.cuda.empty_cache()

            # Benchmark A
            torch.cuda.reset_peak_memory_stats()
            stats_a = benchmark_pipeline(
                pipeline_A, Q, K_packed, V_packed, scale_k, scale_v
            )
            peak_a = torch.cuda.max_memory_allocated()

            # Benchmark B
            torch.cuda.reset_peak_memory_stats()
            stats_b = benchmark_pipeline(
                pipeline_B, Q, K_packed, V_packed, scale_k, scale_v
            )
            peak_b = torch.cuda.max_memory_allocated()

            mem = compute_memory_metrics(B_dim, T_kv)

            speedup = (
                stats_a["mean_ms"] / stats_b["mean_ms"] if stats_b["mean_ms"] > 0 else 0
            )
            lat_imp = (1 - stats_b["mean_ms"] / stats_a["mean_ms"]) * 100
            tp_a = 1000 / stats_a["mean_ms"] * B_dim
            tp_b = 1000 / stats_b["mean_ms"] * B_dim
            tp_imp = (tp_b - tp_a) / tp_a * 100

            row = {
                "B": B_dim,
                "T_kv": T_kv,
                "pipeline_a_mean_ms": stats_a["mean_ms"],
                "pipeline_a_std_ms": stats_a["std_ms"],
                "pipeline_b_mean_ms": stats_b["mean_ms"],
                "pipeline_b_std_ms": stats_b["std_ms"],
                "speedup": round(speedup, 3),
                "latency_improvement_pct": round(lat_imp, 2),
                "pipeline_a_tok_per_sec": round(tp_a, 1),
                "pipeline_b_tok_per_sec": round(tp_b, 1),
                "throughput_improvement_pct": round(tp_imp, 2),
                "peak_mem_a_mb": round(peak_a / 1e6, 1),
                "peak_mem_b_mb": round(peak_b / 1e6, 1),
                "traffic_reduction_pct": mem["traffic_reduction_pct"],
                "pipeline_a_times": stats_a["times_ms"],
                "pipeline_b_times": stats_b["times_ms"],
            }
            results.append(row)
            memory_results.append(mem)

            print(
                f" A={stats_a['mean_ms']:.3f}ms  "
                f"B={stats_b['mean_ms']:.3f}ms  "
                f"speedup={speedup:.2f}x  "
                f"lat={lat_imp:+.1f}%"
            )

            del Q, K_packed, V_packed, scale_k, scale_v
            torch.cuda.empty_cache()
            gc.collect()

    return results, memory_results


# ============================================================
# Plot generation
# ============================================================
def generate_plots(results):
    """Generate benchmark plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Latency vs L (for each B)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, B_dim in enumerate(B_SET):
        ax = axes[i]
        subset = [r for r in results if r["B"] == B_dim]
        Ls = [r["T_kv"] for r in subset]
        a_ms = [r["pipeline_a_mean_ms"] for r in subset]
        b_ms = [r["pipeline_b_mean_ms"] for r in subset]
        ax.plot(Ls, a_ms, "o-", label="A: dequant+SDPA", color="tab:blue")
        ax.plot(Ls, b_ms, "s-", label="B: fused INT4", color="tab:orange")
        ax.set_xlabel("Context Length (T)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Batch={B_dim}")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Latency vs Context Length", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_vs_L.png"), dpi=150)
    plt.close()

    # 2. Throughput vs Batch (for each L)
    fig, axes = plt.subplots(1, len(L_SET), figsize=(5 * len(L_SET), 5))
    if len(L_SET) == 1:
        axes = [axes]
    for i, L in enumerate(L_SET):
        ax = axes[i]
        subset = [r for r in results if r["T_kv"] == L]
        Bs = [r["B"] for r in subset]
        a_tp = [r["pipeline_a_tok_per_sec"] for r in subset]
        b_tp = [r["pipeline_b_tok_per_sec"] for r in subset]
        x = range(len(Bs))
        w = 0.35
        ax.bar(
            [xi - w / 2 for xi in x], a_tp, w, label="A: dequant+SDPA", color="tab:blue"
        )
        ax.bar(
            [xi + w / 2 for xi in x], b_tp, w, label="B: fused INT4", color="tab:orange"
        )
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Tokens/sec")
        ax.set_title(f"L={L}")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(b) for b in Bs])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Throughput vs Batch Size", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "throughput_vs_batch.png"), dpi=150)
    plt.close()

    # 3. KV bytes/token comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    dense = 2 * N_KV_HEADS * HEAD_DIM * 2
    int4_packed = 2 * N_KV_HEADS * (HEAD_DIM // 2)
    int4_scales = 2 * N_KV_HEADS * (HEAD_DIM // GROUP_SIZE) * 2
    int4_total = int4_packed + int4_scales

    categories = [
        "Dense FP16",
        "INT4 packed\n(stored)",
        "Pipeline A\n(read traffic)",
        "Pipeline B\n(read traffic)",
    ]
    # Pipeline A reads: packed + writes fp16 + reads fp16
    pa_traffic = int4_total + dense + dense
    pb_traffic = int4_total
    values = [dense, int4_total, pa_traffic, pb_traffic]
    colors = ["tab:gray", "tab:green", "tab:blue", "tab:orange"]
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel("Bytes per Token per Layer")
    ax.set_title("KV Memory Traffic per Decode Token")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "kv_bytes_per_token.png"), dpi=150)
    plt.close()

    print("  Plots saved to artifacts/v29_flash/")


# ============================================================
# Main
# ============================================================
def main():
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("BPA v29: Fused INT4 FlashAttention A/B Test")
    print(f"GPU: {gpu_name}")
    print("=" * 60)

    validate_correctness()

    results, memory_results = run_ab_benchmark()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "bench_results.csv")
    csv_fields = [
        "B",
        "T_kv",
        "pipeline_a_mean_ms",
        "pipeline_a_std_ms",
        "pipeline_b_mean_ms",
        "pipeline_b_std_ms",
        "speedup",
        "latency_improvement_pct",
        "pipeline_a_tok_per_sec",
        "pipeline_b_tok_per_sec",
        "throughput_improvement_pct",
        "peak_mem_a_mb",
        "peak_mem_b_mb",
        "traffic_reduction_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in csv_fields})
    print(f"\n  Saved: {csv_path}")

    # JSON
    json_path = os.path.join(RESULTS_DIR, "kernel_bench.json")
    decision = {
        "latency_5pct": any(r["latency_improvement_pct"] >= 5 for r in results),
        "throughput_10pct": any(r["throughput_improvement_pct"] >= 10 for r in results),
        "memory_traffic_20pct": any(r["traffic_reduction_pct"] >= 20 for r in results),
    }
    worth_it = any(decision.values())

    output = {
        "version": "v29",
        "experiment": "fused_int4_flash_attention_ab_test",
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "model_config": {
            "n_heads": N_HEADS,
            "n_kv_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "group_size": GROUP_SIZE,
        },
        "benchmark_config": {
            "L_set": L_SET,
            "B_set": B_SET,
            "n_warmup": N_WARMUP,
            "n_repeats": N_REPEATS,
        },
        "results": results,
        "memory_accounting": memory_results,
        "decision_criteria": decision,
        "conclusion": (
            "Fused INT4 FlashAttention IS worth implementing."
            if worth_it
            else "Fused INT4 FlashAttention NOT worth implementing for this hardware regime."
        ),
        "timestamp": datetime.now().isoformat(),
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path}")

    generate_plots(results)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(
        f"{'B':>3} {'T_kv':>6} {'A (ms)':>10} {'B (ms)':>10} "
        f"{'speedup':>8} {'lat%':>7} {'tp%':>7} {'traffic%':>9}"
    )
    for r in results:
        print(
            f"{r['B']:3d} {r['T_kv']:6d} "
            f"{r['pipeline_a_mean_ms']:10.3f} {r['pipeline_b_mean_ms']:10.3f} "
            f"{r['speedup']:8.3f} {r['latency_improvement_pct']:+6.1f}% "
            f"{r['throughput_improvement_pct']:+6.1f}% "
            f"{r['traffic_reduction_pct']:+8.1f}%"
        )

    print(f"\nDecision criteria:")
    for k, v in decision.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\nConclusion: {output['conclusion']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
