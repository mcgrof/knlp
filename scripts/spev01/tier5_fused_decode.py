#!/usr/bin/env python3
"""Tier 5: Fused INT4 decode benchmark with bounded dispatch policy.

Measures the performance difference between baseline FP16 attention
(P0) and fused INT4 dequantization inside the attention kernel (P3/P5)
across a matrix of batch sizes and context lengths.  This is the core
benchmark for validating the BPA fused-KV quantization result on a
given GPU.

The bounded dispatch policy selects P0 (FP16 reference) vs fused
INT4 based on batch size and head dimension, following the rules
documented in docs/fused_kv_quantization.md.

This benchmark operates at the Triton kernel level (synthetic Q/K/V
tensors), not through the vLLM engine.  It isolates the decode
attention kernel so the results are not confounded by model loading,
scheduling, or prompt processing.

Usage:
    python3 scripts/spev01/tier5_fused_decode.py

Environment variables:
    SPEV01_OUTPUT_DIR   Override output directory (default: scripts/spev01/json/)

Output:
    JSON files per model config with P0 and fused latencies, speedup
    ratios, and the dispatch decision at each (batch, context) point.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- portability helpers ------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get(
    "SPEV01_OUTPUT_DIR",
    os.path.join(_SCRIPT_DIR, "json"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    sys.exit(
        "ERROR: tier5_fused_decode requires a CUDA GPU " "(torch.cuda unavailable)"
    )

try:
    import triton
    import triton.language as tl
except ImportError:
    sys.exit("ERROR: tier5_fused_decode requires Triton " "(pip install triton)")


def _detect_gpu_name():
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            return f"{parts[0]}-{int(parts[1])//1024}GB"
        except Exception:
            pass
    return torch.cuda.get_device_name(0)


GPU_TAG = _detect_gpu_name()


# ================================================================
# Model configs — same suite as v35 H100 bench
# ================================================================
@dataclass
class ModelConfig:
    name: str
    n_heads: int
    n_kv_heads: int
    head_dim: int
    group_size: int = 32


CONFIGS = [
    ModelConfig("qwen25_7b", n_heads=28, n_kv_heads=4, head_dim=128),
    ModelConfig("mistral_7b", n_heads=32, n_kv_heads=8, head_dim=128),
    ModelConfig("llama31_8b", n_heads=32, n_kv_heads=8, head_dim=128),
]

# Decode benchmark matrix
DECODE_LENGTHS = [2048, 4096, 8192, 16384, 32768]
DECODE_BATCHES = [1, 2, 4, 8, 16]

N_WARMUP = 5
N_REPEATS = 5


# ================================================================
# INT4 packing / unpacking
# ================================================================
def quantize_and_pack_int4(tensor, group_size=32):
    """Quantize fp16 tensor to INT4 and pack 2 values per byte."""
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


# ================================================================
# Fused INT4 decode kernel (from v35 H100 bench)
# ================================================================
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


# ================================================================
# Pipeline wrappers
# ================================================================
def _launch_fused(Q, K_packed, V_packed, scale_k, scale_v, cfg, block_n):
    """Launch the fused INT4 decode kernel."""
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
    )
    return out


def pipeline_P0(Q, K_fp16, V_fp16, cfg):
    """P0: FP16 reference — dequantized KV through SDPA."""
    n_rep = cfg.n_heads // cfg.n_kv_heads
    K_exp = K_fp16.repeat_interleave(n_rep, dim=1)
    V_exp = V_fp16.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K_exp, V_exp, is_causal=False
    )


def _select_block_n(Q, K_packed, cfg):
    """Bounded dispatch: pick BLOCK_N for the fused kernel.

    Rules from docs/fused_kv_quantization.md:
      - head_dim <= 64 or batch >= 2 or long context: BLOCK_N=128
      - batch == 1, head_dim > 64, short context:     BLOCK_N=64
    """
    B_dim = Q.shape[0]
    T_kv = K_packed.shape[2]
    if cfg.head_dim <= 64:
        return 128
    if B_dim >= 2:
        return 128
    if T_kv >= 8192:
        return 128
    return 64


def pipeline_fused(Q, K_packed, V_packed, scale_k, scale_v, cfg):
    """Fused INT4 decode with bounded dispatch (P3/P5 unified)."""
    bn = _select_block_n(Q, K_packed, cfg)
    return _launch_fused(Q, K_packed, V_packed, scale_k, scale_v, cfg, bn)


# ================================================================
# Benchmark harness
# ================================================================
def benchmark_fn(fn, args):
    """Warm up then time a kernel, return mean latency in seconds."""
    for _ in range(N_WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def estimate_mem_gb(cfg, T_kv, B_dim):
    """Estimate peak GPU memory for one benchmark point."""
    fp16_kv = B_dim * cfg.n_kv_heads * T_kv * cfg.head_dim * 2 * 2
    int4_kv = B_dim * cfg.n_kv_heads * T_kv * (cfg.head_dim // 2) * 2
    scales = B_dim * cfg.n_kv_heads * T_kv * (cfg.head_dim // cfg.group_size) * 2 * 2
    q_mem = B_dim * cfg.n_heads * cfg.head_dim * 2
    return (fp16_kv + int4_kv + scales + q_mem) / 1e9


def run_decode_bench(cfg, T_kv, B_dim):
    """Run P0 and fused decode, return latencies and dispatch info."""
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

    block_n = _select_block_n(Q, K_packed, cfg)

    p0_lat = benchmark_fn(pipeline_P0, (Q, K_fp16, V_fp16, cfg))
    fused_lat = benchmark_fn(
        pipeline_fused,
        (Q, K_packed, V_packed, scale_k, scale_v, cfg),
    )

    del K_fp16, V_fp16, Q, K_packed, V_packed, scale_k, scale_v
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "p0_latency_ms": round(p0_lat * 1000, 4),
        "fused_latency_ms": round(fused_lat * 1000, 4),
        "speedup": round(p0_lat / fused_lat, 3) if fused_lat > 0 else 0,
        "block_n": block_n,
        "dispatch": f"P{'5' if block_n == 64 else '3'}",
    }


# ================================================================
# Main
# ================================================================
def main():
    print(f"Tier 5: Fused INT4 Decode Benchmark")
    print(f"GPU: {GPU_TAG}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Models: {len(CONFIGS)}")
    print(f"Decode lengths: {DECODE_LENGTHS}")
    print(f"Batch sizes: {DECODE_BATCHES}")
    print()

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(
            f"Model: {cfg.name} (H={cfg.n_heads}, KV={cfg.n_kv_heads}, "
            f"D={cfg.head_dim})"
        )
        print(f"{'='*60}")

        model_results = []
        for T_kv in DECODE_LENGTHS:
            for B_dim in DECODE_BATCHES:
                est_gb = estimate_mem_gb(cfg, T_kv, B_dim)
                if est_gb > gpu_mem_gb * 0.85:
                    print(
                        f"  B={B_dim:3d} T={T_kv:6d}: SKIP "
                        f"(est {est_gb:.1f}GB > {gpu_mem_gb*0.85:.1f}GB)"
                    )
                    model_results.append(
                        {
                            "batch": B_dim,
                            "context_len": T_kv,
                            "status": "skipped_oom",
                            "est_mem_gb": round(est_gb, 2),
                        }
                    )
                    continue

                try:
                    result = run_decode_bench(cfg, T_kv, B_dim)
                    result["batch"] = B_dim
                    result["context_len"] = T_kv
                    result["status"] = "ok"
                    model_results.append(result)
                    print(
                        f"  B={B_dim:3d} T={T_kv:6d}: "
                        f"P0={result['p0_latency_ms']:.3f}ms "
                        f"fused={result['fused_latency_ms']:.3f}ms "
                        f"({result['speedup']:.2f}x) "
                        f"[{result['dispatch']} BN={result['block_n']}]"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    model_results.append(
                        {
                            "batch": B_dim,
                            "context_len": T_kv,
                            "status": "oom",
                        }
                    )
                    print(f"  B={B_dim:3d} T={T_kv:6d}: OOM")
                except Exception as e:
                    model_results.append(
                        {
                            "batch": B_dim,
                            "context_len": T_kv,
                            "status": "error",
                            "error": str(e)[:200],
                        }
                    )
                    print(f"  B={B_dim:3d} T={T_kv:6d}: ERROR {e}")

        # Per-model summary
        ok_results = [r for r in model_results if r["status"] == "ok"]
        speedups = [r["speedup"] for r in ok_results]
        summary = {
            "tier": "5",
            "type": "fused_decode_benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": GPU_TAG,
            "model_config": {
                "name": cfg.name,
                "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
                "head_dim": cfg.head_dim,
                "group_size": cfg.group_size,
            },
            "benchmark_params": {
                "n_warmup": N_WARMUP,
                "n_repeats": N_REPEATS,
                "decode_lengths": DECODE_LENGTHS,
                "batch_sizes": DECODE_BATCHES,
            },
            "results": model_results,
            "aggregate": {
                "n_ok": len(ok_results),
                "n_total": len(model_results),
                "mean_speedup": (
                    round(sum(speedups) / len(speedups), 3) if speedups else None
                ),
                "min_speedup": (round(min(speedups), 3) if speedups else None),
                "max_speedup": (round(max(speedups), 3) if speedups else None),
            },
        }

        out_path = os.path.join(OUTPUT_DIR, f"tier5_fused_decode_{cfg.name}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved {out_path}")
        all_results.append(summary)

    # Combined output
    combined_path = os.path.join(OUTPUT_DIR, "tier5_fused_decode_all.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary table
    print(f"\n\n{'='*60}")
    print("FUSED DECODE BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for summary in all_results:
        agg = summary["aggregate"]
        name = summary["model_config"]["name"]
        if agg["mean_speedup"] is not None:
            print(
                f"  {name:20s}  "
                f"mean={agg['mean_speedup']:.2f}x  "
                f"min={agg['min_speedup']:.2f}x  "
                f"max={agg['max_speedup']:.2f}x  "
                f"({agg['n_ok']}/{agg['n_total']} points)"
            )
        else:
            print(f"  {name:20s}  no successful benchmarks")

    print(f"\nResults saved to {combined_path}")


if __name__ == "__main__":
    main()
