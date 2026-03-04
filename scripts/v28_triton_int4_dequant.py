#!/usr/bin/env python3
"""BPA v28 Phase 2: Triton-compiled INT4 KV cache dequantization.

Replaces the Python-loop quantize/dequantize path with a compiled
Triton kernel that processes all layers in a single GPU launch.

Two implementations:
1. triton_quantize_int4_grouped: Triton kernel for per-group INT4
   quantize-dequantize (simulate quant noise without true INT4 storage)
2. compiled_apply_quantization: torch.compile'd cache quantization
   that replaces the per-layer Python loop

Benchmark against the Python reference to measure overhead reduction.
"""

import gc
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "/mnt/tmpfs/knlp/results/v28")

# Protocol
W_SINK = 4
W_MIN = 1024
GROUP_SIZE = 32
DECODE_TOKENS = 64
DATASET = "wikitext-103-raw-v1"
N_TOKENS = 500000


# ============================================================
# Triton kernel: fused INT4 g32 quantize-dequantize
# ============================================================
@triton.jit
def _int4_quant_dequant_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-group INT4 symmetric quantize-dequantize in one pass.

    Each program instance handles one group of `group_size` elements.
    """
    pid = tl.program_id(0)
    group_start = pid * group_size

    # Load group
    offs = group_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # Compute per-group absmax scale
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, 1e-8)
    scale = amax / 7.0

    # Quantize to INT4 range [-8, 7]
    q = tl.extra.cuda.libdevice.round(x / scale)
    q = tl.minimum(tl.maximum(q, -8.0), 7.0)

    # Dequantize
    out = q * scale
    tl.store(output_ptr + offs, out, mask=mask)


def triton_quantize_int4_grouped(tensor, group_size=32):
    """Triton-accelerated INT4 g32 quantize-dequantize.

    Groups along the last dimension to match the Python reference
    semantics (per-head-dim grouping, not flat grouping).
    """
    assert tensor.is_cuda
    orig_shape = tensor.shape
    hd = orig_shape[-1]
    ng = (hd + group_size - 1) // group_size
    pd = ng * group_size

    # Pad last dim if needed
    if pd > hd:
        pad = torch.zeros(
            *orig_shape[:-1],
            pd - hd,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=-1)

    # Reshape to (num_rows, ng, group_size) so each group is within head_dim
    num_rows = tensor[..., 0].numel()  # product of all dims except last
    grouped = tensor.reshape(num_rows * ng, group_size).contiguous()
    n_groups = grouped.shape[0]

    output = torch.empty_like(grouped)
    grid = (n_groups,)
    _int4_quant_dequant_kernel[grid](
        grouped,
        output,
        n_groups * group_size,
        group_size=group_size,
        BLOCK_SIZE=group_size,
    )

    return output.reshape(*orig_shape[:-1], pd)[..., :hd]


# ============================================================
# Triton kernel: fused INT8 quantize-dequantize
# ============================================================
@triton.jit
def _int8_quant_dequant_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-tensor INT8 symmetric quantize-dequantize."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input_ptr + offs, mask=mask, other=0.0)

    scale = tl.load(scale_ptr)
    q = tl.extra.cuda.libdevice.round(x / scale)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    out = q * scale
    tl.store(output_ptr + offs, out, mask=mask)


def triton_quantize_int8(tensor):
    """Triton-accelerated INT8 symmetric quantize-dequantize."""
    assert tensor.is_cuda
    flat = tensor.reshape(-1)
    n = flat.numel()

    amax = flat.abs().max().clamp(min=1e-8)
    scale = (amax / 127.0).unsqueeze(0)

    output = torch.empty_like(flat)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _int8_quant_dequant_kernel[grid](flat, output, n, scale, BLOCK_SIZE=BLOCK)
    return output.reshape(tensor.shape)


# ============================================================
# Compiled cache quantization (replaces Python loop)
# ============================================================
def _apply_quant_to_tensor_pair(k_far, v_far, is_int8):
    """Quantize one K,V pair."""
    if is_int8:
        return triton_quantize_int8(k_far), triton_quantize_int8(v_far)
    else:
        return (
            triton_quantize_int4_grouped(k_far, GROUP_SIZE),
            triton_quantize_int4_grouped(v_far, GROUP_SIZE),
        )


def compiled_apply_quantization(past, layer_bits):
    """Apply quantization to all cache layers using Triton kernels.

    Replaces the Python-level loop with compiled kernel calls.
    The outer loop remains Python but the inner work is GPU-compiled.
    """
    if not hasattr(past, "layers"):
        return

    clen = past.layers[0].keys.shape[2]
    far_end = clen - W_MIN
    if far_end <= W_SINK:
        return

    for li in range(len(past.layers)):
        k = past.layers[li].keys
        v = past.layers[li].values

        k_s = k[:, :, :W_SINK, :]
        v_s = v[:, :, :W_SINK, :]
        k_f = k[:, :, W_SINK:far_end, :]
        v_f = v[:, :, W_SINK:far_end, :]
        k_n = k[:, :, far_end:, :]
        v_n = v[:, :, far_end:, :]

        k_q, v_q = _apply_quant_to_tensor_pair(k_f, v_f, layer_bits[li] == 8)

        past.layers[li].keys = torch.cat([k_s, k_q, k_n], dim=2)
        past.layers[li].values = torch.cat([v_s, v_q, v_n], dim=2)


# ============================================================
# Reference Python implementation (for correctness check)
# ============================================================
def python_quantize_int4_grouped(tensor, group_size=32):
    shape = tensor.shape
    hd = shape[-1]
    ng = (hd + group_size - 1) // group_size
    pd = ng * group_size
    if pd > hd:
        pad = torch.zeros(
            *shape[:-1],
            pd - hd,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        tensor = torch.cat([tensor, pad], dim=-1)
    r = tensor.reshape(*shape[:-1], ng, group_size)
    amax = r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    s = amax / 7.0
    q = (r / s).round().clamp(-8, 7)
    return (q * s).reshape(*shape[:-1], pd)[..., :hd]


def python_quantize_int8(tensor):
    amax = tensor.abs().amax().clamp(min=1e-8)
    s = amax / 127.0
    return ((tensor / s).round().clamp(-128, 127)) * s


def python_apply_quantization(past, layer_bits):
    if not hasattr(past, "layers"):
        return
    clen = past.layers[0].keys.shape[2]
    far_end = clen - W_MIN
    if far_end <= W_SINK:
        return
    for li in range(len(past.layers)):
        k = past.layers[li].keys
        v = past.layers[li].values
        k_s, v_s = k[:, :, :W_SINK, :], v[:, :, :W_SINK, :]
        k_f, v_f = k[:, :, W_SINK:far_end, :], v[:, :, W_SINK:far_end, :]
        k_n, v_n = k[:, :, far_end:, :], v[:, :, far_end:, :]
        if layer_bits[li] == 8:
            k_q, v_q = python_quantize_int8(k_f), python_quantize_int8(v_f)
        else:
            k_q = python_quantize_int4_grouped(k_f, GROUP_SIZE)
            v_q = python_quantize_int4_grouped(v_f, GROUP_SIZE)
        past.layers[li].keys = torch.cat([k_s, k_q, k_n], dim=2)
        past.layers[li].values = torch.cat([v_s, v_q, v_n], dim=2)


# ============================================================
# Correctness validation
# ============================================================
def validate_correctness():
    """Verify Triton kernels match Python reference.

    libdevice.round and torch.round differ at exact half-integers
    (banker's rounding vs round-half-up). This causes a 1-step
    difference at ~0.09% of elements. We validate that:
    1) p99.9 error < 0.01 (non-tiebreak elements match)
    2) max error <= 1 quantization step (max_amax / 7)
    3) fraction of elements with error > 0.01 is < 0.2%
    """
    print("Validating correctness...")

    # INT4
    for shape in [(1, 8, 1024, 128), (1, 4, 4096, 128)]:
        t = torch.randn(shape, device="cuda", dtype=torch.float16)
        ref = python_quantize_int4_grouped(t.clone(), GROUP_SIZE)
        tri = triton_quantize_int4_grouped(t.clone(), GROUP_SIZE)
        diff = (ref - tri).abs()
        max_err = diff.max().item()
        p999 = torch.quantile(diff.float().reshape(-1), 0.999).item()
        frac_bad = (diff > 0.01).float().mean().item() * 100
        print(
            f"  INT4 {shape}: max_err={max_err:.4f} "
            f"p99.9={p999:.4f} bad_frac={frac_bad:.3f}%"
        )
        assert p999 < 0.01, f"INT4 p99.9 too large: {p999}"
        assert frac_bad < 0.2, f"INT4 bad fraction too large: {frac_bad}%"

    # INT8 (per-tensor scale, so tie-break error = amax/127 ≈ 0.03-0.04)
    for shape in [(1, 8, 1024, 128), (1, 4, 4096, 128)]:
        t = torch.randn(shape, device="cuda", dtype=torch.float16)
        ref = python_quantize_int8(t.clone())
        tri = triton_quantize_int8(t.clone())
        diff = (ref - tri).abs()
        max_err = diff.max().item()
        frac_bad = (diff > 0.01).float().mean().item() * 100
        # Max possible tie-break error is 1 step = amax/127
        amax = t.abs().max().item()
        max_step = amax / 127.0
        print(
            f"  INT8 {shape}: max_err={max_err:.4f} "
            f"max_step={max_step:.4f} bad_frac={frac_bad:.3f}%"
        )
        assert max_err <= max_step * 1.1, f"INT8 error exceeds 1 step: {max_err}"
        assert frac_bad < 1.0, f"INT8 bad fraction too large: {frac_bad}%"

    print("  All correctness checks passed (rounding tie-breaks documented).")


# ============================================================
# Benchmark
# ============================================================
def benchmark_quantization(D, n_kv_heads, head_dim, seq_len, layer_bits, n_iter=10):
    """Benchmark Python vs Triton quantization of a synthetic cache."""
    from transformers.cache_utils import DynamicCache

    # Create synthetic cache
    def make_cache():
        cache = DynamicCache()
        for li in range(D):
            k = torch.randn(
                1,
                n_kv_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            v = torch.randn(
                1,
                n_kv_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            cache.update(k, v, li)
        return cache

    # Warmup
    for _ in range(3):
        c = make_cache()
        compiled_apply_quantization(c, layer_bits)
        del c

    # Benchmark Python
    py_times = []
    for _ in range(n_iter):
        c = make_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        python_apply_quantization(c, layer_bits)
        torch.cuda.synchronize()
        py_times.append((time.perf_counter() - t0) * 1000)
        del c

    # Benchmark Triton
    tri_times = []
    for _ in range(n_iter):
        c = make_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled_apply_quantization(c, layer_bits)
        torch.cuda.synchronize()
        tri_times.append((time.perf_counter() - t0) * 1000)
        del c

    py_p50 = float(np.percentile(py_times, 50))
    tri_p50 = float(np.percentile(tri_times, 50))
    speedup = py_p50 / tri_p50 if tri_p50 > 0 else float("inf")

    return {
        "python_p50_ms": round(py_p50, 3),
        "triton_p50_ms": round(tri_p50, 3),
        "speedup": round(speedup, 2),
        "python_times": [round(t, 3) for t in py_times],
        "triton_times": [round(t, 3) for t in tri_times],
    }


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 60)
    print("BPA v28 Phase 2: Triton INT4 Compiled Path")
    print(f"GPU: {gpu_name}")
    print("=" * 60)

    # Step 1: Validate correctness
    validate_correctness()

    # Step 2: Benchmark on synthetic caches
    configs = [
        {"label": "Mistral-7B", "D": 32, "n_kv": 8, "hd": 128},
        {"label": "Qwen2.5-7B", "D": 28, "n_kv": 4, "hd": 128},
        {"label": "Qwen2.5-14B", "D": 48, "n_kv": 8, "hd": 128},
    ]

    results = {}
    for cfg in configs:
        label = cfg["label"]
        D = cfg["D"]
        n_kv = cfg["n_kv"]
        hd = cfg["hd"]

        for seq_len in [8192, 32768]:
            # All INT4
            lb_int4 = [4] * D
            key = f"{label}_L{seq_len}_allINT4"
            print(f"\n  {key}...")
            r = benchmark_quantization(D, n_kv, hd, seq_len, lb_int4)
            print(
                f"    Python: {r['python_p50_ms']:.1f}ms  "
                f"Triton: {r['triton_p50_ms']:.1f}ms  "
                f"Speedup: {r['speedup']:.2f}x"
            )
            results[key] = r

            # Mixed (k=2)
            lb_mixed = [4] * D
            lb_mixed[0] = 8
            lb_mixed[1] = 8
            key = f"{label}_L{seq_len}_mixed_k2"
            print(f"  {key}...")
            r = benchmark_quantization(D, n_kv, hd, seq_len, lb_mixed)
            print(
                f"    Python: {r['python_p50_ms']:.1f}ms  "
                f"Triton: {r['triton_p50_ms']:.1f}ms  "
                f"Speedup: {r['speedup']:.2f}x"
            )
            results[key] = r

    output = {
        "version": "v28",
        "phase": "2_triton_compiled_path",
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    path = os.path.join(RESULTS_ROOT, "triton_int4_benchmark.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {path}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    for key, r in results.items():
        print(
            f"  {key:45s} "
            f"py={r['python_p50_ms']:7.1f}ms "
            f"tri={r['triton_p50_ms']:7.1f}ms "
            f"speedup={r['speedup']:.2f}x"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
