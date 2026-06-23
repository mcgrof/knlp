# SPDX-License-Identifier: GPL-2.0
"""H100 optimization benchmark for fused routed attention kernel.

Implements the bounded optimization plan from:
  routing-h100-kernel-optimization-plan-final-20260401.md

Adds:
- Backend/environment metadata (torch, triton, CUDA, SDPA backend)
- SDPA dense baseline verification (FlashAttention-backed)
- 3-way comparison: Rdense / Rfused-all / Rfused-K
- Autotuned kernel comparison
- Focused on real routing operating points: BS=128 K=8, BS=256 K=8
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from routing.fused_routed_attention import (
    fused_routed_decode,
    reference_routed_decode,
)

try:
    from routing.fused_routed_attention_autotune import (
        fused_routed_decode_autotune,
    )
    HAS_AUTOTUNE = True
except Exception:
    HAS_AUTOTUNE = False


def collect_backend_info():
    """Collect environment/backend metadata for reproducibility."""
    import triton

    info = {
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "python_version": sys.version.split()[0],
    }

    # Check which SDPA backends are available
    info["sdpa_backends"] = {}
    try:
        # Test with a small tensor to determine available backends
        q = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.float16)

        for backend_name, ctx in [
            ("flash", torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False)),
            ("mem_efficient", torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True)),
            ("math", torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False)),
        ]:
            try:
                with ctx:
                    F.scaled_dot_product_attention(q, k, v)
                info["sdpa_backends"][backend_name] = True
            except Exception:
                info["sdpa_backends"][backend_name] = False
    except Exception as e:
        info["sdpa_backends"]["error"] = str(e)

    # Check flash_attn package
    try:
        import flash_attn
        info["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        info["flash_attn_version"] = "not installed"

    info["has_autotune_kernel"] = HAS_AUTOTUNE

    return info


def benchmark_fn(fn, *args, warmup=10, iters=100, **kwargs):
    """Benchmark a function, return median latency in ms."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "p90_ms": times[int(len(times) * 0.9)],
        "p99_ms": times[int(len(times) * 0.99)],
    }


def sdpa_dense_baseline(q_decode, k_cache_flat, v_cache_flat, scale):
    """Dense SDPA baseline: all tokens, FlashAttention-backed if available.

    Args:
        q_decode: [batch, n_heads, head_dim] -> reshaped to [B, H, 1, D]
        k_cache_flat: [batch, n_kv_heads, total_tokens, head_dim]
        v_cache_flat: [batch, n_kv_heads, total_tokens, head_dim]
        scale: attention scale factor
    """
    batch, n_heads, head_dim = q_decode.shape
    n_kv_heads = k_cache_flat.shape[1]
    group_size = n_heads // n_kv_heads

    # Expand KV for GQA
    if group_size > 1:
        k_exp = k_cache_flat.repeat_interleave(group_size, dim=1)
        v_exp = v_cache_flat.repeat_interleave(group_size, dim=1)
    else:
        k_exp = k_cache_flat
        v_exp = v_cache_flat

    q_4d = q_decode.unsqueeze(2)  # [B, H, 1, D]
    out = F.scaled_dot_product_attention(q_4d, k_exp, v_exp, scale=scale)
    return out.squeeze(2)  # [B, H, D]


def prepare_flat_kv(k_cache, v_cache, n_total_blocks, block_size, n_kv_heads, head_dim, batch=1):
    """Flatten paged KV cache to dense format for SDPA baseline."""
    # k_cache: [max_blocks, block_size, n_kv_heads, head_dim]
    # -> [1, n_kv_heads, total_tokens, head_dim]
    k_flat = k_cache[:n_total_blocks].permute(2, 0, 1, 3).reshape(
        n_kv_heads, n_total_blocks * block_size, head_dim
    ).unsqueeze(0).expand(batch, -1, -1, -1)
    v_flat = v_cache[:n_total_blocks].permute(2, 0, 1, 3).reshape(
        n_kv_heads, n_total_blocks * block_size, head_dim
    ).unsqueeze(0).expand(batch, -1, -1, -1)
    return k_flat.contiguous(), v_flat.contiguous()


def run_3way_benchmark(
    batch=1,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    n_total_blocks=32,
    block_size=128,
    K=8,
    dtype=torch.float16,
    iters=200,
):
    """Run the 3-way comparison at a single operating point.

    Returns: dict with Rdense, Rfused-all, Rfused-K timings.
    """
    device = "cuda"
    scale = head_dim ** -0.5

    q = torch.randn(batch, n_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)

    # Prepare dense KV for SDPA baseline
    k_flat, v_flat = prepare_flat_kv(k_cache, v_cache, n_total_blocks, block_size, n_kv_heads, head_dim, batch)

    # --- Rdense: SDPA dense baseline (FlashAttention-backed) ---
    rdense_stats = benchmark_fn(
        sdpa_dense_baseline, q, k_flat, v_flat, scale,
        warmup=20, iters=iters,
    )

    # --- Rfused-all: fused kernel with ALL blocks selected ---
    bt_all = torch.zeros(batch, n_kv_heads, n_total_blocks, dtype=torch.int64, device=device)
    bc_all = torch.zeros(batch, n_kv_heads, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(n_kv_heads):
            bt_all[b, h, :] = torch.arange(n_total_blocks, device=device)
            bc_all[b, h] = n_total_blocks

    rfused_all_stats = benchmark_fn(
        fused_routed_decode, q, k_cache, v_cache, bt_all, bc_all,
        warmup=20, iters=iters,
    )

    # --- Rfused-K: fused kernel with K selected blocks ---
    actual_k = min(K, n_total_blocks)
    bt_k = torch.zeros(batch, n_kv_heads, actual_k, dtype=torch.int64, device=device)
    bc_k = torch.zeros(batch, n_kv_heads, dtype=torch.int32, device=device)
    for b in range(batch):
        for h in range(n_kv_heads):
            perm = torch.randperm(n_total_blocks, device=device)[:actual_k]
            bt_k[b, h, :actual_k] = perm
            bc_k[b, h] = actual_k

    rfused_k_stats = benchmark_fn(
        fused_routed_decode, q, k_cache, v_cache, bt_k, bc_k,
        warmup=20, iters=iters,
    )

    # --- Autotune variant if available ---
    autotune_k_stats = None
    autotune_all_stats = None
    if HAS_AUTOTUNE:
        autotune_k_stats = benchmark_fn(
            fused_routed_decode_autotune, q, k_cache, v_cache, bt_k, bc_k,
            warmup=20, iters=iters,
        )
        autotune_all_stats = benchmark_fn(
            fused_routed_decode_autotune, q, k_cache, v_cache, bt_all, bc_all,
            warmup=20, iters=iters,
        )

    tokens_per_head = actual_k * block_size
    total_kv_tokens = n_total_blocks * block_size
    kv_reduction = 1.0 - tokens_per_head / total_kv_tokens

    result = {
        "config": {
            "batch": batch,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "n_total_blocks": n_total_blocks,
            "K": K,
            "actual_k": actual_k,
            "total_kv_tokens": total_kv_tokens,
            "tokens_per_head_routed": tokens_per_head,
            "kv_reduction_pct": kv_reduction * 100,
        },
        "Rdense_sdpa": rdense_stats,
        "Rfused_all": rfused_all_stats,
        "Rfused_K": rfused_k_stats,
    }

    if autotune_k_stats:
        result["Rfused_K_autotune"] = autotune_k_stats
        result["Rfused_all_autotune"] = autotune_all_stats

    return result


def print_3way_table(results):
    """Print the 3-way comparison table."""
    for r in results:
        cfg = r["config"]
        label = f"BS={cfg['block_size']} K={cfg['K']}"
        rd = r["Rdense_sdpa"]["median_ms"]
        ra = r["Rfused_all"]["median_ms"]
        rk = r["Rfused_K"]["median_ms"]
        kv_red = cfg["kv_reduction_pct"]

        line = (f"  {label:15s}  "
                f"Rdense={rd:.3f}ms  "
                f"Rfused-all={ra:.3f}ms  "
                f"Rfused-K={rk:.3f}ms  "
                f"KV_red={kv_red:.0f}%  "
                f"speedup_vs_dense={rd/max(rk,1e-6):.1f}x")

        if "Rfused_K_autotune" in r:
            rka = r["Rfused_K_autotune"]["median_ms"]
            improvement = (rk - rka) / max(rk, 1e-6) * 100
            line += f"  autotune={rka:.3f}ms ({improvement:+.1f}%)"

        print(line)


def main():
    backend_info = collect_backend_info()

    print("=" * 70)
    print("Routing H100 Optimization Benchmark")
    print("=" * 70)
    print(f"GPU: {backend_info['gpu_name']}")
    print(f"torch: {backend_info['torch_version']}")
    print(f"triton: {backend_info['triton_version']}")
    print(f"CUDA: {backend_info['cuda_version']}")
    print(f"SDPA backends: {backend_info['sdpa_backends']}")
    print(f"flash_attn: {backend_info['flash_attn_version']}")
    print(f"autotune kernel: {backend_info['has_autotune_kernel']}")
    print()

    # Real routing operating points only
    operating_points = [
        {"block_size": 128, "n_total_blocks": 32, "K": 8,
         "label": "BS=128 K=8 (primary candidate)"},
        {"block_size": 256, "n_total_blocks": 16, "K": 8,
         "label": "BS=256 K=8 (high-accuracy candidate)"},
        {"block_size": 256, "n_total_blocks": 16, "K": 4,
         "label": "BS=256 K=4 (latency candidate)"},
    ]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend_info": backend_info,
        "plan": "routing-h100-kernel-optimization-plan-final-20260401.md",
        "operating_points": [],
    }

    print("=" * 70)
    print("3-WAY COMPARISON: Rdense / Rfused-all / Rfused-K")
    print("=" * 70)

    results_list = []
    for op in operating_points:
        print(f"\n--- {op['label']} ---")
        r = run_3way_benchmark(
            n_total_blocks=op["n_total_blocks"],
            block_size=op["block_size"],
            K=op["K"],
        )
        results_list.append(r)
        all_results["operating_points"].append(r)

    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print_3way_table(results_list)

    # Write results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(f"routing_h100_optimization_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
