# SPDX-License-Identifier: GPL-2.0
"""Benchmark harness for fused routed attention kernel.

Measures:
- Fused kernel latency vs reference (Python loop) at various K values
- Throughput in tokens/s
- Speedup ratio

Configurations target Marin 8B architecture:
    n_heads=32, n_kv_heads=8, head_dim=128, block_size=16/128/256
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from routing.fused_routed_attention import (
    fused_routed_decode,
    reference_routed_decode,
)


def benchmark_fn(fn, *args, warmup=10, iters=100, **kwargs):
    """Benchmark a function, return median latency in ms."""
    # Warmup
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


def run_benchmark(
    batch=1,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    n_total_blocks=256,
    block_size=16,
    k_values=(1, 2, 4, 8, 16),
    dtype=torch.float16,
    iters=200,
):
    """Run benchmark across K values, return results dict."""
    device = "cuda"
    results = []

    q = torch.randn(batch, n_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)

    for K in k_values:
        actual_k = min(K, n_total_blocks)

        block_tables = torch.zeros(batch, n_kv_heads, actual_k, dtype=torch.int64, device=device)
        block_counts = torch.zeros(batch, n_kv_heads, dtype=torch.int32, device=device)

        for b in range(batch):
            for h in range(n_kv_heads):
                perm = torch.randperm(n_total_blocks, device=device)[:actual_k]
                block_tables[b, h, :actual_k] = perm
                block_counts[b, h] = actual_k

        # Fused kernel
        fused_stats = benchmark_fn(
            fused_routed_decode, q, k_cache, v_cache, block_tables, block_counts,
            warmup=20, iters=iters,
        )

        # Reference (Python loop)
        ref_stats = benchmark_fn(
            reference_routed_decode, q, k_cache, v_cache, block_tables, block_counts,
            warmup=5, iters=min(iters, 50),  # reference is slow
        )

        tokens_per_head = actual_k * block_size
        total_kv_tokens = n_total_blocks * block_size
        kv_reduction = 1.0 - tokens_per_head / total_kv_tokens

        result = {
            "K": K,
            "actual_k": actual_k,
            "block_size": block_size,
            "tokens_per_head": tokens_per_head,
            "total_kv_tokens": total_kv_tokens,
            "kv_reduction_pct": kv_reduction * 100,
            "fused": fused_stats,
            "reference": ref_stats,
            "speedup": ref_stats["median_ms"] / max(fused_stats["median_ms"], 1e-6),
        }
        results.append(result)

        print(f"  K={K:3d} (bs={block_size:3d}): "
              f"fused={fused_stats['median_ms']:.3f}ms  "
              f"ref={ref_stats['median_ms']:.3f}ms  "
              f"speedup={result['speedup']:.1f}x  "
              f"KV_red={kv_reduction*100:.1f}%")

    return results


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}")
    print()

    # Marin 8B config: 32 layers, 32 QH, 8 KVH, head_dim=128
    # Context 4096 tokens
    configs = [
        {"block_size": 16, "n_total_blocks": 256, "label": "BS=16 (vLLM default)"},
        {"block_size": 128, "n_total_blocks": 32, "label": "BS=128 (coarse)"},
        {"block_size": 256, "n_total_blocks": 16, "label": "BS=256 (very coarse)"},
    ]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "model_config": "Marin-8B-like (32 QH, 8 KVH, head_dim=128)",
        "context_tokens": 4096,
        "configs": {},
    }

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg['label']}")
        print(f"  n_total_blocks={cfg['n_total_blocks']}, block_size={cfg['block_size']}")
        print(f"{'='*60}")

        results = run_benchmark(
            n_total_blocks=cfg["n_total_blocks"],
            block_size=cfg["block_size"],
            k_values=[1, 2, 4, 8],
        )
        all_results["configs"][cfg["label"]] = results

    # Also test dense baseline (all blocks, no routing)
    print(f"\n{'='*60}")
    print("Dense baseline (all blocks, single flash-like call)")
    print(f"{'='*60}")
    for cfg in configs:
        results = run_benchmark(
            n_total_blocks=cfg["n_total_blocks"],
            block_size=cfg["block_size"],
            k_values=[cfg["n_total_blocks"]],  # all blocks = dense
            iters=200,
        )
        all_results["configs"][f"dense_{cfg['label']}"] = results

    # Write results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(f"routing_kernel_benchmark_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
