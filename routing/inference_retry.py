# SPDX-License-Identifier: GPL-2.0
"""Routing inference retry using fused Triton kernel.

Replaces the per-head Python loop from the prior inference test with the
fused kernel, and measures:
- Dense baseline latency (all blocks, fused kernel)
- Routed latency at K=1,2,4,8 for block sizes 16, 128, 256
- Cosine similarity of routed vs dense attention output
- KV-touch reduction
- TTFT comparison (fused routed vs dense)

This is the first honest latency measurement for routed attention,
because the prior test was dominated by 8x kernel launch overhead.
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
    select_top_k_blocks,
)


def benchmark_latency(fn, *args, warmup=20, iters=200, **kwargs):
    """Benchmark latency with proper CUDA sync."""
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
    n = len(times)
    return {
        "median_ms": times[n // 2],
        "mean_ms": sum(times) / n,
        "min_ms": times[0],
        "p90_ms": times[int(n * 0.9)],
        "p99_ms": times[int(n * 0.99)],
    }


def run_inference_retry():
    """Run the routing inference retry with fused kernel."""
    device = "cuda"
    dtype = torch.float16
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}")

    # Marin 8B config
    n_heads = 32
    n_kv_heads = 8
    head_dim = 128
    batch = 1

    configs = [
        {"block_size": 16, "n_total_blocks": 256, "label": "BS=16 (vLLM default)", "context_tokens": 4096},
        {"block_size": 128, "n_total_blocks": 32, "label": "BS=128 (coarse)", "context_tokens": 4096},
        {"block_size": 256, "n_total_blocks": 16, "label": "BS=256 (very coarse)", "context_tokens": 4096},
    ]

    k_values = [1, 2, 4, 8]
    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "experiment": "routing_inference_retry_fused_kernel",
        "model_config": "Marin-8B-like (32 QH, 8 KVH, head_dim=128)",
        "configs": {},
    }

    for cfg in configs:
        block_size = cfg["block_size"]
        n_total_blocks = cfg["n_total_blocks"]
        context_tokens = cfg["context_tokens"]
        label = cfg["label"]

        print(f"\n{'='*70}")
        print(f"Config: {label} — {context_tokens} tokens, {n_total_blocks} blocks")
        print(f"{'='*70}")

        # Create synthetic data
        q = torch.randn(batch, n_heads, head_dim, device=device, dtype=dtype)
        k_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn(n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype)

        # Create synthetic routing prior (random affinities)
        routing_prior = torch.randn(1, n_kv_heads, n_total_blocks, device=device)

        # Dense baseline: all blocks
        dense_tables = torch.arange(n_total_blocks, device=device).unsqueeze(0).unsqueeze(0).expand(batch, n_kv_heads, -1)
        dense_counts = torch.full((batch, n_kv_heads), n_total_blocks, dtype=torch.int32, device=device)

        dense_out = fused_routed_decode(q, k_cache, v_cache, dense_tables, dense_counts)
        dense_latency = benchmark_latency(fused_routed_decode, q, k_cache, v_cache, dense_tables, dense_counts)

        print(f"  Dense baseline: {dense_latency['median_ms']:.4f}ms (median)")

        config_results = {
            "block_size": block_size,
            "n_total_blocks": n_total_blocks,
            "context_tokens": context_tokens,
            "dense_latency": dense_latency,
            "routed": {},
        }

        for K in k_values:
            actual_k = min(K, n_total_blocks)

            # Select top-K blocks per head from routing prior
            block_tables, block_counts = select_top_k_blocks(routing_prior, actual_k)
            block_tables = block_tables.to(device=device, dtype=torch.int64)
            block_counts = block_counts.to(device=device, dtype=torch.int32)

            # Fused routed decode
            routed_out = fused_routed_decode(q, k_cache, v_cache, block_tables, block_counts)
            routed_latency = benchmark_latency(fused_routed_decode, q, k_cache, v_cache, block_tables, block_counts)

            # Reference (Python loop) for comparison
            ref_latency = benchmark_latency(
                reference_routed_decode, q, k_cache, v_cache, block_tables, block_counts,
                warmup=5, iters=50,
            )

            # Cosine similarity: routed vs dense output
            cos_sim = torch.nn.functional.cosine_similarity(
                routed_out.float().reshape(-1), dense_out.float().reshape(-1), dim=0
            ).item()

            # Per-head cosine similarity
            per_head_cos = []
            for h in range(n_heads):
                hcos = torch.nn.functional.cosine_similarity(
                    routed_out[0, h, :].float().unsqueeze(0),
                    dense_out[0, h, :].float().unsqueeze(0),
                    dim=1,
                ).item()
                per_head_cos.append(hcos)

            tokens_per_head = actual_k * block_size
            kv_reduction = 1.0 - tokens_per_head / context_tokens

            # Speedup vs reference (Python loop)
            speedup_vs_ref = ref_latency["median_ms"] / max(routed_latency["median_ms"], 1e-6)
            # Speedup vs dense
            speedup_vs_dense = dense_latency["median_ms"] / max(routed_latency["median_ms"], 1e-6)

            result = {
                "K": K,
                "actual_k": actual_k,
                "tokens_per_head": tokens_per_head,
                "kv_reduction_pct": kv_reduction * 100,
                "fused_latency": routed_latency,
                "ref_latency": ref_latency,
                "dense_latency_ms": dense_latency["median_ms"],
                "fused_latency_ms": routed_latency["median_ms"],
                "ref_latency_ms": ref_latency["median_ms"],
                "speedup_vs_ref": speedup_vs_ref,
                "speedup_vs_dense": speedup_vs_dense,
                "cos_sim_vs_dense": cos_sim,
                "per_head_cos_mean": sum(per_head_cos) / len(per_head_cos),
                "per_head_cos_min": min(per_head_cos),
                "per_head_cos_max": max(per_head_cos),
            }
            config_results["routed"][f"K={K}"] = result

            print(f"  K={K:2d}: fused={routed_latency['median_ms']:.4f}ms  "
                  f"ref={ref_latency['median_ms']:.3f}ms  "
                  f"dense={dense_latency['median_ms']:.4f}ms  "
                  f"vs_ref={speedup_vs_ref:.1f}x  "
                  f"vs_dense={speedup_vs_dense:.1f}x  "
                  f"KV_red={kv_reduction*100:.1f}%  "
                  f"cos={cos_sim:.4f}")

        all_results["configs"][label] = config_results

    # TTFT analysis summary
    print(f"\n{'='*70}")
    print("TTFT / Latency Verdict")
    print(f"{'='*70}")
    for label, cfg_res in all_results["configs"].items():
        print(f"\n{label}:")
        dense_ms = cfg_res["dense_latency"]["median_ms"]
        for k_key, r in cfg_res["routed"].items():
            fused_ms = r["fused_latency_ms"]
            ref_ms = r["ref_latency_ms"]
            kv_red = r["kv_reduction_pct"]
            vs_dense = r["speedup_vs_dense"]
            print(f"  {k_key}: fused={fused_ms:.4f}ms (vs dense {dense_ms:.4f}ms = {vs_dense:.1f}x faster, "
                  f"vs ref {ref_ms:.3f}ms = {r['speedup_vs_ref']:.1f}x faster) "
                  f"KV_red={kv_red:.1f}%")

    # Write results
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(f"routing_inference_retry_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {out_path}")

    return all_results


if __name__ == "__main__":
    run_inference_retry()
