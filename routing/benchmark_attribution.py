# SPDX-License-Identifier: GPL-2.0
"""S3 attribution benchmark for routed decode.

Separately measures:
1. Block selection time (top-K from routing prior)
2. Kernel execution time (fused routed decode)
3. Total end-to-end time (selection + kernel)

This is the minimum instrumentation needed to determine whether the next
performance win comes from kernel tuning (T1), selection optimization (T2),
or regime mismatch (T3).

Usage:
    python -m routing.benchmark_attribution  [--autotune]
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from routing.fused_routed_attention import (
    fused_routed_decode,
    select_top_k_blocks,
)


def _time_fn(fn, *args, warmup=20, iters=200, **kwargs):
    """Time a function, return median and breakdown in ms."""
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


def run_attribution(
    batch=1,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    n_total_blocks=32,
    block_size=128,
    K=8,
    dtype=torch.float16,
    autotune=False,
    iters=200,
):
    """Run attribution benchmark at a single operating point.

    Returns dict with separate timings for selection, kernel, and end-to-end.
    """
    device = "cuda"

    # Synthetic data
    q = torch.randn(batch, n_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        n_total_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=dtype
    )

    # Routing prior: [1, n_kv_heads, n_total_blocks] (single-layer for this bench)
    routing_prior = torch.randn(1, n_kv_heads, n_total_blocks, device=device)

    actual_k = min(K, n_total_blocks)

    # --- Phase 1: measure block selection time ---
    def do_selection():
        bt, bc = select_top_k_blocks(routing_prior, actual_k)
        return bt.to(device=device, dtype=torch.int64), bc.to(
            device=device, dtype=torch.int32
        )

    selection_stats = _time_fn(do_selection, warmup=20, iters=iters)

    # Pre-compute block tables for kernel-only timing
    block_tables, block_counts = do_selection()

    # --- Phase 2: measure kernel-only time ---
    kernel_stats = _time_fn(
        fused_routed_decode,
        q,
        k_cache,
        v_cache,
        block_tables,
        block_counts,
        autotune=autotune,
        warmup=20,
        iters=iters,
    )

    # --- Phase 3: measure end-to-end (selection + kernel) ---
    def do_end_to_end():
        bt, bc = select_top_k_blocks(routing_prior, actual_k)
        bt = bt.to(device=device, dtype=torch.int64)
        bc = bc.to(device=device, dtype=torch.int32)
        return fused_routed_decode(q, k_cache, v_cache, bt, bc, autotune=autotune)

    e2e_stats = _time_fn(do_end_to_end, warmup=20, iters=iters)

    tokens_per_head = actual_k * block_size
    total_kv_tokens = n_total_blocks * block_size
    kv_reduction = 1.0 - tokens_per_head / total_kv_tokens

    # Attribution breakdown
    sel_median = selection_stats["median_ms"]
    kern_median = kernel_stats["median_ms"]
    e2e_median = e2e_stats["median_ms"]
    overhead = e2e_median - kern_median  # selection + launch overhead

    return {
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
            "autotune": autotune,
        },
        "selection": selection_stats,
        "kernel": kernel_stats,
        "end_to_end": e2e_stats,
        "attribution": {
            "selection_median_ms": sel_median,
            "kernel_median_ms": kern_median,
            "e2e_median_ms": e2e_median,
            "overhead_median_ms": overhead,
            "kernel_pct": kern_median / max(e2e_median, 1e-9) * 100,
            "selection_pct": sel_median / max(e2e_median, 1e-9) * 100,
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="S3 attribution benchmark")
    parser.add_argument(
        "--autotune", action="store_true", help="Use autotuned kernel variant"
    )
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}")
    print(f"Autotune: {args.autotune}")
    print()

    operating_points = [
        {
            "block_size": 128,
            "n_total_blocks": 32,
            "K": 8,
            "label": "BS=128 K=8 (promoted)",
        },
        {
            "block_size": 256,
            "n_total_blocks": 16,
            "K": 8,
            "label": "BS=256 K=8 (high-accuracy)",
        },
        {
            "block_size": 256,
            "n_total_blocks": 16,
            "K": 4,
            "label": "BS=256 K=4 (latency)",
        },
    ]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "autotune": args.autotune,
        "operating_points": [],
    }

    print(
        f"{'Label':<25s}  {'Select':>8s}  {'Kernel':>8s}  {'E2E':>8s}  {'Kern%':>6s}  {'Sel%':>6s}"
    )
    print("-" * 75)

    for op in operating_points:
        r = run_attribution(
            n_total_blocks=op["n_total_blocks"],
            block_size=op["block_size"],
            K=op["K"],
            autotune=args.autotune,
            iters=args.iters,
        )
        all_results["operating_points"].append(r)

        a = r["attribution"]
        print(
            f"{op['label']:<25s}  "
            f"{a['selection_median_ms']:>7.3f}ms  "
            f"{a['kernel_median_ms']:>7.3f}ms  "
            f"{a['e2e_median_ms']:>7.3f}ms  "
            f"{a['kernel_pct']:>5.1f}%  "
            f"{a['selection_pct']:>5.1f}%"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(f"routing_attribution_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
