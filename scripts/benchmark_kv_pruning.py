#!/usr/bin/env python3
"""
Benchmark script for comparing KV Pruning attention implementations.

This script benchmarks and compares two implementations:
1. `legacy`: The original `PrunedKVAttention` which materializes the full
   attention matrix, making it memory-intensive and incompatible with
   Flash Attention.
2. `flash`: The new `FlashPrunedKVAttention` which uses a proxy importance
   metric (like L2 norm of keys) to prune before the attention calculation,
   allowing it to use the efficient `scaled_dot_product_attention`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from contextlib import contextmanager

# Add project root to path to allow importing 'ra'
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra import PrunedKVAttention, FlashPrunedKVAttention

@contextmanager
def measure_performance(device, description):
    """A context manager to measure execution time and memory usage."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    start_time = time.perf_counter()

    yield

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        peak_memory_mb = -1 # Memory tracking not implemented for CPU

    print(f"[{description}]")
    print(f"  -> Time: {elapsed_time_ms:.3f} ms")
    if peak_memory_mb != -1:
        print(f"  -> Peak Memory: {peak_memory_mb:.2f} MB")
    return elapsed_time_ms, peak_memory_mb

def run_benchmark(
    implementation,
    batch_size,
    seq_len,
    n_embd,
    n_head,
    k_keep,
    recency,
    device,
    dtype,
    warmup_steps,
    bench_steps,
    importance_metric,
):
    """Runs the benchmark for a given configuration."""
    print("-" * 60)
    print(f"Benchmarking implementation: '{implementation}'")
    print(f"Parameters: B={batch_size}, T={seq_len}, C={n_embd}, H={n_head}, K={k_keep}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("-" * 60)

    if implementation == "legacy":
        model = PrunedKVAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=seq_len,
            k_keep=k_keep,
            recency=recency,
        ).to(device).to(dtype)
    elif implementation == "flash":
        model = FlashPrunedKVAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=seq_len,
            k_keep=k_keep,
            recency=recency,
            importance_metric=importance_metric,
        ).to(device).to(dtype)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    model.eval() # Set to evaluation mode

    x = torch.randn(batch_size, seq_len, n_embd, device=device, dtype=dtype)

    # Warmup
    print("Running warmup...")
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model(x)

    # Benchmark
    print(f"Running benchmark ({bench_steps} steps)...")
    total_time = 0
    total_mem = 0
    for i in range(bench_steps):
        with torch.no_grad():
            with measure_performance(device, f"Step {i+1}/{bench_steps}") as (t, m):
                _ = model(x)
        total_time += t
        total_mem = max(total_mem, m)

    avg_time = total_time / bench_steps
    print("-" * 60)
    print("Benchmark Results:")
    print(f"  Implementation: {implementation}")
    print(f"  Average time per iteration: {avg_time:.3f} ms")
    if device == 'cuda':
        print(f"  Peak memory usage: {total_mem:.2f} MB")
    print("-" * 60)

    return avg_time, total_mem

def main():
    parser = argparse.ArgumentParser(description="Benchmark KV Pruning Implementations")
    parser.add_argument("--implementation", type=str, default="all", choices=["legacy", "flash", "all"], help="Implementation to benchmark.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--n-embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--k-keep", type=int, default=256, help="Number of keys to keep after pruning")
    parser.add_argument("--recency", type=int, default=64, help="Number of recent tokens to always keep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"], help="Data type")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--bench-steps", type=int, default=50, help="Number of benchmark steps")
    parser.add_argument("--importance-metric", type=str, default="l2_norm", choices=["l2_norm", "l2_norm_scaled"], help="Importance metric for flash implementation")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU.")
        args.device = "cpu"

    if args.dtype == "float16" and args.device == "cpu":
        print("Warning: float16 not supported on CPU, using float32.")
        args.dtype = "float32"

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    results = {}

    implementations_to_run = ["legacy", "flash"] if args.implementation == "all" else [args.implementation]

    for impl in implementations_to_run:
        try:
            avg_time, peak_mem = run_benchmark(
                implementation=impl,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                n_embd=args.n_embd,
                n_head=args.n_head,
                k_keep=args.k_keep,
                recency=args.recency,
                device=args.device,
                dtype=torch_dtype,
                warmup_steps=args.warmup_steps,
                bench_steps=args.bench_steps,
                importance_metric=args.importance_metric,
            )
            results[impl] = {"time": avg_time, "mem": peak_mem}
        except Exception as e:
            print(f"\n!!! BENCHMARK FAILED for implementation '{impl}' !!!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


    if len(results) > 1:
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)
        legacy_res = results.get("legacy")
        flash_res = results.get("flash")

        if legacy_res and flash_res:
            time_ratio = legacy_res['time'] / flash_res['time']
            mem_ratio = legacy_res['mem'] / flash_res['mem'] if flash_res['mem'] > 0 else float('inf')
            print(f"Speedup (Flash vs Legacy): {time_ratio:.2f}x faster")
            if args.device == 'cuda':
                print(f"Memory Efficiency (Flash vs Legacy): {mem_ratio:.2f}x less memory")

        print("\nRaw Results:")
        for impl, res in results.items():
            print(f"  - {impl}: {res['time']:.3f} ms | {res['mem']:.2f} MB")
        print("="*60)


if __name__ == "__main__":
    main()
