#!/usr/bin/env python3
"""
Triton Kernel Microbenchmarks for KV Cache Expansion

Uses triton.testing.perf_report for automated benchmarking and visualization.

Benchmarks:
1. FP16 baseline (torch.matmul)
2. PyTorch int8 (dequant + matmul)
3. PyTorch int4 (unpack + dequant + matmul)
4. Triton int8 (fused dequant + matmul)
5. Triton int4 (fused unpack + dequant + matmul)

Usage:
    python scripts/kv_triton_benchmark.py [--output-dir results/]
"""

import argparse
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    import triton
    import triton.testing

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available. Using manual benchmarking.")

from gpt2.compression.kv_plugin import (
    quantize_to_int8,
    quantize_to_int4,
    dequantize_from_int8,
    dequantize_from_int4,
)
from gpt2.compression.triton_kernels import (
    triton_expand_int8,
    triton_expand_int4,
    TRITON_AVAILABLE as TRITON_KERNELS_AVAILABLE,
)


def manual_benchmark(fn, warmup=10, rep=100):
    """Manual benchmarking when Triton is not available."""
    # Warmup
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / rep * 1000  # ms


def benchmark_expand_methods(
    batch_sizes=(1, 2, 4),
    seq_lengths=(128, 256, 512, 1024, 2048, 4096),
    n_heads=12,
    latent_dim=128,
    output_dim=768,
    device="cuda",
    output_dir=None,
):
    """
    Run comprehensive benchmarks comparing all expand methods.

    Returns dict with benchmark results.
    """
    results = []

    for B in batch_sizes:
        for T in seq_lengths:
            print(f"\nBenchmarking B={B}, T={T}...")

            # Create test data
            latent = torch.randn(
                B, n_heads, T, latent_dim, dtype=torch.float16, device=device
            )
            weight = torch.randn(
                latent_dim, output_dim, dtype=torch.float16, device=device
            )

            # Quantize
            qt_int8 = quantize_to_int8(latent)
            qt_int4 = quantize_to_int4(latent)

            # Define benchmark functions
            def fp16_matmul():
                return torch.matmul(latent, weight)

            def pytorch_int8():
                dq = dequantize_from_int8(qt_int8)
                return torch.matmul(dq, weight)

            def pytorch_int4():
                dq = dequantize_from_int4(qt_int4)
                return torch.matmul(dq, weight)

            def triton_int8():
                return triton_expand_int8(qt_int8.data, qt_int8.scale, weight)

            def triton_int4():
                return triton_expand_int4(
                    qt_int4.data, qt_int4.scale, weight, latent_dim
                )

            methods = [
                ("FP16 Baseline", fp16_matmul),
                ("PyTorch Int8", pytorch_int8),
                ("PyTorch Int4", pytorch_int4),
                ("Triton Int8", triton_int8),
                ("Triton Int4", triton_int4),
            ]

            row = {"batch": B, "seq_len": T, "tokens": B * n_heads * T}

            for name, fn in methods:
                try:
                    ms = manual_benchmark(fn)
                    tokens_per_sec = (B * n_heads * T) / (ms / 1000)
                    row[f"{name}_ms"] = ms
                    row[f"{name}_tps"] = tokens_per_sec
                except Exception as e:
                    print(f"  {name}: Failed - {e}")
                    row[f"{name}_ms"] = float("nan")
                    row[f"{name}_tps"] = float("nan")

            results.append(row)

    return results


def run_triton_perf_report(output_dir=None):
    """
    Run benchmarks using triton.testing.perf_report for visualization.

    This generates nice performance plots when Triton is available.
    """
    if not TRITON_AVAILABLE:
        print("Triton not available, skipping perf_report benchmarks")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_heads = 12
    latent_dim = 128
    output_dim = 768

    # Create weight (shared across all tests)
    weight = torch.randn(latent_dim, output_dim, dtype=torch.float16, device=device)

    configs = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[128, 256, 512, 1024, 2048, 4096],
            x_log=True,
            line_arg="provider",
            line_vals=[
                "fp16",
                "pytorch_int8",
                "pytorch_int4",
                "triton_int8",
                "triton_int4",
            ],
            line_names=[
                "FP16",
                "PyTorch Int8",
                "PyTorch Int4",
                "Triton Int8",
                "Triton Int4",
            ],
            styles=[
                ("blue", "-"),
                ("green", "--"),
                ("red", "--"),
                ("green", "-"),
                ("red", "-"),
            ],
            ylabel="Time (ms)",
            plot_name="kv-expand-latency",
            args={"batch": 1, "n_heads": n_heads, "latent_dim": latent_dim},
        ),
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[128, 256, 512, 1024, 2048, 4096],
            x_log=True,
            line_arg="provider",
            line_vals=[
                "fp16",
                "pytorch_int8",
                "pytorch_int4",
                "triton_int8",
                "triton_int4",
            ],
            line_names=[
                "FP16",
                "PyTorch Int8",
                "PyTorch Int4",
                "Triton Int8",
                "Triton Int4",
            ],
            styles=[
                ("blue", "-"),
                ("green", "--"),
                ("red", "--"),
                ("green", "-"),
                ("red", "-"),
            ],
            ylabel="Tokens/sec",
            plot_name="kv-expand-throughput",
            args={"batch": 1, "n_heads": n_heads, "latent_dim": latent_dim},
        ),
    ]

    @triton.testing.perf_report(configs)
    def bench_expand(seq_len, batch, n_heads, latent_dim, provider):
        # Create test data
        latent = torch.randn(
            batch, n_heads, seq_len, latent_dim, dtype=torch.float16, device=device
        )
        qt_int8 = quantize_to_int8(latent)
        qt_int4 = quantize_to_int4(latent)

        if provider == "fp16":
            fn = lambda: torch.matmul(latent, weight)
        elif provider == "pytorch_int8":
            fn = lambda: torch.matmul(dequantize_from_int8(qt_int8), weight)
        elif provider == "pytorch_int4":
            fn = lambda: torch.matmul(dequantize_from_int4(qt_int4), weight)
        elif provider == "triton_int8":
            fn = lambda: triton_expand_int8(qt_int8.data, qt_int8.scale, weight)
        elif provider == "triton_int4":
            fn = lambda: triton_expand_int4(
                qt_int4.data, qt_int4.scale, weight, latent_dim
            )

        ms = triton.testing.do_bench(fn, warmup=10, rep=100)
        return ms

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        bench_expand.run(save_path=output_dir, print_data=True)
    else:
        bench_expand.run(print_data=True)


def print_results_table(results):
    """Print results as formatted table."""
    print("\n" + "=" * 100)
    print("KV EXPAND BENCHMARK RESULTS")
    print("=" * 100)

    # Header
    print(f"{'B':>3} {'T':>6} {'Tokens':>8}  ", end="")
    print(
        f"{'FP16':>10} {'PT-Int8':>10} {'PT-Int4':>10} {'TR-Int8':>10} {'TR-Int4':>10}"
    )
    print("-" * 100)

    for row in results:
        print(f"{row['batch']:>3} {row['seq_len']:>6} {row['tokens']:>8}  ", end="")
        for method in [
            "FP16 Baseline",
            "PyTorch Int8",
            "PyTorch Int4",
            "Triton Int8",
            "Triton Int4",
        ]:
            ms = row.get(f"{method}_ms", float("nan"))
            if ms == ms:  # not nan
                print(f"{ms:>10.3f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    # Speedups
    print("\n" + "=" * 100)
    print("SPEEDUPS vs FP16 Baseline")
    print("=" * 100)
    print(
        f"{'B':>3} {'T':>6}  {'PT-Int8':>10} {'PT-Int4':>10} {'TR-Int8':>10} {'TR-Int4':>10}"
    )
    print("-" * 60)

    for row in results:
        baseline = row.get("FP16 Baseline_ms", 0)
        if baseline == 0 or baseline != baseline:
            continue

        print(f"{row['batch']:>3} {row['seq_len']:>6}  ", end="")
        for method in ["PyTorch Int8", "PyTorch Int4", "Triton Int8", "Triton Int4"]:
            ms = row.get(f"{method}_ms", float("nan"))
            if ms == ms and ms > 0:
                speedup = baseline / ms
                print(f"{speedup:>10.2f}x", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    # Triton vs PyTorch speedup
    print("\n" + "=" * 100)
    print("TRITON vs PYTORCH SPEEDUP (same precision)")
    print("=" * 100)
    print(f"{'B':>3} {'T':>6}  {'Int8':>12} {'Int4':>12}")
    print("-" * 40)

    for row in results:
        pt_int8 = row.get("PyTorch Int8_ms", float("nan"))
        tr_int8 = row.get("Triton Int8_ms", float("nan"))
        pt_int4 = row.get("PyTorch Int4_ms", float("nan"))
        tr_int4 = row.get("Triton Int4_ms", float("nan"))

        print(f"{row['batch']:>3} {row['seq_len']:>6}  ", end="")

        if pt_int8 == pt_int8 and tr_int8 == tr_int8 and tr_int8 > 0:
            print(f"{pt_int8/tr_int8:>12.2f}x", end="")
        else:
            print(f"{'N/A':>12}", end="")

        if pt_int4 == pt_int4 and tr_int4 == tr_int4 and tr_int4 > 0:
            print(f"{pt_int4/tr_int4:>12.2f}x", end="")
        else:
            print(f"{'N/A':>12}", end="")

        print()


def main():
    parser = argparse.ArgumentParser(description="KV Expand Kernel Microbenchmarks")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save benchmark results and plots",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--n-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument(
        "--output-dim", type=int, default=768, help="Output dimension (d_model)"
    )
    parser.add_argument(
        "--use-triton-report",
        action="store_true",
        help="Use triton.testing.perf_report for visualization",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Triton available: {TRITON_AVAILABLE}")
    print(f"Triton kernels available: {TRITON_KERNELS_AVAILABLE}")

    if args.use_triton_report and TRITON_AVAILABLE:
        print("\nRunning Triton perf_report benchmarks...")
        run_triton_perf_report(args.output_dir)
    else:
        print("\nRunning manual benchmarks...")
        results = benchmark_expand_methods(
            batch_sizes=args.batch_sizes,
            seq_lengths=args.seq_lengths,
            n_heads=args.n_heads,
            latent_dim=args.latent_dim,
            output_dim=args.output_dim,
            device=device,
            output_dir=args.output_dir,
        )
        print_results_table(results)

        # Save results if output dir specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            import json

            results_file = os.path.join(args.output_dir, "benchmark_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
