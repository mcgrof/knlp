#!/usr/bin/env python3
"""
Benchmark Tiered Inference

Measures inference latency and throughput impact of hierarchical
memory tiering strategies.

Usage:
    python3 scripts/benchmark_tiered_inference.py \\
        --model gpt2 \\
        --tier-hints tier_hints.json \\
        --mode emulated \\
        --batch-size 1 \\
        --seq-length 128 \\
        --num-iterations 100
"""

import argparse
import json
import time
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np

# Add parent to path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.tiering import load_tier_hints, create_tiering_system


def benchmark_inference(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    num_iterations: int,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Benchmark inference latency and throughput.

    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_length: Sequence length
        num_iterations: Number of iterations
        warmup_iterations: Warmup iterations (excluded from measurements)
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(
        0, 50257, (batch_size, seq_length), dtype=torch.long, device=device
    )

    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    # Benchmark
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.perf_counter()

            _ = model(dummy_input)

            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.perf_counter()

            latencies.append(end_time - start_time)

    # Calculate statistics
    latencies = np.array(latencies)

    return {
        "mean_latency_ms": float(np.mean(latencies) * 1000),
        "median_latency_ms": float(np.median(latencies) * 1000),
        "p95_latency_ms": float(np.percentile(latencies, 95) * 1000),
        "p99_latency_ms": float(np.percentile(latencies, 99) * 1000),
        "std_latency_ms": float(np.std(latencies) * 1000),
        "throughput_samples_per_sec": float(batch_size / np.mean(latencies)),
        "throughput_tokens_per_sec": float(
            batch_size * seq_length / np.mean(latencies)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark tiered inference")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )

    # Tiering configuration
    parser.add_argument(
        "--tier-hints",
        type=str,
        required=True,
        help="Path to tier hints JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="emulated",
        choices=["emulated", "real"],
        help="Tiering mode (emulated or real offloading)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline benchmark without tiering",
    )

    # Benchmark configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading {args.model} model...")
    if args.model == "gpt2":
        from gpt2.model import GPT, GPTConfig

        config = GPTConfig()
        model = GPT(config)

        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model"])

        model = model.to(device)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    results = {}

    # Run baseline benchmark
    if args.baseline:
        print("\nRunning baseline benchmark (no tiering)...")
        baseline_results = benchmark_inference(
            model,
            args.batch_size,
            args.seq_length,
            args.num_iterations,
            args.warmup_iterations,
            device,
        )

        print("\nBaseline Results:")
        print(f"  Mean latency: {baseline_results['mean_latency_ms']:.2f} ms")
        print(
            f"  Throughput: {baseline_results['throughput_tokens_per_sec']:.0f} tokens/s"
        )

        results["baseline"] = baseline_results

    # Run tiered benchmark
    print(f"\nLoading tier hints from {args.tier_hints}...")
    tier_assignments = load_tier_hints(args.tier_hints)

    # Print tier distribution
    tier_counts = {}
    for tier in tier_assignments.values():
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"\nTier distribution:")
    for tier, count in sorted(tier_counts.items()):
        pct = 100 * count / len(tier_assignments)
        print(f"  {tier}: {count} modules ({pct:.1f}%)")

    # Create and install tiering system
    print(f"\nInstalling {args.mode} tiering...")
    tiering_system = create_tiering_system(tier_assignments, mode=args.mode)
    tiering_system.install(model)

    print(f"\nRunning tiered benchmark ({args.mode} mode)...")
    tiered_results = benchmark_inference(
        model,
        args.batch_size,
        args.seq_length,
        args.num_iterations,
        args.warmup_iterations,
        device,
    )

    print("\nTiered Results:")
    print(f"  Mean latency: {tiered_results['mean_latency_ms']:.2f} ms")
    print(f"  Throughput: {tiered_results['throughput_tokens_per_sec']:.0f} tokens/s")

    if args.baseline:
        latency_overhead_pct = (
            100
            * (tiered_results["mean_latency_ms"] - baseline_results["mean_latency_ms"])
            / baseline_results["mean_latency_ms"]
        )
        throughput_degradation_pct = (
            100
            * (
                baseline_results["throughput_tokens_per_sec"]
                - tiered_results["throughput_tokens_per_sec"]
            )
            / baseline_results["throughput_tokens_per_sec"]
        )

        print("\nImpact:")
        print(f"  Latency overhead: {latency_overhead_pct:+.1f}%")
        print(f"  Throughput degradation: {throughput_degradation_pct:+.1f}%")

    results["tiered"] = tiered_results

    # Remove tiering hooks
    tiering_system.remove()

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
