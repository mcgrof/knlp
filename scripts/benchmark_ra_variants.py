#!/usr/bin/env python3
"""
Benchmark inference speed across RA variants.

Tests:
  1. GPT2 (baseline)
  2. GPT2_RA_Learned (learned alternation, broken)
  3. GPT2_RA_Fixed (none) - all standard
  4. GPT2_RA_Fixed (all) - all reciprocal
  5. GPT2_RA_Fixed (alternating) - 50/50 pattern
  6. GPT2_RA_Fixed (late) - last 6 layers reciprocal
  7. GPT2_RA_Fixed (early) - first 6 layers reciprocal

Usage:
    python scripts/benchmark_ra_variants.py
    python scripts/benchmark_ra_variants.py --num-tokens 200 --num-runs 10
"""

import torch
import time
import sys
import os
import argparse
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.model import GPT2, GPTConfig
from ra import GPT2_RA_Learned, GPT2_RA_Fixed


def benchmark_model(model, device, num_tokens=100, num_runs=5, warmup=2):
    """Benchmark inference speed of a model."""
    model.eval()
    model.to(device)

    batch_size = 1
    prompt_len = 10

    # Create prompt
    prompt = torch.randint(0, 50257, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            current_seq = prompt.clone()
            for _ in range(min(num_tokens, 20)):
                logits, _ = model(current_seq)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                current_seq = torch.cat([current_seq, next_token], dim=1)
                if current_seq.size(1) > model.config.block_size:
                    current_seq = current_seq[:, -model.config.block_size :]

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        current_seq = prompt.clone()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_tokens):
                logits, _ = model(current_seq)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                current_seq = torch.cat([current_seq, next_token], dim=1)
                if current_seq.size(1) > model.config.block_size:
                    current_seq = current_seq[:, -model.config.block_size :]

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    avg_tok_per_sec = num_tokens / avg_time

    return avg_tok_per_sec, times


def main():
    parser = argparse.ArgumentParser(description="Benchmark RA variants")
    parser.add_argument(
        "--num-tokens", type=int, default=100, help="Tokens to generate"
    )
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("RA Variants Inference Benchmark")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tokens: {args.num_tokens}, Runs: {args.num_runs}\n")

    # Create config
    config = GPTConfig.from_name("gpt2")
    config.block_size = 1024

    # Test variants
    variants = [
        ("GPT2 (baseline)", GPT2(config)),
        ("GPT2_RA_Learned", GPT2_RA_Learned(config)),
        ("GPT2_RA_Fixed (none)", GPT2_RA_Fixed(config, pattern="none")),
        ("GPT2_RA_Fixed (all)", GPT2_RA_Fixed(config, pattern="all")),
        ("GPT2_RA_Fixed (alternating)", GPT2_RA_Fixed(config, pattern="alternating")),
        ("GPT2_RA_Fixed (late)", GPT2_RA_Fixed(config, pattern="late")),
        ("GPT2_RA_Fixed (early)", GPT2_RA_Fixed(config, pattern="early")),
    ]

    results = {}
    baseline_speed = None

    for name, model in variants:
        print("=" * 80)
        print(f"{name}")
        print("=" * 80)
        print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")

        # Show pattern if applicable
        if hasattr(model, "get_pattern_stats"):
            stats = model.get_pattern_stats()
            print(
                f"Pattern: {stats['n_reciprocal']} reciprocal, {stats['n_standard']} standard"
            )
            print(f"Reciprocal layers: {stats['reciprocal_layers']}")

        speed, times = benchmark_model(model, device, args.num_tokens, args.num_runs)
        results[name] = speed

        print(f"\nResults:")
        for i, t in enumerate(times, 1):
            print(f"  Run {i}: {t:.3f}s ({args.num_tokens / t:.1f} tok/s)")
        print(f"Average: {speed:.1f} tok/s")

        if baseline_speed is None:
            baseline_speed = speed
        else:
            slowdown = baseline_speed / speed
            pct = (slowdown - 1) * 100
            if slowdown > 1:
                print(f"vs Baseline: {slowdown:.3f}x slower ({pct:.1f}% overhead)")
            else:
                print(
                    f"vs Baseline: {1/slowdown:.3f}x faster ({-pct:.1f}% improvement)"
                )

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Variant':<30} {'Speed (tok/s)':<15} {'vs Baseline':<15}")
    print("-" * 80)

    for name, speed in results.items():
        if name == "GPT2 (baseline)":
            comparison = "(baseline)"
        else:
            slowdown = baseline_speed / speed
            if slowdown > 1:
                comparison = f"{slowdown:.3f}x slower"
            else:
                comparison = f"{1/slowdown:.3f}x faster"

        print(f"{name:<30} {speed:>10.1f}      {comparison:<15}")

    print("=" * 80)


if __name__ == "__main__":
    main()
