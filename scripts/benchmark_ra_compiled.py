#!/usr/bin/env python3
"""
Benchmark RA variants with torch.compile optimization.

Tests whether torch.compile can eliminate the 3-6% overhead
by fusing operations, eliminating branching, and optimizing
the execution graph.
"""

import torch
import time
import sys
import os
import argparse

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
    parser = argparse.ArgumentParser(
        description="Benchmark RA variants with torch.compile"
    )
    parser.add_argument(
        "--num-tokens", type=int, default=200, help="Tokens to generate"
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("RA Variants Benchmark: torch.compile Optimization")
    print("=" * 80)
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tokens: {args.num_tokens}, Runs: {args.num_runs}")
    print(f"torch.compile backend: inductor")
    print()

    # Create config
    config = GPTConfig.from_name("gpt2")
    config.block_size = 1024

    # Test variants with and without compile
    variants = [
        ("GPT2 (baseline)", GPT2(config)),
        ("GPT2_RA_Fixed (none)", GPT2_RA_Fixed(config, pattern="none")),
        ("GPT2_RA_Fixed (early)", GPT2_RA_Fixed(config, pattern="early")),
        ("GPT2_RA_Fixed (late)", GPT2_RA_Fixed(config, pattern="late")),
        ("GPT2_RA_Fixed (all)", GPT2_RA_Fixed(config, pattern="all")),
    ]

    results = {}

    for name, model in variants:
        print("=" * 80)
        print(f"{name}")
        print("=" * 80)

        # Uncompiled
        print("\n[Without torch.compile]")
        speed_uncompiled, times = benchmark_model(
            model, device, args.num_tokens, args.num_runs
        )
        print(f"  Average: {speed_uncompiled:.1f} tok/s")
        print(f"  Min: {args.num_tokens / max(times):.1f} tok/s")
        print(f"  Max: {args.num_tokens / min(times):.1f} tok/s")

        # Compiled
        print("\n[With torch.compile]")
        try:
            # Recreate model for clean compile
            if "baseline" in name:
                model_compiled = GPT2(config)
            elif "none" in name:
                model_compiled = GPT2_RA_Fixed(config, pattern="none")
            elif "early" in name:
                model_compiled = GPT2_RA_Fixed(config, pattern="early")
            elif "late" in name:
                model_compiled = GPT2_RA_Fixed(config, pattern="late")
            elif "all" in name:
                model_compiled = GPT2_RA_Fixed(config, pattern="all")

            # Compile with inductor backend
            model_compiled = torch.compile(model_compiled, backend="inductor")

            speed_compiled, times = benchmark_model(
                model_compiled, device, args.num_tokens, args.num_runs
            )
            print(f"  Average: {speed_compiled:.1f} tok/s")
            print(f"  Min: {args.num_tokens / max(times):.1f} tok/s")
            print(f"  Max: {args.num_tokens / min(times):.1f} tok/s")

            speedup = speed_compiled / speed_uncompiled
            print(f"\n  Speedup from torch.compile: {speedup:.3f}x")

            results[name] = {
                "uncompiled": speed_uncompiled,
                "compiled": speed_compiled,
                "speedup": speedup,
            }

        except Exception as e:
            print(f"  Compilation failed: {e}")
            results[name] = {
                "uncompiled": speed_uncompiled,
                "compiled": None,
                "speedup": None,
            }

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY: torch.compile Impact")
    print("=" * 80)
    print(f"\n{'Variant':<30} {'Uncompiled':<15} {'Compiled':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_compiled = results["GPT2 (baseline)"]["compiled"]

    for name, data in results.items():
        uncompiled = f"{data['uncompiled']:.1f} tok/s"
        compiled = f"{data['compiled']:.1f} tok/s" if data["compiled"] else "N/A"
        speedup = f"{data['speedup']:.3f}x" if data["speedup"] else "N/A"
        print(f"{name:<30} {uncompiled:<15} {compiled:<15} {speedup:<10}")

        # Show overhead vs compiled baseline
        if data["compiled"] and baseline_compiled and "baseline" not in name:
            overhead = (baseline_compiled / data["compiled"] - 1) * 100
            if overhead > 0:
                print(f"  {'':30} vs compiled baseline: {overhead:.1f}% slower")
            else:
                print(f"  {'':30} vs compiled baseline: {-overhead:.1f}% faster")

    print("=" * 80)


if __name__ == "__main__":
    main()
