#!/usr/bin/env python3
"""Compare inference speed between baseline GPT-2 and RA models."""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra import GPT2_RA_Model
from gpt2.model import GPT, GPTConfig


def benchmark_inference(model, num_tokens=100, num_runs=5, batch_size=1):
    """Benchmark inference speed using autoregressive generation."""
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    for _ in range(3):
        prompt = torch.randint(0, 50257, (batch_size, 32), device=device)
        with torch.no_grad():
            for _ in range(10):
                logits, _ = model(prompt)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                prompt = torch.cat([prompt, next_token], dim=1)

    # Benchmark
    times = []
    for run in range(num_runs):
        prompt = torch.randint(0, 50257, (batch_size, 32), device=device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_tokens):
                logits, _ = model(prompt)
                next_token = logits[:, -1:, :].argmax(dim=-1)
                prompt = torch.cat([prompt, next_token], dim=1)

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        tokens_per_sec = num_tokens / elapsed
        print(f"  Run {run+1}: {elapsed:.3f}s ({tokens_per_sec:.1f} tok/s)")

    avg_time = sum(times) / len(times)
    avg_tokens_per_sec = num_tokens / avg_time

    return {
        "avg_time": avg_time,
        "tokens_per_sec": avg_tokens_per_sec,
        "times": times,
    }


def main():
    print("=" * 80)
    print("GPT-2 vs RA Inference Speed Comparison")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load configs
    gpt_config = GPTConfig.from_name("gpt2")
    gpt_config.block_size = 1024

    # Create models
    print("\n" + "-" * 80)
    print("Baseline GPT-2")
    print("-" * 80)
    baseline = GPT(gpt_config).to(device)
    print(f"Parameters: {baseline.get_num_params() / 1e6:.2f}M")
    baseline_results = benchmark_inference(baseline, num_tokens=100, num_runs=5)

    print("\n" + "-" * 80)
    print("GPT-2 + RA (Reciprocal Attention)")
    print("-" * 80)
    ra_model = GPT2_RA_Model(gpt_config).to(device)
    print(f"Parameters: {ra_model.get_num_params() / 1e6:.2f}M")
    ra_results = benchmark_inference(ra_model, num_tokens=100, num_runs=5)

    # Compare
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nBaseline GPT-2:")
    print(f"  Avg time: {baseline_results['avg_time']:.3f}s")
    print(f"  Throughput: {baseline_results['tokens_per_sec']:.1f} tok/s")

    print(f"\nGPT-2 + RA:")
    print(f"  Avg time: {ra_results['avg_time']:.3f}s")
    print(f"  Throughput: {ra_results['tokens_per_sec']:.1f} tok/s")

    speedup = baseline_results["tokens_per_sec"] / ra_results["tokens_per_sec"]
    if speedup > 1:
        print(f"\n⚠️  RA is {speedup:.2f}x SLOWER than baseline")
    else:
        print(f"\n🚀 RA is {1/speedup:.2f}x FASTER than baseline")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
