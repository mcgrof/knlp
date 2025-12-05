#!/usr/bin/env python3
"""
KV Cache Performance Profiling.

Measures latency and throughput across context lengths and batch sizes
to quantify the performance impact of KV compression.

Metrics:
- TTFT (Time To First Token)
- Tokens/sec (generation throughput)
- Memory usage

Usage:
    python scripts/profile_kv_performance.py --model Qwen/Qwen2.5-0.5B
    python scripts/profile_kv_performance.py --model Qwen/Qwen2.5-7B \
        --preset kv_preset_qwen-qwen2.5-7b_v9.json
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


def create_cache_factory(
    preset_path: Optional[str],
    num_layers: int,
    device: str,
):
    """Create a cache factory function from preset."""
    if preset_path is None:
        return None

    with open(preset_path) as f:
        preset = json.load(f)

    k_comp, v_comp, metadata = load_calibrated_compressors(
        preset["calibration_file"],
        device=torch.device(device),
        dtype=torch.float16,
        quantize_bits=preset["bits"] if preset["bits"] < 16 else None,
    )

    if preset["target"] == "v":
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
    elif preset["target"] == "k":
        v_comp = [IdentityCompressor() for _ in range(num_layers)]

    def factory():
        return CompressedDynamicCache(k_comp, v_comp, num_layers)

    return factory


def warmup_model(model, tokenizer, device: str, seq_len: int = 64):
    """Warmup model to ensure accurate timing."""
    dummy_input = torch.randint(0, 1000, (1, seq_len), device=device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize() if device == "cuda" else None


@torch.no_grad()
def measure_ttft(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cache_factory,
    device: str,
    num_runs: int = 5,
) -> Tuple[float, float]:
    """
    Measure Time To First Token.

    Returns:
        (mean_ttft_ms, std_ttft_ms)
    """
    ttfts = []

    for _ in range(num_runs):
        cache = cache_factory() if cache_factory else None

        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        if cache is not None:
            outputs = model(input_ids, past_key_values=cache, use_cache=True)
        else:
            outputs = model(input_ids, use_cache=True)

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        ttfts.append(elapsed * 1000)  # Convert to ms

        if cache is not None:
            del cache
        del outputs
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    return float(sum(ttfts) / len(ttfts)), float(
        (sum((t - sum(ttfts) / len(ttfts)) ** 2 for t in ttfts) / len(ttfts)) ** 0.5
    )


@torch.no_grad()
def measure_generation_throughput(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cache_factory,
    device: str,
    num_new_tokens: int = 32,
    num_runs: int = 3,
) -> Tuple[float, float]:
    """
    Measure generation throughput (tokens/sec).

    Returns:
        (mean_tokens_per_sec, std_tokens_per_sec)
    """
    throughputs = []

    for _ in range(num_runs):
        cache = cache_factory() if cache_factory else None

        # Prefill
        if cache is not None:
            outputs = model(input_ids, past_key_values=cache, use_cache=True)
            past = outputs.past_key_values
        else:
            outputs = model(input_ids, use_cache=True)
            past = outputs.past_key_values

        # Generation
        generated_ids = input_ids
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()

        for _ in range(num_new_tokens):
            outputs = model(generated_ids[:, -1:], past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.perf_counter() - start

        throughputs.append(num_new_tokens / elapsed)

        del past, outputs
        if cache is not None:
            del cache
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None

    mean = sum(throughputs) / len(throughputs)
    std = (sum((t - mean) ** 2 for t in throughputs) / len(throughputs)) ** 0.5
    return float(mean), float(std)


def measure_memory(
    model,
    input_ids: torch.Tensor,
    cache_factory,
    device: str,
) -> float:
    """Measure peak GPU memory during forward pass."""
    if device != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    cache = cache_factory() if cache_factory else None

    with torch.no_grad():
        if cache is not None:
            _ = model(input_ids, past_key_values=cache, use_cache=True)
        else:
            _ = model(input_ids, use_cache=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB

    if cache is not None:
        del cache
    gc.collect()
    torch.cuda.empty_cache()

    return peak_mem


def profile_model(
    model_name: str,
    preset_path: Optional[str],
    context_lengths: List[int],
    batch_sizes: List[int],
    device: str = "cuda",
    output_dir: str = "plots/performance",
) -> Dict:
    """
    Run performance profiling across configurations.

    Returns:
        Results dict
    """
    print("=" * 70)
    print("KV Cache Performance Profiling")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Preset: {preset_path or 'Baseline'}")
    print(f"Context lengths: {context_lengths}")
    print(f"Batch sizes: {batch_sizes}")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Create cache factories
    baseline_factory = None
    compressed_factory = create_cache_factory(preset_path, num_layers, device)

    # Warmup
    print("Warming up...")
    warmup_model(model, tokenizer, device)

    results = {
        "model": model_name,
        "preset": preset_path,
        "device": device,
        "context_lengths": context_lengths,
        "batch_sizes": batch_sizes,
        "baseline": {},
        "compressed": {},
    }

    for batch_size in batch_sizes:
        print(f"\n--- Batch size: {batch_size} ---")
        results["baseline"][f"batch_{batch_size}"] = {}
        results["compressed"][f"batch_{batch_size}"] = {}

        for ctx_len in context_lengths:
            print(f"  Context length: {ctx_len}")

            # Create input
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (batch_size, ctx_len), device=device
            )

            # Baseline measurements
            ttft_mean, ttft_std = measure_ttft(
                model, tokenizer, input_ids, baseline_factory, device
            )
            tput_mean, tput_std = measure_generation_throughput(
                model, tokenizer, input_ids, baseline_factory, device
            )
            mem = measure_memory(model, input_ids, baseline_factory, device)

            results["baseline"][f"batch_{batch_size}"][f"ctx_{ctx_len}"] = {
                "ttft_ms": {"mean": ttft_mean, "std": ttft_std},
                "tokens_per_sec": {"mean": tput_mean, "std": tput_std},
                "memory_gb": mem,
            }
            print(
                f"    Baseline: TTFT={ttft_mean:.1f}ms, {tput_mean:.0f} tok/s, {mem:.2f}GB"
            )

            # Compressed measurements (if preset provided)
            if compressed_factory is not None:
                ttft_mean, ttft_std = measure_ttft(
                    model, tokenizer, input_ids, compressed_factory, device
                )
                tput_mean, tput_std = measure_generation_throughput(
                    model, tokenizer, input_ids, compressed_factory, device
                )
                mem = measure_memory(model, input_ids, compressed_factory, device)

                results["compressed"][f"batch_{batch_size}"][f"ctx_{ctx_len}"] = {
                    "ttft_ms": {"mean": ttft_mean, "std": ttft_std},
                    "tokens_per_sec": {"mean": tput_mean, "std": tput_std},
                    "memory_gb": mem,
                }
                print(
                    f"    Compressed: TTFT={ttft_mean:.1f}ms, {tput_mean:.0f} tok/s, {mem:.2f}GB"
                )

            gc.collect()
            torch.cuda.empty_cache() if device == "cuda" else None

    return results


def generate_plots(results: Dict, output_dir: str):
    """Generate performance plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib/numpy not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    model_short = results["model"].split("/")[-1]
    context_lengths = results["context_lengths"]
    batch_sizes = results["batch_sizes"]
    has_compressed = bool(results["compressed"].get(f"batch_{batch_sizes[0]}"))

    # Plot 1: TTFT vs Context Length
    fig, ax = plt.subplots(figsize=(10, 6))

    for batch_size in batch_sizes:
        baseline_ttfts = [
            results["baseline"][f"batch_{batch_size}"][f"ctx_{ctx}"]["ttft_ms"]["mean"]
            for ctx in context_lengths
        ]
        ax.plot(
            context_lengths,
            baseline_ttfts,
            "o-",
            label=f"Baseline (batch={batch_size})",
        )

        if has_compressed:
            compressed_ttfts = [
                results["compressed"][f"batch_{batch_size}"][f"ctx_{ctx}"]["ttft_ms"][
                    "mean"
                ]
                for ctx in context_lengths
            ]
            ax.plot(
                context_lengths,
                compressed_ttfts,
                "s--",
                label=f"Compressed (batch={batch_size})",
            )

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Time To First Token (ms)", fontsize=12)
    ax.set_title(f"TTFT vs Context Length: {model_short}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ttft_vs_context_{model_short}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/ttft_vs_context_{model_short}.png")

    # Plot 2: Throughput vs Context Length
    fig, ax = plt.subplots(figsize=(10, 6))

    for batch_size in batch_sizes:
        baseline_tput = [
            results["baseline"][f"batch_{batch_size}"][f"ctx_{ctx}"]["tokens_per_sec"][
                "mean"
            ]
            for ctx in context_lengths
        ]
        ax.plot(
            context_lengths,
            baseline_tput,
            "o-",
            label=f"Baseline (batch={batch_size})",
        )

        if has_compressed:
            compressed_tput = [
                results["compressed"][f"batch_{batch_size}"][f"ctx_{ctx}"][
                    "tokens_per_sec"
                ]["mean"]
                for ctx in context_lengths
            ]
            ax.plot(
                context_lengths,
                compressed_tput,
                "s--",
                label=f"Compressed (batch={batch_size})",
            )

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title(f"Generation Throughput vs Context: {model_short}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_vs_context_{model_short}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/throughput_vs_context_{model_short}.png")

    # Plot 3: Memory vs Context Length
    fig, ax = plt.subplots(figsize=(10, 6))

    batch_size = batch_sizes[0]  # Use first batch size for memory plot
    baseline_mem = [
        results["baseline"][f"batch_{batch_size}"][f"ctx_{ctx}"]["memory_gb"]
        for ctx in context_lengths
    ]
    ax.plot(context_lengths, baseline_mem, "o-", label="Baseline", linewidth=2)

    if has_compressed:
        compressed_mem = [
            results["compressed"][f"batch_{batch_size}"][f"ctx_{ctx}"]["memory_gb"]
            for ctx in context_lengths
        ]
        ax.plot(
            context_lengths, compressed_mem, "s--", label="Compressed (v9)", linewidth=2
        )

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title(f"Memory Usage vs Context: {model_short}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_vs_context_{model_short}.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/memory_vs_context_{model_short}.png")


def print_summary(results: Dict):
    """Print performance summary."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    batch_sizes = results["batch_sizes"]
    context_lengths = results["context_lengths"]
    has_compressed = bool(results["compressed"].get(f"batch_{batch_sizes[0]}"))

    # Find representative config (middle context, batch=1)
    batch = batch_sizes[0]
    ctx = context_lengths[len(context_lengths) // 2]

    baseline = results["baseline"][f"batch_{batch}"][f"ctx_{ctx}"]
    print(f"\nRepresentative config: batch={batch}, context={ctx}")
    print(f"\nBaseline:")
    print(f"  TTFT: {baseline['ttft_ms']['mean']:.1f} +/- {baseline['ttft_ms']['std']:.1f} ms")
    print(
        f"  Throughput: {baseline['tokens_per_sec']['mean']:.0f} +/- {baseline['tokens_per_sec']['std']:.0f} tok/s"
    )
    print(f"  Memory: {baseline['memory_gb']:.2f} GB")

    if has_compressed:
        compressed = results["compressed"][f"batch_{batch}"][f"ctx_{ctx}"]
        print(f"\nCompressed (v9):")
        print(
            f"  TTFT: {compressed['ttft_ms']['mean']:.1f} +/- {compressed['ttft_ms']['std']:.1f} ms"
        )
        print(
            f"  Throughput: {compressed['tokens_per_sec']['mean']:.0f} +/- {compressed['tokens_per_sec']['std']:.0f} tok/s"
        )
        print(f"  Memory: {compressed['memory_gb']:.2f} GB")

        # Compute deltas
        ttft_delta = (
            (compressed["ttft_ms"]["mean"] - baseline["ttft_ms"]["mean"])
            / baseline["ttft_ms"]["mean"]
            * 100
        )
        tput_delta = (
            (compressed["tokens_per_sec"]["mean"] - baseline["tokens_per_sec"]["mean"])
            / baseline["tokens_per_sec"]["mean"]
            * 100
        )
        mem_delta = (
            (compressed["memory_gb"] - baseline["memory_gb"])
            / baseline["memory_gb"]
            * 100
        )

        print(f"\nDeltas:")
        print(f"  TTFT: {ttft_delta:+.1f}%")
        print(f"  Throughput: {tput_delta:+.1f}%")
        print(f"  Memory: {mem_delta:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="KV cache performance profiling")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to profile",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Compression preset JSON file",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="256,512,1024,2048",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/performance",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = profile_model(
        model_name=args.model,
        preset_path=args.preset,
        context_lengths=context_lengths,
        batch_sizes=batch_sizes,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Print summary
    print_summary(results)

    # Generate plots
    generate_plots(results, args.output_dir)

    # Save JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
