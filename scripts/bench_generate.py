#!/usr/bin/env python3
"""
Benchmark generation speed and quality with KV compression.

Compares baseline vs compressed generation:
- Throughput (tokens/sec)
- Time to first token (TTFT)
- Generation quality (visual comparison)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    load_calibrated_compressors,
)


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cache=None,
    num_runs: int = 3,
) -> Dict:
    """Benchmark a single generation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    times = []
    outputs_text = []

    for _ in range(num_runs):
        if cache is not None and hasattr(cache, "reset"):
            cache.reset()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=cache,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        outputs_text.append(text)

    output_len = outputs.shape[1] - input_len
    avg_time = sum(times) / len(times)
    throughput = output_len / avg_time

    return {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "avg_time_sec": avg_time,
        "throughput_tok_sec": throughput,
        "times": times,
        "sample_output": outputs_text[0][:500],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark generation with compression"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B", help="Model to benchmark"
    )
    parser.add_argument("--calib", type=str, default=None, help="Calibration file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-runs", type=int, default=3, help="Runs per benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("GENERATION BENCHMARK")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Calibration: {args.calib or 'None (identity)'}")

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()

    config = AutoConfig.from_pretrained(args.model)
    num_layers = config.num_hidden_layers

    # Create compressors
    if args.calib:
        k_comp, v_comp, meta = load_calibrated_compressors(
            args.calib,
            device=torch.device(args.device),
            dtype=torch.float16,
            quantize_bits=8,
        )
        compression_ratio = meta.get("compression_ratio", "unknown")
    else:
        k_comp = [IdentityCompressor() for _ in range(num_layers)]
        v_comp = [IdentityCompressor() for _ in range(num_layers)]
        compression_ratio = 1.0

    # Test prompts
    prompts = [
        ("short", "The capital of France is"),
        ("medium", "Explain the concept of machine learning in simple terms:"),
        (
            "long",
            "Write a detailed analysis of the impact of artificial intelligence on "
            "modern healthcare, including benefits, challenges, and future prospects:",
        ),
    ]

    token_lengths = [32, 128, 256]

    results = {
        "model": args.model,
        "compression_ratio": compression_ratio,
        "benchmarks": [],
    }

    # Warmup
    print("\nWarming up...")
    _ = benchmark_generation(model, tokenizer, "Hello", 10, cache=None, num_runs=1)

    for prompt_name, prompt in prompts:
        for max_tokens in token_lengths:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt_name}, Max tokens: {max_tokens}")
            print(f"{'='*60}")

            # Baseline (no cache)
            print("\nBaseline (no cache):")
            baseline = benchmark_generation(
                model, tokenizer, prompt, max_tokens, cache=None, num_runs=args.num_runs
            )
            print(f"  Throughput: {baseline['throughput_tok_sec']:.1f} tok/s")
            print(f"  Time: {baseline['avg_time_sec']:.2f}s")

            # Compressed
            print("\nCompressed:")
            cache = CompressedDynamicCache(k_comp, v_comp, num_layers)
            compressed = benchmark_generation(
                model,
                tokenizer,
                prompt,
                max_tokens,
                cache=cache,
                num_runs=args.num_runs,
            )
            print(f"  Throughput: {compressed['throughput_tok_sec']:.1f} tok/s")
            print(f"  Time: {compressed['avg_time_sec']:.2f}s")

            speedup = compressed["throughput_tok_sec"] / baseline["throughput_tok_sec"]
            print(f"\n  Speedup: {speedup:.2f}x")

            # Compare outputs
            print("\n  Output comparison:")
            print(f"    Baseline: {baseline['sample_output'][:200]}...")
            print(f"    Compressed: {compressed['sample_output'][:200]}...")

            results["benchmarks"].append(
                {
                    "prompt_type": prompt_name,
                    "max_tokens": max_tokens,
                    "baseline": baseline,
                    "compressed": compressed,
                    "speedup": speedup,
                }
            )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Prompt':<10} {'Tokens':<8} {'Baseline':<12} {'Compressed':<12} {'Speedup'}"
    )
    print("-" * 60)
    for b in results["benchmarks"]:
        print(
            f"{b['prompt_type']:<10} {b['max_tokens']:<8} "
            f"{b['baseline']['throughput_tok_sec']:.1f} tok/s{'':<4} "
            f"{b['compressed']['throughput_tok_sec']:.1f} tok/s{'':<4} "
            f"{b['speedup']:.2f}x"
        )

    # Save results
    if args.output is None:
        model_short = args.model.replace("/", "-").lower()
        args.output = f"results/bench_generate_{model_short}.json"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
