#!/usr/bin/env python3
"""
Performance Evaluation Script for KV Plugin

Measures key performance metrics:
- TTFT (Time to First Token)
- Tokens per second at various context lengths
- GPU memory usage
- KV cache memory

Follows methodology from:
- Palu (ICLR 2025) - Table 2
- MiniCache (NeurIPS 2024)
- PyramidKV (NeurIPS 2024) - Figure 3-5
- AsymKV (NeurIPS 2025) - Figure 2-3

Usage:
    python scripts/eval_performance.py --model gpt2 --preset orthogonal_int4
    python scripts/eval_performance.py --model Qwen/Qwen2.5-7B-Instruct --context-lengths 1024 4096 8192
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from gpt2.compression.kv_plugin import KVPlugin


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


@torch.no_grad()
def measure_ttft(
    model,
    tokenizer,
    prompt_length: int,
    device: str = "cuda",
    warmup: int = 3,
    repeat: int = 10,
) -> Dict:
    """
    Measure Time to First Token.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        prompt_length: Length of input prompt
        device: Device
        warmup: Warmup iterations
        repeat: Measurement iterations

    Returns:
        Dict with TTFT statistics
    """
    # Create dummy prompt
    prompt_ids = torch.randint(
        100,
        tokenizer.vocab_size - 100,
        (1, prompt_length),
        device=device,
    )

    # Warmup
    for _ in range(warmup):
        model.generate(
            prompt_ids,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(repeat):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        model.generate(
            prompt_ids,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "prompt_length": prompt_length,
        "ttft_mean_ms": sum(times) / len(times) * 1000,
        "ttft_min_ms": min(times) * 1000,
        "ttft_max_ms": max(times) * 1000,
        "ttft_std_ms": (
            sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)
        )
        ** 0.5
        * 1000,
    }


@torch.no_grad()
def measure_throughput(
    model,
    tokenizer,
    context_length: int,
    gen_length: int = 128,
    device: str = "cuda",
    warmup: int = 2,
    repeat: int = 5,
) -> Dict:
    """
    Measure generation throughput.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        context_length: Input context length
        gen_length: Tokens to generate
        device: Device
        warmup: Warmup iterations
        repeat: Measurement iterations

    Returns:
        Dict with throughput statistics
    """
    # Create dummy context
    context_ids = torch.randint(
        100,
        tokenizer.vocab_size - 100,
        (1, context_length),
        device=device,
    )

    # Warmup
    for _ in range(warmup):
        model.generate(
            context_ids,
            max_new_tokens=gen_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(repeat):
        reset_memory_stats()

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        outputs = model.generate(
            context_ids,
            max_new_tokens=gen_length,
            pad_token_id=tokenizer.eos_token_id,
        )

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        peak_memory = get_gpu_memory()

    tokens_generated = outputs.size(1) - context_length
    avg_time = sum(times) / len(times)

    return {
        "context_length": context_length,
        "tokens_generated": tokens_generated,
        "time_seconds": avg_time,
        "tokens_per_second": tokens_generated / avg_time,
        "peak_memory_gb": peak_memory,
    }


@torch.no_grad()
def measure_kv_cache_memory(
    model,
    tokenizer,
    context_lengths: List[int],
    device: str = "cuda",
) -> List[Dict]:
    """
    Measure KV cache memory at different context lengths.

    Returns list of measurements.
    """
    results = []

    for ctx_len in context_lengths:
        # Create dummy context
        context_ids = torch.randint(
            100,
            tokenizer.vocab_size - 100,
            (1, ctx_len),
            device=device,
        )

        reset_memory_stats()

        # Forward pass with cache
        outputs = model(context_ids, use_cache=True)
        past_kv = outputs.past_key_values

        if device == "cuda":
            torch.cuda.synchronize()

        # Measure cache memory
        cache_bytes = 0
        if past_kv is not None:
            for layer_kv in past_kv:
                if layer_kv is not None:
                    for tensor in layer_kv:
                        if tensor is not None:
                            cache_bytes += tensor.numel() * tensor.element_size()

        results.append(
            {
                "context_length": ctx_len,
                "kv_cache_mb": cache_bytes / 1e6,
                "kv_cache_gb": cache_bytes / 1e9,
                "peak_memory_gb": get_gpu_memory(),
            }
        )

    return results


def run_evaluation(
    model_name: str,
    preset: str = "none",
    context_lengths: List[int] = [512, 1024, 2048, 4096],
    gen_length: int = 128,
    output_file: Optional[str] = None,
) -> Dict:
    """
    Run full performance evaluation.

    Args:
        model_name: HuggingFace model name
        preset: KV plugin preset
        context_lengths: Context lengths to test
        gen_length: Tokens to generate
        output_file: Output JSON file

    Returns:
        Dict with all results
    """
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Preset: {preset}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        model = model.to(device)

    model.eval()

    # Get model size
    model_params = sum(p.numel() for p in model.parameters())
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    # Create plugin if needed
    plugin = None
    if preset != "none":
        print(f"Creating KV plugin with preset: {preset}")
        plugin = KVPlugin.from_preset(preset, model)

    # Results
    results = {
        "model": model_name,
        "preset": preset,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "parameters": model_params,
            "memory_gb": model_memory,
        },
        "ttft": [],
        "throughput": [],
        "kv_cache": [],
    }

    # TTFT measurements
    print("\nMeasuring TTFT...")
    for ctx_len in tqdm(context_lengths):
        try:
            ttft = measure_ttft(model, tokenizer, ctx_len, device)
            results["ttft"].append(ttft)
            print(f"  {ctx_len} tokens: {ttft['ttft_mean_ms']:.1f} ms")
        except Exception as e:
            print(f"  {ctx_len} tokens: ERROR - {e}")

    # Throughput measurements
    print("\nMeasuring throughput...")
    for ctx_len in tqdm(context_lengths):
        try:
            throughput = measure_throughput(
                model, tokenizer, ctx_len, gen_length, device
            )
            results["throughput"].append(throughput)
            print(f"  {ctx_len} tokens: {throughput['tokens_per_second']:.1f} tok/s")
        except Exception as e:
            print(f"  {ctx_len} tokens: ERROR - {e}")

    # KV cache measurements
    print("\nMeasuring KV cache...")
    try:
        kv_results = measure_kv_cache_memory(model, tokenizer, context_lengths, device)
        results["kv_cache"] = kv_results
        for r in kv_results:
            print(f"  {r['context_length']} tokens: {r['kv_cache_mb']:.1f} MB")
    except Exception as e:
        print(f"  ERROR: {e}")

    # GPU info
    if device == "cuda":
        results["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def print_table(results: Dict):
    """Print results as formatted table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: {results['model']}")
    print(f"Preset: {results['preset']}")

    print("\n--- TTFT (Time to First Token) ---")
    print(f"{'Context':>10} {'TTFT (ms)':>12} {'Std':>10}")
    print("-" * 35)
    for r in results["ttft"]:
        print(
            f"{r['prompt_length']:>10} {r['ttft_mean_ms']:>12.1f} {r['ttft_std_ms']:>10.1f}"
        )

    print("\n--- Throughput ---")
    print(f"{'Context':>10} {'Tok/sec':>12} {'Peak Mem (GB)':>15}")
    print("-" * 40)
    for r in results["throughput"]:
        print(
            f"{r['context_length']:>10} {r['tokens_per_second']:>12.1f} {r['peak_memory_gb']:>15.2f}"
        )

    print("\n--- KV Cache Memory ---")
    print(f"{'Context':>10} {'Cache (MB)':>12}")
    print("-" * 25)
    for r in results["kv_cache"]:
        print(f"{r['context_length']:>10} {r['kv_cache_mb']:>12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate performance with KV compression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument("--preset", type=str, default="none", help="KV plugin preset")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--gen-length", type=int, default=128, help="Tokens to generate"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if not HF_AVAILABLE:
        print("Error: transformers library required")
        sys.exit(1)

    results = run_evaluation(
        model_name=args.model,
        preset=args.preset,
        context_lengths=args.context_lengths,
        gen_length=args.gen_length,
        output_file=args.output,
    )

    print_table(results)


if __name__ == "__main__":
    main()
