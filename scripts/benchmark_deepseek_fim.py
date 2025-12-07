#!/usr/bin/env python3
"""
Benchmark FIM-guided selective KVSplice on DeepSeek models.

Compares three approaches:
1. Baseline (no compression)
2. Uniform KVSplice (all layers compressed)
3. FIM-guided KVSplice (only low-FIM layers compressed)

Usage:
    python scripts/benchmark_deepseek_fim.py \
        --model deepseek-ai/DeepSeek-V2-Lite \
        --compression-ratio 0.5 \
        --fim-threshold 0.7

The key insight from FIM analysis:
- Early layers have HIGH FIM trace = critical representational work = protect
- Later layers have LOW FIM trace = less important = safe to compress

This mirrors the GPT2_MLA_KV_FIM architecture which showed 63.08 PPL vs 71.04
baseline by compressing only the last 4 layers.
"""

import argparse
import gc
import time
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset

# Import plugin functions
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepseek_kvsplice_plugin import (
    compute_fim_trace_per_layer,
    patch_model_with_kvsplice,
    patch_model_with_kvsplice_fim,
    get_kv_cache_size,
)


def get_calibration_texts(n_samples: int = 20) -> List[str]:
    """Load calibration texts from WikiText."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in dataset["text"] if len(t) > 100 and not t.startswith(" =")][
            :n_samples
        ]
        return texts
    except Exception as e:
        print(f"Warning: Could not load WikiText, using dummy texts: {e}")
        return ["The quick brown fox jumps over the lazy dog. " * 20] * n_samples


def evaluate_perplexity(
    model, tokenizer, texts: List[str], device: str = "cuda", max_length: int = 512
) -> float:
    """Evaluate perplexity on test texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            n_tokens = inputs["input_ids"].numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def benchmark_throughput(
    model, tokenizer, device: str = "cuda", seq_len: int = 256, n_trials: int = 5
) -> Tuple[float, float]:
    """Benchmark generation throughput."""
    model.eval()

    # Warmup
    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(2):
            model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Benchmark
    torch.cuda.synchronize()
    times = []

    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=seq_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_per_sec = seq_len / avg_time

    return avg_time, tokens_per_sec


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def run_benchmark(args):
    """Run full benchmark comparing all three approaches."""
    print("=" * 80)
    print("FIM-GUIDED KVSPLICE BENCHMARK")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression ratio: {args.compression_ratio}")
    print(f"FIM threshold: {args.fim_threshold}")
    print()

    # Load tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get calibration and test texts
    print("Loading calibration data...")
    calib_texts = get_calibration_texts(20)
    test_texts = get_calibration_texts(50)[20:50]  # Different samples for test

    results = {}

    # =========================================================================
    # 1. Baseline (no compression)
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. BASELINE (No Compression)")
    print("=" * 80)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    baseline_mem = get_gpu_memory_mb()
    print(f"GPU memory: {baseline_mem:.1f} MB")

    print("Evaluating perplexity...")
    baseline_ppl = evaluate_perplexity(model, tokenizer, test_texts, args.device)
    print(f"Perplexity: {baseline_ppl:.2f}")

    print("Benchmarking throughput...")
    baseline_time, baseline_tps = benchmark_throughput(
        model, tokenizer, args.device, args.seq_len
    )
    print(f"Throughput: {baseline_tps:.1f} tokens/sec")

    results["baseline"] = {
        "perplexity": baseline_ppl,
        "throughput": baseline_tps,
        "memory_mb": baseline_mem,
        "compressed_layers": 0,
    }

    # Run FIM calibration on baseline model
    print("\nRunning FIM calibration...")
    fim_traces = compute_fim_trace_per_layer(model, tokenizer, calib_texts, args.device)

    # Clean up baseline model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 2. Uniform KVSplice (all layers)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. UNIFORM KVSPLICE (All Layers)")
    print("=" * 80)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Patching all layers with KVSplice...")
    patch_model_with_kvsplice(model, args.compression_ratio)

    uniform_mem = get_gpu_memory_mb()
    print(f"GPU memory: {uniform_mem:.1f} MB")

    print("Evaluating perplexity...")
    uniform_ppl = evaluate_perplexity(model, tokenizer, test_texts, args.device)
    print(f"Perplexity: {uniform_ppl:.2f}")

    print("Benchmarking throughput...")
    uniform_time, uniform_tps = benchmark_throughput(
        model, tokenizer, args.device, args.seq_len
    )
    print(f"Throughput: {uniform_tps:.1f} tokens/sec")

    n_layers = len(fim_traces)
    results["uniform"] = {
        "perplexity": uniform_ppl,
        "throughput": uniform_tps,
        "memory_mb": uniform_mem,
        "compressed_layers": n_layers,
    }

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 3. FIM-guided KVSplice (selective)
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. FIM-GUIDED KVSPLICE (Selective)")
    print("=" * 80)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Patching low-FIM layers with KVSplice...")
    model, compressed_layers = patch_model_with_kvsplice_fim(
        model, fim_traces, args.compression_ratio, args.fim_threshold
    )

    fim_mem = get_gpu_memory_mb()
    print(f"GPU memory: {fim_mem:.1f} MB")

    print("Evaluating perplexity...")
    fim_ppl = evaluate_perplexity(model, tokenizer, test_texts, args.device)
    print(f"Perplexity: {fim_ppl:.2f}")

    print("Benchmarking throughput...")
    fim_time, fim_tps = benchmark_throughput(
        model, tokenizer, args.device, args.seq_len
    )
    print(f"Throughput: {fim_tps:.1f} tokens/sec")

    results["fim_guided"] = {
        "perplexity": fim_ppl,
        "throughput": fim_tps,
        "memory_mb": fim_mem,
        "compressed_layers": len(compressed_layers),
    }

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(
        f"{'Method':<20} {'PPL':>10} {'Tokens/s':>12} {'Mem (MB)':>12} {'Layers':>10}"
    )
    print("-" * 70)

    for name, r in results.items():
        print(
            f"{name:<20} {r['perplexity']:>10.2f} {r['throughput']:>12.1f} "
            f"{r['memory_mb']:>12.1f} {r['compressed_layers']:>10}"
        )

    print()
    print("Analysis:")

    # PPL degradation
    uniform_degrad = (
        (results["uniform"]["perplexity"] - results["baseline"]["perplexity"])
        / results["baseline"]["perplexity"]
        * 100
    )
    fim_degrad = (
        (results["fim_guided"]["perplexity"] - results["baseline"]["perplexity"])
        / results["baseline"]["perplexity"]
        * 100
    )

    print(f"  Uniform PPL degradation: {uniform_degrad:+.1f}%")
    print(f"  FIM PPL degradation: {fim_degrad:+.1f}%")
    print(
        f"  FIM improvement over uniform: {uniform_degrad - fim_degrad:.1f}% less degradation"
    )

    # Compression efficiency
    uniform_layers = results["uniform"]["compressed_layers"]
    fim_layers = results["fim_guided"]["compressed_layers"]
    print(f"\n  Uniform: {uniform_layers}/{n_layers} layers compressed")
    print(f"  FIM: {fim_layers}/{n_layers} layers compressed")
    print(f"  FIM protects {uniform_layers - fim_layers} high-importance early layers")

    print("\n" + "=" * 80)

    # Save FIM traces for analysis
    print("\nFIM Traces (for reference):")
    for layer_idx, trace in sorted(fim_traces.items()):
        marker = " *" if layer_idx in compressed_layers else ""
        print(f"  Layer {layer_idx:2d}: {trace:.4f}{marker}")
    print("  (* = compressed)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark FIM-guided KVSplice")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model to benchmark",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="KVSplice compression ratio (0.5 = 2x compression)",
    )
    parser.add_argument(
        "--fim-threshold",
        type=float,
        default=0.7,
        help="FIM threshold for selective compression (lower = fewer layers compressed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length for throughput benchmark",
    )
    args = parser.parse_args()

    results = run_benchmark(args)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
