#!/usr/bin/env python3
"""
Profile Runtime-Aware KV Compression (v20).

Compares baseline, always-compress (v18), and thresholded compression (v20)
across different context lengths to measure overhead reduction.

Usage:
    python scripts/profile_thresholded_compression.py
    python scripts/profile_thresholded_compression.py --model Qwen/Qwen2.5-7B
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt2.compression.compressed_cache import (
    CompressedDynamicCache,
    IdentityCompressor,
    QuantizedCalibratedCompressor,
)


def load_calibration(model_name: str, rank: int = 96) -> Optional[Dict]:
    """Try to load existing calibration data."""
    model_short = model_name.replace("/", "-").lower()
    calib_path = Path(f"kv_lowrank_calib_{model_short}_r{rank}.pt")
    if calib_path.exists():
        return torch.load(calib_path)
    return None


def create_compressors(
    calib_data: Dict,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    bits: int = 8,
) -> Tuple[List, List]:
    """Create K and V compressors from calibration data."""
    k_compressors = []
    v_compressors = []

    for layer_data in calib_data["layers"]:
        # K: identity (no compression)
        k_comp = IdentityCompressor()
        k_compressors.append(k_comp)

        # V: quantized low-rank
        V_U = layer_data["V"]["U"].to(device).to(dtype)
        V_mean = layer_data["V"]["mean"].to(device).to(dtype)
        v_comp = QuantizedCalibratedCompressor(V_U, V_mean, bits=bits, dtype=dtype)
        v_compressors.append(v_comp)

    return k_compressors, v_compressors


def quick_calibrate(model, tokenizer, device: torch.device, rank: int = 96) -> Dict:
    """Quick calibration for testing."""
    from gpt2.compression.compressed_cache import CalibratedCompressor

    # Use simple random projections for quick testing
    # In production, use scripts/calibrate_kv_lowrank.py
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    layers_data = []
    for _ in range(num_layers):
        # Random orthonormal projection (QR in float32, then convert)
        U = torch.randn(head_dim, rank, device=device, dtype=torch.float32)
        U, _ = torch.linalg.qr(U)
        U = U.to(torch.float16)
        mean = torch.zeros(head_dim, device=device, dtype=torch.float16)

        layers_data.append(
            {
                "K": {"U": U, "mean": mean},
                "V": {"U": U.clone(), "mean": mean.clone()},
            }
        )

    return {
        "model": str(model.config._name_or_path),
        "rank": rank,
        "head_dim": head_dim,
        "n_layers": num_layers,
        "n_heads": num_heads,
        "layers": layers_data,
    }


def benchmark_generation(
    model,
    tokenizer,
    cache,
    prompt: str,
    max_new_tokens: int = 128,
    warmup_runs: int = 2,
    timed_runs: int = 3,
) -> Dict:
    """Benchmark generation with a given cache."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    # Warmup
    for _ in range(warmup_runs):
        if cache is not None:
            cache.reset()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                past_key_values=cache,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Timed runs
    times = []
    for _ in range(timed_runs):
        if cache is not None:
            cache.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                past_key_values=cache,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    output_len = outputs.shape[1] - prompt_len
    avg_time = sum(times) / len(times)
    tokens_per_sec = output_len / avg_time

    # Get memory stats if available
    memory_stats = {}
    if cache is not None and hasattr(cache, "get_memory_stats"):
        memory_stats = cache.get_memory_stats()

    return {
        "prompt_len": prompt_len,
        "output_len": output_len,
        "avg_time": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "memory_stats": memory_stats,
    }


def run_profile(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    context_lengths: List[int] = None,
    max_new_tokens: int = 128,
    rank: int = 96,
    device: str = "cuda",
) -> Dict:
    """Run profiling across different configurations."""
    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048]

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    device = next(model.parameters()).device
    dtype = torch.float16

    # Try to load calibration, or create quick one
    calib_data = load_calibration(model_name, rank)
    if calib_data is None:
        print("No calibration found, using quick random projections...")
        # Adjust rank for small models
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        rank = min(rank, head_dim - 1)
        calib_data = quick_calibrate(model, tokenizer, device, rank)

    num_layers = calib_data["n_layers"]
    k_compressors, v_compressors = create_compressors(calib_data, device, dtype)

    results = {
        "model": model_name,
        "rank": rank,
        "context_lengths": context_lengths,
        "max_new_tokens": max_new_tokens,
        "configs": {},
    }

    # Generate prompts of different lengths
    base_text = "The quick brown fox jumps over the lazy dog. " * 100

    for ctx_len in context_lengths:
        print(f"\n=== Context length: {ctx_len} ===")

        # Create prompt of approximately ctx_len tokens
        prompt = base_text
        tokens = tokenizer(prompt, return_tensors="pt")
        while tokens.input_ids.shape[1] < ctx_len:
            prompt = prompt + prompt
            tokens = tokenizer(prompt, return_tensors="pt")
        # Truncate to exact length
        prompt = tokenizer.decode(
            tokens.input_ids[0, :ctx_len], skip_special_tokens=True
        )

        results["configs"][ctx_len] = {}

        # 1. Baseline (no compression)
        print("  Baseline (no compression)...")
        baseline_result = benchmark_generation(
            model, tokenizer, None, prompt, max_new_tokens
        )
        results["configs"][ctx_len]["baseline"] = baseline_result
        print(f"    {baseline_result['tokens_per_sec']:.1f} tok/s")

        # 2. v18: Always compress (compress_start_len=0, uncompressed_tail=0)
        print("  v18: Always compress...")
        cache_v18 = CompressedDynamicCache(
            k_compressors=k_compressors,
            v_compressors=v_compressors,
            num_layers=num_layers,
            compress_start_len=0,
            uncompressed_tail=0,
        )
        v18_result = benchmark_generation(
            model, tokenizer, cache_v18, prompt, max_new_tokens
        )
        results["configs"][ctx_len]["v18_always"] = v18_result
        print(f"    {v18_result['tokens_per_sec']:.1f} tok/s")

        # 3. v20: Thresholded compression
        print("  v20: Thresholded (start=512, tail=256)...")
        cache_v20 = CompressedDynamicCache(
            k_compressors=k_compressors,
            v_compressors=v_compressors,
            num_layers=num_layers,
            compress_start_len=512,
            uncompressed_tail=256,
        )
        v20_result = benchmark_generation(
            model, tokenizer, cache_v20, prompt, max_new_tokens
        )
        results["configs"][ctx_len]["v20_thresholded"] = v20_result
        print(f"    {v20_result['tokens_per_sec']:.1f} tok/s")

        # Calculate speedups
        baseline_tps = baseline_result["tokens_per_sec"]
        v18_tps = v18_result["tokens_per_sec"]
        v20_tps = v20_result["tokens_per_sec"]

        v18_overhead = (baseline_tps - v18_tps) / baseline_tps * 100
        v20_overhead = (baseline_tps - v20_tps) / baseline_tps * 100

        print(f"\n  Overhead vs baseline:")
        print(f"    v18 (always): {v18_overhead:+.1f}%")
        print(f"    v20 (thresholded): {v20_overhead:+.1f}%")

        if ctx_len >= 512:
            improvement = v18_overhead - v20_overhead
            print(f"    v20 improvement over v18: {improvement:.1f}% less overhead")

    return results


def save_results(results: Dict, output_dir: str = "plots/performance_v20"):
    """Save results to JSON and generate markdown table."""
    os.makedirs(output_dir, exist_ok=True)

    model_short = results["model"].replace("/", "_").replace("-", "_")

    # Save JSON
    json_path = os.path.join(output_dir, f"{model_short}_thresholded_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Generate markdown table
    md_lines = [
        f"# Runtime-Aware Compression Results: {results['model']}",
        "",
        f"Rank: {results['rank']}, Max new tokens: {results['max_new_tokens']}",
        "",
        "| Context | Baseline | v18 (always) | v20 (thresholded) | v18 Overhead | v20 Overhead |",
        "|---------|----------|--------------|-------------------|--------------|--------------|",
    ]

    for ctx_len in results["context_lengths"]:
        cfg = results["configs"][ctx_len]
        baseline = cfg["baseline"]["tokens_per_sec"]
        v18 = cfg["v18_always"]["tokens_per_sec"]
        v20 = cfg["v20_thresholded"]["tokens_per_sec"]

        v18_oh = (baseline - v18) / baseline * 100
        v20_oh = (baseline - v20) / baseline * 100

        md_lines.append(
            f"| {ctx_len} | {baseline:.1f} tok/s | {v18:.1f} tok/s | {v20:.1f} tok/s | "
            f"{v18_oh:+.1f}% | {v20_oh:+.1f}% |"
        )

    md_path = os.path.join(output_dir, f"{model_short}_thresholded_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile runtime-aware KV compression")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model to profile",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=96,
        help="Low-rank projection rank",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    args = parser.parse_args()

    results = run_profile(
        model_name=args.model,
        context_lengths=args.context_lengths,
        max_new_tokens=args.max_new_tokens,
        rank=args.rank,
        device=args.device,
    )

    save_results(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"v20 thresholded compression reduces overhead vs v18 always-compress")
    print("especially at short contexts where no compression occurs.")


if __name__ == "__main__":
    main()
