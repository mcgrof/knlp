#!/usr/bin/env python3
"""Benchmark KVSplice inference throughput improvements."""

import torch
import torch.nn.functional as F
import gc
import sys
import os
import time
from typing import Tuple, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.mla import GPT2_MLA, GPT2_MLA_KV, MLA_Config


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_gpu_memory_info():
    """Get GPU memory info."""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        free = total - reserved
        return {
            "total_mb": total,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
        }
    return None


def benchmark_generation(
    model,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    warmup: int = 3,
    trials: int = 10,
) -> Dict:
    """
    Benchmark autoregressive generation throughput.

    Args:
        model: The model to benchmark
        batch_size: Number of parallel sequences
        prompt_len: Length of input prompt
        gen_len: Number of tokens to generate
        warmup: Number of warmup iterations
        trials: Number of benchmark trials

    Returns:
        Dict with throughput metrics
    """
    device = next(model.parameters()).device
    model.eval()

    # Create prompt
    prompt = torch.randint(0, 50257, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(prompt)

    clear_gpu_memory()
    mem_before = get_gpu_memory_mb()

    # Benchmark generation with KV caching
    times = []
    with torch.no_grad():
        for _ in range(trials):
            start_time = time.perf_counter()

            # Process prompt and initialize cache
            # For GPT2_MLA models, we need to go through blocks manually
            # to use caching since the top-level forward doesn't expose it
            B, T = prompt.shape
            tok_emb = model.wte(prompt)
            pos_emb = model.wpe(torch.arange(T, device=prompt.device))
            hidden = model.drop(tok_emb + pos_emb)

            # Run through blocks to get initial cache
            caches = []
            for block in model.blocks:
                hidden, cache = block(hidden, cache=None, use_cache=True)
                caches.append(cache)

            # Get logits from prompt
            hidden = model.ln_f(hidden)
            logits = model.lm_head(hidden)

            # Generate tokens one at a time using cache
            for i in range(gen_len):
                # Sample next token
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)

                # Get embeddings for new token
                tok_emb = model.wte(next_token)
                pos = T + i
                pos_emb = model.wpe(torch.tensor([pos], device=next_token.device))
                hidden = model.drop(tok_emb + pos_emb)

                # Run through blocks with cache
                new_caches = []
                for block, cache in zip(model.blocks, caches):
                    hidden, new_cache = block(hidden, cache=cache, use_cache=True)
                    new_caches.append(new_cache)
                caches = new_caches

                # Get logits
                hidden = model.ln_f(hidden)
                logits = model.lm_head(hidden)

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

    mem_after = get_gpu_memory_mb()
    mem_used = mem_after - mem_before

    # Calculate metrics
    total_tokens = batch_size * gen_len * trials
    total_time = sum(times)
    throughput = total_tokens / total_time

    avg_time = sum(times) / len(times)
    tokens_per_trial = batch_size * gen_len
    trial_throughput = tokens_per_trial / avg_time

    return {
        "batch_size": batch_size,
        "avg_time_s": avg_time,
        "throughput_tok_s": trial_throughput,
        "total_throughput_tok_s": throughput,
        "memory_used_mb": mem_used,
        "memory_per_batch_mb": mem_used / batch_size if batch_size > 0 else 0,
    }


def find_max_batch_size(
    model, seq_len: int, start_batch: int = 1, max_batch: int = 256
) -> int:
    """
    Binary search to find maximum batch size that fits in memory.

    Args:
        model: The model to test
        seq_len: Sequence length to test
        start_batch: Starting batch size
        max_batch: Maximum batch size to try

    Returns:
        Maximum batch size that doesn't OOM
    """
    device = next(model.parameters()).device
    model.eval()

    def test_batch_size(batch_size: int) -> bool:
        """Test if batch_size fits in memory."""
        try:
            clear_gpu_memory()
            x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                clear_gpu_memory()
                return False
            raise

    # Binary search
    low, high = start_batch, max_batch
    best = start_batch

    while low <= high:
        mid = (low + high) // 2
        if test_batch_size(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def main():
    print("=" * 80)
    print("KVSplice Inference Throughput Benchmark")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires GPU.")
        return

    # GPU info
    gpu_info = get_gpu_memory_info()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {gpu_info['total_mb']:.0f} MB")
    print(f"Available memory: {gpu_info['free_mb']:.0f} MB")

    # Model config (GPT-2 124M equivalent)
    config = MLA_Config(
        d_model=768,
        n_heads=12,
        head_dim=64,
        d_latent=256,
        block_size=1024,
        n_layers=12,
        dropout=0.0,
    )

    print(f"\nModel: GPT-2 124M (d_model={config.d_model}, n_layers={config.n_layers})")
    print(f"MLA latent: d_latent={config.d_latent}")

    # Test parameters
    prompt_len = 128
    gen_len = 128
    seq_len = prompt_len + gen_len

    print(f"\nBenchmark parameters:")
    print(f"  Prompt length: {prompt_len} tokens")
    print(f"  Generation length: {gen_len} tokens")
    print(f"  Total sequence: {seq_len} tokens")

    # Find maximum batch sizes
    print("\n" + "=" * 80)
    print("Finding Maximum Batch Sizes")
    print("=" * 80)

    print("\nTesting MLA (6x compression)...")
    clear_gpu_memory()
    model_mla = GPT2_MLA(config).to(device)
    mla_model_mem = get_gpu_memory_mb()
    print(f"  Model memory: {mla_model_mem:.2f} MB")

    max_batch_mla = find_max_batch_size(model_mla, seq_len)
    print(f"  Maximum batch size: {max_batch_mla}")

    del model_mla
    clear_gpu_memory()

    print("\nTesting MLA+KVSplice (12x compression)...")
    model_kvsplice = GPT2_MLA_KV(config, compression_ratio=0.5).to(device)
    kv_model_mem = get_gpu_memory_mb()
    print(f"  Model memory: {kv_model_mem:.2f} MB")

    max_batch_kv = find_max_batch_size(model_kvsplice, seq_len)
    print(f"  Maximum batch size: {max_batch_kv}")

    batch_increase = (max_batch_kv / max_batch_mla - 1) * 100
    print(
        f"\n  Batch size increase: {max_batch_kv - max_batch_mla} (+{batch_increase:.1f}%)"
    )

    del model_kvsplice
    clear_gpu_memory()

    # Throughput benchmarks at various batch sizes
    print("\n" + "=" * 80)
    print("Throughput Benchmarks")
    print("=" * 80)

    # Test at multiple batch sizes
    test_batches = [1, 2, 4, 8, 16, 32]
    # Always add max batch sizes
    test_batches.append(max_batch_mla)
    test_batches.append(max_batch_kv)
    test_batches = sorted(set(test_batches))

    # Filter to only test batches that fit on both models
    common_batches = [b for b in test_batches if b <= max_batch_mla]

    print("\n" + "-" * 80)
    print("MLA (6x compression)")
    print("-" * 80)

    model_mla = GPT2_MLA(config).to(device)
    mla_results = {}

    for batch_size in common_batches:
        print(f"\nBatch size: {batch_size}")
        result = benchmark_generation(model_mla, batch_size, prompt_len, gen_len)
        mla_results[batch_size] = result

        print(f"  Throughput: {result['throughput_tok_s']:.1f} tokens/sec")
        print(f"  Avg time: {result['avg_time_s']*1000:.1f} ms")
        print(f"  Memory used: {result['memory_used_mb']:.2f} MB")

    del model_mla
    clear_gpu_memory()

    print("\n" + "-" * 80)
    print("MLA+KVSplice (12x compression)")
    print("-" * 80)

    model_kvsplice = GPT2_MLA_KV(config, compression_ratio=0.5).to(device)
    kv_results = {}

    for batch_size in common_batches:
        print(f"\nBatch size: {batch_size}")
        result = benchmark_generation(model_kvsplice, batch_size, prompt_len, gen_len)
        kv_results[batch_size] = result

        print(f"  Throughput: {result['throughput_tok_s']:.1f} tokens/sec")
        print(f"  Avg time: {result['avg_time_s']*1000:.1f} ms")
        print(f"  Memory used: {result['memory_used_mb']:.2f} MB")

    # Test KVSplice at higher batches if it supports them
    kv_only_batches = [
        b for b in test_batches if b > max_batch_mla and b <= max_batch_kv
    ]
    if kv_only_batches:
        print("\n(KVSplice-only batch sizes due to higher capacity)")
        for batch_size in kv_only_batches:
            print(f"\nBatch size: {batch_size}")
            result = benchmark_generation(
                model_kvsplice, batch_size, prompt_len, gen_len
            )
            kv_results[batch_size] = result

            print(f"  Throughput: {result['throughput_tok_s']:.1f} tokens/sec")
            print(f"  Avg time: {result['avg_time_s']*1000:.1f} ms")
            print(f"  Memory used: {result['memory_used_mb']:.2f} MB")

    del model_kvsplice
    clear_gpu_memory()

    # Summary comparison
    print("\n" + "=" * 80)
    print("THROUGHPUT COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Batch':<8} | {'MLA (tok/s)':<12} | {'KVSplice (tok/s)':<15} | {'Speedup':<10} | {'Memory Savings'}"
    )
    print("-" * 80)

    for batch_size in common_batches:
        mla = mla_results[batch_size]
        kv = kv_results[batch_size]

        speedup = kv["throughput_tok_s"] / mla["throughput_tok_s"]
        mem_savings = (1 - kv["memory_used_mb"] / mla["memory_used_mb"]) * 100

        print(
            f"{batch_size:<8} | {mla['throughput_tok_s']:>11.1f} | "
            f"{kv['throughput_tok_s']:>14.1f} | {speedup:>9.2f}x | {mem_savings:>12.1f}%"
        )

    # Maximum throughput comparison
    print("\n" + "=" * 80)
    print("MAXIMUM THROUGHPUT (at max batch size)")
    print("=" * 80)

    mla_max_throughput = mla_results[max_batch_mla]["throughput_tok_s"]
    kv_max_throughput = kv_results[max_batch_kv]["throughput_tok_s"]
    max_speedup = kv_max_throughput / mla_max_throughput

    print(f"\nMLA (batch={max_batch_mla}):")
    print(f"  Throughput: {mla_max_throughput:.1f} tokens/sec")
    print(f"  Memory: {mla_results[max_batch_mla]['memory_used_mb']:.2f} MB")

    print(f"\nKVSplice (batch={max_batch_kv}):")
    print(f"  Throughput: {kv_max_throughput:.1f} tokens/sec")
    print(f"  Memory: {kv_results[max_batch_kv]['memory_used_mb']:.2f} MB")

    print(
        f"\nMaximum throughput improvement: {max_speedup:.2f}x ({(max_speedup-1)*100:.1f}% faster)"
    )

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    print(
        f"\nCache compression enables {max_batch_kv / max_batch_mla:.1f}x larger batch sizes"
    )
    print(f"This translates to {max_speedup:.2f}x higher inference throughput")
    print(f"\nFor production serving with {gpu_info['total_mb']:.0f} MB GPU memory:")
    print(f"  MLA: ~{max_batch_mla} parallel sequences")
    print(f"  KVSplice: ~{max_batch_kv} parallel sequences")
    print(
        f"  Throughput gain: {kv_max_throughput:.0f} vs {mla_max_throughput:.0f} tokens/sec"
    )

    if max_speedup >= 1.5:
        print(
            f"\n✅ KVSplice provides {max_speedup:.1f}x throughput improvement - significant gain!"
        )
    elif max_speedup >= 1.1:
        print(
            f"\n✅ KVSplice provides {max_speedup:.1f}x throughput improvement - modest gain"
        )
    else:
        print(
            f"\n⚠️  KVSplice provides {max_speedup:.1f}x throughput - minimal improvement"
        )
        print("   Cache reduction may not be the bottleneck on this GPU")


if __name__ == "__main__":
    main()
