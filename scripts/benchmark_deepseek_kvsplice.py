#!/usr/bin/env python3
"""
Benchmark KVSplice compression on DeepSeek models.

Compares inference throughput and memory usage between:
- Original model (with native MLA compression)
- Model with KVSplice (additional compression on top of MLA)

Usage:
    python scripts/benchmark_deepseek_kvsplice.py \
        --model deepseek-ai/DeepSeek-V2-Lite \
        --compression-ratio 0.5 \
        --batch-sizes 1,4,8,16 \
        --seq-len 512

Requirements:
    pip install transformers torch accelerate
"""

import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_kvsplice_plugin import patch_model_with_kvsplice, get_kv_cache_size


def benchmark_generation(
    model,
    tokenizer,
    batch_size: int,
    prompt_len: int = 64,
    gen_len: int = 64,
    trials: int = 3,
    device: str = "cuda",
):
    """
    Benchmark autoregressive generation with KV caching.

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch_size: Batch size
        prompt_len: Prompt length in tokens
        gen_len: Number of tokens to generate
        trials: Number of trials to average
        device: Device to run on

    Returns:
        (tokens_per_second, memory_allocated_gb)
    """
    model.eval()
    model = model.to(device)

    # Create dummy prompt
    prompt = torch.randint(
        0, tokenizer.vocab_size, (batch_size, prompt_len), device=device
    )

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            prompt,
            max_new_tokens=gen_len,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model.generate(
                prompt,
                max_new_tokens=gen_len,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    # Compute metrics
    avg_time = sum(times) / len(times)
    total_tokens = batch_size * gen_len
    tokens_per_sec = total_tokens / avg_time

    # Memory usage
    memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB

    return tokens_per_sec, memory_allocated


def main():
    parser = argparse.ArgumentParser(description="Benchmark KVSplice on DeepSeek")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model name or path",
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.5,
        help="KVSplice compression ratio (0.5 = 2x compression)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length to generate"
    )
    parser.add_argument(
        "--prompt-len", type=int, default=64, help="Prompt length in tokens"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of trials per configuration"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--use-layernorm",
        action="store_true",
        default=True,
        help="Use LayerNorm in KVSplice latent space",
    )
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    gen_len = args.seq_len - args.prompt_len

    print("=" * 80)
    print("KVSplice Benchmark on DeepSeek Models")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression ratio: {args.compression_ratio}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prompt length: {args.prompt_len}")
    print(f"Generation length: {gen_len}")
    print(f"Trials: {args.trials}")
    print(f"Device: {args.device}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Benchmark 1: Original model
    print("\n" + "=" * 80)
    print("Benchmark 1: Original Model (MLA only)")
    print("=" * 80)

    print("Loading model...")
    model_original = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("\nRunning benchmarks...")
    results_original = []

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        tokens_per_sec, memory_gb = benchmark_generation(
            model_original,
            tokenizer,
            batch_size,
            args.prompt_len,
            gen_len,
            args.trials,
            args.device,
        )

        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
        print(f"  Memory: {memory_gb:.2f} GB")

        results_original.append((batch_size, tokens_per_sec, memory_gb))

    # Clear memory
    del model_original
    torch.cuda.empty_cache()

    # Benchmark 2: Model with KVSplice
    print("\n" + "=" * 80)
    print(f"Benchmark 2: Model with KVSplice ({args.compression_ratio}x compression)")
    print("=" * 80)

    print("Loading model...")
    model_kvsplice = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Patching with KVSplice...")
    patch_model_with_kvsplice(
        model_kvsplice,
        compression_ratio=args.compression_ratio,
        use_layernorm=args.use_layernorm,
    )

    # Estimate KV cache sizes
    orig_cache_mb, compressed_cache_mb = get_kv_cache_size(model_kvsplice)
    print(f"\nEstimated KV cache (seq_len=2048):")
    print(f"  Original: {orig_cache_mb:.1f} MB")
    print(f"  Compressed: {compressed_cache_mb:.1f} MB")
    print(
        f"  Reduction: {orig_cache_mb - compressed_cache_mb:.1f} MB ({100 * (1 - compressed_cache_mb/orig_cache_mb):.1f}%)"
    )

    print("\nRunning benchmarks...")
    results_kvsplice = []

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        tokens_per_sec, memory_gb = benchmark_generation(
            model_kvsplice,
            tokenizer,
            batch_size,
            args.prompt_len,
            gen_len,
            args.trials,
            args.device,
        )

        print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
        print(f"  Memory: {memory_gb:.2f} GB")

        results_kvsplice.append((batch_size, tokens_per_sec, memory_gb))

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Original vs KVSplice")
    print("=" * 80)
    print(
        f"{'Batch':>6} | {'Original (tok/s)':>17} | {'KVSplice (tok/s)':>17} | {'Speedup':>8} | {'Mem (GB)':>10} | {'Mem Savings':>12}"
    )
    print("-" * 80)

    for (b_orig, tps_orig, mem_orig), (b_kv, tps_kv, mem_kv) in zip(
        results_original, results_kvsplice
    ):
        assert b_orig == b_kv
        speedup = tps_kv / tps_orig
        mem_savings = mem_orig - mem_kv

        print(
            f"{b_orig:6d} | {tps_orig:17.1f} | {tps_kv:17.1f} | {speedup:8.2f}x | "
            f"{mem_orig:5.2f} → {mem_kv:4.2f} | {mem_savings:+11.2f} GB"
        )

    print("\n" + "=" * 80)
    print("Notes:")
    print("- Speedup > 1.0 means KVSplice is faster (unlikely due to extra compute)")
    print("- Speedup < 1.0 means KVSplice is slower (expected ~0.89-0.92x)")
    print("- Memory savings from reduced KV cache (positive = good)")
    print("- Trade-off: Accept ~11% speed cost for reduced memory usage")
    print("=" * 80)


if __name__ == "__main__":
    main()
