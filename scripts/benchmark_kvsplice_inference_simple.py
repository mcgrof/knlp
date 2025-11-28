#!/usr/bin/env python3
"""Simple benchmark comparing MLA vs KVSplice inference throughput."""

import torch
import gc
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.mla import GPT2_MLA, GPT2_MLA_KV, MLA_Config


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure_generation(model, batch_size, prompt_len=64, gen_len=64, trials=5):
    """Measure generation throughput with KV caching."""
    device = next(model.parameters()).device
    model.eval()

    prompt = torch.randint(0, 50257, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        B, T = prompt.shape
        tok_emb = model.wte(prompt)
        pos_emb = model.wpe(torch.arange(T, device=device))
        hidden = model.drop(tok_emb + pos_emb)
        caches = []
        for block in model.blocks:
            hidden, cache = block(hidden, cache=None, use_cache=True)
            caches.append(cache)

    clear_gpu()
    mem_before = torch.cuda.memory_allocated() / 1024 / 1024

    times = []
    with torch.no_grad():
        for _ in range(trials):
            # Process prompt
            B, T = prompt.shape
            tok_emb = model.wte(prompt)
            pos_emb = model.wpe(torch.arange(T, device=device))
            hidden = model.drop(tok_emb + pos_emb)

            caches = []
            for block in model.blocks:
                hidden, cache = block(hidden, cache=None, use_cache=True)
                caches.append(cache)

            hidden = model.ln_f(hidden)
            logits = model.lm_head(hidden)

            start = time.perf_counter()

            # Generate with caching
            for i in range(gen_len):
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)
                tok_emb = model.wte(next_token)
                pos = T + i
                pos_emb = model.wpe(torch.tensor([pos], device=device))
                hidden = model.drop(tok_emb + pos_emb)

                new_caches = []
                for block, cache in zip(model.blocks, caches):
                    hidden, new_cache = block(hidden, cache=cache, use_cache=True)
                    new_caches.append(new_cache)
                caches = new_caches

                hidden = model.ln_f(hidden)
                logits = model.lm_head(hidden)

            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    mem_after = torch.cuda.memory_allocated() / 1024 / 1024
    avg_time = sum(times) / len(times)
    throughput = (batch_size * gen_len) / avg_time

    # Measure final cache size
    cache_size = sum(c.numel() * c.element_size() for c in caches) / 1024 / 1024

    return {
        "throughput": throughput,
        "avg_time_ms": avg_time * 1000,
        "cache_mb": cache_size,
        "mem_used_mb": mem_after - mem_before,
    }


def main():
    print("=" * 70)
    print("KVSplice Inference Throughput Benchmark (Simple)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA required")
        return

    device = torch.device("cuda")
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"Total memory: {total_mem:.0f} MB")

    config = MLA_Config(
        d_model=768,
        n_heads=12,
        head_dim=64,
        d_latent=256,
        block_size=1024,
        n_layers=12,
        dropout=0.0,
    )

    print(f"\nModel: GPT-2 124M")
    print(f"d_latent: {config.d_latent}")

    # Test parameters - use longer sequences to stress memory
    prompt_len = 512
    gen_len = 512
    test_batches = [1, 2, 4, 8, 16, 24, 32]

    print(f"\nPrompt: {prompt_len} tokens, Generate: {gen_len} tokens")
    print(f"Testing batch sizes: {test_batches}")

    # MLA
    print("\n" + "=" * 70)
    print("MLA (6x compression)")
    print("=" * 70)

    clear_gpu()
    model_mla = GPT2_MLA(config).to(device)
    model_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Model memory: {model_mem:.1f} MB\n")

    mla_results = {}
    for batch in test_batches:
        try:
            result = measure_generation(model_mla, batch, prompt_len, gen_len)
            mla_results[batch] = result
            print(
                f"Batch {batch:2d}: {result['throughput']:6.1f} tok/s, "
                f"cache: {result['cache_mb']:5.2f} MB, "
                f"time: {result['avg_time_ms']:6.1f} ms"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch {batch:2d}: OOM")
                clear_gpu()
                break
            raise

    del model_mla
    clear_gpu()

    # KVSplice
    print("\n" + "=" * 70)
    print("MLA+KVSplice (12x compression)")
    print("=" * 70)

    model_kv = GPT2_MLA_KV(config, compression_ratio=0.5).to(device)
    model_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Model memory: {model_mem:.1f} MB\n")

    kv_results = {}
    for batch in test_batches:
        try:
            result = measure_generation(model_kv, batch, prompt_len, gen_len)
            kv_results[batch] = result
            print(
                f"Batch {batch:2d}: {result['throughput']:6.1f} tok/s, "
                f"cache: {result['cache_mb']:5.2f} MB, "
                f"time: {result['avg_time_ms']:6.1f} ms"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch {batch:2d}: OOM")
                clear_gpu()
                break
            raise

    del model_kv
    clear_gpu()

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Batch':>5} | {'MLA tok/s':>10} | {'KV tok/s':>10} | "
        f"{'Speedup':>8} | {'Cache Savings':>13}"
    )
    print("-" * 70)

    common = sorted(set(mla_results.keys()) & set(kv_results.keys()))
    for batch in common:
        mla = mla_results[batch]
        kv = kv_results[batch]
        speedup = kv["throughput"] / mla["throughput"]
        cache_save = (1 - kv["cache_mb"] / mla["cache_mb"]) * 100

        print(
            f"{batch:>5} | {mla['throughput']:>10.1f} | {kv['throughput']:>10.1f} | "
            f"{speedup:>7.2f}x | {cache_save:>12.1f}%"
        )

    # Summary
    if common:
        batch = max(common)
        mla = mla_results[batch]
        kv = kv_results[batch]
        speedup = kv["throughput"] / mla["throughput"]

        print(f"\nAt batch={batch}:")
        print(f"  Cache: {mla['cache_mb']:.1f} MB → {kv['cache_mb']:.1f} MB")
        print(f"  Throughput: {mla['throughput']:.0f} → {kv['throughput']:.0f} tok/s")
        print(f"  Speedup: {speedup:.2f}x")

        if speedup >= 1.1:
            print(f"\n✅ KVSplice provides {speedup:.2f}x throughput improvement")
        elif speedup >= 0.95:
            print(f"\n✅ KVSplice matches MLA throughput ({speedup:.2f}x)")
        else:
            print(f"\n⚠️  KVSplice slower than MLA ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
