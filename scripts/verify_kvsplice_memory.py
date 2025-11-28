#!/usr/bin/env python3
"""Verify KVSplice actually reduces KV cache memory during inference."""

import torch
import torch.nn.functional as F
import gc
import sys
import os

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


def measure_cache_size_from_blocks(model, seq_len, batch_size=1):
    """
    Measure KV cache size by extracting it from the model's blocks.

    This approach directly measures the cache tensors returned by the attention
    layers, giving us the actual memory footprint.
    """
    device = next(model.parameters()).device
    model.eval()

    # Create input
    x = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    with torch.no_grad():
        # Get embeddings (same as forward pass)
        B, T = x.shape
        tok_emb = model.wte(x)
        pos_emb = model.wpe(torch.arange(T, device=x.device))
        hidden = model.drop(tok_emb + pos_emb)

        # Run through blocks and collect caches
        caches = []
        for block in model.blocks:
            hidden, cache = block(hidden, cache=None, use_cache=True)
            caches.append(cache)

        # Measure cache sizes
        total_cache_bytes = 0
        cache_shapes = []

        for i, cache in enumerate(caches):
            if cache is not None:
                cache_bytes = cache.numel() * cache.element_size()
                total_cache_bytes += cache_bytes
                cache_shapes.append(
                    {
                        "layer": i,
                        "shape": tuple(cache.shape),
                        "numel": cache.numel(),
                        "bytes": cache_bytes,
                        "mb": cache_bytes / (1024**2),
                    }
                )

        total_cache_mb = total_cache_bytes / (1024**2)

        return {
            "total_cache_mb": total_cache_mb,
            "total_cache_bytes": total_cache_bytes,
            "cache_shapes": cache_shapes,
            "n_layers": len(caches),
        }


def main():
    print("=" * 70)
    print("KVSplice Memory Verification")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Test sequence lengths
    seq_lengths = [256, 512, 1024]

    print(f"\nDevice: {device}")
    print(f"Model: GPT-2 124M (d_model={config.d_model}, n_layers={config.n_layers})")
    print(f"MLA latent: d_latent={config.d_latent}")

    # Test MLA (6x compression)
    print("\n" + "=" * 70)
    print("Testing MLA (6x compression)")
    print("=" * 70)

    clear_gpu_memory()
    model_mla = GPT2_MLA(config).to(device)

    mla_results = {}
    for seq_len in seq_lengths:
        print(f"\nSeq length: {seq_len}")
        result = measure_cache_size_from_blocks(model_mla, seq_len)
        mla_results[seq_len] = result

        print(f"  Total cache memory: {result['total_cache_mb']:.2f} MB")
        print(
            f"  Cache per layer: {result['total_cache_mb'] / result['n_layers']:.2f} MB"
        )
        if seq_len == 256:
            # Show first layer details for smallest sequence
            print(f"  First layer cache shape: {result['cache_shapes'][0]['shape']}")

    del model_mla
    clear_gpu_memory()

    # Test MLA+KVSplice (12x compression)
    print("\n" + "=" * 70)
    print("Testing MLA+KVSplice (12x compression, ratio=0.5)")
    print("=" * 70)

    model_kvsplice = GPT2_MLA_KV(config, compression_ratio=0.5).to(device)

    kvsplice_results = {}
    for seq_len in seq_lengths:
        print(f"\nSeq length: {seq_len}")
        result = measure_cache_size_from_blocks(model_kvsplice, seq_len)
        kvsplice_results[seq_len] = result

        print(f"  Total cache memory: {result['total_cache_mb']:.2f} MB")
        print(
            f"  Cache per layer: {result['total_cache_mb'] / result['n_layers']:.2f} MB"
        )
        if seq_len == 256:
            # Show first layer details for smallest sequence
            print(f"  First layer cache shape: {result['cache_shapes'][0]['shape']}")

    # Compare results
    print("\n" + "=" * 70)
    print("MEMORY SAVINGS COMPARISON")
    print("=" * 70)

    print(
        "\n{:>10} | {:>12} | {:>12} | {:>12} | {:>10}".format(
            "Seq Len", "MLA (MB)", "KVSplice (MB)", "Reduction", "Savings %"
        )
    )
    print("-" * 70)

    for seq_len in seq_lengths:
        mla_mem = mla_results[seq_len]["total_cache_mb"]
        kv_mem = kvsplice_results[seq_len]["total_cache_mb"]
        reduction = mla_mem - kv_mem
        savings_pct = (reduction / mla_mem * 100) if mla_mem > 0 else 0

        print(
            "{:>10} | {:>12.2f} | {:>12.2f} | {:>12.2f} | {:>9.1f}%".format(
                seq_len, mla_mem, kv_mem, reduction, savings_pct
            )
        )

    # Theoretical calculation
    print("\n" + "=" * 70)
    print("THEORETICAL vs ACTUAL")
    print("=" * 70)

    # Standard KV cache: 2 * n_layers * seq_len * d_model * 2 bytes (fp16)
    # MLA cache: n_layers * seq_len * d_latent * 2 bytes
    # KVSplice cache: n_layers * seq_len * (d_latent * compression_ratio) * 2 bytes

    d_model = config.d_model
    d_latent = config.d_latent
    n_layers = config.n_layers
    compression_ratio = 0.5

    print(f"\nTheoretical calculation:")
    print(
        f"  Standard GPT-2: 2 * {n_layers} layers * seq_len * {d_model} dims * 2 bytes"
    )
    print(f"  MLA (6x): {n_layers} layers * seq_len * {d_latent} dims * 2 bytes")
    print(
        f"  KVSplice (12x): {n_layers} layers * seq_len * {int(d_latent * compression_ratio)} dims * 2 bytes"
    )

    print(
        "\n{:>10} | {:>15} | {:>15} | {:>15}".format(
            "Seq Len", "Standard (MB)", "MLA Theory (MB)", "KVSplice Theory (MB)"
        )
    )
    print("-" * 70)

    for seq_len in seq_lengths:
        standard_mb = (2 * n_layers * seq_len * d_model * 2) / (1024**2)
        mla_theory_mb = (n_layers * seq_len * d_latent * 2) / (1024**2)
        kv_theory_mb = (n_layers * seq_len * int(d_latent * compression_ratio) * 2) / (
            1024**2
        )

        print(
            "{:>10} | {:>15.2f} | {:>15.2f} | {:>15.2f}".format(
                seq_len, standard_mb, mla_theory_mb, kv_theory_mb
            )
        )

    # Compare theoretical vs actual
    print("\n" + "=" * 70)
    print("THEORETICAL vs ACTUAL COMPARISON")
    print("=" * 70)

    print(
        "\n{:>10} | {:>15} | {:>15} | {:>10}".format(
            "Seq Len", "MLA Actual (MB)", "MLA Theory (MB)", "Diff %"
        )
    )
    print("-" * 70)

    for seq_len in seq_lengths:
        actual = mla_results[seq_len]["total_cache_mb"]
        theory = (n_layers * seq_len * d_latent * 2) / (1024**2)
        diff_pct = ((actual - theory) / theory * 100) if theory > 0 else 0

        print(
            "{:>10} | {:>15.2f} | {:>15.2f} | {:>9.1f}%".format(
                seq_len, actual, theory, diff_pct
            )
        )

    print(
        "\n{:>10} | {:>15} | {:>15} | {:>10}".format(
            "Seq Len", "KV Actual (MB)", "KV Theory (MB)", "Diff %"
        )
    )
    print("-" * 70)

    for seq_len in seq_lengths:
        actual = kvsplice_results[seq_len]["total_cache_mb"]
        theory = (n_layers * seq_len * int(d_latent * compression_ratio) * 2) / (
            1024**2
        )
        diff_pct = ((actual - theory) / theory * 100) if theory > 0 else 0

        print(
            "{:>10} | {:>15.2f} | {:>15.2f} | {:>9.1f}%".format(
                seq_len, actual, theory, diff_pct
            )
        )

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if actual savings match theory
    seq_len = 1024  # Use longest sequence for comparison
    actual_savings = (
        (
            mla_results[seq_len]["total_cache_mb"]
            - kvsplice_results[seq_len]["total_cache_mb"]
        )
        / mla_results[seq_len]["total_cache_mb"]
        * 100
    )
    theoretical_savings = 50.0  # 2x compression from 0.5 ratio

    print(f"\nAt seq_len={seq_len}:")
    print(f"  Actual savings: {actual_savings:.1f}%")
    print(f"  Theoretical savings: {theoretical_savings:.1f}%")

    if abs(actual_savings - theoretical_savings) < 5:
        print("\n✅ KVSplice compression is working as expected!")
        print(f"   Cache memory reduced by ~{actual_savings:.0f}% during inference")
    else:
        print(
            f"\n⚠️  Discrepancy detected: {abs(actual_savings - theoretical_savings):.1f}% difference"
        )
        print("   Difference might be due to tensor padding or dtype variations")


if __name__ == "__main__":
    main()
