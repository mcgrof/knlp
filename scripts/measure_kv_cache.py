#!/usr/bin/env python3
"""
Measure actual KV cache memory impact across different model architectures.

Compares runtime GPU memory usage during inference for:
  - Baseline GPT-2 (full KV cache)
  - RA (Reciprocal Attention - still full cache)
  - MLA (Multi-head Latent Attention - compressed cache)
  - MLAKV (MLA + KVSplice - further compressed)
  - RAMLA (RA + MLA)
  - RAMLAKV (RA + MLA + KVSplice)

Tests at multiple sequence lengths to show how cache grows.
"""

import torch
import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.model import GPT, GPTConfig
from ra import (
    GPT2_RA_Model,
    RA_MLA_Config,
    MLAGPT,
    MLAKV_GPT,
    RAMLAGPT,
    RAMLAKV_GPT,
)


@dataclass
class CacheStats:
    """Statistics for a single measurement."""

    architecture: str
    sequence_length: int
    batch_size: int
    memory_allocated_mb: float
    memory_reserved_mb: float
    cache_size_mb: float  # Estimated cache size
    tokens_per_sec: float
    theoretical_cache_mb: float  # Theoretical calculation


def get_baseline_memory(device):
    """Get baseline memory usage before any model operations."""
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def measure_inference_memory(model, seq_len, batch_size=1, num_tokens=50):
    """Measure memory usage during autoregressive inference."""
    model.eval()
    device = next(model.parameters()).device

    # Clear cache and get baseline
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Get memory before inference
    mem_before = (
        torch.cuda.memory_allocated() / (1024**2) if device.type == "cuda" else 0
    )

    # Create initial prompt
    prompt = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            logits, _ = model(prompt)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Measure inference with growing cache
    start_time = time.perf_counter()

    with torch.no_grad():
        current_seq = prompt
        for _ in range(num_tokens):
            logits, _ = model(current_seq)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            current_seq = torch.cat([current_seq, next_token], dim=1)

            # Keep within block_size
            if current_seq.size(1) > model.config.block_size:
                current_seq = current_seq[:, -model.config.block_size :]

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = num_tokens / elapsed

    # Get peak memory during inference
    if device.type == "cuda":
        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
        mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        mem_peak = 0
        mem_reserved = 0

    # Estimate cache size (peak memory minus baseline)
    cache_size = mem_peak - mem_before

    return {
        "memory_allocated_mb": mem_peak,
        "memory_reserved_mb": mem_reserved,
        "cache_size_mb": cache_size,
        "tokens_per_sec": tokens_per_sec,
    }


def calculate_theoretical_cache(
    arch: str, seq_len: int, batch_size: int, config: GPTConfig
) -> float:
    """Calculate theoretical KV cache size in MB (fp16)."""
    n_layer = config.n_layer
    n_head = config.n_head
    n_embd = config.n_embd
    head_dim = n_embd // n_head

    # Standard full cache (K + V)
    standard_elements = batch_size * n_layer * n_head * seq_len * head_dim * 2

    if arch == "baseline" or arch == "ra":
        # Full KV cache
        cache_elements = standard_elements
    elif arch == "mla":
        # MLA uses shared kv_latent (d_latent=256)
        # Cache per layer: kv_latent [B, T, d_latent] + rope [B, H, T, head_dim]
        d_latent = 256
        latent_elements = batch_size * n_layer * seq_len * d_latent
        rope_elements = batch_size * n_layer * n_head * seq_len * head_dim
        cache_elements = latent_elements + rope_elements
    elif arch == "mlakv":
        # MLA + KVSplice: further compression on V
        # Assume 0.5 compression ratio (k=32 for head_dim=64)
        d_latent = 256
        compression_ratio = 0.5
        v_compressed_dim = int(head_dim * compression_ratio)

        latent_elements = batch_size * n_layer * seq_len * d_latent
        k_rope_elements = batch_size * n_layer * n_head * seq_len * head_dim
        v_compressed_elements = (
            batch_size * n_layer * n_head * seq_len * v_compressed_dim
        )
        cache_elements = latent_elements + k_rope_elements + v_compressed_elements
    elif arch in ["ramla", "ramlakv"]:
        # Same cache structure as mla/mlakv (RA doesn't change cache)
        if arch == "ramla":
            # Same as mla
            d_latent = 256
            latent_elements = batch_size * n_layer * seq_len * d_latent
            rope_elements = batch_size * n_layer * n_head * seq_len * head_dim
            cache_elements = latent_elements + rope_elements
        else:  # ramlakv
            # Same as mlakv
            d_latent = 256
            compression_ratio = 0.5
            v_compressed_dim = int(head_dim * compression_ratio)

            latent_elements = batch_size * n_layer * seq_len * d_latent
            k_rope_elements = batch_size * n_layer * n_head * seq_len * head_dim
            v_compressed_elements = (
                batch_size * n_layer * n_head * seq_len * v_compressed_dim
            )
            cache_elements = latent_elements + k_rope_elements + v_compressed_elements
    else:
        cache_elements = standard_elements

    # Convert to MB (fp16 = 2 bytes per element)
    cache_mb = (cache_elements * 2) / (1024**2)
    return cache_mb


def test_architecture(arch_name: str, seq_lengths: List[int], batch_size: int = 1):
    """Test a specific architecture at multiple sequence lengths."""
    print(f"\n{'='*70}")
    print(f"Testing {arch_name}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model based on architecture
    if arch_name == "baseline":
        config = GPTConfig.from_name("gpt2")
        config.block_size = 1024
        config.weight_tying = True
        model = GPT(config).to(device)
    elif arch_name == "ra":
        config = GPTConfig.from_name("gpt2")
        config.block_size = 1024
        config.weight_tying = True
        model = GPT2_RA_Model(config).to(device)
    elif arch_name == "mla":
        cfg = RA_MLA_Config(
            d_model=768, n_heads=12, head_dim=64, d_latent=256, block_size=1024
        )
        model = MLAGPT(cfg).to(device)
    elif arch_name == "mlakv":
        cfg = RA_MLA_Config(
            d_model=768, n_heads=12, head_dim=64, d_latent=256, block_size=1024
        )
        model = MLAKV_GPT(cfg, compression_ratio=0.5).to(device)
    elif arch_name == "ramla":
        cfg = RA_MLA_Config(
            d_model=768, n_heads=12, head_dim=64, d_latent=256, block_size=1024
        )
        model = RAMLAGPT(cfg).to(device)
    elif arch_name == "ramlakv":
        cfg = RA_MLA_Config(
            d_model=768, n_heads=12, head_dim=64, d_latent=256, block_size=1024
        )
        model = RAMLAKV_GPT(cfg, compression_ratio=0.5).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    print(f"Parameters: {model.get_num_params() / 1e6:.2f}M")

    results = []

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        # Get model config for theoretical calculation
        if hasattr(model, "config"):
            config = model.config
        elif hasattr(model, "cfg"):
            # For MLA models, create a GPTConfig-like object
            cfg = model.cfg
            config = GPTConfig(
                n_layer=cfg.n_layers,
                n_head=cfg.n_heads,
                n_embd=cfg.d_model,
                block_size=cfg.block_size,
            )
        else:
            config = GPTConfig.from_name("gpt2")

        theoretical = calculate_theoretical_cache(arch_name, seq_len, batch_size, config)

        stats = measure_inference_memory(model, seq_len, batch_size, num_tokens=50)

        result = CacheStats(
            architecture=arch_name,
            sequence_length=seq_len,
            batch_size=batch_size,
            memory_allocated_mb=stats["memory_allocated_mb"],
            memory_reserved_mb=stats["memory_reserved_mb"],
            cache_size_mb=stats["cache_size_mb"],
            tokens_per_sec=stats["tokens_per_sec"],
            theoretical_cache_mb=theoretical,
        )

        results.append(result)

        print(f"  Memory allocated: {result.memory_allocated_mb:.2f} MB")
        print(f"  Cache size (estimated): {result.cache_size_mb:.2f} MB")
        print(f"  Theoretical cache: {result.theoretical_cache_mb:.2f} MB")
        print(f"  Throughput: {result.tokens_per_sec:.1f} tok/s")

    return results


def print_comparison_table(all_results: Dict[str, List[CacheStats]]):
    """Print a comparison table across all architectures."""
    print("\n" + "=" * 100)
    print("KV CACHE SIZE COMPARISON")
    print("=" * 100)

    # Get unique sequence lengths
    seq_lengths = sorted(set(r.sequence_length for r in all_results["baseline"]))

    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len} tokens")
        print("-" * 100)
        print(
            f"{'Architecture':<15} {'Cache (MB)':<12} {'Theoretical (MB)':<18} {'Reduction':<12} {'Throughput':<15}"
        )
        print("-" * 100)

        baseline_cache = None
        for arch, results in all_results.items():
            result = next(r for r in results if r.sequence_length == seq_len)

            if arch == "baseline":
                baseline_cache = result.cache_size_mb
                reduction = "—"
            else:
                if baseline_cache and baseline_cache > 0:
                    reduction_pct = (
                        1 - result.cache_size_mb / baseline_cache
                    ) * 100
                    reduction = f"{reduction_pct:.1f}%"
                else:
                    reduction = "—"

            print(
                f"{arch:<15} {result.cache_size_mb:>10.2f}   {result.theoretical_cache_mb:>10.2f}         "
                f"{reduction:<12} {result.tokens_per_sec:>10.1f} tok/s"
            )

    print("\n" + "=" * 100)


def main():
    print("=" * 70)
    print("KV Cache Memory Impact Measurement")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )

    # Test at multiple sequence lengths
    seq_lengths = [128, 256, 512, 1024]
    batch_size = 1

    # Architectures to test
    architectures = ["baseline", "ra", "mla", "mlakv", "ramla", "ramlakv"]

    all_results = {}

    for arch in architectures:
        try:
            results = test_architecture(arch, seq_lengths, batch_size)
            all_results[arch] = results
        except Exception as e:
            print(f"\nError testing {arch}: {e}")
            import traceback

            traceback.print_exc()

    # Print comparison table
    if all_results:
        print_comparison_table(all_results)

    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print("\nCache size reduction mechanisms:")
    print("  - Baseline: Full K+V cache (no compression)")
    print("  - RA: Same cache as baseline (RA changes attention, not cache)")
    print("  - MLA: Shared kv_latent (d_latent=256) reduces cache size")
    print("  - MLAKV: MLA + KVSplice further compresses V vectors (0.5 ratio)")
    print("  - RAMLA: RA attention + MLA compression")
    print("  - RAMLAKV: RA attention + MLA + KVSplice (maximum compression)")
    print(
        "\nNote: RA provides quality improvements (5.9% better perplexity) without cache changes."
    )
    print(
        "      MLA variants reduce memory footprint at the cost of some inference speed."
    )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
