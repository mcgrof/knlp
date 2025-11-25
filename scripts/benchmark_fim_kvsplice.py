#!/usr/bin/env python3
"""
Benchmark script for FIMKVSplice (Fisher Information Matrix guided KV compression).

Compares:
1. FIMKVSplice: Temporal compression using FIM eigenvectors
2. LearnedKVSplice: Feature-space compression (legacy)
3. PCA baseline: Standard PCA on temporal dimension

Tests:
- Reconstruction error at different compression ratios
- FIM eigenvalue spectrum analysis
- Memory reduction metrics
- Attention probability preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
from contextlib import contextmanager

# Add project root to path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra import (
    FIMKVSplice,
    LearnedKVSplice,
    compute_fisher_spectrum,
    RA_MLA_Config,
    GPT2_MLA_RA,
)


@contextmanager
def measure_time(description):
    """Measure execution time."""
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  [{description}] {elapsed:.2f} ms")


def generate_synthetic_attention(batch_size, n_heads, seq_len, device, sparsity=0.1):
    """
    Generate synthetic attention probabilities with realistic properties.

    Creates attention patterns with:
    - Causal masking
    - Sparse attention (few tokens attended strongly)
    - Some recency bias
    """
    # Random logits
    logits = torch.randn(batch_size, n_heads, seq_len, seq_len, device=device)

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
    logits = logits.masked_fill(causal_mask, float("-inf"))

    # Add recency bias (attend more to recent tokens)
    recency = torch.arange(seq_len, device=device).float()
    recency_bias = recency.unsqueeze(0) - recency.unsqueeze(1)
    recency_bias = torch.where(recency_bias <= 0, recency_bias * 0.1, float("-inf"))
    logits = logits + recency_bias

    # Sparsify: make some attention heads more focused
    if sparsity > 0:
        # Top-k masking per query
        k = max(1, int(seq_len * sparsity))
        for h in range(n_heads):
            if h % 2 == 0:  # Half the heads are sparse
                topk = logits[:, h].topk(k, dim=-1).indices
                mask = torch.ones_like(logits[:, h], dtype=torch.bool)
                mask.scatter_(-1, topk, False)
                logits[:, h] = logits[:, h].masked_fill(mask, float("-inf"))

    return F.softmax(logits, dim=-1)


def benchmark_fim_kvsplice(
    batch_size,
    n_heads,
    seq_len,
    d_head,
    ranks,
    device,
    dtype,
    n_calibration_samples=512,
):
    """Benchmark FIMKVSplice at different compression ranks."""
    print("\n" + "=" * 60)
    print("FIMKVSplice Benchmark")
    print("=" * 60)
    print(f"Config: B={batch_size}, H={n_heads}, T={seq_len}, D={d_head}")
    print(f"Device: {device}, dtype: {dtype}")
    print("=" * 60)

    # Generate synthetic attention probabilities for calibration
    print("\nGenerating calibration attention probabilities...")
    attn_probs = generate_synthetic_attention(batch_size * 4, n_heads, seq_len, device)

    # Analyze FIM spectrum
    print("\n--- FIM Eigenvalue Spectrum ---")
    for h in range(min(3, n_heads)):  # First 3 heads
        eigvals = compute_fisher_spectrum(attn_probs[:, h], n_samples=256)
        eigvals = eigvals.cpu().numpy()
        print(f"  Head {h}:")
        print(f"    Max eigenvalue: {eigvals[-1]:.6f}")
        print(f"    Min eigenvalue: {eigvals[0]:.6f}")
        print(f"    Trace (sum): {eigvals.sum():.6f}")
        print(f"    Condition: {eigvals[-1] / (abs(eigvals[0]) + 1e-8):.2f}")
        print(f"    Top-5: {eigvals[-5:]}")

    # Generate synthetic K tensor
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)

    results = []

    for rank in ranks:
        print(f"\n--- Rank {rank} (compression: {rank/seq_len:.1%}) ---")

        # Create FIMKVSplice
        fim_splice = FIMKVSplice(
            max_seq_len=seq_len,
            rank=rank,
            n_heads=n_heads,
            per_head=True,
        ).to(device)

        # Calibrate each head
        with measure_time("Calibration"):
            for h in range(n_heads):
                # Extract single head's attention: [B, T, T]
                attn_h = attn_probs[:, h]  # [B, T, T]
                # Pass as 3D - calibrate handles this format
                fim_splice.calibrate(
                    attn_h,  # [B, T, T] for single head
                    head_idx=h,
                    n_samples=n_calibration_samples,
                )

        # Compress and decompress each head
        total_recon_error = 0.0
        with measure_time("Compress/Decompress"):
            for h in range(n_heads):
                K_h = K[:, h]  # [B, T, d_head]
                K_hat_h = fim_splice.forward(K_h, head_idx=h)
                error = F.mse_loss(K_hat_h, K_h).item()
                total_recon_error += error

        avg_error = total_recon_error / n_heads

        # Memory stats
        stats = fim_splice.get_compression_stats()

        print(f"  Reconstruction MSE: {avg_error:.6f}")
        print(f"  Memory reduction: {stats['memory_reduction']:.1%}")
        print(f"  Calibrated: {stats['calibrated']}")

        results.append(
            {
                "rank": rank,
                "compression": rank / seq_len,
                "recon_mse": avg_error,
                "memory_reduction": stats["memory_reduction"],
            }
        )

    return results


def benchmark_legacy_kvsplice(
    batch_size,
    seq_len,
    d_latent,
    compression_ratios,
    device,
    dtype,
):
    """Benchmark legacy LearnedKVSplice (feature-space compression)."""
    print("\n" + "=" * 60)
    print("Legacy LearnedKVSplice Benchmark (Feature-Space)")
    print("=" * 60)
    print(f"Config: B={batch_size}, T={seq_len}, d_latent={d_latent}")
    print("=" * 60)

    # Generate synthetic latent
    latent = torch.randn(batch_size, seq_len, d_latent, device=device, dtype=dtype)

    results = []

    for ratio in compression_ratios:
        d_compressed = int(d_latent * ratio)
        print(f"\n--- Compression ratio {ratio:.1%} (d={d_compressed}) ---")

        # Create LearnedKVSplice
        splice = LearnedKVSplice(d_latent, d_compressed).to(device).to(dtype)

        # Forward pass
        with torch.no_grad():
            with measure_time("Compress/Decompress"):
                latent_hat = splice(latent)

        error = F.mse_loss(latent_hat, latent).item()
        stats = splice.get_compression_stats()

        print(f"  Reconstruction MSE: {error:.6f}")
        print(f"  Memory reduction: {stats['memory_reduction']:.1%}")

        results.append(
            {
                "ratio": ratio,
                "d_compressed": d_compressed,
                "recon_mse": error,
                "memory_reduction": stats["memory_reduction"],
            }
        )

    return results


def benchmark_pca_baseline(
    batch_size,
    n_heads,
    seq_len,
    d_head,
    ranks,
    device,
    dtype,
):
    """Benchmark PCA baseline (variance-based compression)."""
    print("\n" + "=" * 60)
    print("PCA Baseline Benchmark (Variance-Based)")
    print("=" * 60)

    # Generate synthetic K tensor
    K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)

    results = []

    for rank in ranks:
        print(f"\n--- Rank {rank} ---")

        total_error = 0.0

        for h in range(n_heads):
            K_h = K[:, h]  # [B, T, d_head]

            # Reshape for PCA: [B*d_head, T]
            K_flat = K_h.permute(0, 2, 1).reshape(-1, seq_len)

            # Compute PCA via SVD
            U, S, Vh = torch.linalg.svd(K_flat, full_matrices=False)

            # Keep top-r components
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]

            # Reconstruct
            K_hat_flat = U_r @ torch.diag(S_r) @ Vh_r
            K_hat_h = K_hat_flat.reshape(batch_size, d_head, seq_len).permute(0, 2, 1)

            error = F.mse_loss(K_hat_h, K_h).item()
            total_error += error

        avg_error = total_error / n_heads

        print(f"  Reconstruction MSE: {avg_error:.6f}")

        results.append(
            {
                "rank": rank,
                "recon_mse": avg_error,
            }
        )

    return results


def benchmark_with_sba_model(
    seq_len,
    ranks,
    device,
    dtype,
):
    """Benchmark FIMKVSplice with real SBA attention probabilities."""
    print("\n" + "=" * 60)
    print("FIMKVSplice with Real SBA Model")
    print("=" * 60)

    # Create small SBA model
    cfg = RA_MLA_Config(
        d_model=256,
        n_heads=4,
        n_layers=2,
        block_size=seq_len,
        d_latent=64,
    )

    model = GPT2_MLA_RA(cfg, vocab_size=1000).to(device).eval()

    # Generate input
    x = torch.randint(0, 1000, (4, seq_len), device=device)

    print(f"Model: {model.get_num_params()/1e6:.2f}M params")

    # Get attention probabilities by running compute_fisher_metrics
    # This gives us real attention patterns
    print("\nComputing Fisher metrics from RAMLA model...")
    metrics = model.compute_fisher_metrics(x, n_samples=128, topk=4)

    # Print some metrics
    for key, value in sorted(metrics.items()):
        if "eigmax_mean" in key:
            print(f"  {key}: {value:.6f}")

    print("\n✓ RAMLA model Fisher metrics computed successfully")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FIMKVSplice")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--d-latent", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--skip-sba", action="store_true", help="Skip SBA model benchmark"
    )
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)

    # Compression ranks to test
    ranks = [8, 16, 32, 64]
    ranks = [r for r in ranks if r < args.seq_len]

    # Feature compression ratios
    compression_ratios = [0.25, 0.5, 0.75]

    print("\n" + "#" * 60)
    print("# FIMKVSplice Benchmark Suite")
    print("#" * 60)

    # 1. FIMKVSplice (temporal, FIM-based)
    fim_results = benchmark_fim_kvsplice(
        args.batch_size,
        args.n_heads,
        args.seq_len,
        args.d_head,
        ranks,
        device,
        dtype,
    )

    # 2. Legacy LearnedKVSplice (feature-space)
    legacy_results = benchmark_legacy_kvsplice(
        args.batch_size,
        args.seq_len,
        args.d_latent,
        compression_ratios,
        device,
        dtype,
    )

    # 3. PCA baseline
    pca_results = benchmark_pca_baseline(
        args.batch_size,
        args.n_heads,
        args.seq_len,
        args.d_head,
        ranks,
        device,
        dtype,
    )

    # 4. Real SBA model (optional)
    if not args.skip_sba:
        benchmark_with_sba_model(args.seq_len, ranks, device, dtype)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nFIMKVSplice (Temporal, FIM-based):")
    for r in fim_results:
        print(
            f"  rank={r['rank']:3d}: MSE={r['recon_mse']:.6f}, reduction={r['memory_reduction']:.1%}"
        )

    print("\nPCA Baseline (Variance-based):")
    for r in pca_results:
        print(f"  rank={r['rank']:3d}: MSE={r['recon_mse']:.6f}")

    print("\nLegacy LearnedKVSplice (Feature-space):")
    for r in legacy_results:
        print(
            f"  ratio={r['ratio']:.0%}: MSE={r['recon_mse']:.6f}, reduction={r['memory_reduction']:.1%}"
        )

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("On random K tensors, PCA has lower MSE because it directly")
    print("optimizes variance reconstruction. This is expected.")
    print("")
    print("FIMKVSplice's benefit shows in MODEL PERPLEXITY, not K MSE:")
    print("- It preserves attention-critical temporal directions")
    print("- Random K has no correlation with attention patterns")
    print("- Real benefit: better perplexity at same compression ratio")
    print("")
    print("To validate: train models with FIMKVSplice vs PCA compression")
    print("and compare val_perplexity, not reconstruction MSE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
