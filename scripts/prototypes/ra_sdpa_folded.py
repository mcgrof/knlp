#!/usr/bin/env python3
"""
RA via SDPA with algebraic folding trick.

Key insight: Instead of computing w_std * S + w_rec * S^T explicitly,
fold both into augmented Q and K:

    Q̃ = [√w_std * Q, √w_rec * K]  (dimension: D → 2D)
    K̃ = [√w_std * K, √w_rec * Q]  (dimension: D → 2D)

Then Q̃ @ K̃^T = w_std * S + w_rec * S^T in a single matmul!

This allows using SDPA's kernel fusion for RA.

Cost: 2x head dimension → roughly 2x FLOPs
Benefit: SDPA fusion instead of custom Triton kernel
Expected: Faster than Triton RA (4.85ms), close to SDPA baseline (1.98ms)
"""

import torch
import torch.nn.functional as F
import math


def ra_sdpa_folded(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """
    RA via SDPA with algebraic folding.

    Args:
        Q, K, V: [B, H, T, D]
        d_bias: [B, H, T] - discoverability bias
        w_std, w_rec, w_disc: [B, H] - per-head gate weights

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape
    device = Q.device

    # Scale weights (need sqrt for algebraic identity)
    sqrt_w_std = torch.sqrt(w_std).view(B, H, 1, 1)  # [B, H, 1, 1]
    sqrt_w_rec = torch.sqrt(w_rec).view(B, H, 1, 1)

    # Augment Q and K: concatenate along head dimension
    # Q̃ = [√w_std * Q, √w_rec * K]
    # K̃ = [√w_std * K, √w_rec * Q]
    Q_aug = torch.cat([sqrt_w_std * Q, sqrt_w_rec * K], dim=-1)  # [B, H, T, 2D]
    K_aug = torch.cat([sqrt_w_std * K, sqrt_w_rec * Q], dim=-1)  # [B, H, T, 2D]

    # Augment V: pad with zeros (we only care about first D dims of output)
    V_aug = torch.cat([V, torch.zeros_like(V)], dim=-1)  # [B, H, T, 2D]

    # Create attention bias for discoverability
    # d_bias is [B, H, T], need to broadcast to [B, H, T, T]
    # Each query position i gets bias d_bias[:, :, j] added to its score with key j
    w_disc_bc = w_disc.view(B, H, 1, 1)  # [B, H, 1, 1]
    attn_bias = w_disc_bc * d_bias.unsqueeze(-2)  # [B, H, 1, T]
    attn_bias = attn_bias.expand(B, H, T, T)  # [B, H, T, T]

    # Call SDPA with augmented tensors
    # This computes: softmax(Q̃ @ K̃^T / √(2D) + bias) @ Ṽ
    # Q̃ @ K̃^T already contains w_std * S + w_rec * S^T
    out_aug = F.scaled_dot_product_attention(
        Q_aug, K_aug, V_aug,
        attn_mask=attn_bias,
        is_causal=True
    )

    # Extract first D dimensions (second half is zeros @ V = 0)
    out = out_aug[..., :D]

    return out


def ra_sdpa_folded_lowrank(Q, K, V, d_bias, w_std, w_rec, w_disc, rank=16):
    """
    RA via SDPA with low-rank approximation for reciprocity.

    Uses compressed reciprocal term: √w_rec * K @ W instead of full K
    where W is [D, R] projection matrix (R << D).

    This reduces head dimension from 2D to D+R, closer to baseline cost.

    Args:
        Q, K, V: [B, H, T, D]
        d_bias: [B, H, T]
        w_std, w_rec, w_disc: [B, H]
        rank: Rank of low-rank approximation (default 16)

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape
    device = Q.device

    # Low-rank projection matrix (shared across batch and heads for now)
    W = torch.randn(D, rank, device=device) * 0.02  # [D, R]

    # Scale weights
    sqrt_w_std = torch.sqrt(w_std).view(B, H, 1, 1)
    sqrt_w_rec = torch.sqrt(w_rec).view(B, H, 1, 1)

    # Augment with low-rank reciprocity
    # Q̃ = [√w_std * Q, √w_rec * K @ W]  (dimension: D + R)
    # K̃ = [√w_std * K, √w_rec * Q @ W]
    K_compressed = torch.matmul(K, W)  # [B, H, T, R]
    Q_compressed = torch.matmul(Q, W)  # [B, H, T, R]

    Q_aug = torch.cat([sqrt_w_std * Q, sqrt_w_rec * K_compressed], dim=-1)  # [B, H, T, D+R]
    K_aug = torch.cat([sqrt_w_std * K, sqrt_w_rec * Q_compressed], dim=-1)  # [B, H, T, D+R]

    # Augment V: pad with zeros for the R dimensions
    V_aug = torch.cat([V, torch.zeros(B, H, T, rank, device=device)], dim=-1)  # [B, H, T, D+R]

    # Attention bias
    w_disc_bc = w_disc.view(B, H, 1, 1)
    attn_bias = w_disc_bc * d_bias.unsqueeze(-2)
    attn_bias = attn_bias.expand(B, H, T, T)

    # SDPA call
    out_aug = F.scaled_dot_product_attention(
        Q_aug, K_aug, V_aug,
        attn_mask=attn_bias,
        is_causal=True
    )

    # Extract first D dimensions
    out = out_aug[..., :D]

    return out


def benchmark_folded_variants():
    """Benchmark folded SDPA variants against baselines."""
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D = 8, 12, 1024, 64

    # Create test inputs
    Q = torch.randn(B, H, T, D, device=device)
    K = torch.randn(B, H, T, D, device=device)
    V = torch.randn(B, H, T, D, device=device)
    d_bias = torch.randn(B, H, T, device=device)
    w_std = torch.full((B, H), 0.5, device=device)
    w_rec = torch.full((B, H), 0.3, device=device)
    w_disc = torch.full((B, H), 0.2, device=device)

    print("="*70)
    print("Benchmarking RA SDPA Folded Variants")
    print("="*70)
    print(f"Config: B={B}, H={H}, T={T}, D={D}")
    print(f"Device: {device}")
    print()

    # Baseline: Standard SDPA
    print("1. Baseline SDPA...")
    for _ in range(10):  # Warmup
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    sdpa_time = (time.time() - start) / 100 * 1000
    print(f"   {sdpa_time:.2f} ms/iter")

    # Triton RA (for comparison)
    print("\n2. RA Triton (current best)...")
    try:
        from triton_ra_attention import triton_ra_attention
        for _ in range(10):  # Warmup
            _ = triton_ra_attention(Q, K, V, d_bias, w_std, w_rec, w_disc)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = triton_ra_attention(Q, K, V, d_bias, w_std, w_rec, w_disc)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100 * 1000
        print(f"   {triton_time:.2f} ms/iter")
    except Exception as e:
        print(f"   ⚠️  Triton not available: {e}")
        triton_time = None

    # Folded SDPA (full rank)
    print("\n3. RA SDPA Folded (2D head dimension)...")
    for _ in range(10):  # Warmup
        _ = ra_sdpa_folded(Q, K, V, d_bias, w_std, w_rec, w_disc)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = ra_sdpa_folded(Q, K, V, d_bias, w_std, w_rec, w_disc)
    torch.cuda.synchronize()
    folded_time = (time.time() - start) / 100 * 1000
    print(f"   {folded_time:.2f} ms/iter")

    # Folded SDPA (low rank)
    for rank in [16, 32]:
        print(f"\n4. RA SDPA Folded (low-rank R={rank})...")
        for _ in range(10):  # Warmup
            _ = ra_sdpa_folded_lowrank(Q, K, V, d_bias, w_std, w_rec, w_disc, rank=rank)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = ra_sdpa_folded_lowrank(Q, K, V, d_bias, w_std, w_rec, w_disc, rank=rank)
        torch.cuda.synchronize()
        lowrank_time = (time.time() - start) / 100 * 1000
        print(f"   {lowrank_time:.2f} ms/iter")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Implementation':<35} {'ms/iter':>12} {'vs SDPA':>12}")
    print("-"*70)
    print(f"{'SDPA baseline':<35} {sdpa_time:>12.2f} {1.00:>11.2f}x")
    if triton_time:
        print(f"{'RA Triton (current)':<35} {triton_time:>12.2f} {triton_time/sdpa_time:>11.2f}x")
    print(f"{'RA SDPA Folded (full)':<35} {folded_time:>12.2f} {folded_time/sdpa_time:>11.2f}x")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if folded_time < (triton_time if triton_time else float('inf')):
        improvement = (triton_time - folded_time) / triton_time * 100 if triton_time else 0
        print(f"✅ SDPA Folded WINS: {improvement:.1f}% faster than Triton!")
        print(f"   This is the clever trick we were looking for!")
        print(f"   Overhead vs baseline: {folded_time/sdpa_time:.2f}x (down from {triton_time/sdpa_time:.2f}x)")
    else:
        print(f"❌ SDPA Folded slower than Triton")
        print(f"   Folded: {folded_time:.2f}ms vs Triton: {triton_time:.2f}ms")
        print(f"   Algebraic trick doesn't help on this hardware")

    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA required")
        exit(1)

    benchmark_folded_variants()
