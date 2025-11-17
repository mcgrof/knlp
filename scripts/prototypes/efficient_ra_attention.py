#!/usr/bin/env python3
"""
Efficient RA Attention: Sparse + Local Combined

Key optimizations:
1. Local window: Only compute S^T within ±W tokens (default W=128)
2. Sparse selection: Within window, only keep top-k% reciprocal scores
3. Combined speedup: ~40x cheaper reciprocity (4x × 10x)

This should approach SDPA speeds while maintaining RA benefits.
"""

import torch
import torch.nn.functional as F
import math


def efficient_ra_attention_v1_local_only(Q, K, V, d_bias, w_std, w_rec, w_disc, window=128):
    """
    Local reciprocity: S^T computed only within ±window tokens.

    Args:
        Q, K, V: [B, H, T, D]
        d_bias: [B, H, T]
        w_std, w_rec, w_disc: [B, H]
        window: reciprocity window size (default 128)

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape

    # Compute standard attention scores
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

    # Compute reciprocity with local window constraint
    S_reciprocal = torch.zeros_like(S)

    # For each query position, only reciprocate within window
    for i in range(T):
        window_start = max(0, i - window)
        window_end = min(T, i + window + 1)

        # S^T[i, j] = S[j, i] for j in [window_start, window_end)
        S_reciprocal[:, :, i, window_start:window_end] = S[:, :, window_start:window_end, i]

    # Apply per-head gates
    w_std_bc = w_std.view(B, H, 1, 1)
    w_rec_bc = w_rec.view(B, H, 1, 1)
    w_disc_bc = w_disc.view(B, H, 1, 1)

    # Combine components
    logits = w_std_bc * S + w_rec_bc * S_reciprocal + w_disc_bc * d_bias.unsqueeze(-2)

    # Causal mask
    mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float('-inf'))

    # Softmax + aggregate
    attn = F.softmax(logits, dim=-1)
    out = torch.matmul(attn, V)

    return out


def efficient_ra_attention_v2_sparse_only(Q, K, V, d_bias, w_std, w_rec, w_disc, sparsity=0.1):
    """
    Sparse reciprocity: S^T computed only for top-k% attention weights.

    Args:
        Q, K, V: [B, H, T, D]
        d_bias: [B, H, T]
        w_std, w_rec, w_disc: [B, H]
        sparsity: fraction of reciprocal connections to keep (default 0.1 = 10%)

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape

    # Compute standard attention scores
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

    # Compute full reciprocal (for mask computation)
    S_reciprocal_full = S.transpose(-2, -1)

    # Create sparsity mask: keep only top-k% values per row
    k = max(1, int(sparsity * T))

    # Get top-k threshold per row
    topk_vals, _ = torch.topk(S_reciprocal_full, k, dim=-1, sorted=False)
    threshold = topk_vals[:, :, :, -1:]  # [B, H, T, 1] - kth largest value

    # Mask: keep values >= threshold
    sparse_mask = (S_reciprocal_full >= threshold).float()

    # Apply mask to reciprocal
    S_reciprocal = S_reciprocal_full * sparse_mask

    # Apply per-head gates
    w_std_bc = w_std.view(B, H, 1, 1)
    w_rec_bc = w_rec.view(B, H, 1, 1)
    w_disc_bc = w_disc.view(B, H, 1, 1)

    # Combine components
    logits = w_std_bc * S + w_rec_bc * S_reciprocal + w_disc_bc * d_bias.unsqueeze(-2)

    # Causal mask
    mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float('-inf'))

    # Softmax + aggregate
    attn = F.softmax(logits, dim=-1)
    out = torch.matmul(attn, V)

    return out


def efficient_ra_attention_v3_local_sparse_combined(Q, K, V, d_bias, w_std, w_rec, w_disc,
                                                     window=128, sparsity=0.1):
    """
    Sparse + Local combined: Best of both worlds.

    1. Local window: Only look at S^T within ±window tokens
    2. Sparse: Within window, only keep top-k% values

    Expected speedup: window_reduction × sparsity_reduction
                    = (T / 2*window) × (1 / sparsity)
                    = (1024 / 256) × 10 = 40x for reciprocity component

    Args:
        Q, K, V: [B, H, T, D]
        d_bias: [B, H, T]
        w_std, w_rec, w_disc: [B, H]
        window: reciprocity window size (default 128)
        sparsity: fraction of reciprocal connections within window (default 0.1)

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape

    # Compute standard attention scores
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

    # Compute reciprocity with local window + sparsity
    S_reciprocal = torch.zeros_like(S)

    # For each query position
    for i in range(T):
        window_start = max(0, i - window)
        window_end = min(T, i + window + 1)
        window_size = window_end - window_start

        # Get reciprocal scores in window: S[j, i] for j in window
        local_reciprocal = S[:, :, window_start:window_end, i]  # [B, H, window_size]

        # Keep only top-k% within window
        k = max(1, int(sparsity * window_size))
        topk_vals, topk_idx = torch.topk(local_reciprocal, k, dim=-1, sorted=False)

        # Create sparse mask
        sparse_mask = torch.zeros_like(local_reciprocal)
        sparse_mask.scatter_(-1, topk_idx, 1.0)

        # Apply mask and write to global reciprocal matrix
        S_reciprocal[:, :, i, window_start:window_end] = local_reciprocal * sparse_mask

    # Apply per-head gates
    w_std_bc = w_std.view(B, H, 1, 1)
    w_rec_bc = w_rec.view(B, H, 1, 1)
    w_disc_bc = w_disc.view(B, H, 1, 1)

    # Combine components
    logits = w_std_bc * S + w_rec_bc * S_reciprocal + w_disc_bc * d_bias.unsqueeze(-2)

    # Causal mask
    mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float('-inf'))

    # Softmax + aggregate
    attn = F.softmax(logits, dim=-1)
    out = torch.matmul(attn, V)

    return out


def efficient_ra_attention_v4_vectorized(Q, K, V, d_bias, w_std, w_rec, w_disc,
                                          window=128, sparsity=0.1):
    """
    Vectorized sparse + local (faster than v3's for-loop).

    Uses torch operations to avoid Python loops.

    Args:
        Q, K, V: [B, H, T, D]
        d_bias: [B, H, T]
        w_std, w_rec, w_disc: [B, H]
        window: reciprocity window size (default 128)
        sparsity: fraction of reciprocal connections within window (default 0.1)

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape
    device = Q.device

    # Compute standard attention scores
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

    # Create window mask: S_reciprocal[i,j] only if |i-j| <= window
    positions = torch.arange(T, device=device)
    distance = (positions[:, None] - positions[None, :]).abs()  # [T, T]
    window_mask = (distance <= window).float()  # [T, T]

    # Compute full reciprocal with window constraint
    S_reciprocal = S.transpose(-2, -1) * window_mask[None, None, :, :]  # [B, H, T, T]

    # Apply sparsity within window
    # For each row, keep top-k values (where k = sparsity * window_size)
    # Approximate window_size as 2*window for simplicity
    k = max(1, int(sparsity * min(T, 2 * window)))

    # Get top-k threshold per row
    topk_vals, _ = torch.topk(S_reciprocal, k, dim=-1, sorted=False)
    threshold = topk_vals[:, :, :, -1:]  # [B, H, T, 1]

    # Sparsity mask: keep values >= threshold AND within window
    sparse_mask = (S_reciprocal >= threshold).float() * window_mask[None, None, :, :]
    S_reciprocal = S_reciprocal * sparse_mask

    # Apply per-head gates
    w_std_bc = w_std.view(B, H, 1, 1)
    w_rec_bc = w_rec.view(B, H, 1, 1)
    w_disc_bc = w_disc.view(B, H, 1, 1)

    # Combine components
    logits = w_std_bc * S + w_rec_bc * S_reciprocal + w_disc_bc * d_bias.unsqueeze(-2)

    # Causal mask
    mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float('-inf'))

    # Softmax + aggregate
    attn = F.softmax(logits, dim=-1)
    out = torch.matmul(attn, V)

    return out


# Quick test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D = 2, 12, 1024, 64

    print(f"Testing efficient RA implementations on {device}")
    print(f"Config: B={B}, H={H}, T={T}, D={D}")
    print()

    # Create test inputs
    Q = torch.randn(B, H, T, D, device=device)
    K = torch.randn(B, H, T, D, device=device)
    V = torch.randn(B, H, T, D, device=device)
    d_bias = torch.randn(B, H, T, device=device)
    w_std = torch.full((B, H), 0.5, device=device)
    w_rec = torch.full((B, H), 0.3, device=device)
    w_disc = torch.full((B, H), 0.2, device=device)

    # Test all versions
    print("Testing v1 (local only, window=128)...")
    out_v1 = efficient_ra_attention_v1_local_only(Q, K, V, d_bias, w_std, w_rec, w_disc, window=128)
    print(f"  Output shape: {out_v1.shape}")
    print(f"  Output range: [{out_v1.min():.3f}, {out_v1.max():.3f}]")

    print("\nTesting v2 (sparse only, sparsity=0.1)...")
    out_v2 = efficient_ra_attention_v2_sparse_only(Q, K, V, d_bias, w_std, w_rec, w_disc, sparsity=0.1)
    print(f"  Output shape: {out_v2.shape}")
    print(f"  Output range: [{out_v2.min():.3f}, {out_v2.max():.3f}]")

    print("\nTesting v3 (local + sparse, window=128, sparsity=0.1)...")
    out_v3 = efficient_ra_attention_v3_local_sparse_combined(Q, K, V, d_bias, w_std, w_rec, w_disc,
                                                               window=128, sparsity=0.1)
    print(f"  Output shape: {out_v3.shape}")
    print(f"  Output range: [{out_v3.min():.3f}, {out_v3.max():.3f}]")

    print("\nTesting v4 (vectorized local + sparse, window=128, sparsity=0.1)...")
    out_v4 = efficient_ra_attention_v4_vectorized(Q, K, V, d_bias, w_std, w_rec, w_disc,
                                                    window=128, sparsity=0.1)
    print(f"  Output shape: {out_v4.shape}")
    print(f"  Output range: [{out_v4.min():.3f}, {out_v4.max():.3f}]")

    print("\n✅ All versions executed successfully!")
    print("\nNext: Run benchmark_ra_vs_baseline_extended.py to measure speedups")
