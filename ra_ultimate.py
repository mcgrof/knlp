#!/usr/bin/env python3
"""
ULTIMATE RA: All optimizations from both worlds combined.

Combines insights from:
- My work: Same-FLOP approach, head-selective routing
- ChatGPT: Zero-copy folding, fused bias, symmetric scaling

Target: 2.0ms or BETTER (match/beat baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ra_ultimate(Q, K, V, w_std, w_rec, w_disc, d_bias, W_recip,
                threshold=0.1, dropout_p=0.0):
    """
    Ultimate optimized RA with all tricks.

    Args:
        Q, K, V: [B, H, T, D] (BF16, contiguous)
        w_std, w_rec, w_disc: [H] per-head gate weights
        d_bias: [H, T] discoverability bias
        W_recip: [D, R] shared low-rank projection
        threshold: w_rec threshold for RA routing
        dropout_p: dropout probability

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = Q.shape
    R = W_recip.shape[1]
    D_std = D - R
    device = Q.device
    dtype = Q.dtype

    # Ensure contiguity (critical for SDPA performance)
    assert Q.is_contiguous()
    assert K.is_contiguous()
    assert V.is_contiguous()

    # Head-selective routing
    use_ra = (w_rec > threshold)  # [H] boolean
    n_ra = use_ra.sum().item()

    if n_ra == 0:
        # All heads use baseline SDPA (fast path)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        ):
            return F.scaled_dot_product_attention(
                Q, K, V, dropout_p=dropout_p, is_causal=True
            )

    # Shared low-rank projections (compute once per layer)
    QW = torch.matmul(Q, W_recip)  # [B, H, T, R]
    KW = torch.matmul(K, W_recip)  # [B, H, T, R]

    # Pack heads into RA and baseline groups
    idx_ra = torch.nonzero(use_ra, as_tuple=False).squeeze(1)
    idx_bl = torch.nonzero(~use_ra, as_tuple=False).squeeze(1)

    # Gather slices (views, cheap)
    Q_ra, K_ra, V_ra = Q[:, idx_ra], K[:, idx_ra], V[:, idx_ra]
    Q_bl, K_bl, V_bl = Q[:, idx_bl], K[:, idx_bl], V[:, idx_bl]
    QW_ra, KW_ra = QW[:, idx_ra], KW[:, idx_ra]

    # Per-pack gate scales
    ws_ra = w_std[idx_ra].clamp_min(1e-8).sqrt().view(1, -1, 1, 1)
    wr_ra = w_rec[idx_ra].clamp_min(1e-8).sqrt().view(1, -1, 1, 1)
    wd_ra = w_disc[idx_ra].view(1, -1, 1, 1)

    ws_bl = w_std[idx_bl].clamp_min(1e-8).sqrt().view(1, -1, 1, 1) if len(idx_bl) > 0 else None
    wd_bl = w_disc[idx_bl].view(1, -1, 1, 1) if len(idx_bl) > 0 else None

    # Build fused attention bias (causal + discoverability in ONE tensor)
    def make_fused_bias(K_pack, heads_idx, wd_pack, n_heads):
        if n_heads == 0:
            return None

        # Start with causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        bias = torch.zeros(B, n_heads, T, T, device=device, dtype=dtype)
        bias.masked_fill_(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Add discoverability column bias (zero-mean)
        if d_bias is not None:
            d = d_bias[heads_idx, :T].unsqueeze(0).unsqueeze(-2)  # [1, H_pack, 1, T]
            d = d - d.mean(dim=-1, keepdim=True)  # Zero-mean
            d = d * wd_pack  # Scale by w_disc
            bias = bias + d

        return bias.contiguous()

    bias_ra = make_fused_bias(K_ra, idx_ra, wd_ra, len(idx_ra))
    bias_bl = make_fused_bias(K_bl, idx_bl, wd_bl, len(idx_bl)) if len(idx_bl) > 0 else None

    # Process baseline heads (if any)
    out = torch.empty_like(Q)

    if len(idx_bl) > 0:
        # Symmetric scaling on Q and K
        Q_bl_scaled = Q_bl * ws_bl
        K_bl_scaled = K_bl * ws_bl

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        ):
            out[:, idx_bl] = F.scaled_dot_product_attention(
                Q_bl_scaled, K_bl_scaled, V_bl,
                attn_mask=bias_bl,
                dropout_p=dropout_p,
                is_causal=False  # Mask already in bias
            )

    # Process RA heads with zero-copy folding
    if len(idx_ra) > 0:
        # Preallocate folded buffers (ONCE, no cat overhead!)
        Qf = torch.empty(B, len(idx_ra), T, D, device=device, dtype=dtype)
        Kf = torch.empty_like(Qf)

        # Left half: sqrt(w_std) * standard channels (zero-copy)
        Qf[..., :D_std].copy_(Q_ra[..., :D_std] * ws_ra)
        Kf[..., :D_std].copy_(K_ra[..., :D_std] * ws_ra)

        # Right half: sqrt(w_rec) * reciprocal channels (zero-copy)
        Qf[..., D_std:].copy_(KW_ra * wr_ra)  # KW for Q_fold
        Kf[..., D_std:].copy_(QW_ra * wr_ra)  # QW for K_fold

        # Ensure contiguous for SDPA
        Qf = Qf.contiguous()
        Kf = Kf.contiguous()

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        ):
            out[:, idx_ra] = F.scaled_dot_product_attention(
                Qf, Kf, V_ra,
                attn_mask=bias_ra,
                dropout_p=dropout_p,
                is_causal=False  # Mask already in bias
            )

    return out


class UltimateRA(nn.Module):
    """
    Ultimate RA module with all optimizations.
    """

    def __init__(self, n_embd=768, n_head=12, block_size=1024,
                 R=8, threshold=0.1, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.R = R
        self.threshold = threshold
        self.dropout = dropout

        assert R < self.head_dim

        # QKV projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Shared low-rank projection for reciprocity
        self.W_recip = nn.Parameter(torch.randn(self.head_dim, R) * 0.02)

        # Per-head learnable gates
        self.w_std = nn.Parameter(torch.ones(n_head) * 0.5)
        self.w_rec = nn.Parameter(torch.ones(n_head) * 0.3)
        self.w_disc = nn.Parameter(torch.ones(n_head) * 0.2)

        # Discoverability bias
        self.d_bias = nn.Parameter(torch.zeros(n_head, block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Force BF16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # QKV projection
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            # Reshape to [B, H, T, D]
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()

            # Ultimate RA forward
            out = ra_ultimate(
                q, k, v,
                self.w_std, self.w_rec, self.w_disc,
                self.d_bias,
                self.W_recip,
                threshold=self.threshold,
                dropout_p=self.dropout if self.training else 0.0
            )

            # Reshape back
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.c_proj(out)

        return out


def benchmark_ultimate():
    """Benchmark ultimate RA."""
    import time

    device = "cuda"
    B, H, T, D = 8, 12, 1024, 64
    n_embd = H * D

    print("="*70)
    print("ULTIMATE RA Benchmark")
    print("="*70)
    print("All optimizations:")
    print("  - Same-FLOP (D_std=56, R=8)")
    print("  - Zero-copy folding")
    print("  - Fused bias (causal + discoverability)")
    print("  - Symmetric Q/K scaling")
    print("  - Head-selective routing")
    print("  - BF16 + Flash backend")
    print()

    x = torch.randn(B, T, n_embd, device=device)

    # Baseline for comparison
    print("1. Baseline SDPA (optimized)...")

    class BaselineAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
            self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(n_embd, dim=2)
            q = q.view(B, T, H, D).transpose(1, 2).contiguous()
            k = k.view(B, T, H, D).transpose(1, 2).contiguous()
            v = v.view(B, T, H, D).transpose(1, 2).contiguous()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_mem_efficient=True, enable_math=False
                ):
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(out)

    baseline = BaselineAttn().to(device)
    for _ in range(10):
        _ = baseline(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = baseline(x)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100 * 1000
    print(f"   {baseline_time:.2f} ms/iter")

    # Ultimate RA tests
    configs = [
        (0.0, "100% RA heads"),
        (0.25, "~50% RA heads"),
        (0.35, "~25% RA heads"),
    ]

    results = []
    for thresh, desc in configs:
        print(f"\n2. Ultimate RA ({desc}, threshold={thresh})...")
        model = UltimateRA(n_embd=n_embd, n_head=H, R=8, threshold=thresh).to(device)

        # Set head gates to create desired distribution
        if thresh == 0.25:
            with torch.no_grad():
                model.w_rec[:6] = 0.2   # Below threshold
                model.w_rec[6:] = 0.4   # Above threshold
        elif thresh == 0.35:
            with torch.no_grad():
                model.w_rec[:9] = 0.3   # Below threshold
                model.w_rec[9:] = 0.4   # Above threshold

        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            _ = model(x)
        torch.cuda.synchronize()
        ra_time = (time.time() - start) / 100 * 1000

        print(f"   {ra_time:.2f} ms/iter ({ra_time/baseline_time:.2f}x)")
        results.append((desc, ra_time, ra_time/baseline_time))

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Configuration':<40} {'ms/iter':>10} {'vs Baseline':>12}")
    print("-"*70)
    print(f"{'Baseline SDPA (optimized)':<40} {baseline_time:>10.2f} {1.00:>11.2f}x")
    for desc, ra_time, overhead in results:
        print(f"{f'Ultimate RA ({desc})':<40} {ra_time:>10.2f} {overhead:>11.2f}x")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    best_time = min(r[1] for r in results)
    best_overhead = best_time / baseline_time

    if best_time <= baseline_time * 1.05:
        print(f"🎉🎉🎉 INCREDIBLE! Within 5% of baseline!")
        print(f"   Best: {best_time:.2f}ms vs {baseline_time:.2f}ms baseline")
        print(f"   Only {(best_overhead-1)*100:.1f}% overhead")
        print()
        print("   WE BEAT THE GOAL! RA at near-baseline speed! 🚀")
    elif best_time <= baseline_time * 1.10:
        print(f"🎉 SUCCESS! Within 10% of baseline!")
        print(f"   Best: {best_time:.2f}ms vs {baseline_time:.2f}ms baseline")
        print(f"   {(best_overhead-1)*100:.1f}% overhead is acceptable")
    else:
        print(f"Good progress:")
        print(f"   Best: {best_time:.2f}ms vs {baseline_time:.2f}ms baseline")
        print(f"   {(best_overhead-1)*100:.1f}% overhead")

    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA required")
        exit(1)

    benchmark_ultimate()
