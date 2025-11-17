#!/usr/bin/env python3
"""
Fully optimized RA: All quick wins combined.

1. Same-FLOP approach (D_std + R = D)
2. Optimal R=8 (best speed/capacity tradeoff)
3. BF16 precision
4. Force Flash Attention backend
5. Shared W matrix
6. Head-selective routing (learned)
7. Proper memory layout

Target: 2.1-2.2ms (1.05-1.1x overhead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OptimizedRA(nn.Module):
    """
    Fully optimized RA attention module.

    Combines all optimizations:
    - Same-FLOP folding
    - Learnable head-selective routing
    - Shared W matrix per layer
    - Proper precision and backend control
    """

    def __init__(self, n_embd=768, n_head=12, block_size=1024,
                 R=8, head_threshold=0.1, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.R = R
        self.D_std = self.head_dim - R
        self.head_threshold = head_threshold

        assert self.D_std > 0, f"R ({R}) must be less than head_dim ({self.head_dim})"

        # QKV projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Shared low-rank projection for reciprocity (ONE per layer)
        self.W_recip = nn.Parameter(torch.randn(self.head_dim, R) * 0.02)

        # Per-head learnable gates
        self.w_std = nn.Parameter(torch.ones(n_head) * 0.5)
        self.w_rec = nn.Parameter(torch.ones(n_head) * 0.3)  # Learned - can go to 0
        self.w_disc = nn.Parameter(torch.ones(n_head) * 0.2)

        # Discoverability bias
        self.d_bias = nn.Parameter(torch.zeros(n_head, block_size))

        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()

        # Force BF16 for efficiency
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Force Flash Attention backend (no fallback to math)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_mem_efficient=True,
                enable_math=False
            ):
                return self._forward_impl(x, B, T, C)

    def _forward_impl(self, x, B, T, C):
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to [B, H, T, D]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Ensure contiguous layout for optimal performance
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Head-selective routing: decide which heads use RA
        # Heads with low w_rec use baseline SDPA (faster)
        use_ra = (self.w_rec.detach() > self.head_threshold)  # [H]
        n_ra_heads = use_ra.sum().item()

        if n_ra_heads == 0:
            # All heads use baseline SDPA (fast path)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        elif n_ra_heads == self.n_head:
            # All heads use RA (no splitting overhead)
            out = self._ra_attention_all_heads(q, k, v, B, T)
        else:
            # Mixed: some heads RA, some baseline
            out = self._ra_attention_selective(q, k, v, use_ra, B, T)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out

    def _ra_attention_all_heads(self, q, k, v, B, T):
        """RA for all heads (same-FLOP approach with R=8)."""

        # Compute shared low-rank projections ONCE
        # This is the key optimization - reuse across all heads
        q_low = torch.matmul(q, self.W_recip)  # [B, H, T, R]
        k_low = torch.matmul(k, self.W_recip)  # [B, H, T, R]

        # Split Q, K into standard and reciprocal parts
        q_std = q[..., :self.D_std]  # [B, H, T, D_std]
        k_std = k[..., :self.D_std]

        # Expand weights for batch dimension
        sqrt_w_std = torch.sqrt(self.w_std).view(1, self.n_head, 1, 1)
        sqrt_w_rec = torch.sqrt(self.w_rec).view(1, self.n_head, 1, 1)

        # Augmented Q and K (same dimension as baseline!)
        q_aug = torch.cat([sqrt_w_std * q_std, sqrt_w_rec * k_low], dim=-1)
        k_aug = torch.cat([sqrt_w_std * k_std, sqrt_w_rec * q_low], dim=-1)

        # Ensure contiguous for SDPA
        q_aug = q_aug.contiguous()
        k_aug = k_aug.contiguous()

        # Discoverability bias
        w_disc = self.w_disc.view(1, self.n_head, 1, 1)
        d_bias = self.d_bias[:, :T].unsqueeze(0)  # [1, H, T]
        attn_bias = w_disc * d_bias.unsqueeze(-2)  # [1, H, 1, T]
        attn_bias = attn_bias.expand(B, self.n_head, T, T).contiguous()

        # SDPA (same FLOPs as baseline!)
        out = F.scaled_dot_product_attention(
            q_aug, k_aug, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        return out

    def _ra_attention_selective(self, q, k, v, use_ra, B, T):
        """Selective: some heads RA, some baseline."""

        # Identify head indices
        ra_idx = torch.where(use_ra)[0]
        baseline_idx = torch.where(~use_ra)[0]

        out = torch.zeros_like(q)

        # Process baseline heads (fast path)
        if len(baseline_idx) > 0:
            q_base = q[:, baseline_idx].contiguous()
            k_base = k[:, baseline_idx].contiguous()
            v_base = v[:, baseline_idx].contiguous()

            out[:, baseline_idx] = F.scaled_dot_product_attention(
                q_base, k_base, v_base,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )

        # Process RA heads (same-FLOP path)
        if len(ra_idx) > 0:
            q_ra = q[:, ra_idx]
            k_ra = k[:, ra_idx]
            v_ra = v[:, ra_idx]

            # Same logic as _ra_attention_all_heads but for subset
            q_low = torch.matmul(q_ra, self.W_recip)
            k_low = torch.matmul(k_ra, self.W_recip)

            q_std = q_ra[..., :self.D_std]
            k_std = k_ra[..., :self.D_std]

            sqrt_w_std = torch.sqrt(self.w_std[ra_idx]).view(1, -1, 1, 1)
            sqrt_w_rec = torch.sqrt(self.w_rec[ra_idx]).view(1, -1, 1, 1)

            q_aug = torch.cat([sqrt_w_std * q_std, sqrt_w_rec * k_low], dim=-1).contiguous()
            k_aug = torch.cat([sqrt_w_std * k_std, sqrt_w_rec * q_low], dim=-1).contiguous()

            w_disc = self.w_disc[ra_idx].view(1, -1, 1, 1)
            d_bias = self.d_bias[ra_idx, :T].unsqueeze(0)
            attn_bias = (w_disc * d_bias.unsqueeze(-2)).expand(B, -1, T, T).contiguous()

            out[:, ra_idx] = F.scaled_dot_product_attention(
                q_aug, k_aug, v_ra,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )

        return out


def benchmark_optimized():
    """Benchmark fully optimized RA."""
    import time

    device = "cuda"
    B, H, T, D = 8, 12, 1024, 64
    n_embd = H * D  # 768

    print("="*70)
    print("Fully Optimized RA Benchmark")
    print("="*70)
    print(f"Config: B={B}, H={H}, T={T}, D={D}")
    print(f"Optimizations: BF16 + Flash + R=8 + Shared W + Head-selective")
    print()

    # Test input
    x = torch.randn(B, T, n_embd, device=device)

    # Baseline SDPA
    print("1. Baseline SDPA (for comparison)...")

    class BaselineAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
            self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(n_embd, dim=2)
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)

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

    # Optimized RA (all heads use RA for now)
    print("\n2. Optimized RA (R=8, all heads use RA)...")
    ra_all = OptimizedRA(n_embd=n_embd, n_head=H, R=8, head_threshold=0.0).to(device)

    for _ in range(10):
        _ = ra_all(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = ra_all(x)
    torch.cuda.synchronize()
    ra_all_time = (time.time() - start) / 100 * 1000
    print(f"   {ra_all_time:.2f} ms/iter ({ra_all_time/baseline_time:.2f}x)")

    # Optimized RA (50% heads selective)
    print("\n3. Optimized RA (R=8, 50% heads use RA)...")
    ra_50 = OptimizedRA(n_embd=n_embd, n_head=H, R=8, head_threshold=0.25).to(device)
    # Manually set half heads to low w_rec
    with torch.no_grad():
        ra_50.w_rec[:6] = 0.05  # First 6 heads low
        ra_50.w_rec[6:] = 0.35  # Last 6 heads high

    for _ in range(10):
        _ = ra_50(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = ra_50(x)
    torch.cuda.synchronize()
    ra_50_time = (time.time() - start) / 100 * 1000
    print(f"   {ra_50_time:.2f} ms/iter ({ra_50_time/baseline_time:.2f}x)")

    # Optimized RA (75% heads selective)
    print("\n4. Optimized RA (R=8, 75% heads use baseline)...")
    ra_75 = OptimizedRA(n_embd=n_embd, n_head=H, R=8, head_threshold=0.25).to(device)
    with torch.no_grad():
        ra_75.w_rec[:9] = 0.05   # First 9 heads low (use baseline)
        ra_75.w_rec[9:] = 0.35   # Last 3 heads high (use RA)

    for _ in range(10):
        _ = ra_75(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = ra_75(x)
    torch.cuda.synchronize()
    ra_75_time = (time.time() - start) / 100 * 1000
    print(f"   {ra_75_time:.2f} ms/iter ({ra_75_time/baseline_time:.2f}x)")

    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Configuration':<40} {'ms/iter':>10} {'vs Baseline':>12}")
    print("-"*70)
    print(f"{'Baseline SDPA (optimized)':<40} {baseline_time:>10.2f} {1.00:>11.2f}x")
    print(f"{'RA (100% heads, R=8)':<40} {ra_all_time:>10.2f} {ra_all_time/baseline_time:>11.2f}x")
    print(f"{'RA (50% heads, R=8)':<40} {ra_50_time:>10.2f} {ra_50_time/baseline_time:>11.2f}x")
    print(f"{'RA (25% heads, R=8)':<40} {ra_75_time:>10.2f} {ra_75_time/baseline_time:>11.2f}x")

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if ra_75_time < baseline_time * 1.1:
        print(f"🎉 SUCCESS! RA with selective heads is within 10% of baseline!")
        print(f"   {ra_75_time:.2f}ms vs {baseline_time:.2f}ms baseline")
        print(f"   Only {(ra_75_time/baseline_time - 1)*100:.1f}% overhead")
    elif ra_50_time < baseline_time * 1.15:
        print(f"✅ GOOD! RA with 50% selective is within 15% of baseline")
        print(f"   {ra_50_time:.2f}ms vs {baseline_time:.2f}ms baseline")
    else:
        print(f"Progress made, but still work needed:")
        print(f"   Best: {min(ra_all_time, ra_50_time, ra_75_time):.2f}ms")
        print(f"   Target: {baseline_time * 1.1:.2f}ms (10% overhead)")

    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA required")
        exit(1)

    benchmark_optimized()
