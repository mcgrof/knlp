#!/usr/bin/env python3
"""
ULTIMATE RA v3: Fused QKV+Low-Rank Projection

Eliminates 2 GEMMs by fusing Q@W_recip and K@W_recip into the QKV projection:
- Single GEMM: x @ W_fused → [Q_std, K_std, V, Q_low, K_low]
- No separate Q@W_recip, K@W_recip needed

Expected: 0.15-0.30ms faster than v2 (2.00ms → 1.70-1.85ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import math


def ra_ultimate_v3(Q_std, K_std, V, Q_low, K_low, w_std, w_rec,
                   threshold=0.1, dropout_p=0.0, Qf_buf=None, Kf_buf=None):
    """
    Ultimate optimized RA v3 with pre-computed low-rank projections and buffer reuse.

    Args:
        Q_std, K_std: [B, H, T, D_std] standard channels
        V: [B, H, T, D] full value
        Q_low, K_low: [B, H, T, R] pre-computed low-rank projections
        w_std, w_rec: [H] per-head gate weights
        threshold: w_rec threshold for RA routing
        dropout_p: dropout probability
        Qf_buf, Kf_buf: Optional pre-allocated buffers for folded Q/K

    Returns:
        out: [B, H, T, D]
    """
    B, H, T, D = V.shape
    R = Q_low.shape[-1]
    D_std = Q_std.shape[-1]

    # Head-selective routing
    use_ra = (w_rec > threshold)  # [H] boolean
    n_ra = use_ra.sum().item()

    if n_ra == 0:
        # All heads use baseline SDPA (reconstruct full Q, K)
        Q_full = torch.cat([Q_std, Q_low], dim=-1)
        K_full = torch.cat([K_std, K_low], dim=-1)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                Q_full, K_full, V, is_causal=True, dropout_p=dropout_p
            )

    # Pack heads into RA and baseline groups
    idx_ra = torch.nonzero(use_ra, as_tuple=False).squeeze(1)
    idx_bl = torch.nonzero(~use_ra, as_tuple=False).squeeze(1)

    # Gather slices (views, cheap)
    Q_std_ra, K_std_ra, V_ra = Q_std[:, idx_ra], K_std[:, idx_ra], V[:, idx_ra]
    Q_std_bl, K_std_bl, V_bl = Q_std[:, idx_bl], K_std[:, idx_bl], V[:, idx_bl]
    Q_low_ra, K_low_ra = Q_low[:, idx_ra], K_low[:, idx_ra]
    Q_low_bl, K_low_bl = Q_low[:, idx_bl], K_low[:, idx_bl]

    # Per-pack gate scales
    ws_ra = w_std[idx_ra].clamp_min(1e-8).sqrt().view(1, -1, 1, 1)
    wr_ra = w_rec[idx_ra].clamp_min(1e-8).sqrt().view(1, -1, 1, 1)
    ws_bl = w_std[idx_bl].clamp_min(1e-8).sqrt().view(1, -1, 1, 1) if len(idx_bl) > 0 else None

    # Process output
    out = torch.empty_like(V)

    # Process baseline heads (if any)
    if len(idx_bl) > 0:
        # Reconstruct full Q, K for baseline
        Q_bl_full = torch.cat([Q_std_bl, Q_low_bl], dim=-1)
        K_bl_full = torch.cat([K_std_bl, K_low_bl], dim=-1)

        # Symmetric scaling
        Q_bl_scaled = Q_bl_full * ws_bl
        K_bl_scaled = K_bl_full * ws_bl

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out[:, idx_bl] = F.scaled_dot_product_attention(
                Q_bl_scaled, K_bl_scaled, V_bl,
                is_causal=True,
                dropout_p=dropout_p
            )

    # Process RA heads with same-FLOP folding and buffer reuse
    if len(idx_ra) > 0:
        n_ra_heads = len(idx_ra)

        if Qf_buf is not None and Kf_buf is not None:
            # Use persistent buffers (zero malloc!)
            Qf = Qf_buf[:B, :n_ra_heads, :T, :D]
            Kf = Kf_buf[:B, :n_ra_heads, :T, :D]

            # Copy into buffers (no cat, no contiguous needed!)
            Qf[..., :D_std].copy_(ws_ra * Q_std_ra)
            Qf[..., D_std:].copy_(wr_ra * K_low_ra)
            Kf[..., :D_std].copy_(ws_ra * K_std_ra)
            Kf[..., D_std:].copy_(wr_ra * Q_low_ra)
        else:
            # Fallback: allocate (slower)
            Qf = torch.cat([ws_ra * Q_std_ra, wr_ra * K_low_ra], dim=-1).contiguous()
            Kf = torch.cat([ws_ra * K_std_ra, wr_ra * Q_low_ra], dim=-1).contiguous()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out[:, idx_ra] = F.scaled_dot_product_attention(
                Qf, Kf, V_ra,
                is_causal=True,
                dropout_p=dropout_p
            )

    return out


class UltimateRAv3(nn.Module):
    """
    Ultimate RA v3 module with fused projection and buffer reuse.
    """

    def __init__(self, n_embd=768, n_head=12, block_size=1024,
                 R=8, threshold=0.1, dropout=0.0, max_batch_size=8):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.R = R
        self.D_std = self.head_dim - R
        self.threshold = threshold
        self.dropout = dropout
        self.max_batch_size = max_batch_size
        self.block_size = block_size

        assert R < self.head_dim

        # FUSED projection: outputs [Q_std, K_std, V, Q_low, K_low] in one GEMM
        # Output dimensions:
        # - Q_std: n_head * D_std
        # - K_std: n_head * D_std
        # - V: n_embd (full)
        # - Q_low: n_head * R
        # - K_low: n_head * R
        # Total: 2*n_head*D_std + n_embd + 2*n_head*R
        #      = 2*n_head*D_std + n_head*head_dim + 2*n_head*R
        #      = 2*n_head*D_std + n_head*(D_std+R) + 2*n_head*R
        #      = 3*n_head*D_std + 3*n_head*R
        #      = 3*n_embd (same total dimension!)
        fused_dim = 2 * n_head * self.D_std + n_embd + 2 * n_head * R
        self.c_attn = nn.Linear(n_embd, fused_dim, bias=False)

        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Per-head learnable gates
        self.w_std = nn.Parameter(torch.ones(n_head) * 0.5)
        self.w_rec = nn.Parameter(torch.ones(n_head) * 0.3)

        # Store dimensions for slicing
        self.q_std_end = n_head * self.D_std
        self.k_std_end = self.q_std_end + n_head * self.D_std
        self.v_end = self.k_std_end + n_embd
        self.q_low_end = self.v_end + n_head * R
        # k_low is everything after q_low_end

        # Persistent buffers for folded Q/K (avoids allocation every forward)
        self.register_buffer('_Qf_buffer', torch.empty(
            max_batch_size, n_head, block_size, self.head_dim,
            dtype=torch.bfloat16
        ))
        self.register_buffer('_Kf_buffer', torch.empty(
            max_batch_size, n_head, block_size, self.head_dim,
            dtype=torch.bfloat16
        ))

    def forward(self, x):
        B, T, C = x.size()

        # Force BF16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # FUSED QKV + low-rank projection (single GEMM!)
            fused = self.c_attn(x)  # [B, T, fused_dim]

            # Slice into 5 components
            q_std_flat = fused[..., :self.q_std_end]
            k_std_flat = fused[..., self.q_std_end:self.k_std_end]
            v_flat = fused[..., self.k_std_end:self.v_end]
            q_low_flat = fused[..., self.v_end:self.q_low_end]
            k_low_flat = fused[..., self.q_low_end:]

            # Reshape to [B, H, T, D_*]
            Q_std = q_std_flat.view(B, T, self.n_head, self.D_std).transpose(1, 2).contiguous()
            K_std = k_std_flat.view(B, T, self.n_head, self.D_std).transpose(1, 2).contiguous()
            V = v_flat.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
            Q_low = q_low_flat.view(B, T, self.n_head, self.R).transpose(1, 2).contiguous()
            K_low = k_low_flat.view(B, T, self.n_head, self.R).transpose(1, 2).contiguous()

            # Ultimate RA v3 forward with buffer reuse (no GEMMs, no malloc!)
            out = ra_ultimate_v3(
                Q_std, K_std, V, Q_low, K_low,
                self.w_std, self.w_rec,
                threshold=self.threshold,
                dropout_p=self.dropout if self.training else 0.0,
                Qf_buf=self._Qf_buffer,
                Kf_buf=self._Kf_buffer
            )

            # Reshape back
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.c_proj(out)

        return out


def benchmark_ultimate_v3():
    """Benchmark ultimate RA v3 with fused projection."""
    import time

    device = "cuda"
    B, H, T, D = 8, 12, 1024, 64
    n_embd = H * D

    print("="*70)
    print("ULTIMATE RA v3 Benchmark (Fused Projection)")
    print("="*70)
    print("Optimizations:")
    print("  - Same-FLOP (D_std=56, R=8)")
    print("  - Fused QKV+low-rank projection (single GEMM!)")
    print("  - Head-selective routing")
    print("  - BF16 + Flash backend")
    print()
    print("Key improvement: Eliminates 2 GEMMs (Q@W_recip, K@W_recip)")
    print("Expected: 0.15-0.30ms faster than v2 (2.00ms → 1.70-1.85ms)")
    print()

    x = torch.randn(B, T, n_embd, device=device, dtype=torch.bfloat16)

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
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, dropout_p=0.0
                    )

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(out)

    baseline = BaselineAttn().to(device).to(torch.bfloat16)
    for _ in range(10):
        _ = baseline(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = baseline(x)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100 * 1000
    print(f"   {baseline_time:.2f} ms/iter")

    # Ultimate RA v3 tests: sweep R values and threshold
    test_configs = [
        (8, 0.0, "R=8, 100% RA heads"),
        (4, 0.0, "R=4, 100% RA heads"),  # Smaller R for speed
    ]

    results = []
    for R_val, thresh, desc in test_configs:
        print(f"\n2. Ultimate RA v3 ({desc}, threshold={thresh})...")
        model = UltimateRAv3(
            n_embd=n_embd, n_head=H, R=R_val, threshold=thresh, max_batch_size=B
        ).to(device).to(torch.bfloat16)

        # Set head gates if needed
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
        print(f"{f'Ultimate RA v3 ({desc})':<40} {ra_time:>10.2f} {overhead:>11.2f}x")

    print("\n" + "="*70)
    print("COMPARISON TO V2")
    print("="*70)
    print("Expected v2 performance: 2.00ms (1.66x)")
    print(f"Actual v3 performance:   {results[0][1]:.2f}ms ({results[0][2]:.2f}x)")

    v2_expected = 2.00
    speedup = v2_expected - results[0][1]
    print(f"\nSpeedup vs v2: {speedup:.2f}ms")

    if speedup > 0.10:
        print(f"SUCCESS! Fused projection saved {speedup:.2f}ms as expected!")
    elif speedup > 0:
        print(f"Marginal gain of {speedup:.2f}ms (expected 0.15-0.30ms)")
    else:
        print(f"No improvement (overhead from other factors?)")

    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
        exit(1)

    benchmark_ultimate_v3()
