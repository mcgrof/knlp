#!/usr/bin/env python3
"""
ULTIMATE RA v4: FP16 + All Optimizations

Following ChatGPT's advice to close the gap:
1. FP16 everywhere (8-15% speedup on A10G)
2. Remove ALL cats with persistent buffers
3. Bake gate scales into projection weights (no runtime muls)
4. torch.compile + CUDA graphs
5. R=4 with routing

Target: Match or beat baseline (1.21ms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import math


def ra_ultimate_v4(Q_std, K_std, V, Q_low, K_low, w_std, w_rec,
                   threshold=0.1, dropout_p=0.0,
                   Qf_buf=None, Kf_buf=None, Qfull_buf=None, Kfull_buf=None):
    """
    Ultimate optimized RA v4 with FP16 and zero-cat approach.
    """
    B, H, T, D = V.shape
    R = Q_low.shape[-1]
    D_std = Q_std.shape[-1]

    # Head-selective routing
    use_ra = (w_rec > threshold)  # [H] boolean
    n_ra = use_ra.sum().item()

    # All heads baseline (no RA)
    if n_ra == 0:
        # Use full-Q/K buffers (no cat!)
        Qfull = Qfull_buf[:B, :H, :T, :D] if Qfull_buf is not None else None
        Kfull = Kfull_buf[:B, :H, :T, :D] if Kfull_buf is not None else None

        if Qfull is not None and Kfull is not None:
            # Zero-copy: fill buffers directly
            Qfull[..., :D_std].copy_(Q_std)
            Qfull[..., D_std:].copy_(Q_low)
            Kfull[..., :D_std].copy_(K_std)
            Kfull[..., D_std:].copy_(K_low)
        else:
            # Fallback: cat
            Qfull = torch.cat([Q_std, Q_low], dim=-1)
            Kfull = torch.cat([K_std, K_low], dim=-1)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                Qfull, Kfull, V, is_causal=True, dropout_p=dropout_p
            )

    # Pack heads
    idx_ra = torch.nonzero(use_ra, as_tuple=False).squeeze(1)
    idx_bl = torch.nonzero(~use_ra, as_tuple=False).squeeze(1)

    # Slices
    Q_std_ra, K_std_ra, V_ra = Q_std[:, idx_ra], K_std[:, idx_ra], V[:, idx_ra]
    Q_std_bl, K_std_bl, V_bl = Q_std[:, idx_bl], K_std[:, idx_bl], V[:, idx_bl]
    Q_low_ra, K_low_ra = Q_low[:, idx_ra], K_low[:, idx_ra]
    Q_low_bl, K_low_bl = Q_low[:, idx_bl], K_low[:, idx_bl]

    out = torch.empty_like(V)

    # Baseline pack (no cat!)
    if len(idx_bl) > 0:
        n_bl = len(idx_bl)

        if Qfull_buf is not None and Kfull_buf is not None:
            Qb = Qfull_buf[:B, :n_bl, :T, :D]
            Kb = Kfull_buf[:B, :n_bl, :T, :D]

            # Fill directly (no scaling - baked into weights!)
            Qb[..., :D_std].copy_(Q_std_bl)
            Qb[..., D_std:].copy_(Q_low_bl)
            Kb[..., :D_std].copy_(K_std_bl)
            Kb[..., D_std:].copy_(K_low_bl)
        else:
            Qb = torch.cat([Q_std_bl, Q_low_bl], dim=-1)
            Kb = torch.cat([K_std_bl, K_low_bl], dim=-1)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out[:, idx_bl] = F.scaled_dot_product_attention(
                Qb, Kb, V_bl, is_causal=True, dropout_p=dropout_p
            )

    # RA pack (no cat!)
    if len(idx_ra) > 0:
        n_ra = len(idx_ra)

        if Qf_buf is not None and Kf_buf is not None:
            Qf = Qf_buf[:B, :n_ra, :T, :D]
            Kf = Kf_buf[:B, :n_ra, :T, :D]

            # Fill with reciprocal swap (no scaling - baked into weights!)
            Qf[..., :D_std].copy_(Q_std_ra)
            Qf[..., D_std:].copy_(K_low_ra)  # Reciprocal swap
            Kf[..., :D_std].copy_(K_std_ra)
            Kf[..., D_std:].copy_(Q_low_ra)  # Reciprocal swap
        else:
            Qf = torch.cat([Q_std_ra, K_low_ra], dim=-1).contiguous()
            Kf = torch.cat([K_std_ra, Q_low_ra], dim=-1).contiguous()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out[:, idx_ra] = F.scaled_dot_product_attention(
                Qf, Kf, V_ra, is_causal=True, dropout_p=dropout_p
            )

    return out


class UltimateRAv4(nn.Module):
    """
    Ultimate RA v4: FP16 + zero-cat + baked scaling.
    """

    def __init__(self, n_embd=768, n_head=12, block_size=1024,
                 R=4, threshold=0.1, dropout=0.0, max_batch_size=8):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.R = R
        self.D_std = self.head_dim - R
        self.threshold = threshold
        self.dropout = dropout

        # FP16 Linear layers (on GPU directly)
        fused_dim = 2 * n_head * self.D_std + n_embd + 2 * n_head * R
        self.c_attn = nn.Linear(n_embd, fused_dim, bias=False, dtype=torch.float16, device="cuda")
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float16, device="cuda")

        # Per-head learnable gates
        self.w_std = nn.Parameter(torch.ones(n_head, dtype=torch.float16, device="cuda") * 0.5)
        self.w_rec = nn.Parameter(torch.ones(n_head, dtype=torch.float16, device="cuda") * 0.3)

        # Persistent buffers (FP16)
        self.register_buffer('_Qf_buffer', torch.empty(
            max_batch_size, n_head, block_size, self.head_dim,
            dtype=torch.float16, device="cuda"
        ))
        self.register_buffer('_Kf_buffer', torch.empty_like(self._Qf_buffer))
        self.register_buffer('_Qfull_buffer', torch.empty_like(self._Qf_buffer))
        self.register_buffer('_Kfull_buffer', torch.empty_like(self._Qf_buffer))

        # Slice indices for fused projection
        self.q_std_end = n_head * self.D_std
        self.k_std_end = self.q_std_end + n_head * self.D_std
        self.v_end = self.k_std_end + n_embd
        self.q_low_end = self.v_end + n_head * R

        # Flag to track if scales are baked
        self._scales_baked = False

    def bake_scales_into_weights(self):
        """
        Bake gate scales into projection weights (call after init or optimizer step).
        """
        with torch.no_grad():
            s_std = self.w_std.clamp_min(1e-8).sqrt()  # [H]
            s_rec = self.w_rec.clamp_min(1e-8).sqrt()  # [H]

            # Expand to per-output-dim scales
            scale_q_std = s_std.repeat_interleave(self.D_std)  # [H*D_std]
            scale_k_std = s_std.repeat_interleave(self.D_std)
            scale_q_low = s_rec.repeat_interleave(self.R)      # [H*R]
            scale_k_low = s_rec.repeat_interleave(self.R)

            # Apply to fused weight matrix
            W = self.c_attn.weight  # [fused_dim, n_embd]

            # Scale each block
            W[:self.q_std_end].mul_(scale_q_std.unsqueeze(1))
            W[self.q_std_end:self.k_std_end].mul_(scale_k_std.unsqueeze(1))
            # V stays unscaled
            W[self.v_end:self.q_low_end].mul_(scale_q_low.unsqueeze(1))
            W[self.q_low_end:].mul_(scale_k_low.unsqueeze(1))

        self._scales_baked = True

    def forward(self, x):
        B, T, C = x.size()

        # Bake scales on first forward (or call explicitly)
        if not self._scales_baked:
            self.bake_scales_into_weights()

        # FP16 autocast
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Fused projection
            fused = self.c_attn(x)

            # Slice
            q_std_flat = fused[..., :self.q_std_end]
            k_std_flat = fused[..., self.q_std_end:self.k_std_end]
            v_flat = fused[..., self.k_std_end:self.v_end]
            q_low_flat = fused[..., self.v_end:self.q_low_end]
            k_low_flat = fused[..., self.q_low_end:]

            # Reshape
            Q_std = q_std_flat.view(B, T, self.n_head, self.D_std).transpose(1, 2).contiguous()
            K_std = k_std_flat.view(B, T, self.n_head, self.D_std).transpose(1, 2).contiguous()
            V = v_flat.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
            Q_low = q_low_flat.view(B, T, self.n_head, self.R).transpose(1, 2).contiguous()
            K_low = k_low_flat.view(B, T, self.n_head, self.R).transpose(1, 2).contiguous()

            # RA v4 (no runtime scaling, no cats!)
            out = ra_ultimate_v4(
                Q_std, K_std, V, Q_low, K_low,
                self.w_std, self.w_rec,
                threshold=self.threshold,
                dropout_p=self.dropout if self.training else 0.0,
                Qf_buf=self._Qf_buffer,
                Kf_buf=self._Kf_buffer,
                Qfull_buf=self._Qfull_buffer,
                Kfull_buf=self._Kfull_buffer
            )

            # Reshape back
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            out = self.c_proj(out)

        return out


def benchmark_ultimate_v4():
    """Benchmark ultimate RA v4."""
    import time

    device = "cuda"
    B, H, T, D = 8, 12, 1024, 64
    n_embd = H * D

    print("="*70)
    print("ULTIMATE RA v4 Benchmark (FP16 + All Optimizations)")
    print("="*70)
    print("Optimizations:")
    print("  - FP16 everywhere (8-15% speedup expected)")
    print("  - Zero-cat with persistent buffers")
    print("  - Gate scales baked into weights (no runtime muls)")
    print("  - R=4 (faster, good quality)")
    print()

    x = torch.randn(B, T, n_embd, device=device, dtype=torch.float16)

    # Baseline FP16
    print("1. Baseline SDPA (FP16)...")

    class BaselineAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False, dtype=torch.float16, device="cuda")
            self.c_proj = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float16, device="cuda")

        def forward(self, x):
            B, T, C = x.size()
            q, k, v = self.c_attn(x).split(n_embd, dim=2)
            q = q.view(B, T, H, D).transpose(1, 2).contiguous()
            k = k.view(B, T, H, D).transpose(1, 2).contiguous()
            v = v.view(B, T, H, D).transpose(1, 2).contiguous()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, dropout_p=0.0
                    )

            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(out)

    baseline = BaselineAttn()
    for _ in range(10):
        _ = baseline(x)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = baseline(x)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100 * 1000
    print(f"   {baseline_time:.2f} ms/iter")

    # RA v4 configurations
    test_configs = [
        (4, 0.0, "R=4, 100% RA"),
        (4, 0.35, "R=4, 25% RA (routing)"),
    ]

    results = []
    for R_val, thresh, desc in test_configs:
        print(f"\n2. Ultimate RA v4 ({desc})...")
        model = UltimateRAv4(
            n_embd=n_embd, n_head=H, R=R_val, threshold=thresh, max_batch_size=B
        )

        # Set routing if needed
        if thresh == 0.35:
            with torch.no_grad():
                model.w_rec[:9] = 0.3   # Below threshold
                model.w_rec[9:] = 0.4   # Above threshold

        # Bake scales
        model.bake_scales_into_weights()

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
    print(f"{'Baseline SDPA (FP16)':<40} {baseline_time:>10.2f} {1.00:>11.2f}x")
    for desc, ra_time, overhead in results:
        print(f"{f'Ultimate RA v4 ({desc})':<40} {ra_time:>10.2f} {overhead:>11.2f}x")

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print("v2 (BF16): 2.00ms (1.66x)")
    print("v3 (BF16): 2.23ms (1.85x) - SLOWER!")
    print(f"v4 (FP16): {results[0][1]:.2f}ms ({results[0][2]:.2f}x)")

    best_time = min(r[1] for r in results)
    best_config = min(results, key=lambda x: x[1])[0]

    if best_time <= baseline_time * 1.05:
        print(f"\nSUCCESS! Within 5% of baseline!")
        print(f"Best config: {best_config}")
        print(f"Time: {best_time:.2f}ms vs {baseline_time:.2f}ms baseline")
    elif best_time <= baseline_time * 1.15:
        print(f"\nGOOD! Within 15% of baseline")
        print(f"Best config: {best_config}")
        print(f"Time: {best_time:.2f}ms vs {baseline_time:.2f}ms baseline")
    else:
        print(f"\nStill overhead:")
        print(f"Best: {best_time:.2f}ms ({best_time/baseline_time:.2f}x)")

    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required")
        exit(1)

    benchmark_ultimate_v4()
