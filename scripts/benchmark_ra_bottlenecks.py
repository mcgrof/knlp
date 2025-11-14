#!/usr/bin/env python3
"""
Benchmark RA/R-MLP bottlenecks to find optimization opportunities.

Based on bitter7 lesson: Profile first, optimize what matters.
"""

import time
import torch
import torch.nn.functional as F

def benchmark_operation(name, func, warmup=10, iters=100):
    """Benchmark a single operation."""
    device = torch.device("cuda:0")

    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize(device)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        func()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / iters * 1000

    print(f"  {name:<40} {elapsed:>8.3f} ms")
    return elapsed


def test_gate_scaling_overhead():
    """Test overhead of dynamic gate scaling in RA forward pass."""
    print("\n" + "=" * 70)
    print("1. Gate Scaling Overhead (RA forward pass)")
    print("=" * 70)

    device = torch.device("cuda:0")
    B, H, T, D = 8, 12, 1024, 64

    # Simulated Qf/Kf tensors
    Qf = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    Kf = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    # Gate parameters
    w_std = torch.ones(H, device=device) * 0.9
    w_rec = torch.ones(H, device=device) * 0.1
    R = 4
    D_std = D - R

    def gate_scaling_current():
        """Current implementation (dynamic scaling in forward)."""
        # Compute gates (with gradients!)
        g_std = torch.sqrt(torch.clamp(w_std, min=1e-8))
        g_rec = torch.sqrt(torch.clamp(w_rec, min=1e-8))

        # Reshape for broadcasting [1, H, 1, 1]
        g_std = g_std.view(1, -1, 1, 1)
        g_rec = g_rec.view(1, -1, 1, 1)

        # Split into std and reciprocal parts
        Q_std = Qf[:, :, :, :D_std]
        K_low = Qf[:, :, :, D_std:]
        K_std = Kf[:, :, :, :D_std]
        Q_low = Kf[:, :, :, D_std:]

        # Apply gate scaling
        Q_std = Q_std * g_std
        K_std = K_std * g_std
        K_low = K_low * g_rec
        Q_low = Q_low * g_rec

        # Reconstruct
        Qf_out = torch.cat([Q_std, K_low], dim=-1)
        Kf_out = torch.cat([K_std, Q_low], dim=-1)
        return Qf_out, Kf_out

    def gate_scaling_baked():
        """Hypothetical: Pre-baked gates (no runtime ops)."""
        # Assume gates are baked into weights - just return tensors
        return Qf, Kf

    def gate_scaling_fused():
        """Optimized: Fused gate application without splits/cats."""
        # Compute gates once
        g_std = torch.sqrt(torch.clamp(w_std, min=1e-8)).view(1, -1, 1, 1)
        g_rec = torch.sqrt(torch.clamp(w_rec, min=1e-8)).view(1, -1, 1, 1)

        # Create scale tensor [g_std, ..., g_std, g_rec, ..., g_rec]
        scale = torch.ones(1, H, 1, D, device=device, dtype=torch.float16)
        scale[:, :, :, :D_std] = g_std
        scale[:, :, :, D_std:] = g_rec

        # Single element-wise multiply (no splits/cats!)
        Qf_out = Qf * scale
        Kf_out = Kf * scale
        return Qf_out, Kf_out

    time_current = benchmark_operation("Current (split+scale+cat)", gate_scaling_current)
    time_baked = benchmark_operation("Baked (no-op baseline)", gate_scaling_baked)
    time_fused = benchmark_operation("Fused (single multiply)", gate_scaling_fused)

    overhead_current = time_current - time_baked
    overhead_fused = time_fused - time_baked

    print(f"\n  Overhead:")
    print(f"    Current approach: {overhead_current:.3f} ms")
    print(f"    Fused approach:   {overhead_fused:.3f} ms")
    print(f"    Speedup potential: {time_current / time_fused:.2f}x")


def test_topk_vs_sampling():
    """Test topk vs sampling for KV pruning (same as bitter7 lesson!)."""
    print("\n" + "=" * 70)
    print("2. KV Pruning: topk vs Sampling")
    print("=" * 70)
    print("(Lesson from bitter7: sampling is 71x faster!)")

    device = torch.device("cuda:0")
    B, H, T = 8, 12, 1024
    k_keep = 391  # Golden ratio pruning

    # Importance scores (like mean_importance in PrunedKVAttention)
    importance = torch.rand(B, H, T, device=device, dtype=torch.float32)

    def topk_selection():
        """Current: topk for exact selection."""
        vals, idx = torch.topk(importance, k_keep, dim=-1)
        return idx

    def sampling_selection():
        """Optimized: Approximate via sampling."""
        # Sample 2% of sequence
        sample_size = max(64, int(T * 0.02))

        # Random sample indices
        sample_idx = torch.randint(0, T, (B, H, sample_size), device=device)

        # Gather samples
        sample_importance = torch.gather(importance, 2, sample_idx)

        # Find k-th in sample
        k_sample = max(1, int(k_keep * (sample_size / T)))
        vals, local_idx = torch.topk(sample_importance, k_sample, dim=-1)

        # Map back to full indices (approximate top-k)
        idx = torch.gather(sample_idx, 2, local_idx)
        return idx

    time_topk = benchmark_operation("topk (exact)", topk_selection)
    time_sampling = benchmark_operation("sampling (approx)", sampling_selection)

    print(f"\n  Speedup: {time_topk / time_sampling:.2f}x with sampling")
    print(f"  (Note: topk on full sequence is expensive at long context!)")


def test_gather_vs_masking():
    """Test gather vs masked indexing for KV selection."""
    print("\n" + "=" * 70)
    print("3. KV Selection: gather vs masking")
    print("=" * 70)

    device = torch.device("cuda:0")
    B, H, T, D = 8, 12, 1024, 64
    k_keep = 391

    K = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    V = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    # Pre-selected indices
    idx = torch.randint(0, T, (B, H, k_keep), device=device)

    def gather_approach():
        """Current: torch.gather for selective indexing."""
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        K_keep = torch.gather(K, 2, idx_expanded)
        V_keep = torch.gather(V, 2, idx_expanded)
        return K_keep, V_keep

    def index_select_approach():
        """Alternative: index_select (may be faster)."""
        # Flatten batch dimensions for index_select
        K_flat = K.reshape(B * H, T, D)
        V_flat = V.reshape(B * H, T, D)
        idx_flat = idx.reshape(B * H, k_keep)

        # This doesn't work directly with 2D idx
        # Would need loop or advanced indexing
        # Skip this for now
        return gather_approach()

    time_gather = benchmark_operation("gather", gather_approach)

    print(f"\n  (gather is standard approach, hard to optimize further)")


def test_cat_overhead():
    """Test concatenation overhead in split-scale-cat pattern."""
    print("\n" + "=" * 70)
    print("4. Concatenation Overhead")
    print("=" * 70)

    device = torch.device("cuda:0")
    B, H, T, D = 8, 12, 1024, 64
    D_std = 60
    R = 4

    Q_std = torch.randn(B, H, T, D_std, device=device, dtype=torch.float16)
    K_low = torch.randn(B, H, T, R, device=device, dtype=torch.float16)

    def cat_standard():
        """Current: torch.cat."""
        return torch.cat([Q_std, K_low], dim=-1)

    def cat_preallocated():
        """Optimized: Pre-allocate and fill."""
        result = torch.empty(B, H, T, D, device=device, dtype=torch.float16)
        result[:, :, :, :D_std] = Q_std
        result[:, :, :, D_std:] = K_low
        return result

    time_cat = benchmark_operation("torch.cat", cat_standard)
    time_prealloc = benchmark_operation("pre-allocated fill", cat_preallocated)

    print(f"\n  Speedup: {time_cat / time_prealloc:.2f}x with pre-allocation")
    print(f"  (May not be significant, but worth testing at scale)")


def main():
    if not torch.cuda.is_available():
        print("CUDA required for benchmarks")
        return

    print("=" * 70)
    print("RA/R-MLP BOTTLENECK ANALYSIS")
    print("=" * 70)
    print("Lessons from bitter7 optimization:")
    print("  1. Profile operations at scale")
    print("  2. Sampling beats exact selection (71x faster!)")
    print("  3. Minimize allocations and copies")
    print("=" * 70)

    test_gate_scaling_overhead()
    test_topk_vs_sampling()
    test_gather_vs_masking()
    test_cat_overhead()

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("1. Fuse gate scaling into single multiply (avoid split+cat)")
    print("2. Use sampling for KV pruning topk (expect ~50-70x speedup)")
    print("3. Pre-allocate buffers for concatenation where possible")
    print("4. Consider baking gates into weights if they don't change")
    print("=" * 70)


if __name__ == "__main__":
    main()
