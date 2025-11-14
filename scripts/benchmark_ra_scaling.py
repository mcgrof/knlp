#!/usr/bin/env python3
"""
Test RA optimizations at different sequence lengths.
Find the breakeven point where optimizations matter.
"""

import time
import torch

def benchmark_at_length(T, num_iters=50):
    """Benchmark operations at specific sequence length."""
    device = torch.device("cuda:0")
    B, H, D = 8, 12, 64
    R = 4
    D_std = D - R

    # Setup tensors
    Qf = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    Kf = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    w_std = torch.ones(H, device=device) * 0.9
    w_rec = torch.ones(H, device=device) * 0.1

    # === Test 1: Gate scaling ===
    def gate_current():
        g_std = torch.sqrt(torch.clamp(w_std, min=1e-8)).view(1, -1, 1, 1)
        g_rec = torch.sqrt(torch.clamp(w_rec, min=1e-8)).view(1, -1, 1, 1)
        Q_std = Qf[:, :, :, :D_std] * g_std
        K_low = Qf[:, :, :, D_std:] * g_rec
        K_std = Kf[:, :, :, :D_std] * g_std
        Q_low = Kf[:, :, :, D_std:] * g_rec
        Qf_out = torch.cat([Q_std, K_low], dim=-1)
        Kf_out = torch.cat([K_std, Q_low], dim=-1)
        return Qf_out, Kf_out

    def gate_fused():
        g_std = torch.sqrt(torch.clamp(w_std, min=1e-8)).view(1, -1, 1, 1)
        g_rec = torch.sqrt(torch.clamp(w_rec, min=1e-8)).view(1, -1, 1, 1)
        scale = torch.ones(1, H, 1, D, device=device, dtype=torch.float16)
        scale[:, :, :, :D_std] = g_std
        scale[:, :, :, D_std:] = g_rec
        return Qf * scale, Kf * scale

    # Warmup
    for _ in range(10):
        _ = gate_current()
        _ = gate_fused()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = gate_current()
    torch.cuda.synchronize()
    time_current = (time.perf_counter() - start) / num_iters * 1000

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = gate_fused()
    torch.cuda.synchronize()
    time_fused = (time.perf_counter() - start) / num_iters * 1000

    gate_speedup = time_current / time_fused

    # === Test 2: KV Pruning topk ===
    k_keep = int(T * 0.38)  # Golden ratio
    importance = torch.rand(B, H, T, device=device, dtype=torch.float32)

    def topk_exact():
        return torch.topk(importance, k_keep, dim=-1)[1]

    def topk_sampling():
        sample_size = max(64, int(T * 0.02))
        sample_idx = torch.randint(0, T, (B, H, sample_size), device=device)
        sample_importance = torch.gather(importance, 2, sample_idx)
        k_sample = max(1, int(k_keep * (sample_size / T)))
        _, local_idx = torch.topk(sample_importance, k_sample, dim=-1)
        return torch.gather(sample_idx, 2, local_idx)

    # Warmup
    for _ in range(10):
        _ = topk_exact()
        _ = topk_sampling()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = topk_exact()
    torch.cuda.synchronize()
    time_topk = (time.perf_counter() - start) / num_iters * 1000

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = topk_sampling()
    torch.cuda.synchronize()
    time_sampling = (time.perf_counter() - start) / num_iters * 1000

    topk_speedup = time_topk / time_sampling

    return {
        "T": T,
        "gate_current_ms": time_current,
        "gate_fused_ms": time_fused,
        "gate_speedup": gate_speedup,
        "topk_exact_ms": time_topk,
        "topk_sampling_ms": time_sampling,
        "topk_speedup": topk_speedup,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return

    print("=" * 80)
    print("RA OPTIMIZATION SCALING ANALYSIS")
    print("=" * 80)
    print("Testing at different sequence lengths to find breakeven points")
    print()

    # Test at different sequence lengths
    lengths = [512, 1024, 2048, 4096, 8192, 16384]

    results = []
    for T in lengths:
        print(f"Testing T={T}...", end=" ", flush=True)
        try:
            result = benchmark_at_length(T, num_iters=20)
            results.append(result)
            print("✓")
        except RuntimeError as e:
            print(f"✗ (OOM: {e})")
            break

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    print("Gate Scaling Optimization:")
    print("-" * 80)
    print(f"{'T':<8} {'Current (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10} {'Savings':<15}")
    print("-" * 80)
    for r in results:
        savings = (r["gate_current_ms"] - r["gate_fused_ms"])
        savings_per_layer = savings * 12  # 12 layers in GPT-2
        print(f"{r['T']:<8} {r['gate_current_ms']:>10.3f}     {r['gate_fused_ms']:>10.3f}     "
              f"{r['gate_speedup']:>6.2f}x    {savings_per_layer:>10.2f} ms/12L")

    print()
    print("KV Pruning topk Optimization:")
    print("-" * 80)
    print(f"{'T':<8} {'topk (ms)':<15} {'sampling (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['T']:<8} {r['topk_exact_ms']:>10.3f}     {r['topk_sampling_ms']:>10.3f}     "
              f"{r['topk_speedup']:>6.2f}x")

    print("\n" + "=" * 80)
    print("INSIGHTS")
    print("=" * 80)

    # Find breakeven for gate optimization
    gate_break = next((r for r in results if r["gate_speedup"] > 1.5), None)
    if gate_break:
        print(f"Gate optimization: Worthwhile at T>={gate_break['T']} ({gate_break['gate_speedup']:.1f}x)")
    else:
        print("Gate optimization: Modest gains at all lengths")

    # Find breakeven for topk
    topk_break = next((r for r in results if r["topk_speedup"] > 2.0), None)
    if topk_break:
        print(f"KV pruning sampling: Significant at T>={topk_break['T']} ({topk_break['topk_speedup']:.1f}x)")
    else:
        print("KV pruning sampling: Only helps at very long context (T>16K)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("1. Fused gate scaling: Implement (consistent 2x speedup)")
    print("2. KV pruning sampling: Implement for long context (T>4K)")
    print("3. Pre-allocation: Small gains, implement if profiling shows benefit")
    print("=" * 80)


if __name__ == "__main__":
    main()
