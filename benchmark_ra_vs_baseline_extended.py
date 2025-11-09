#!/usr/bin/env python3
"""
Extended benchmark: Test efficient RA variants vs baseline SDPA.

Compares:
1. SDPA (baseline)
2. RA (Triton fused) - our current best
3. RA (local window) - reciprocity within ±128 tokens only
4. RA (sparse) - reciprocity for top-10% attention weights only
5. RA (local + sparse) - combined optimization
6. RA (vectorized) - fast implementation of local + sparse

Goal: Find which optimization gets closest to SDPA speed while maintaining RA benefits.
"""

import torch
import torch.nn.functional as F
import time
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
B, H, T, D = 8, 12, 1024, 64  # GPT-2 training config


def standard_attention_sdpa(Q, K, V):
    """Baseline: PyTorch SDPA with kernel fusion."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def ra_attention_triton(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """Our Triton fused kernel."""
    try:
        from triton_ra_attention import triton_ra_attention
        return triton_ra_attention(Q, K, V, d_bias, w_std, w_rec, w_disc)
    except Exception as e:
        print(f"⚠️  Triton failed: {e}")
        return torch.zeros(B, H, T, D, device=device)


def ra_attention_local(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """Local window reciprocity (±128 tokens)."""
    from efficient_ra_attention import efficient_ra_attention_v1_local_only
    return efficient_ra_attention_v1_local_only(Q, K, V, d_bias, w_std, w_rec, w_disc, window=128)


def ra_attention_sparse(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """Sparse reciprocity (top-10% attention weights)."""
    from efficient_ra_attention import efficient_ra_attention_v2_sparse_only
    return efficient_ra_attention_v2_sparse_only(Q, K, V, d_bias, w_std, w_rec, w_disc, sparsity=0.1)


def ra_attention_local_sparse(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """Combined local + sparse reciprocity."""
    from efficient_ra_attention import efficient_ra_attention_v3_local_sparse_combined
    return efficient_ra_attention_v3_local_sparse_combined(Q, K, V, d_bias, w_std, w_rec, w_disc,
                                                             window=128, sparsity=0.1)


def ra_attention_vectorized(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """Vectorized local + sparse (fastest implementation)."""
    from efficient_ra_attention import efficient_ra_attention_v4_vectorized
    return efficient_ra_attention_v4_vectorized(Q, K, V, d_bias, w_std, w_rec, w_disc,
                                                 window=128, sparsity=0.1)


def benchmark_throughput(attention_fn, name, time_budget_sec=10.0, use_ra=False):
    """Run attention_fn for time_budget_sec, measure iterations/sec."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"{'='*70}")

    # Create test data
    Q = torch.randn(B, H, T, D, device=device)
    K = torch.randn(B, H, T, D, device=device)
    V = torch.randn(B, H, T, D, device=device)

    if use_ra:
        d_bias = torch.randn(B, H, T, device=device)
        w_std = torch.full((B, H), 0.5, device=device)
        w_rec = torch.full((B, H), 0.3, device=device)
        w_disc = torch.full((B, H), 0.2, device=device)

    # Warmup
    print("Warming up...", end=" ", flush=True)
    for _ in range(10):
        try:
            if use_ra:
                _ = attention_fn(Q, K, V, d_bias, w_std, w_rec, w_disc)
            else:
                _ = attention_fn(Q, K, V)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"\n❌ Error during warmup: {e}")
            return None, None
    print("done")

    # Benchmark
    print(f"Running for {time_budget_sec}s...", end=" ", flush=True)
    iterations = 0
    start_time = time.time()

    try:
        while (time.time() - start_time) < time_budget_sec:
            if use_ra:
                _ = attention_fn(Q, K, V, d_bias, w_std, w_rec, w_disc)
            else:
                _ = attention_fn(Q, K, V)
            iterations += 1

        torch.cuda.synchronize()
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        return None, None

    elapsed = time.time() - start_time
    print("done")

    iters_per_sec = iterations / elapsed
    ms_per_iter = (elapsed / iterations) * 1000

    print(f"\nResults:")
    print(f"  Iterations completed: {iterations}")
    print(f"  Throughput: {iters_per_sec:.2f} iters/sec")
    print(f"  Latency: {ms_per_iter:.2f} ms/iter")

    return iters_per_sec, ms_per_iter


def main():
    """Run extended benchmark comparison."""
    print("="*70)
    print("Extended RA Optimization Benchmark")
    print("="*70)
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Config: B={B}, H={H}, T={T}, D={D}")
    print()

    time_budget = 10.0  # 10 seconds per test
    results = {}

    tests = [
        ("sdpa", "SDPA (baseline)", standard_attention_sdpa, False),
        ("ra_triton", "RA (Triton fused)", ra_attention_triton, True),
        ("ra_local", "RA (local window ±128)", ra_attention_local, True),
        ("ra_sparse", "RA (sparse 10%)", ra_attention_sparse, True),
        ("ra_local_sparse", "RA (local + sparse)", ra_attention_local_sparse, True),
        ("ra_vectorized", "RA (vectorized)", ra_attention_vectorized, True),
    ]

    # Run all tests
    for key, display_name, fn, use_ra in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {display_name}")
        print(f"{'='*70}")

        iters_sec, ms_iter = benchmark_throughput(fn, display_name, time_budget, use_ra)

        if iters_sec is not None:
            results[key] = {'iters_sec': iters_sec, 'ms_iter': ms_iter, 'name': display_name}
        else:
            print(f"⚠️  Skipping {display_name} due to errors")

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: Throughput Comparison")
    print("="*70)
    print(f"{'Configuration':<35} {'ms/iter':>12} {'iters/sec':>12} {'vs SDPA':>12}")
    print("-"*70)

    if 'sdpa' not in results:
        print("❌ SDPA baseline failed, cannot compute relative speedups")
        return

    sdpa_ms = results['sdpa']['ms_iter']
    sdpa_iters = results['sdpa']['iters_sec']

    for key in ['sdpa', 'ra_triton', 'ra_local', 'ra_sparse', 'ra_local_sparse', 'ra_vectorized']:
        if key not in results:
            continue

        name = results[key]['name']
        ms = results[key]['ms_iter']
        iters = results[key]['iters_sec']
        slowdown = ms / sdpa_ms

        print(f"{name:<35} {ms:>12.2f} {iters:>12.2f} {slowdown:>11.2f}x")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS: Which optimization gets closest to SDPA?")
    print("="*70)

    best_ra = None
    best_slowdown = float('inf')

    for key in ['ra_triton', 'ra_local', 'ra_sparse', 'ra_local_sparse', 'ra_vectorized']:
        if key not in results:
            continue

        slowdown = results[key]['ms_iter'] / sdpa_ms
        if slowdown < best_slowdown:
            best_slowdown = slowdown
            best_ra = key

    if best_ra:
        print(f"\n🏆 Best RA variant: {results[best_ra]['name']}")
        print(f"   Slowdown vs SDPA: {best_slowdown:.2f}x")
        print(f"   Throughput: {results[best_ra]['iters_sec']:.2f} iters/sec")

        # Calculate what quality improvement is needed
        iterations_ratio = results[best_ra]['iters_sec'] / sdpa_iters
        print(f"\n📊 In 1 hour of training:")
        print(f"   SDPA completes: {int(sdpa_iters * 3600)} iterations")
        print(f"   Best RA completes: {int(results[best_ra]['iters_sec'] * 3600)} iterations")
        print(f"   RA does {(1 - iterations_ratio)*100:.1f}% fewer iterations")
        print(f"\n💡 For RA to win: must achieve better val_loss with {(1 - iterations_ratio)*100:.1f}% fewer iterations!")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if best_ra == 'ra_triton':
        print("✓ Triton kernel is already the best RA implementation")
        print("  → Consider hybrid architecture (RA in some layers, SDPA in others)")
        print("  → Or progressive annealing (full RA early, reduce late)")
    elif best_slowdown < 1.5:
        print(f"✓ {results[best_ra]['name']} is within 1.5x of SDPA!")
        print("  → This is competitive - run quality comparison training")
        print("  → If quality is better, RA is worth using")
    elif best_slowdown < 2.0:
        print(f"✓ {results[best_ra]['name']} is within 2x of SDPA")
        print("  → Marginal - needs significant quality improvement to justify")
        print("  → Consider further optimizations or hybrid approach")
    else:
        print(f"⚠️  Best RA ({best_slowdown:.2f}x slower) is still far from SDPA")
        print("  → RA needs major quality improvement to be worth the cost")
        print("  → Or consider fundamentally different approach")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"1. Run actual training comparison with best variant: {results[best_ra]['name']}")
    print("   python3 train_comparison.py --baseline sdpa --test", best_ra)
    print()
    print("2. Measure validation loss curves over 1 hour")
    print("   - Does RA reach better loss despite fewer iterations?")
    print()
    print("3. If RA shows promise, integrate into full training pipeline")
    print("   - Add to ra_lens_gpt2.py")
    print("   - Run ablation study with optimized RA")
    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available - this benchmark requires GPU")
        exit(1)

    main()
