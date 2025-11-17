#!/usr/bin/env python3
"""
Quick benchmark: Does RA provide better validation loss than standard attention?

Tests 4 configurations over fixed wall-clock time:
1. Standard attention (SDPA fused) - fastest baseline
2. Standard attention (open-coded) - fair comparison baseline
3. RA attention (open-coded) - to prove RA value
4. RA attention (Triton) - to prove optimization works

Measures: iterations completed and val_loss achieved in fixed time budget.
"""

import torch
import torch.nn.functional as F
import time
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
B, H, T, D = 8, 12, 1024, 64  # GPT-2 training config


def standard_attention_sdpa(Q, K, V):
    """Fastest baseline: PyTorch SDPA with kernel fusion."""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def standard_attention_opencoded(Q, K, V):
    """Fair comparison: open-coded standard attention."""
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    S = S.masked_fill(~mask, float('-inf'))
    attn = F.softmax(S, dim=-1)
    return torch.matmul(attn, V)


def ra_attention_opencoded(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """RA with reciprocity: open-coded implementation."""
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    # Per-head gates (broadcast)
    w_std_bc = w_std.view(B, H, 1, 1)
    w_rec_bc = w_rec.view(B, H, 1, 1)
    w_disc_bc = w_disc.view(B, H, 1, 1)

    # Combine: standard + reciprocity + discoverability
    logits = w_std_bc * S + w_rec_bc * S.transpose(-2, -1) + w_disc_bc * d_bias.unsqueeze(-2)

    # Causal mask
    mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float('-inf'))

    attn = F.softmax(logits, dim=-1)
    return torch.matmul(attn, V)


def ra_attention_triton(Q, K, V, d_bias, w_std, w_rec, w_disc):
    """RA with reciprocity: Triton fused kernel."""
    try:
        from triton_ra_attention import triton_ra_attention
        return triton_ra_attention(Q, K, V, d_bias, w_std, w_rec, w_disc)
    except ImportError:
        print("⚠️  Triton not available, falling back to open-coded")
        return ra_attention_opencoded(Q, K, V, d_bias, w_std, w_rec, w_disc)


def benchmark_throughput(attention_fn, name, time_budget_sec=10.0, use_ra=False):
    """
    Run attention_fn for time_budget_sec, measure iterations/sec.

    Args:
        attention_fn: Function to benchmark
        name: Display name
        time_budget_sec: How long to run
        use_ra: Whether this is RA (needs extra args)

    Returns:
        (iterations_per_sec, ms_per_iter)
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {name}")
    print(f"Time budget: {time_budget_sec}s")
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
        if use_ra:
            _ = attention_fn(Q, K, V, d_bias, w_std, w_rec, w_disc)
        else:
            _ = attention_fn(Q, K, V)
    torch.cuda.synchronize()
    print("done")

    # Benchmark
    print(f"Running for {time_budget_sec}s...", end=" ", flush=True)
    iterations = 0
    start_time = time.time()

    while (time.time() - start_time) < time_budget_sec:
        if use_ra:
            _ = attention_fn(Q, K, V, d_bias, w_std, w_rec, w_disc)
        else:
            _ = attention_fn(Q, K, V)
        iterations += 1

    torch.cuda.synchronize()
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
    """Run benchmark comparison."""
    print("="*70)
    print("RA Value Proposition Benchmark")
    print("="*70)
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Config: B={B}, H={H}, T={T}, D={D}")
    print()

    time_budget = 10.0  # 10 seconds per test

    results = {}

    # Test 1: SDPA (fastest possible baseline)
    print("\n" + "="*70)
    print("TEST 1: Standard Attention (SDPA fused)")
    print("="*70)
    iters_sec, ms_iter = benchmark_throughput(
        standard_attention_sdpa,
        "SDPA (fused)",
        time_budget,
        use_ra=False
    )
    results['sdpa'] = {'iters_sec': iters_sec, 'ms_iter': ms_iter}

    # Test 2: Open-coded standard (fair comparison)
    print("\n" + "="*70)
    print("TEST 2: Standard Attention (open-coded)")
    print("="*70)
    iters_sec, ms_iter = benchmark_throughput(
        standard_attention_opencoded,
        "Standard (open-coded)",
        time_budget,
        use_ra=False
    )
    results['opencoded'] = {'iters_sec': iters_sec, 'ms_iter': ms_iter}

    # Test 3: RA open-coded
    print("\n" + "="*70)
    print("TEST 3: RA Attention (open-coded)")
    print("="*70)
    iters_sec, ms_iter = benchmark_throughput(
        ra_attention_opencoded,
        "RA (open-coded)",
        time_budget,
        use_ra=True
    )
    results['ra_opencoded'] = {'iters_sec': iters_sec, 'ms_iter': ms_iter}

    # Test 4: RA Triton
    print("\n" + "="*70)
    print("TEST 4: RA Attention (Triton fused)")
    print("="*70)
    iters_sec, ms_iter = benchmark_throughput(
        ra_attention_triton,
        "RA (Triton)",
        time_budget,
        use_ra=True
    )
    results['ra_triton'] = {'iters_sec': iters_sec, 'ms_iter': ms_iter}

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: Throughput Comparison")
    print("="*70)
    print(f"{'Configuration':<30} {'ms/iter':>12} {'iters/sec':>12} {'vs SDPA':>12}")
    print("-"*70)

    sdpa_ms = results['sdpa']['ms_iter']
    for name, display in [
        ('sdpa', 'SDPA (fused)'),
        ('opencoded', 'Standard (open-coded)'),
        ('ra_opencoded', 'RA (open-coded)'),
        ('ra_triton', 'RA (Triton)')
    ]:
        ms = results[name]['ms_iter']
        iters = results[name]['iters_sec']
        slowdown = ms / sdpa_ms
        print(f"{display:<30} {ms:>12.2f} {iters:>12.2f} {slowdown:>11.2f}x")

    print("\n" + "="*70)
    print("ANALYSIS: What does RA need to win?")
    print("="*70)

    sdpa_iters = results['sdpa']['iters_sec']
    ra_triton_iters = results['ra_triton']['iters_sec']
    slowdown = sdpa_iters / ra_triton_iters

    print(f"\nRA (Triton) is {slowdown:.2f}x slower than SDPA baseline.")
    print(f"\nFor RA to be worth it in 1 hour of training:")
    print(f"  SDPA completes: {int(sdpa_iters * 3600)} iterations")
    print(f"  RA completes: {int(ra_triton_iters * 3600)} iterations")
    print(f"\nRA must achieve BETTER validation loss with {(1 - ra_triton_iters/sdpa_iters)*100:.1f}% fewer iterations!")
    print(f"Example: If SDPA reaches val_loss=3.57, RA must reach val_loss < 3.57")
    print(f"         or reach 3.57 in significantly fewer iterations.")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run actual training comparison:")
    print("   - Standard attention (SDPA) for 1 hour → measure final val_loss")
    print("   - RA attention (Triton) for 1 hour → measure final val_loss")
    print("   - Compare: does RA's quality improvement justify the slowdown?")
    print()
    print("2. If RA shows promise, optimize further:")
    print("   - Sparse reciprocity (only high-attention positions)")
    print("   - Local reciprocity (windowed S^T)")
    print("   - Layer-selective RA (only middle layers)")
    print("   - Learned gating (adaptive reciprocity)")
    print()
    print("3. If RA doesn't show value:")
    print("   - Reconsider the architectural approach")
    print("   - Focus on other mechanisms (MLA, cross-layer)")
    print("="*70)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available - this benchmark requires GPU")
        exit(1)

    main()
