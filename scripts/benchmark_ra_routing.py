#!/usr/bin/env python3
"""
Benchmark script for RA (Reciprocal Attention) with compute routing.

Tests performance of:
1. RAAttention (SDPA vs manual attention)
2. Router overhead (contextual hardness computation)
3. Full RABlock throughput
4. Comparison vs baseline CausalSelfAttention

Run before full training to validate performance expectations.
"""

import argparse
import sys
import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

# Add parent to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ra import (
    RAConfig,
    RAAttention,
    ContextShiftGate,
    ContextRouter,
    RoutedMixer,
    RABlock,
)
from gpt2.model import CausalSelfAttention, GPTConfig


@contextmanager
def timer(name, warmup=False):
    """Context manager for timing with optional warmup skip."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    if not warmup:
        print(f"  {name}: {elapsed:.2f} ms")
    return elapsed


def benchmark_attention(
    device, batch_size=8, seq_len=1024, n_embd=768, n_heads=12, n_iters=100
):
    """Benchmark RAAttention vs CausalSelfAttention."""
    print(f"\n{'='*60}")
    print("ATTENTION BENCHMARK")
    print(f"  batch={batch_size}, seq={seq_len}, d={n_embd}, heads={n_heads}")
    print(f"  iterations={n_iters}")
    print(f"{'='*60}")

    # Create configs
    ra_config = RAConfig(
        d_model=n_embd,
        n_heads=n_heads,
        block_size=seq_len,
        ra_head_frac=0.25,
    )

    gpt_config = GPTConfig(
        block_size=seq_len,
        vocab_size=50257,
        n_layer=12,
        n_head=n_heads,
        n_embd=n_embd,
        bias=False,  # Match RAAttention which uses bias=False
    )

    # Create models
    ra_attn = RAAttention(ra_config).to(device).eval()
    baseline_attn = CausalSelfAttention(gpt_config).to(device).eval()

    # Debug: check if flash attention is enabled
    print(f"\nBaseline flash={baseline_attn.flash}, k_eq_vt={baseline_attn.k_eq_vt}")

    # Input tensor
    x = torch.randn(batch_size, seq_len, n_embd, device=device)

    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = ra_attn(x)
            _ = baseline_attn(x)

    # Benchmark RAAttention
    print("\nRAAttention (SDPA):")
    ra_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out_full, out_ra, _ = ra_attn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ra_times.append((time.perf_counter() - start) * 1000)

    ra_mean = sum(ra_times) / len(ra_times)
    ra_std = (sum((t - ra_mean) ** 2 for t in ra_times) / len(ra_times)) ** 0.5
    print(f"  Mean: {ra_mean:.2f} ms ± {ra_std:.2f} ms")

    # Benchmark baseline
    print("\nCausalSelfAttention (baseline):")
    baseline_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = baseline_attn(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_times.append((time.perf_counter() - start) * 1000)

    baseline_mean = sum(baseline_times) / len(baseline_times)
    baseline_std = (
        sum((t - baseline_mean) ** 2 for t in baseline_times) / len(baseline_times)
    ) ** 0.5
    print(f"  Mean: {baseline_mean:.2f} ms ± {baseline_std:.2f} ms")

    # Comparison
    overhead = ((ra_mean / baseline_mean) - 1) * 100
    print(f"\nRA overhead vs baseline: {overhead:+.1f}%")

    return ra_mean, baseline_mean


def benchmark_router(device, batch_size=8, seq_len=1024, n_embd=768, n_iters=100):
    """Benchmark router components."""
    print(f"\n{'='*60}")
    print("ROUTER BENCHMARK")
    print(f"  batch={batch_size}, seq={seq_len}, d={n_embd}")
    print(f"  iterations={n_iters}")
    print(f"{'='*60}")

    ra_config = RAConfig(
        d_model=n_embd,
        n_heads=12,
        block_size=seq_len,
        router_hidden=16,
    )

    # Create components
    shift_gate = ContextShiftGate().to(device).eval()
    router = ContextRouter(ra_config).to(device).eval()
    mixer = RoutedMixer().to(device).eval()

    # Inputs
    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    tok_emb = torch.randn(batch_size, seq_len, n_embd, device=device)
    out_ra = torch.randn(batch_size, seq_len, n_embd, device=device)
    out_full = torch.randn(batch_size, seq_len, n_embd, device=device)

    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        with torch.no_grad():
            shift = shift_gate(x, tok_emb)
            probs = router(x, tok_emb, shift)
            _ = mixer(out_ra, out_full, probs)

    # Benchmark shift computation
    print("\nContextShiftGate (|x - E(x)|):")
    shift_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            shift = shift_gate(x, tok_emb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        shift_times.append((time.perf_counter() - start) * 1000)

    shift_mean = sum(shift_times) / len(shift_times)
    print(f"  Mean: {shift_mean:.3f} ms")

    # Benchmark router MLP
    print("\nContextRouter (MLP + softmax):")
    router_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            probs = router(x, tok_emb, shift)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        router_times.append((time.perf_counter() - start) * 1000)

    router_mean = sum(router_times) / len(router_times)
    print(f"  Mean: {router_mean:.3f} ms")

    # Benchmark mixer
    print("\nRoutedMixer (weighted sum):")
    mixer_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = mixer(out_ra, out_full, probs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mixer_times.append((time.perf_counter() - start) * 1000)

    mixer_mean = sum(mixer_times) / len(mixer_times)
    print(f"  Mean: {mixer_mean:.3f} ms")

    total = shift_mean + router_mean + mixer_mean
    print(f"\nTotal router overhead: {total:.3f} ms")

    return shift_mean, router_mean, mixer_mean


def benchmark_full_block(
    device, batch_size=8, seq_len=1024, n_embd=768, n_heads=12, n_iters=100
):
    """Benchmark full RABlock in both phases."""
    print(f"\n{'='*60}")
    print("FULL BLOCK BENCHMARK")
    print(f"  batch={batch_size}, seq={seq_len}, d={n_embd}, heads={n_heads}")
    print(f"  iterations={n_iters}")
    print(f"{'='*60}")

    ra_config = RAConfig(
        d_model=n_embd,
        n_heads=n_heads,
        block_size=seq_len,
        ra_head_frac=0.25,
    )

    # Create block
    block = RABlock(ra_config, layer_idx=0).to(device).eval()

    # Inputs
    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    e_tok = torch.randn(batch_size, seq_len, n_embd, device=device)

    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = block(x, e_tok=e_tok)

    # Phase 1 (no routing)
    block.phase1 = True
    print("\nPhase 1 (warmup, no routing):")
    phase1_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = block(x, e_tok=e_tok)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase1_times.append((time.perf_counter() - start) * 1000)

    phase1_mean = sum(phase1_times) / len(phase1_times)
    phase1_std = (
        sum((t - phase1_mean) ** 2 for t in phase1_times) / len(phase1_times)
    ) ** 0.5
    print(f"  Mean: {phase1_mean:.2f} ms ± {phase1_std:.2f} ms")

    # Phase 2 (routing enabled)
    block.phase1 = False
    print("\nPhase 2 (routing enabled):")
    phase2_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = block(x, e_tok=e_tok)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase2_times.append((time.perf_counter() - start) * 1000)

    phase2_mean = sum(phase2_times) / len(phase2_times)
    phase2_std = (
        sum((t - phase2_mean) ** 2 for t in phase2_times) / len(phase2_times)
    ) ** 0.5
    print(f"  Mean: {phase2_mean:.2f} ms ± {phase2_std:.2f} ms")

    overhead = ((phase2_mean / phase1_mean) - 1) * 100
    print(f"\nRouting overhead (phase2 vs phase1): {overhead:+.1f}%")

    return phase1_mean, phase2_mean


def benchmark_head_groups(
    device, batch_size=8, seq_len=1024, n_embd=768, n_heads=12, n_iters=100
):
    """Benchmark FULL-only vs RA-only attention to show potential savings."""
    print(f"\n{'='*60}")
    print("HEAD GROUP BENCHMARK (potential savings)")
    print(f"  batch={batch_size}, seq={seq_len}, d={n_embd}, heads={n_heads}")
    print(f"  iterations={n_iters}")
    print(f"{'='*60}")

    # With ra_head_frac=0.25, we have 9 FULL + 3 RA heads
    ra_head_frac = 0.25
    n_ra = max(1, int(round(ra_head_frac * n_heads)))
    n_ra = min(n_ra, n_heads - 1)
    n_full = n_heads - n_ra

    print(f"\n  Head split: {n_full} FULL + {n_ra} RA = {n_heads} total")

    # Create configs for different head counts
    # Simulate "RA-only" by creating attention with only n_ra heads
    ra_only_config = RAConfig(
        d_model=n_embd,
        n_heads=n_ra,  # Only RA heads
        block_size=seq_len,
        ra_head_frac=1.0,  # All heads are "RA"
    )

    # Simulate "FULL-only" with n_full heads
    full_only_config = RAConfig(
        d_model=n_embd,
        n_heads=n_full,  # Only FULL heads
        block_size=seq_len,
        ra_head_frac=0.0,  # All heads are "FULL"
    )

    # Standard config with all heads
    all_heads_config = RAConfig(
        d_model=n_embd,
        n_heads=n_heads,
        block_size=seq_len,
        ra_head_frac=0.0,
    )

    # Create attention modules
    # Note: We need to adjust d_model to match head count for fair comparison
    # Actually, let's just measure projection cost difference

    # Measure full SDPA with different head counts
    head_dim = n_embd // n_heads

    # Create baseline config for comparison
    gpt_config_all = GPTConfig(
        block_size=seq_len,
        vocab_size=50257,
        n_layer=12,
        n_head=n_heads,
        n_embd=n_embd,
        bias=False,
    )

    # RA-only config (3 heads, same head_dim)
    # To keep head_dim constant, we need d_model = n_ra * head_dim
    d_ra_model = n_ra * head_dim
    gpt_config_ra = GPTConfig(
        block_size=seq_len,
        vocab_size=50257,
        n_layer=12,
        n_head=n_ra,
        n_embd=d_ra_model,
        bias=False,
    )

    # Create attention modules
    attn_all = CausalSelfAttention(gpt_config_all).to(device).eval()
    attn_ra = CausalSelfAttention(gpt_config_ra).to(device).eval()

    # Input tensors
    x_all = torch.randn(batch_size, seq_len, n_embd, device=device)
    x_ra = torch.randn(batch_size, seq_len, d_ra_model, device=device)

    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = attn_all(x_all)
            _ = attn_ra(x_ra)

    # Benchmark all heads attention
    print(f"\nAll heads attention ({n_heads} heads, d={n_embd}):")
    all_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = attn_all(x_all)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        all_times.append((time.perf_counter() - start) * 1000)

    all_mean = sum(all_times) / len(all_times)
    print(f"  Mean: {all_mean:.3f} ms")

    # Benchmark RA-only attention
    print(f"\nRA-only attention ({n_ra} heads, d={d_ra_model}):")
    ra_times = []
    for _ in range(n_iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = attn_ra(x_ra)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ra_times.append((time.perf_counter() - start) * 1000)

    ra_mean = sum(ra_times) / len(ra_times)
    savings_ra = (1 - ra_mean / all_mean) * 100
    print(f"  Mean: {ra_mean:.3f} ms ({savings_ra:+.1f}% vs all)")

    print(f"\nPotential savings when router chooses RA-only: {savings_ra:.1f}%")

    return all_mean, ra_mean


def benchmark_memory(device, batch_size=8, seq_len=1024, n_embd=768, n_heads=12):
    """Benchmark memory usage."""
    if not torch.cuda.is_available():
        print("\nMemory benchmark requires CUDA")
        return

    print(f"\n{'='*60}")
    print("MEMORY BENCHMARK")
    print(f"  batch={batch_size}, seq={seq_len}, d={n_embd}, heads={n_heads}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    ra_config = RAConfig(
        d_model=n_embd,
        n_heads=n_heads,
        block_size=seq_len,
        ra_head_frac=0.25,
    )

    # Create block and inputs
    block = RABlock(ra_config, layer_idx=0).to(device)
    x = torch.randn(batch_size, seq_len, n_embd, device=device, requires_grad=True)
    e_tok = torch.randn(batch_size, seq_len, n_embd, device=device)

    # Forward + backward
    block.phase1 = False
    out = block(x, e_tok=e_tok)
    loss = out.sum()
    loss.backward()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak memory (forward + backward): {peak_mem:.2f} GB")

    return peak_mem


def main():
    parser = argparse.ArgumentParser(description="Benchmark RA routing components")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Run benchmarks
    benchmark_attention(
        device, args.batch_size, args.seq_len, args.n_embd, args.n_heads, args.n_iters
    )

    benchmark_router(device, args.batch_size, args.seq_len, args.n_embd, args.n_iters)

    benchmark_full_block(
        device, args.batch_size, args.seq_len, args.n_embd, args.n_heads, args.n_iters
    )

    benchmark_head_groups(
        device, args.batch_size, args.seq_len, args.n_embd, args.n_heads, args.n_iters
    )

    benchmark_memory(device, args.batch_size, args.seq_len, args.n_embd, args.n_heads)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
