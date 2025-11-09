#!/usr/bin/env python3
"""
Test different D_std/R ratios to find optimal speed/quality tradeoff.

The Q@W and K@W projections cost scales with R.
Lower R = faster, but less reciprocal capacity.

Find the sweet spot!
"""

import torch
import torch.nn.functional as F
import time
from ra_same_flop import ra_same_flop_v1_basic

device = "cuda" if torch.cuda.is_available() else "cpu"
B, H, T, D = 8, 12, 1024, 64

# Create test inputs
Q = torch.randn(B, H, T, D, device=device)
K = torch.randn(B, H, T, D, device=device)
V = torch.randn(B, H, T, D, device=device)
d_bias = torch.randn(B, H, T, device=device)
w_std = torch.full((B, H), 0.5, device=device)
w_rec = torch.full((B, H), 0.3, device=device)
w_disc = torch.full((B, H), 0.2, device=device)

print("="*70)
print("Same-FLOP RA: Testing D_std/R Ratios")
print("="*70)
print(f"Config: B={B}, H={H}, T={T}, D={D}")
print()

# Baseline
print("Baseline SDPA...")
for _ in range(10):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
torch.cuda.synchronize()
sdpa_time = (time.time() - start) / 100 * 1000
print(f"   {sdpa_time:.2f} ms/iter")
print()

# Test different ratios
ratios = [
    (60, 4),   # 60 standard, 4 reciprocal (minimal overhead)
    (56, 8),   # 56 standard, 8 reciprocal
    (52, 12),  # 52 standard, 12 reciprocal
    (48, 16),  # 48 standard, 16 reciprocal (current)
    (40, 24),  # 40 standard, 24 reciprocal (more reciprocal)
    (32, 32),  # 32 standard, 32 reciprocal (balanced)
]

results = []

for D_std, R in ratios:
    print(f"Testing D_std={D_std}, R={R}...")

    # Warmup
    for _ in range(10):
        _ = ra_same_flop_v1_basic(Q, K, V, d_bias, w_std, w_rec, w_disc, D_std=D_std, R=R)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = ra_same_flop_v1_basic(Q, K, V, d_bias, w_std, w_rec, w_disc, D_std=D_std, R=R)
    torch.cuda.synchronize()
    ra_time = (time.time() - start) / 100 * 1000

    overhead = ra_time / sdpa_time
    results.append((D_std, R, ra_time, overhead))

    print(f"   {ra_time:.2f} ms/iter ({overhead:.2f}x vs SDPA)")
    print()

# Summary
print("="*70)
print("SUMMARY: Finding Optimal Ratio")
print("="*70)
print(f"{'D_std':>6} {'R':>4} {'ms/iter':>10} {'vs SDPA':>10} {'Overhead':>10}")
print("-"*70)

for D_std, R, ra_time, overhead in results:
    print(f"{D_std:>6} {R:>4} {ra_time:>10.2f} {overhead:>10.2f}x {(overhead-1)*100:>9.1f}%")

# Find best
best_idx = min(range(len(results)), key=lambda i: results[i][2])
best_D_std, best_R, best_time, best_overhead = results[best_idx]

print()
print("="*70)
print("OPTIMAL CONFIGURATION")
print("="*70)
print(f"Best ratio: D_std={best_D_std}, R={best_R}")
print(f"Time: {best_time:.2f} ms/iter ({best_overhead:.2f}x vs SDPA)")
print(f"Overhead: {(best_overhead-1)*100:.1f}%")
print()

if best_overhead < 1.2:
    print("🎉 Within 20% of baseline! This is very promising!")
    print("   If RA provides ANY quality benefit, it's worth using.")
elif best_overhead < 1.3:
    print("✅ Within 30% of baseline - good progress!")
    print("   RA needs moderate quality improvement to justify.")
else:
    print("⚠️  Still >30% overhead")
    print("   RA needs significant quality improvement.")

print()
print("Next optimization: Head-selective routing")
print(f"  If 50% of heads learn w_rec≈0 → {best_overhead * 0.5 + 0.5:.2f}x overhead")
print(f"  If 75% of heads learn w_rec≈0 → {best_overhead * 0.25 + 0.75:.2f}x overhead")
print("="*70)
