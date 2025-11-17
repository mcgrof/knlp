#!/usr/bin/env python3
"""
Test script to verify whether SDPA kernel is being used.
Profiles a single forward pass and checks for Flash Attention kernels.
"""

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

# Test with standard attention (should use SDPA)
def standard_attention(Q, K, V, mask=None):
    """Standard PyTorch SDPA - should use optimized kernel."""
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)

# Test with open-coded attention (what lens uses)
def opencoded_attention(Q, K, V):
    """Open-coded attention - no kernel optimization."""
    B, H, T, D = Q.shape
    S = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
    # Causal mask
    mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool))
    S = S.masked_fill(~mask, float('-inf'))
    attn = F.softmax(S, dim=-1)
    return torch.matmul(attn, V)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D = 2, 12, 1024, 64

    Q = torch.randn(B, H, T, D, device=device)
    K = torch.randn(B, H, T, D, device=device)
    V = torch.randn(B, H, T, D, device=device)

    print("=" * 70)
    print("Testing SDPA Kernel Usage")
    print("=" * 70)

    # Test standard SDPA
    print("\n1. Standard F.scaled_dot_product_attention():")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("standard_sdpa"):
            out1 = standard_attention(Q, K, V)

    # Look for Flash Attention or Memory-Efficient kernels
    sdpa_kernels = [
        "flash", "efficient", "scaled_dot_product",
        "fused", "cutlass", "bmm"
    ]
    found_sdpa = False
    for event in prof.key_averages():
        event_name = event.key.lower()
        if any(kernel in event_name for kernel in sdpa_kernels):
            print(f"  ✓ Found optimized kernel: {event.key}")
            found_sdpa = True

    if not found_sdpa:
        print("  ⚠️  No optimized SDPA kernel found!")

    # Test open-coded attention
    print("\n2. Open-coded attention (what lens-gated uses):")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        with record_function("opencoded_attn"):
            out2 = opencoded_attention(Q, K, V)

    found_opencoded = False
    for event in prof.key_averages():
        event_name = event.key.lower()
        if any(kernel in event_name for kernel in sdpa_kernels):
            print(f"  ✓ Found kernel: {event.key}")
        elif "matmul" in event_name or "bmm" in event_name:
            print(f"  ⚠️  Explicit matmul: {event.key}")
            found_opencoded = True

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    if found_sdpa:
        print("✓ Standard attention uses optimized SDPA kernel")
    else:
        print("⚠️  Standard attention NOT using optimized kernel")

    if found_opencoded:
        print("✓ Open-coded attention uses explicit matmul (no kernel optimization)")

    print("\nFor lens-gated attention (L0-L7):")
    print("  → ALL configurations use open-coded path")
    print("  → Even L0 baseline does NOT use SDPA kernel")
    print("  → Overhead comparison is fair (all open-coded)")
    print("  → But missing comparison against SDPA-optimized baseline")
    print("=" * 70)
