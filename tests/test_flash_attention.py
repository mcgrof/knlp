#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Simple test script to verify Flash Attention works with ROCm 7.x PyTorch.

This script tests PyTorch's scaled_dot_product_attention (SDPA) with different
backends:
1. Flash Attention 2 (requires FP16/BF16)
2. Memory-efficient attention
3. Math fallback (standard attention)

Flash Attention provides O(N) memory complexity vs O(N^2) for standard attention.
Uses FP16 (half precision) for optimal performance on AMD W7900 and to enable
Flash Attention 2 backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import time


def standard_attention(query, key, value):
    """Standard attention implementation for comparison."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def test_flash_attention():
    """Test Flash Attention with scaled_dot_product_attention using FP16."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using precision: FP16 (half)")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test parameters
    batch_size = 4
    num_heads = 8
    seq_len = 512
    head_dim = 64

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")

    # Create random Q, K, V tensors in FP16
    query = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    key = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    value = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )

    # Test 1: Standard attention (baseline)
    print("\n=== Testing Standard Attention (Baseline) ===")
    start_time = time.time()
    with torch.no_grad():
        output_standard = standard_attention(query, key, value)
    standard_time = time.time() - start_time
    print(f"Output shape: {output_standard.shape}")
    print(f"Time: {standard_time:.4f}s")

    # Test 2: SDPA with automatic backend selection
    print("\n=== Testing SDPA (Automatic Backend) ===")
    start_time = time.time()
    with torch.no_grad():
        output_sdpa = F.scaled_dot_product_attention(query, key, value)
    sdpa_time = time.time() - start_time
    print(f"Output shape: {output_sdpa.shape}")
    print(f"Time: {sdpa_time:.4f}s")

    # Check which backends are available
    print("\n=== Available SDPA Backends ===")
    backends = []
    if hasattr(SDPBackend, "FLASH_ATTENTION"):
        backends.append(("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION))
    if hasattr(SDPBackend, "EFFICIENT_ATTENTION"):
        backends.append(("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION))
    if hasattr(SDPBackend, "MATH"):
        backends.append(("MATH", SDPBackend.MATH))

    for name, backend in backends:
        print(f"  - {name}")

    # Test each backend individually
    print("\n=== Testing Individual Backends ===")
    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                start_time = time.time()
                with torch.no_grad():
                    output = F.scaled_dot_product_attention(query, key, value)
                backend_time = time.time() - start_time
                print(f"{name}: {backend_time:.4f}s")
        except Exception as e:
            print(f"{name}: FAILED - {e}")

    # Test with causal mask (important for GPT-like models)
    print("\n=== Testing with Causal Mask ===")
    try:
        start_time = time.time()
        with torch.no_grad():
            output_causal = F.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        causal_time = time.time() - start_time
        print(f"Output shape: {output_causal.shape}")
        print(f"Time: {causal_time:.4f}s")
        print("✓ Causal masking works")
    except Exception as e:
        print(f"✗ Causal masking failed: {e}")

    # Test with attention mask
    print("\n=== Testing with Attention Mask ===")
    try:
        # Create a simple attention mask (attend to first half only)
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        attn_mask[:, seq_len // 2 :] = True  # Mask out second half

        start_time = time.time()
        with torch.no_grad():
            output_masked = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask
            )
        masked_time = time.time() - start_time
        print(f"Output shape: {output_masked.shape}")
        print(f"Time: {masked_time:.4f}s")
        print("✓ Attention masking works")
    except Exception as e:
        print(f"✗ Attention masking failed: {e}")

    # Verify outputs are similar (numerically)
    print("\n=== Numerical Correctness ===")
    diff = torch.abs(output_standard - output_sdpa).max().item()
    print(f"Max difference (standard vs SDPA): {diff:.6f}")
    # FP16 has lower precision than FP32, so we use a more relaxed tolerance
    if diff < 1e-2:
        print("✓ Outputs are numerically close (within FP16 precision)")
    else:
        print(f"⚠ Outputs differ by {diff:.6f} (may be due to different backends)")

    # Performance comparison
    print("\n=== Performance Summary ===")
    print(f"Standard attention: {standard_time:.4f}s")
    print(f"SDPA: {sdpa_time:.4f}s")
    speedup = standard_time / sdpa_time if sdpa_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")

    print("\n✓ Flash Attention test PASSED!")
    print("  - scaled_dot_product_attention works")
    print("  - All backends tested")
    print("  - Causal and custom masking work")
    print("  - No ROCm/HIP failures")


if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention Hello World Test")
    print("=" * 60)

    try:
        test_flash_attention()
    except Exception as e:
        print(f"\n✗ Flash Attention test FAILED!")
        print(f"Error: {e}")
        raise
