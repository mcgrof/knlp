#!/usr/bin/env python3
"""Test potential improvements to Bitter7 pruning."""

import time
import torch
import torch.nn as nn


def benchmark_kthvalue_dtype(size=402_653_184, k_frac=0.5, num_calls=10):
    """Test if kthvalue is faster in bf16 vs fp32."""
    print(f"\nTesting kthvalue performance ({size/1e6:.1f}M elements):")

    device = torch.device("cuda:0")
    k = int(size * k_frac)

    # Test bf16 kthvalue
    data_bf16 = torch.rand(size, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.kthvalue(data_bf16, k)
    torch.cuda.synchronize()
    bf16_time = (time.perf_counter() - start) / num_calls

    # Test fp32 kthvalue
    data_fp32 = data_bf16.float()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.kthvalue(data_fp32, k)
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / num_calls

    print(f"  bf16 kthvalue: {bf16_time*1000:.1f} ms")
    print(f"  fp32 kthvalue: {fp32_time*1000:.1f} ms")
    print(f"  Speedup: {fp32_time/bf16_time:.2f}x")

    del data_bf16, data_fp32
    torch.cuda.empty_cache()


def benchmark_topk_vs_kthvalue(size=402_653_184, k_frac=0.5, num_calls=10):
    """Test if topk is faster than kthvalue."""
    print(f"\nTesting topk vs kthvalue ({size/1e6:.1f}M elements):")

    device = torch.device("cuda:0")
    k = int(size * k_frac)

    data = torch.rand(size, device=device, dtype=torch.float32)

    # Test kthvalue
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.kthvalue(data, k)
    torch.cuda.synchronize()
    kthvalue_time = (time.perf_counter() - start) / num_calls

    # Test topk (smallest=True to get bottom k)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.topk(data, k, largest=False)
    torch.cuda.synchronize()
    topk_time = (time.perf_counter() - start) / num_calls

    print(f"  kthvalue: {kthvalue_time*1000:.1f} ms")
    print(f"  topk:     {topk_time*1000:.1f} ms")
    print(f"  Speedup: {kthvalue_time/topk_time:.2f}x")

    del data
    torch.cuda.empty_cache()


def benchmark_quantile(size=402_653_184, q=0.5, num_calls=10):
    """Test if quantile is faster for approximate selection."""
    print(f"\nTesting quantile (approximate) ({size/1e6:.1f}M elements):")

    device = torch.device("cuda:0")
    k = int(size * q)

    data = torch.rand(size, device=device, dtype=torch.float32)

    # Test kthvalue (exact)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.kthvalue(data, k)
    torch.cuda.synchronize()
    kthvalue_time = (time.perf_counter() - start) / num_calls

    # Test quantile (approximate)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.quantile(data, q)
    torch.cuda.synchronize()
    quantile_time = (time.perf_counter() - start) / num_calls

    print(f"  kthvalue (exact):  {kthvalue_time*1000:.1f} ms")
    print(f"  quantile (approx): {quantile_time*1000:.1f} ms")
    print(f"  Speedup: {kthvalue_time/quantile_time:.2f}x")

    del data
    torch.cuda.empty_cache()


def benchmark_avoid_conversion(size=402_653_184, num_calls=10):
    """Test if avoiding bf16↔fp32 conversion helps."""
    print(f"\nTesting dtype conversion overhead ({size/1e6:.1f}M elements):")

    device = torch.device("cuda:0")
    data_fp32 = torch.rand(size, device=device, dtype=torch.float32)

    # Current approach: fp32 → bf16 → fp32 → kthvalue
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        bf16 = data_fp32.to(torch.bfloat16)
        fp32 = bf16.float()
        _ = torch.kthvalue(fp32, size // 2)
    torch.cuda.synchronize()
    with_conversion = (time.perf_counter() - start) / num_calls

    # Direct: fp32 → kthvalue (no conversions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = torch.kthvalue(data_fp32, size // 2)
    torch.cuda.synchronize()
    no_conversion = (time.perf_counter() - start) / num_calls

    print(f"  With bf16 conversion: {with_conversion*1000:.1f} ms")
    print(f"  Direct fp32:          {no_conversion*1000:.1f} ms")
    print(f"  Overhead: {(with_conversion - no_conversion)*1000:.1f} ms")

    del data_fp32
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 60)
    print("Bitter7 Optimization Experiments")
    print("=" * 60)

    benchmark_kthvalue_dtype()
    benchmark_topk_vs_kthvalue()
    benchmark_quantile()
    benchmark_avoid_conversion()

    print("\n" + "=" * 60)
    print("Summary:")
    print("If bf16 kthvalue is faster: use it directly")
    print("If topk is faster: switch from kthvalue to topk")
    print("If quantile is faster: use for approximate pruning")
    print("If conversion overhead is large: keep in fp32")
    print("=" * 60)
