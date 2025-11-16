#!/usr/bin/env python3
"""
Benchmark importance calculation overhead for different bitter variants.

This tests the REAL bottleneck hypothesis: the importance calculation
(|w| × (|v| + ε)^0.25) is expensive and happens every 50-100 iterations,
creating a periodic spike in iteration time.

The gradient masking is cheap, but the importance recalculation requires:
1. Reading all weights and Adam second moments (exp_avg_sq)
2. Computing importance scores
3. Finding global kthvalue threshold
4. Updating masks based on threshold

For bitter7 (baseline): (|v| + ε) ** 0.25 using two sqrt()
For bitter8 (FP16): same but with FP16 conversion overhead
For bitter9 (FP16+compile): same but with torch.compile kernel fusion

We test whether the importance calculation itself is slow enough to
cause the observed 20-30% performance degradation.
"""

import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn


class GPT2Weights:
    """Simulated GPT-2 weights for benchmarking."""

    def __init__(self, n_layers: int = 12, n_embd: int = 768, device: str = "cpu"):
        self.device = device
        self.layers = []

        # Create weight tensors matching GPT-2 architecture
        for i in range(n_layers):
            layer_weights = {
                "attn_qkv": torch.randn(n_embd, 3 * n_embd, device=device),
                "attn_proj": torch.randn(n_embd, n_embd, device=device),
                "mlp_fc": torch.randn(n_embd, 4 * n_embd, device=device),
                "mlp_proj": torch.randn(4 * n_embd, n_embd, device=device),
            }
            self.layers.append(layer_weights)

        # LM head
        self.lm_head = torch.randn(50257, n_embd, device=device)

    def get_all_weights(self) -> List[torch.Tensor]:
        """Get list of all weight tensors."""
        weights = []
        for layer in self.layers:
            weights.extend(layer.values())
        weights.append(self.lm_head)
        return weights

    def create_fake_adam_state(self) -> dict:
        """Create fake Adam state with exp_avg_sq."""
        state = {}
        for w in self.get_all_weights():
            state[id(w)] = {"exp_avg_sq": torch.rand_like(w)}
        return state


def compute_bitter7_importance(w: torch.Tensor, v: torch.Tensor, eps: float = 1e-5):
    """Bitter7: |w| × (|v| + ε)^0.25 using double sqrt."""
    v_abs = torch.abs(v) + eps
    fourth_root = torch.sqrt(torch.sqrt(v_abs))
    importance = torch.abs(w) * fourth_root
    return importance


def compute_bitter8_importance(w: torch.Tensor, v: torch.Tensor, eps: float = 1e-5):
    """Bitter8: Same as bitter7 but with FP16 conversion."""
    w_fp16 = w.to(torch.float16)
    v_fp16 = v.to(torch.float16)
    v_abs = torch.abs(v_fp16) + eps
    fourth_root = torch.sqrt(torch.sqrt(v_abs))
    importance = torch.abs(w_fp16) * fourth_root
    return importance.float()


@torch.compile(mode="default")
def compute_bitter9_importance(w: torch.Tensor, v: torch.Tensor, eps: float = 1e-5):
    """Bitter9: Same as bitter8 but with torch.compile."""
    w_fp16 = w.to(torch.float16)
    v_fp16 = v.to(torch.float16)
    v_abs = torch.abs(v_fp16) + eps
    fourth_root = torch.sqrt(torch.sqrt(v_abs))
    importance = torch.abs(w_fp16) * fourth_root
    return importance.float()


def compute_bitter8_rsqrt_importance(
    w: torch.Tensor, v: torch.Tensor, eps: float = 1e-5
):
    """Bitter8 rsqrt: Using rsqrt instead of sqrt for 4th root."""
    w_fp16 = w.to(torch.float16)
    v_fp16 = v.to(torch.float16)
    v_abs = torch.abs(v_fp16) + eps
    # rsqrt(rsqrt(x)) = x^0.25
    fourth_root = torch.rsqrt(torch.rsqrt(v_abs))
    importance = torch.abs(w_fp16) * fourth_root
    return importance.float()


@torch.compile(mode="default")
def compute_bitter9_rsqrt_importance(
    w: torch.Tensor, v: torch.Tensor, eps: float = 1e-5
):
    """Bitter9 rsqrt: Using rsqrt + torch.compile."""
    w_fp16 = w.to(torch.float16)
    v_fp16 = v.to(torch.float16)
    v_abs = torch.abs(v_fp16) + eps
    fourth_root = torch.rsqrt(torch.rsqrt(v_abs))
    importance = torch.abs(w_fp16) * fourth_root
    return importance.float()


def benchmark_importance_calculation(
    weights: List[torch.Tensor],
    adam_state: dict,
    importance_fn,
    sparsity: float,
    num_calls: int,
    device: str,
    label: str,
):
    """Benchmark importance calculation + threshold finding."""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    if device == "cuda":
        torch.cuda.synchronize()

    all_importances = []
    for w in weights:
        v = adam_state[id(w)]["exp_avg_sq"]
        imp = importance_fn(w, v)
        all_importances.append(imp.flatten())

    all_imp = torch.cat(all_importances)
    k = int(sparsity * all_imp.numel())
    _ = torch.kthvalue(all_imp, k).values

    if device == "cuda":
        torch.cuda.synchronize()

    # Actual benchmark
    start = time.perf_counter()

    for _ in range(num_calls):
        all_importances = []
        for w in weights:
            v = adam_state[id(w)]["exp_avg_sq"]
            imp = importance_fn(w, v)
            all_importances.append(imp.flatten())

        all_imp = torch.cat(all_importances)
        k = int(sparsity * all_imp.numel())
        threshold = torch.kthvalue(all_imp, k).values

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_time = ((end - start) / num_calls) * 1000  # ms

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        peak_mem = 0.0

    print(f"\n{label}:")
    print(f"  Avg time: {avg_time:.2f} ms")
    if peak_mem > 0:
        print(f"  Peak mem: {peak_mem:.2f} MB")

    return avg_time, peak_mem


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark importance calculation overhead"
    )
    parser.add_argument(
        "--n-layer", type=int, default=12, help="Number of layers (default: 12)"
    )
    parser.add_argument(
        "--n-embd", type=int, default=768, help="Embedding dim (default: 768)"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Sparsity (default: 0.5)"
    )
    parser.add_argument(
        "--num-calls", type=int, default=20, help="Number of calls (default: 20)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU\n")
        device = "cpu"

    print("=" * 80)
    print("Importance Calculation Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Layers: {args.n_layer}")
    print(f"  Embedding: {args.n_embd}")
    print(f"  Sparsity: {args.sparsity:.1%}")
    print(f"  Calls: {args.num_calls}")
    print(f"  Device: {device}")

    # Create weights
    print(f"\nCreating weights...")
    gpt2_weights = GPT2Weights(n_layers=args.n_layer, n_embd=args.n_embd, device=device)
    weights = gpt2_weights.get_all_weights()
    adam_state = gpt2_weights.create_fake_adam_state()

    total_params = sum(w.numel() for w in weights)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Weight tensors: {len(weights)}")

    print(f"\n{'=' * 80}")
    print("IMPORTANCE CALCULATION BENCHMARKS")
    print("=" * 80)

    # Benchmark bitter7
    t7, _ = benchmark_importance_calculation(
        weights,
        adam_state,
        compute_bitter7_importance,
        args.sparsity,
        args.num_calls,
        device,
        "Bitter7 (FP32, double sqrt)",
    )

    # Benchmark bitter8
    t8, _ = benchmark_importance_calculation(
        weights,
        adam_state,
        compute_bitter8_importance,
        args.sparsity,
        args.num_calls,
        device,
        "Bitter8 (FP16, double sqrt)",
    )

    # Benchmark bitter8 rsqrt
    t8r, _ = benchmark_importance_calculation(
        weights,
        adam_state,
        compute_bitter8_rsqrt_importance,
        args.sparsity,
        args.num_calls,
        device,
        "Bitter8-rsqrt (FP16, double rsqrt)",
    )

    # Benchmark bitter9 (with compile warmup)
    if device == "cuda":
        print("\nWarming up torch.compile (bitter9)...")
        for w in weights[:3]:  # Warmup on a few tensors
            v = adam_state[id(w)]["exp_avg_sq"]
            _ = compute_bitter9_importance(w, v)
        torch.cuda.synchronize()

    t9, _ = benchmark_importance_calculation(
        weights,
        adam_state,
        compute_bitter9_importance,
        args.sparsity,
        args.num_calls,
        device,
        "Bitter9 (FP16, double sqrt, compiled)",
    )

    # Benchmark bitter9 rsqrt
    if device == "cuda":
        print("\nWarming up torch.compile (bitter9-rsqrt)...")
        for w in weights[:3]:
            v = adam_state[id(w)]["exp_avg_sq"]
            _ = compute_bitter9_rsqrt_importance(w, v)
        torch.cuda.synchronize()

    t9r, _ = benchmark_importance_calculation(
        weights,
        adam_state,
        compute_bitter9_rsqrt_importance,
        args.sparsity,
        args.num_calls,
        device,
        "Bitter9-rsqrt (FP16, double rsqrt, compiled)",
    )

    # Analysis
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print("=" * 80)

    print(f"\nSpeedup vs Bitter7:")
    print(f"  Bitter8 (FP16):        {(t7/t8 - 1)*100:+.1f}%")
    print(f"  Bitter8-rsqrt:         {(t7/t8r - 1)*100:+.1f}%")
    print(f"  Bitter9 (compiled):    {(t7/t9 - 1)*100:+.1f}%")
    print(f"  Bitter9-rsqrt:         {(t7/t9r - 1)*100:+.1f}%")

    print(f"\n{'=' * 80}")
    print("HYPOTHESIS TEST")
    print("=" * 80)

    # Assuming importance calc happens every 50 iters and baseline iter is 100ms
    baseline_iter_ms = 100.0
    importance_interval = 50

    avg_overhead_bitter7 = t7 / importance_interval
    avg_overhead_bitter9 = t9r / importance_interval

    print(f"\nAssuming:")
    print(f"  - Baseline iteration time: {baseline_iter_ms:.1f} ms")
    print(f"  - Importance recalc interval: {importance_interval} iterations")
    print(f"\nAmortized overhead per iteration:")
    print(
        f"  Bitter7: +{avg_overhead_bitter7:.2f} ms ({avg_overhead_bitter7/baseline_iter_ms*100:.1f}%)"
    )
    print(
        f"  Bitter9-rsqrt: +{avg_overhead_bitter9:.2f} ms ({avg_overhead_bitter9/baseline_iter_ms*100:.1f}%)"
    )

    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print("=" * 80)

    if avg_overhead_bitter7 / baseline_iter_ms > 0.15:  # >15% overhead
        print(
            f"\n✓ Importance calculation adds {avg_overhead_bitter7/baseline_iter_ms*100:.1f}% overhead"
        )
        print(f"  This could partially explain the performance gap")
    else:
        print(
            f"\n✗ Importance calculation only adds {avg_overhead_bitter7/baseline_iter_ms*100:.1f}% overhead"
        )
        print(f"  This is NOT the main bottleneck")

    print()


if __name__ == "__main__":
    main()
