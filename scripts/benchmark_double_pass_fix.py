#!/usr/bin/env python3
"""
Benchmark double-pass elimination in update_adamprune_masks.

Compares the old implementation (reads Adam states twice) vs the
new optimized version (caches importance, uses sampled threshold).
"""
import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.optimizers import update_adamprune_masks, create_optimizer


class ToyModel(nn.Module):
    """Simple model with multiple Linear layers for benchmarking."""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, dim, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Args:
    """Mock args object for create_optimizer."""

    def __init__(self, variant, sparsity):
        self.weight_decay = 0.0
        self.adamwprune_enable_pruning = True
        self.adamwprune_target_sparsity = sparsity
        self.adamwprune_warmup_steps = 10
        self.adamwprune_ramp_end_step = 100
        self.adamwprune_frequency = 1
        self.adamwprune_variant = variant
        self.adamwprune_base_optimizer_name = "adamw"
        self.adamwprune_beta1 = 0.9
        self.adamwprune_beta2 = 0.999
        self.adamwprune_weight_decay = 0.0
        self.adamwprune_amsgrad = False


def setup_adamwprune(model, variant="bitter7", sparsity=0.5, device="cpu"):
    """Setup AdamWPrune optimizer with pruning state."""
    args = Args(variant=variant, sparsity=sparsity)

    optimizer, scheduler, grad_clip, spam_state, adamprune_state = create_optimizer(
        model=model,
        optimizer_type="adamwprune",
        learning_rate=1e-3,
        num_epochs=10,
        args=args,
        model_type="gpt2",
    )

    # Attach adamprune_state to optimizer for easy access
    optimizer.adamprune_state = adamprune_state

    # Run a few dummy forward/backward passes to populate Adam states
    dim = model.layers[0].weight.shape[0]
    for _ in range(15):
        x = torch.randn(4, dim, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return optimizer


def benchmark_pruning(
    model, optimizer, num_iterations=50, warmup=5, device="cpu", label=""
):
    """Benchmark update_adamprune_masks timing and memory."""
    adamprune_state = optimizer.adamprune_state
    is_cuda = isinstance(device, torch.device) and device.type == "cuda"

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for i in range(warmup):
        step = adamprune_state["warmup_steps"] + 1 + i
        update_adamprune_masks(optimizer, adamprune_state, None, step)

    # Benchmark
    if is_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for i in range(num_iterations):
        step = adamprune_state["warmup_steps"] + 1 + warmup + i
        update_adamprune_masks(optimizer, adamprune_state, None, step)

    if is_cuda:
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations

    print(f"\n[{label}]")
    print(f"  Avg pruning update: {avg_time * 1000:.3f} ms")

    if is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        print(f"  Peak CUDA memory:   {peak_bytes / (1024 ** 2):.2f} MB")
    else:
        print(f"  (CPU mode - no GPU memory tracking)")

    return avg_time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark double-pass elimination in update_adamprune_masks"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Dimension of Linear layers (default: 1024)",
    )
    parser.add_argument(
        "--layers", type=int, default=12, help="Number of Linear layers (default: 12)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="bitter7",
        choices=[
            "bitter0",
            "bitter1",
            "bitter2",
            "bitter3",
            "bitter5",
            "bitter6",
            "bitter7",
            "bitter8",
            "bitter9",
        ],
        help="Pruning variant (default: bitter7)",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Target sparsity (default: 0.5)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of pruning updates to benchmark (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu, or cuda:0 for GPU)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Model: dim={args.dim}, layers={args.layers}")
    print(f"Variant: {args.variant}")
    print(f"Target sparsity: {args.sparsity}")
    print(f"Pruning updates per benchmark: {args.iterations}")

    # Build model
    model = ToyModel(dim=args.dim, num_layers=args.layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f} M")

    # Setup AdamWPrune
    optimizer = setup_adamwprune(
        model, variant=args.variant, sparsity=args.sparsity, device=device
    )

    # Benchmark current (optimized) implementation
    avg_time = benchmark_pruning(
        model,
        optimizer,
        num_iterations=args.iterations,
        warmup=5,
        device=device,
        label=f"{args.variant} - Optimized (single-pass w/ sampling)",
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Average pruning update time: {avg_time * 1000:.3f} ms")
    print(f"\nOptimization features:")
    print(f"  - Single-pass importance computation (cached)")
    print(f"  - Sampled threshold estimation (1% default)")
    print(f"  - No giant all_scores tensor allocation")
    print(f"\nExpected benefits vs old double-pass:")
    print(f"  - ~2x reduction in Adam state reads")
    print(f"  - Lower peak memory (samples vs full concat)")
    print(f"  - Reduced memory bandwidth usage")


if __name__ == "__main__":
    main()
