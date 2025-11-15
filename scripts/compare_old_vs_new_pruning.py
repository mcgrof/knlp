#!/usr/bin/env python3
"""
Compare old double-pass vs new single-pass update_adamprune_masks.

Shows actual speedup from the optimization by implementing both
versions and benchmarking them side by side.
"""
import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.optimizers import create_optimizer, _kth_threshold_sampling


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


@torch.no_grad()
def update_adamprune_masks_OLD(optimizer, adamprune_state, train_loader, step):
    """OLD implementation with double-pass (reads Adam states twice)."""
    if adamprune_state is None or not adamprune_state.get("pruning_enabled", False):
        return

    if train_loader is None:
        adamprune_state["step_count"] = step
    else:
        adamprune_state["step_count"] += 1

    step_count = adamprune_state["step_count"]

    # Skip if not a pruning step
    if (
        step_count <= adamprune_state["warmup_steps"]
        or step_count % adamprune_state["pruning_frequency"] != 0
    ):
        if step_count > adamprune_state["warmup_steps"]:
            for module in adamprune_state["masks"].keys():
                module.weight.data.mul_(
                    adamprune_state["masks"][module].to(module.weight.dtype)
                )
        for module, mask in adamprune_state["masks"].items():
            state = optimizer.state.get(module.weight, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
        return

    # Calculate sparsity
    if train_loader is not None:
        ramp_end_epoch = adamprune_state.get("ramp_end_epoch", 75)
        ramp_end_step = len(train_loader) * ramp_end_epoch
    else:
        ramp_end_step = adamprune_state.get("ramp_end_step", 10000)

    warmup_steps = adamprune_state["warmup_steps"]
    progress = min(
        1.0, (step_count - warmup_steps) / max(1, (ramp_end_step - warmup_steps))
    )
    progress = progress**3
    current_sparsity = adamprune_state["target_sparsity"] * progress

    variant = adamprune_state.get("variant", "bitter0")

    # === PASS 1: Build all_scores (reads Adam states) ===
    all_scores = []
    for module in adamprune_state["masks"].keys():
        state = optimizer.state.get(module.weight, {})
        w = module.weight.data

        # Compute importance (bitter7 as example)
        if variant == "bitter7" and "exp_avg_sq" in state:
            v = state["exp_avg_sq"]
            variance_importance = (torch.abs(v) + 1e-8) ** 0.25
            importance = torch.abs(w) * variance_importance
        else:
            importance = torch.abs(w)

        all_scores.append(importance.flatten())

    if not all_scores:
        return

    # Concatenate (giant tensor allocation)
    all_scores = torch.cat(all_scores)
    k = int(current_sparsity * all_scores.numel())
    if k <= 0:
        threshold = all_scores.min() - 1
    else:
        threshold = _kth_threshold_sampling(all_scores, k)

    # === PASS 2: Recompute importance and apply masks (reads Adam states AGAIN) ===
    for module in adamprune_state["masks"].keys():
        state = optimizer.state.get(module.weight, {})
        w = module.weight.data

        # Recompute same importance (redundant memory access!)
        if variant == "bitter7" and "exp_avg_sq" in state:
            v = state["exp_avg_sq"]
            variance_importance = (torch.abs(v) + 1e-8) ** 0.25
            importance = torch.abs(w) * variance_importance
        else:
            importance = torch.abs(w)

        new_mask = importance > threshold
        adamprune_state["masks"][module].data = new_mask.to(torch.bool)
        module.weight.data.mul_(
            adamprune_state["masks"][module].to(module.weight.dtype)
        )

    # Mask optimizer states
    for module, mask in adamprune_state["masks"].items():
        state = optimizer.state.get(module.weight, {})
        if "exp_avg" in state:
            state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))


@torch.no_grad()
def update_adamprune_masks_NEW(optimizer, adamprune_state, train_loader, step):
    """NEW implementation with single-pass (caches importance, uses sampling)."""
    if adamprune_state is None or not adamprune_state.get("pruning_enabled", False):
        return

    if train_loader is None:
        adamprune_state["step_count"] = step
    else:
        adamprune_state["step_count"] += 1

    step_count = adamprune_state["step_count"]

    # Skip if not a pruning step
    if (
        step_count <= adamprune_state["warmup_steps"]
        or step_count % adamprune_state["pruning_frequency"] != 0
    ):
        if step_count > adamprune_state["warmup_steps"]:
            for module in adamprune_state["masks"].keys():
                module.weight.data.mul_(
                    adamprune_state["masks"][module].to(module.weight.dtype)
                )
        for module, mask in adamprune_state["masks"].items():
            state = optimizer.state.get(module.weight, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))
        return

    # Calculate sparsity
    if train_loader is not None:
        ramp_end_epoch = adamprune_state.get("ramp_end_epoch", 75)
        ramp_end_step = len(train_loader) * ramp_end_epoch
    else:
        ramp_end_step = adamprune_state.get("ramp_end_step", 10000)

    warmup_steps = adamprune_state["warmup_steps"]
    progress = min(
        1.0, (step_count - warmup_steps) / max(1, (ramp_end_step - warmup_steps))
    )
    progress = progress**3
    current_sparsity = adamprune_state["target_sparsity"] * progress

    variant = adamprune_state.get("variant", "bitter0")

    # === SINGLE PASS: Compute importance once, cache, sample ===
    importance_cache = {}
    sampled_scores = []
    total_params = 0
    sample_frac = float(adamprune_state.get("sample_frac", 0.01))

    for module in adamprune_state["masks"].keys():
        state = optimizer.state.get(module.weight, {})
        w = module.weight.data

        # Compute importance ONCE
        if variant == "bitter7" and "exp_avg_sq" in state:
            v = state["exp_avg_sq"]
            variance_importance = (torch.abs(v) + 1e-8) ** 0.25
            importance = torch.abs(w) * variance_importance
        else:
            importance = torch.abs(w)

        # Cache it
        importance_cache[module] = importance
        numel = importance.numel()
        total_params += numel

        # Sample (small tensor, not full concat)
        if sample_frac > 0.0:
            sample_size = max(1, int(numel * sample_frac))
            idx = torch.randint(0, numel, (sample_size,), device=importance.device)
            sampled_scores.append(importance.view(-1)[idx])

    if not sampled_scores:
        return

    # Estimate threshold from samples
    samples = torch.cat(sampled_scores)
    global_k = int(current_sparsity * total_params)
    if global_k <= 0:
        threshold = samples.min() - 1
    else:
        k_sample = max(1, int(global_k * (samples.numel() / max(total_params, 1))))
        k_sample = min(k_sample, samples.numel())
        threshold = torch.kthvalue(samples, k_sample).values

    # Apply masks using cached importance (NO recomputation)
    for module, importance in importance_cache.items():
        new_mask = importance > threshold
        adamprune_state["masks"][module].data = new_mask.to(torch.bool)
        module.weight.data.mul_(
            adamprune_state["masks"][module].to(module.weight.dtype)
        )

    # Mask optimizer states
    for module, mask in adamprune_state["masks"].items():
        state = optimizer.state.get(module.weight, {})
        if "exp_avg" in state:
            state["exp_avg"].mul_(mask.to(state["exp_avg"].dtype))
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask.to(state["exp_avg_sq"].dtype))


def setup_optimizer(model, variant="bitter7", sparsity=0.5, device="cpu"):
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

    optimizer.adamprune_state = adamprune_state

    # Populate Adam states
    dim = model.layers[0].weight.shape[0]
    for _ in range(15):
        x = torch.randn(4, dim, device=device)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return optimizer


def benchmark_version(
    model, optimizer, update_fn, num_iterations=50, warmup=5, device="cpu", label=""
):
    """Benchmark a specific version of update_adamprune_masks."""
    adamprune_state = optimizer.adamprune_state
    is_cuda = isinstance(device, torch.device) and device.type == "cuda"

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for i in range(warmup):
        step = adamprune_state["warmup_steps"] + 1 + i
        update_fn(optimizer, adamprune_state, None, step)

    # Benchmark
    if is_cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for i in range(num_iterations):
        step = adamprune_state["warmup_steps"] + 1 + warmup + i
        update_fn(optimizer, adamprune_state, None, step)

    if is_cuda:
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations

    print(f"\n[{label}]")
    print(f"  Avg pruning update: {avg_time * 1000:.3f} ms")

    if is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        print(f"  Peak CUDA memory:   {peak_bytes / (1024 ** 2):.2f} MB")

    return avg_time


def main():
    parser = argparse.ArgumentParser(
        description="Compare old double-pass vs new single-pass pruning"
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
        "--device", type=str, default="cpu", help="Device to use (default: cpu)"
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

    # Test OLD version
    print("\n" + "=" * 60)
    print("Testing OLD implementation (double-pass)")
    print("=" * 60)
    optimizer_old = setup_optimizer(
        model, variant=args.variant, sparsity=args.sparsity, device=device
    )
    time_old = benchmark_version(
        model,
        optimizer_old,
        update_adamprune_masks_OLD,
        num_iterations=args.iterations,
        warmup=5,
        device=device,
        label="OLD: Double-pass (reads Adam states twice)",
    )

    # Test NEW version
    print("\n" + "=" * 60)
    print("Testing NEW implementation (single-pass)")
    print("=" * 60)
    optimizer_new = setup_optimizer(
        model, variant=args.variant, sparsity=args.sparsity, device=device
    )
    time_new = benchmark_version(
        model,
        optimizer_new,
        update_adamprune_masks_NEW,
        num_iterations=args.iterations,
        warmup=5,
        device=device,
        label="NEW: Single-pass (cached importance + sampling)",
    )

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nOLD (double-pass):  {time_old * 1000:.3f} ms")
    print(f"NEW (single-pass):  {time_new * 1000:.3f} ms")
    speedup = time_old / time_new
    improvement = ((time_old - time_new) / time_old) * 100
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}% faster")

    print("\n" + "=" * 60)
    print("KEY OPTIMIZATIONS")
    print("=" * 60)
    print("1. Single-pass importance computation (cached)")
    print("   - OLD: Read exp_avg_sq twice (pass 1 + pass 2)")
    print("   - NEW: Read exp_avg_sq once (cached)")
    print("   - Result: ~2x reduction in Adam state memory reads")
    print("\n2. Sampled threshold estimation")
    print("   - OLD: torch.cat(all_scores) - giant tensor")
    print("   - NEW: torch.cat(samples) - 1% sample")
    print(f"   - Result: ~99% reduction in concat size")


if __name__ == "__main__":
    main()
