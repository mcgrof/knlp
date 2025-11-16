#!/usr/bin/env python3
"""
Benchmark gradient masking overhead in AdamWPrune vs baseline.

Hypothesis: The per-iteration gradient masking (48+ kernel launches for GPT-2)
is the real bottleneck, not the importance calculation which happens every
50-100 iterations.

Tests:
1. Baseline: Forward + Backward + Optimizer step + Weight masking
2. AdamWPrune: Forward + Backward + GRADIENT masking + Optimizer step + Weight masking

The difference isolates the gradient masking overhead.
"""

import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn


class GPT2LikeModel(nn.Module):
    """Simplified GPT-2 architecture for benchmarking."""

    def __init__(self, n_layer: int = 12, n_embd: int = 768, block_size: int = 1024):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd

        # Build transformer blocks (simplified)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, 50257, bias=False)  # vocab size

    def forward(self, x):
        """x: (B, T, n_embd) - already embedded"""
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""

    def __init__(self, n_embd: int):
        super().__init__()
        # Attention (simplified - just the linear layers)
        self.attn_qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.attn_proj = nn.Linear(n_embd, n_embd, bias=False)

        # MLP
        self.mlp_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Attention (simplified - skip actual attention computation)
        x = x + self.attn_proj(self.attn_qkv(self.ln_1(x)))
        # MLP
        x = x + self.mlp_proj(torch.relu(self.mlp_fc(self.ln_2(x))))
        return x


def get_linear_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all Linear modules from model."""
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules.append((name, module))
    return modules


def create_random_masks(
    modules: List[Tuple[str, nn.Module]], sparsity: float, device: torch.device
) -> dict:
    """Create random binary masks for pruning."""
    masks = {}
    for name, module in modules:
        mask = torch.rand_like(module.weight) > sparsity
        masks[module] = mask.to(device)
    return masks


def apply_gradient_masking(modules: List[Tuple[str, nn.Module]], masks: dict):
    """Apply masks to gradients (AdamWPrune style)."""
    for name, module in modules:
        if module.weight.grad is not None:
            mask = masks[module]
            module.weight.grad.data.mul_(mask.to(module.weight.grad.dtype))


def apply_weight_masking(modules: List[Tuple[str, nn.Module]], masks: dict):
    """Apply masks to weights (both variants)."""
    for name, module in modules:
        mask = masks[module]
        module.weight.data.mul_(mask.to(module.weight.dtype))


def benchmark_training_iteration(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    modules: List[Tuple[str, nn.Module]],
    masks: dict,
    batch_size: int,
    seq_len: int,
    n_embd: int,
    device: torch.device,
    use_gradient_masking: bool,
    num_iters: int,
) -> Tuple[float, float]:
    """
    Benchmark one training iteration.

    Returns:
        (avg_time_ms, peak_memory_mb)
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for _ in range(5):
        # Forward
        x = torch.randn(batch_size, seq_len, n_embd, device=device, requires_grad=False)
        target = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1)
        )

        # Backward
        loss.backward()

        # Gradient masking (if enabled)
        if use_gradient_masking:
            apply_gradient_masking(modules, masks)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Weight masking (always)
        apply_weight_masking(modules, masks)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Actual benchmark
    start = time.perf_counter()

    for _ in range(num_iters):
        # Forward
        x = torch.randn(batch_size, seq_len, n_embd, device=device, requires_grad=False)
        target = torch.randint(0, 50257, (batch_size, seq_len), device=device)

        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1)
        )

        # Backward
        loss.backward()

        # Gradient masking (if enabled)
        if use_gradient_masking:
            apply_gradient_masking(modules, masks)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Weight masking (always)
        apply_weight_masking(modules, masks)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    avg_time = ((end - start) / num_iters) * 1000  # ms
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    else:
        peak_memory = 0.0  # CPU memory tracking not available

    return avg_time, peak_memory


def main():
    parser = argparse.ArgumentParser(description="Benchmark gradient masking overhead")
    parser.add_argument(
        "--n-layer",
        type=int,
        default=12,
        help="Number of transformer layers (default: 12)",
    )
    parser.add_argument(
        "--n-embd", type=int, default=768, help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Sequence length (default: 256)"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="Pruning sparsity (default: 0.5)"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=50,
        help="Number of iterations to benchmark (default: 50)",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index (default: 0)"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        print("Results will not reflect actual GPU performance!\n")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)

    print("=" * 80)
    print("Gradient Masking Overhead Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.n_layer} layers, {args.n_embd} dim")
    print(f"  Batch: {args.batch_size} × {args.seq_len} tokens")
    print(f"  Sparsity: {args.sparsity:.1%}")
    print(f"  Iterations: {args.num_iters}")
    print(f"  Device: {device}")

    # Create model
    print(f"\nBuilding model...")
    model = GPT2LikeModel(n_layer=args.n_layer, n_embd=args.n_embd).to(device)

    # Get linear modules
    modules = get_linear_modules(model)
    total_params = sum(m.weight.numel() for _, m in modules)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Linear modules: {len(modules)}")

    # Create masks
    print(f"\nCreating pruning masks...")
    masks = create_random_masks(modules, args.sparsity, device)

    # Create optimizer
    optimizer_baseline = torch.optim.AdamW(model.parameters(), lr=6e-4)
    optimizer_adamprune = torch.optim.AdamW(model.parameters(), lr=6e-4)

    # Benchmark baseline (no gradient masking)
    print(f"\n{'=' * 80}")
    print("BASELINE: Weight masking only (magnitude pruning)")
    print("=" * 80)

    time_baseline, mem_baseline = benchmark_training_iteration(
        model=model,
        optimizer=optimizer_baseline,
        modules=modules,
        masks=masks,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_embd=args.n_embd,
        device=device,
        use_gradient_masking=False,
        num_iters=args.num_iters,
    )

    print(f"\nResults:")
    print(f"  Avg iteration time: {time_baseline:.2f} ms")
    print(f"  Peak memory: {mem_baseline:.2f} MB")

    # Benchmark AdamWPrune (with gradient masking)
    print(f"\n{'=' * 80}")
    print("ADAMWPRUNE: Gradient masking + Weight masking")
    print("=" * 80)

    time_adamprune, mem_adamprune = benchmark_training_iteration(
        model=model,
        optimizer=optimizer_adamprune,
        modules=modules,
        masks=masks,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_embd=args.n_embd,
        device=device,
        use_gradient_masking=True,
        num_iters=args.num_iters,
    )

    print(f"\nResults:")
    print(f"  Avg iteration time: {time_adamprune:.2f} ms")
    print(f"  Peak memory: {mem_adamprune:.2f} MB")

    # Analysis
    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print("=" * 80)

    time_overhead = time_adamprune - time_baseline
    time_overhead_pct = (time_overhead / time_baseline) * 100

    mem_overhead = mem_adamprune - mem_baseline
    mem_overhead_pct = (mem_overhead / mem_baseline) * 100

    print(f"\nGradient masking overhead:")
    print(f"  Time: +{time_overhead:.2f} ms ({time_overhead_pct:+.1f}%)")
    print(f"  Memory: +{mem_overhead:.2f} MB ({mem_overhead_pct:+.1f}%)")
    print(f"\nPer-module overhead:")
    print(f"  Time per module: {time_overhead / len(modules):.3f} ms")
    print(f"  Modules masked: {len(modules)}")

    print(f"\n{'=' * 80}")
    print("HYPOTHESIS TEST")
    print("=" * 80)

    if time_overhead_pct > 10:
        print(f"\n✓ HYPOTHESIS CONFIRMED:")
        print(f"  Gradient masking adds {time_overhead_pct:.1f}% overhead")
        print(f"  This explains the slower training in bitter variants")
        print(f"\nRecommendation:")
        print(f"  - Consider gradient masking less frequently")
        print(f"  - Or fuse gradient masking with gradient clipping")
        print(f"  - Or eliminate gradient masking entirely")
    else:
        print(f"\n✗ HYPOTHESIS REJECTED:")
        print(f"  Gradient masking only adds {time_overhead_pct:.1f}% overhead")
        print(f"  The bottleneck must be elsewhere")

    print()


if __name__ == "__main__":
    main()
