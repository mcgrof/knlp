#!/usr/bin/env python3
import argparse
import time

import torch
import torch.nn as nn


# -------------------------------
# Toy model with many Linear layers
# -------------------------------

class ToyModel(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            # No bias, just big weight matrices
            layers.append(nn.Linear(dim, dim, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------------
# Magnitude pruning (baseline)
# -------------------------------

class MagnitudePrunerBaseline:
    def __init__(self, prunable_layers, sparsity: float):
        """
        prunable_layers: list of (name, module) pairs
        sparsity: fraction of weights to prune (0.0 - 1.0)
        """
        self.prunable_layers = prunable_layers
        self.sparsity = sparsity

    @torch.no_grad()
    def get_threshold(self) -> torch.Tensor:
        """
        Compute global magnitude threshold via kthvalue over |w|.
        Allocates a fresh concatenated vector each call (baseline behavior).
        """
        mags = []
        for name, module in self.prunable_layers:
            w = module.weight.data
            mags.append(w.abs().flatten())
        all_mags = torch.cat(mags)

        k = int(self.sparsity * all_mags.numel())
        if k <= 0:
            # No pruning
            return torch.tensor(0.0, device=all_mags.device, dtype=all_mags.dtype)
        if k >= all_mags.numel():
            # Extreme sparsity; everything pruned
            return all_mags.max()

        # kthvalue is 1-indexed in PyTorch
        kth = torch.kthvalue(all_mags, k).values
        return kth


# -------------------------------
# Bitter7-style pruner (optimized)
# -------------------------------

class Bitter7PrunerOptimized:
    def __init__(
        self,
        prunable_layers,
        optimizer_state,
        sparsity: float,
        imp_dtype: torch.dtype = torch.float16,
        eps: float = 1e-8,
    ):
        """
        prunable_layers: list of (name, module) pairs
        optimizer_state: dict[param] -> {"exp_avg_sq": Tensor}
        sparsity: fraction of weights to prune (0.0 - 1.0)
        imp_dtype: dtype for storing importance (fp16 or bf16)
        """
        self.prunable_layers = prunable_layers
        self.optimizer_state = optimizer_state
        self.sparsity = sparsity
        self.imp_dtype = imp_dtype
        self.eps = eps

        # Infer device / dtype from first layer
        first_module = prunable_layers[0][1]
        self.device = first_module.weight.device
        self.param_dtype = first_module.weight.dtype

        # Precompute shapes
        self.total_params = sum(m.weight.numel() for _, m in self.prunable_layers)
        self.max_layer_elems = max(m.weight.numel() for _, m in self.prunable_layers)

        # Allocate reusable buffers: one scratch + one global importance
        self.scratch = torch.empty(
            self.max_layer_elems, device=self.device, dtype=self.param_dtype
        )
        self.all_importances = torch.empty(
            self.total_params, device=self.device, dtype=self.imp_dtype
        )

    @torch.no_grad()
    def _compute_importances(self) -> torch.Tensor:
        """
        Fill self.all_importances with:
            importance = |w| * (|v| + eps) ** 0.25
        using a single reusable scratch buffer and fp16/bf16 storage.
        """
        offset = 0
        for name, module in self.prunable_layers:
            w = module.weight.data
            state = self.optimizer_state[module.weight]
            v = state["exp_avg_sq"]  # same shape as w

            n = w.numel()
            buf = self.scratch[:n].view_as(w)

            # buf = (|v| + eps) ** 0.25, done in-place
            buf.copy_(v)
            buf.abs_()
            buf.add_(self.eps)
            buf.sqrt_()
            buf.sqrt_()

            # buf = importance = |w| * buf
            buf.mul_(w.abs())

            # Store in fp16 / bf16 in the global vector
            self.all_importances[offset : offset + n] = buf.flatten().to(self.imp_dtype)
            offset += n

        return self.all_importances

    @torch.no_grad()
    def get_threshold(self) -> torch.Tensor:
        """
        Compute global kthvalue threshold over importance scores.
        Importance is stored in fp16/bf16 but kthvalue is done in fp32
        for numerical stability.
        """
        imps = self._compute_importances()
        numel = imps.numel()
        k = int(self.sparsity * numel)
        if k <= 0:
            return torch.tensor(0.0, device=imps.device, dtype=torch.float32)
        if k >= numel:
            return imps.float().max()

        imps32 = imps.float()
        kth = torch.kthvalue(imps32, k).values
        return kth


# -------------------------------
# Utility: build fake Adam exp_avg_sq
# -------------------------------

def build_fake_adam_state(model: nn.Module, device: torch.device):
    """
    Build a minimal Adam-like state dict:
        state[param]["exp_avg_sq"] = random Tensor same shape as param
    """
    state = {}
    for p in model.parameters():
        if p.requires_grad:
            state[p] = {
                "exp_avg_sq": torch.rand_like(p.data, device=device),
            }
    return state


# -------------------------------
# Benchmark helpers
# -------------------------------

def benchmark_pruner(pruner, num_calls: int, device: torch.device, label: str):
    """
    Measure average time per get_threshold() call and peak CUDA memory.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup (to avoid one-time kernel compilation / caching noise)
    torch.cuda.synchronize(device)
    _ = pruner.get_threshold()
    torch.cuda.synchronize(device)

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = pruner.get_threshold()
    torch.cuda.synchronize(device)
    end = time.perf_counter()

    avg_time = (end - start) / num_calls
    peak_bytes = torch.cuda.max_memory_allocated(device)

    print(f"\n[{label}]")
    print(f"  Avg prune-op time: {avg_time * 1000:.3f} ms")
    print(f"  Peak CUDA memory:  {peak_bytes / (1024 ** 2):.2f} MB")

    return avg_time, peak_bytes


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark magnitude pruning vs Bitter7-style pruning "
                    "with optimized memory usage."
    )
    parser.add_argument("--dim", type=int, default=2048,
                        help="Dimension of each Linear layer (default: 2048)")
    parser.add_argument("--layers", type=int, default=24,
                        help="Number of Linear layers (default: 24)")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Fraction of weights to prune (default: 0.5)")
    parser.add_argument("--num-calls", type=int, default=20,
                        help="Number of pruning calls to average over (default: 20)")
    parser.add_argument("--device-index", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument(
        "--imp-dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help="dtype for importance vector (default: bf16)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)

    print(f"Using device: {device}")
    print(f"Model: dim={args.dim}, layers={args.layers}")
    print(f"Sparsity: {args.sparsity:.3f}")
    print(f"Importance dtype: {args.imp_dtype}")
    print(f"Prune calls per benchmark: {args.num_calls}")

    # Build model
    model = ToyModel(dim=args.dim, num_layers=args.layers).to(device)

    # Build fake Adam state
    adam_state = build_fake_adam_state(model, device=device)

    # Collect prunable layers (all Linear layers)
    prunable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    total_params = sum(m.weight.numel() for _, m in prunable_layers)
    print(f"Total prunable params: {total_params / 1e6:.2f} M")

    # Instantiate pruners
    mag_pruner = MagnitudePrunerBaseline(
        prunable_layers=prunable_layers,
        sparsity=args.sparsity,
    )

    imp_dtype = torch.float16 if args.imp_dtype == "fp16" else torch.bfloat16

    bitter_pruner = Bitter7PrunerOptimized(
        prunable_layers=prunable_layers,
        optimizer_state=adam_state,
        sparsity=args.sparsity,
        imp_dtype=imp_dtype,
    )

    # Benchmark both
    benchmark_pruner(
        pruner=mag_pruner,
        num_calls=args.num_calls,
        device=device,
        label="MagnitudePruning (baseline)",
    )

    benchmark_pruner(
        pruner=bitter_pruner,
        num_calls=args.num_calls,
        device=device,
        label=f"Bitter7 Optimized (scratch + {args.imp_dtype})",
    )


if __name__ == "__main__":
    main()
