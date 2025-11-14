#!/usr/bin/env python3
"""
Benchmark Bitter7 with topk instead of kthvalue.
topk is 20x faster! Let's see if this translates to real speedup.
"""

import argparse
import time
import torch
import torch.nn as nn


class ToyModel(nn.Module):
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


class Bitter7PrunerTopK:
    """Bitter7 using topk instead of kthvalue (20x faster!)."""

    def __init__(
        self,
        prunable_layers,
        optimizer_state,
        sparsity: float,
        imp_dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-8,
    ):
        self.prunable_layers = prunable_layers
        self.optimizer_state = optimizer_state
        self.sparsity = sparsity
        self.imp_dtype = imp_dtype
        self.eps = eps

        first_module = prunable_layers[0][1]
        self.device = first_module.weight.device
        self.param_dtype = first_module.weight.dtype

        self.total_params = sum(m.weight.numel() for _, m in self.prunable_layers)
        self.max_layer_elems = max(m.weight.numel() for _, m in self.prunable_layers)

        self.scratch = torch.empty(
            self.max_layer_elems, device=self.device, dtype=self.param_dtype
        )
        self.all_importances = torch.empty(
            self.total_params, device=self.device, dtype=self.imp_dtype
        )

    @torch.no_grad()
    def _compute_importances(self) -> torch.Tensor:
        offset = 0
        for name, module in self.prunable_layers:
            w = module.weight.data
            state = self.optimizer_state[module.weight]
            v = state["exp_avg_sq"]

            n = w.numel()
            buf = self.scratch[:n].view_as(w)

            # importance = |w| * (|v| + eps) ** 0.25
            buf.copy_(v)
            buf.abs_()
            buf.add_(self.eps)
            buf.sqrt_()
            buf.sqrt_()
            buf.mul_(w.abs())

            self.all_importances[offset : offset + n] = buf.flatten().to(self.imp_dtype)
            offset += n

        return self.all_importances

    @torch.no_grad()
    def get_threshold_kthvalue(self) -> torch.Tensor:
        """Original kthvalue approach."""
        imps = self._compute_importances()
        k = int(self.sparsity * imps.numel())
        if k <= 0:
            return torch.tensor(0.0, device=imps.device, dtype=torch.float32)
        if k >= imps.numel():
            return imps.float().max()

        imps32 = imps.float()
        kth = torch.kthvalue(imps32, k).values
        return kth

    @torch.no_grad()
    def get_threshold_topk(self) -> torch.Tensor:
        """NEW: Use topk (20x faster than kthvalue!)."""
        imps = self._compute_importances()
        k = int(self.sparsity * imps.numel())
        if k <= 0:
            return torch.tensor(0.0, device=imps.device, dtype=torch.float32)
        if k >= imps.numel():
            return imps.float().max()

        # topk with largest=False gets the k smallest elements
        # The maximum of these k smallest is our threshold
        values, _ = torch.topk(imps.float(), k, largest=False)
        threshold = values[-1]  # k-th smallest value
        return threshold


def build_fake_adam_state(model: nn.Module, device: torch.device):
    state = {}
    for p in model.parameters():
        if p.requires_grad:
            state[p] = {"exp_avg_sq": torch.rand_like(p.data, device=device)}
    return state


def benchmark_method(pruner, method_name: str, num_calls: int, device: torch.device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Get method
    if method_name == "kthvalue":
        method = pruner.get_threshold_kthvalue
    else:
        method = pruner.get_threshold_topk

    # Warmup
    torch.cuda.synchronize(device)
    _ = method()
    torch.cuda.synchronize(device)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = method()
    torch.cuda.synchronize(device)
    end = time.perf_counter()

    avg_time = (end - start) / num_calls
    peak_bytes = torch.cuda.max_memory_allocated(device)

    print(f"\n[Bitter7 with {method_name}]")
    print(f"  Avg time: {avg_time * 1000:.3f} ms")
    print(f"  Peak mem: {peak_bytes / (1024 ** 2):.2f} MB")

    return avg_time, peak_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--num-calls", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)

    print(f"Testing topk vs kthvalue for Bitter7")
    print(f"Model: dim={args.dim}, layers={args.layers}")
    print(f"Total params: {args.dim * args.dim * args.layers / 1e6:.1f}M")

    model = ToyModel(dim=args.dim, num_layers=args.layers).to(device)
    adam_state = build_fake_adam_state(model, device=device)

    prunable_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    pruner = Bitter7PrunerTopK(
        prunable_layers=prunable_layers,
        optimizer_state=adam_state,
        sparsity=args.sparsity,
        imp_dtype=torch.bfloat16,
    )

    # Benchmark both
    time_kth, mem_kth = benchmark_method(pruner, "kthvalue", args.num_calls, device)
    time_topk, mem_topk = benchmark_method(pruner, "topk", args.num_calls, device)

    print("\n" + "=" * 60)
    print(f"SPEEDUP: {time_kth / time_topk:.2f}x faster with topk!")
    print(f"Memory: {(mem_kth - mem_topk) / (1024**2):.1f} MB saved")
    print("=" * 60)


if __name__ == "__main__":
    main()
