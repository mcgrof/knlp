#!/usr/bin/env python3
"""
Build weight access graph for mobile weight packing optimization.

Traces the forward pass of a model to record which weights are accessed
in what order. This graph can be used for layout optimization (METIS-style
partitioning) to minimize page faults during inference.

IMPORTANT: This script enforces CPU-only execution.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, asdict

# Force CPU before importing torch (covers NVIDIA CUDA and AMD ROCm)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn

# Verify CPU-only mode
assert not torch.cuda.is_available(), "CUDA should be disabled but is available!"
DEVICE = torch.device("cpu")
print(f"[CPU ENFORCED] Running on: {DEVICE}")
print(f"[CPU ENFORCED] torch.cuda.is_available() = {torch.cuda.is_available()}")


@dataclass
class WeightInfo:
    """Information about a weight tensor."""

    name: str
    shape: List[int]
    numel: int
    dtype: str
    size_bytes: int
    module_type: str


@dataclass
class AccessEvent:
    """Record of a weight access during forward pass."""

    weight_name: str
    order: int
    module_name: str


class WeightAccessTracer:
    """
    Trace weight accesses during forward pass.

    Uses forward hooks to record which modules (and their weights) are
    accessed in what order during inference.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.access_order: List[AccessEvent] = []
        self.weight_info: Dict[str, WeightInfo] = {}
        self.hooks: List = []
        self._order_counter = 0

        # Build weight info map
        self._build_weight_info()

    def _build_weight_info(self):
        """Catalog all weights in the model."""
        for name, param in self.model.named_parameters():
            dtype_str = str(param.dtype).replace("torch.", "")
            size_bytes = param.numel() * param.element_size()

            # Find module type
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                module_path, param_name = parts
                try:
                    module = self.model.get_submodule(module_path)
                    module_type = type(module).__name__
                except AttributeError:
                    module_type = "Unknown"
            else:
                module_type = "Root"

            self.weight_info[name] = WeightInfo(
                name=name,
                shape=list(param.shape),
                numel=param.numel(),
                dtype=dtype_str,
                size_bytes=size_bytes,
                module_type=module_type,
            )

    def _make_forward_hook(self, module_name: str):
        """Create a forward hook that records access events."""

        def hook(module, inputs, outputs):
            # Record access for each parameter in this module
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if full_name in self.weight_info:
                    event = AccessEvent(
                        weight_name=full_name,
                        order=self._order_counter,
                        module_name=module_name,
                    )
                    self.access_order.append(event)
                    self._order_counter += 1

        return hook

    def install_hooks(self):
        """Install forward hooks on all modules."""
        for name, module in self.model.named_modules():
            # Only hook modules that have their own parameters (not inherited)
            has_direct_params = any(
                True for _ in module.named_parameters(recurse=False)
            )
            if has_direct_params:
                hook = module.register_forward_hook(self._make_forward_hook(name))
                self.hooks.append(hook)

        print(f"Installed {len(self.hooks)} forward hooks")

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def trace(self, input_ids: torch.Tensor) -> List[AccessEvent]:
        """
        Run forward pass and record weight access order.

        Args:
            input_ids: Input tensor for the model

        Returns:
            List of access events in order
        """
        self._order_counter = 0
        self.access_order = []

        self.install_hooks()
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            self.remove_hooks()

        return self.access_order

    def build_access_graph(self) -> Dict[str, List[str]]:
        """
        Build adjacency graph from access order.

        Weights accessed consecutively are connected with an edge.

        Returns:
            Adjacency list mapping weight names to neighbors
        """
        graph: Dict[str, Set[str]] = defaultdict(set)

        # Connect consecutively accessed weights
        for i in range(len(self.access_order) - 1):
            w1 = self.access_order[i].weight_name
            w2 = self.access_order[i + 1].weight_name
            if w1 != w2:
                graph[w1].add(w2)
                graph[w2].add(w1)

        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in graph.items()}

    def get_forward_order(self) -> List[str]:
        """
        Get unique weights in forward-pass order.

        Returns:
            List of weight names in the order they're first accessed
        """
        seen = set()
        order = []
        for event in self.access_order:
            if event.weight_name not in seen:
                seen.add(event.weight_name)
                order.append(event.weight_name)
        return order


def estimate_page_layout(
    forward_order: List[str],
    weight_info: Dict[str, WeightInfo],
    page_size: int = 4096,
) -> Dict[str, any]:
    """
    Estimate page layout statistics for the forward-order weight arrangement.

    Args:
        forward_order: Weights in forward-pass access order
        weight_info: Weight metadata
        page_size: Page size in bytes (default 4KB)

    Returns:
        Statistics about the page layout
    """
    current_page = 0
    current_offset = 0
    page_assignments: Dict[str, int] = {}
    weights_per_page: Dict[int, List[str]] = defaultdict(list)

    for weight_name in forward_order:
        info = weight_info.get(weight_name)
        if info is None:
            continue

        # Check if weight fits in current page
        if current_offset + info.size_bytes > page_size:
            current_page += 1
            current_offset = 0

        page_assignments[weight_name] = current_page
        weights_per_page[current_page].append(weight_name)
        current_offset += info.size_bytes

    # Calculate statistics
    total_bytes = sum(info.size_bytes for info in weight_info.values())
    total_pages = current_page + 1
    theoretical_pages = (total_bytes + page_size - 1) // page_size

    # Count page transitions during forward pass
    page_transitions = 0
    prev_page = None
    for weight_name in forward_order:
        page = page_assignments.get(weight_name)
        if page is not None and page != prev_page:
            page_transitions += 1
            prev_page = page

    return {
        "total_bytes": total_bytes,
        "page_size": page_size,
        "total_pages": total_pages,
        "theoretical_min_pages": theoretical_pages,
        "page_transitions": page_transitions,
        "weights_count": len(forward_order),
        "avg_weights_per_page": len(forward_order) / max(total_pages, 1),
        "page_assignments": page_assignments,
    }


def load_public_model(model_name: str = "openai-community/gpt2"):
    """Load a public HuggingFace model on CPU."""
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    print(f"Loading {model_name} on CPU...")

    if "gpt2" in model_name.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Build weight access graph for mobile packing optimization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for tracing (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weight_access_graph.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="Page size in bytes for layout estimation (default: 4096)",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_public_model(args.model)

    # Create tracer
    tracer = WeightAccessTracer(model)

    # Generate sample input
    print(f"\nTracing forward pass with seq_len={args.seq_len}...")
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=DEVICE)

    # Run trace
    import time

    start = time.time()
    access_events = tracer.trace(input_ids)
    elapsed = time.time() - start
    print(f"Traced {len(access_events)} weight accesses in {elapsed:.2f}s")

    # Build graph
    access_graph = tracer.build_access_graph()
    forward_order = tracer.get_forward_order()

    # Estimate page layout
    page_stats = estimate_page_layout(forward_order, tracer.weight_info, args.page_size)

    # Prepare output
    results = {
        "model": args.model,
        "seq_len": args.seq_len,
        "device": str(DEVICE),
        "weight_info": {k: asdict(v) for k, v in tracer.weight_info.items()},
        "forward_order": forward_order,
        "access_graph": access_graph,
        "page_layout": {k: v for k, v in page_stats.items() if k != "page_assignments"},
    }

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== Summary ===")
    print(f"Model: {args.model}")
    print(f"Total weights: {len(tracer.weight_info)}")
    print(f"Weights accessed in forward pass: {len(forward_order)}")
    print(f"Access events recorded: {len(access_events)}")
    print(f"\n=== Page Layout Statistics ===")
    print(
        f"Total model size: {page_stats['total_bytes']:,} bytes ({page_stats['total_bytes']/1e6:.1f} MB)"
    )
    print(f"Page size: {page_stats['page_size']} bytes")
    print(f"Pages needed: {page_stats['total_pages']}")
    print(f"Page transitions during forward: {page_stats['page_transitions']}")
    print(f"Avg weights per page: {page_stats['avg_weights_per_page']:.1f}")

    # Show forward order sample
    print(f"\n=== Forward Access Order (first 20) ===")
    for i, name in enumerate(forward_order[:20]):
        info = tracer.weight_info[name]
        print(f"  {i+1}. {name} ({info.size_bytes:,} bytes)")


if __name__ == "__main__":
    main()
