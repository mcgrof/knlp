#!/usr/bin/env python
"""
RGSA v18 Phase 3: Policy Test

Compare allocation policies:
- A) uniform allocation
- B) signal-weighted allocation (far_mass or impact_kl)
- C) inverted (sanity check)

This script provides:
1. Allocation computation from head metrics
2. PPL evaluation comparison at fixed budget
3. Multi-seed aggregation (when run with actual training)

For full experiment, use with gpt2/train.py with different allocations.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.rgsa import GPT2_RGSA, RGSAConfig, HeadMetrics, ImpactKLTracker
from utils.sensitivity import (
    compute_per_head_top_b_exact,
    compute_head_weights_from_metrics,
)


def compute_allocations(
    model: GPT2_RGSA,
    idx: torch.Tensor,
    total_budget: int,
    gamma: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute allocation policies from model metrics.

    Args:
        model: RGSA model
        idx: Sample input for metric computation
        total_budget: Total budget to allocate
        gamma: Exponent for weighting

    Returns:
        Dict of allocation tensors {policy_name: top_b[n_layer, n_head]}
    """
    n_layer = model.config.n_layer
    n_head = model.config.n_head

    # Compute head metrics
    head_metrics = model.compute_head_metrics(idx)

    # A) Uniform allocation
    uniform_weight = torch.ones(n_layer, n_head) / (n_layer * n_head)
    uniform_alloc = compute_per_head_top_b_exact(
        uniform_weight, total_budget, n_layer, n_head
    )

    # B) Far-mass weighted (heads with more far attention get more budget)
    far_mass_weight = compute_head_weights_from_metrics(
        head_metrics.far_mass, gamma=gamma
    )
    far_mass_alloc = compute_per_head_top_b_exact(
        far_mass_weight, total_budget, n_layer, n_head
    )

    # C) Inverted far-mass (sanity check: should be worse)
    eps = 1e-8
    inverted_metrics = 1.0 / (head_metrics.far_mass + eps)
    inverted_weight = compute_head_weights_from_metrics(inverted_metrics, gamma=gamma)
    inverted_alloc = compute_per_head_top_b_exact(
        inverted_weight, total_budget, n_layer, n_head
    )

    # Verify all sum to total_budget
    assert uniform_alloc.sum().item() == total_budget
    assert far_mass_alloc.sum().item() == total_budget
    assert inverted_alloc.sum().item() == total_budget

    return {
        "uniform": uniform_alloc,
        "far_mass": far_mass_alloc,
        "inverted": inverted_alloc,
    }


def evaluate_perplexity(
    model: GPT2_RGSA,
    idx: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute perplexity on a batch."""
    with torch.no_grad():
        logits, loss = model(idx, targets)
        ppl = torch.exp(loss).item()
    return ppl


def run_policy_comparison(
    cfg: RGSAConfig,
    n_samples: int = 5,
    total_budget: int = 48,
    seed: int = 1,
) -> Dict:
    """
    Run policy comparison on random data (for demonstration).

    For real experiments, this would:
    1. Load trained model checkpoint
    2. Compute allocations
    3. Run training with each allocation
    4. Compare final PPL

    Args:
        cfg: RGSA config
        n_samples: Number of evaluation samples
        total_budget: Total top_b budget
        seed: Random seed

    Returns:
        Dict with comparison results
    """
    torch.manual_seed(seed)

    # Create model
    model = GPT2_RGSA(cfg)
    model.eval()

    # Generate sample data
    B, T = 4, 256
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    # Compute allocations
    allocations = compute_allocations(model, idx, total_budget)

    results = {
        "seed": seed,
        "total_budget": total_budget,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "allocations": {},
        "ppl": {},
    }

    # Store allocation details
    for name, alloc in allocations.items():
        results["allocations"][name] = {
            "top_b": alloc.tolist(),
            "sum": alloc.sum().item(),
            "per_layer_sum": [alloc[l].sum().item() for l in range(cfg.n_layer)],
        }

    # Compute PPL for each allocation
    # NOTE: For real experiment, this would train with each allocation
    # Here we just evaluate untrained model as demonstration
    baseline_ppl = evaluate_perplexity(model, idx, targets)
    for name in allocations:
        results["ppl"][name] = baseline_ppl  # All same for untrained model

    return results


def aggregate_multi_seed(results_list: List[Dict]) -> Dict:
    """Aggregate results from multiple seeds."""
    if not results_list:
        return {}

    policies = list(results_list[0]["ppl"].keys())
    aggregated = {
        "seeds": [r["seed"] for r in results_list],
        "n_seeds": len(results_list),
        "total_budget": results_list[0]["total_budget"],
        "ppl_summary": {},
    }

    for policy in policies:
        ppls = [r["ppl"][policy] for r in results_list]
        aggregated["ppl_summary"][policy] = {
            "mean": sum(ppls) / len(ppls),
            "std": (
                sum((p - sum(ppls) / len(ppls)) ** 2 for p in ppls) / len(ppls)
            )
            ** 0.5,
            "min": min(ppls),
            "max": max(ppls),
            "values": ppls,
        }

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="RGSA v18 Phase 3 Policy Test")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds to test")
    parser.add_argument("--total-budget", type=int, default=48, help="Total budget")
    parser.add_argument("--n-layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-head", type=int, default=6, help="Heads per layer")
    args = parser.parse_args()

    print("RGSA v18 Phase 3: Policy Test")
    print("=" * 60)

    # Create config
    cfg = RGSAConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_layer * 64,
        local_window=128,
        chunk_size=32,
        top_b=4,
    )

    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"Config: n_layer={cfg.n_layer}, n_head={cfg.n_head}")
    print(f"Total budget: {args.total_budget}")
    print(f"Seeds: {seeds}")
    print()

    # Run for each seed
    results_list = []
    for seed in seeds:
        print(f"Running seed {seed}...")
        result = run_policy_comparison(
            cfg, total_budget=args.total_budget, seed=seed
        )
        results_list.append(result)

        # Print allocation summary
        for name, alloc_data in result["allocations"].items():
            print(f"  {name}: sum={alloc_data['sum']}, per_layer={alloc_data['per_layer_sum']}")

    # Aggregate
    aggregated = aggregate_multi_seed(results_list)

    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    print(f"\nSeeds: {aggregated['seeds']}")
    print(f"Total budget: {aggregated['total_budget']}")

    print("\nPPL Summary:")
    print(f"{'Policy':<15} {'Mean':<10} {'Std':<10} {'Range'}")
    print("-" * 50)
    for policy, data in aggregated["ppl_summary"].items():
        print(
            f"{policy:<15} {data['mean']:>8.2f}   {data['std']:>8.2f}   "
            f"[{data['min']:.2f}, {data['max']:.2f}]"
        )

    # Note about real experiment
    print("\n" + "-" * 60)
    print("NOTE: These results are on UNTRAINED model (random weights).")
    print("All policies show same PPL because model hasn't learned.")
    print()
    print("For real experiment:")
    print("1. Train RGSA model to convergence with uniform allocation")
    print("2. Compute head metrics and allocations")
    print("3. Run short training with each allocation policy")
    print("4. Compare final PPL across seeds")
    print("-" * 60)

    # Save results
    os.makedirs("rgsa_v18_results", exist_ok=True)
    output = {
        "config": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "total_budget": args.total_budget,
        },
        "per_seed": results_list,
        "aggregated": aggregated,
        "note": "Results on untrained model - all policies equivalent",
    }
    with open("rgsa_v18_results/phase3_policy_test.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to rgsa_v18_results/phase3_policy_test.json")


if __name__ == "__main__":
    main()
