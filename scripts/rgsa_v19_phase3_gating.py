#!/usr/bin/env python
"""
RGSA v19 Phase 3: Dynamic Gating Policies

Test conditional gating using boundary_pressure as predictor:
- A) Uniform RGSA (control)
- B) Conditional gate: enable far-context when boundary_pressure > threshold
- C) Budget-capped: top-B heads by boundary_pressure per token
- D) Hybrid: minimal uniform + conditional

This script provides infrastructure for gating policy comparison.
Full experiment requires actual training runs.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt2.rgsa import GPT2_RGSA, RGSAConfig


class ConditionalGatingPolicy:
    """Dynamic gating policy for RGSA based on boundary_pressure signal."""

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        policy: str = "uniform",
        threshold: float = 0.1,
        budget_cap: int = 8,
        min_uniform_budget: int = 2,
    ):
        """
        Initialize gating policy.

        Args:
            n_layer: Number of layers
            n_head: Number of heads per layer
            policy: One of 'uniform', 'threshold', 'topk', 'hybrid'
            threshold: Boundary pressure threshold for 'threshold' policy
            budget_cap: Max heads to enable per token for 'topk' policy
            min_uniform_budget: Minimum heads always enabled for 'hybrid'
        """
        self.n_layer = n_layer
        self.n_head = n_head
        self.policy = policy
        self.threshold = threshold
        self.budget_cap = budget_cap
        self.min_uniform_budget = min_uniform_budget

        # Statistics for logging
        self.total_tokens = 0
        self.far_context_enabled = 0
        self.heads_enabled_per_token = []

    def compute_gate(
        self,
        boundary_pressure: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gating mask based on boundary_pressure.

        Args:
            boundary_pressure: [B, T, n_layer, n_head] attention at boundary

        Returns:
            gate: [B, T, n_layer, n_head] boolean mask (True = enable far-context)
        """
        B, T, n_layer, n_head = boundary_pressure.shape
        device = boundary_pressure.device

        if self.policy == "uniform":
            # All heads always enabled
            gate = torch.ones(B, T, n_layer, n_head, dtype=torch.bool, device=device)

        elif self.policy == "threshold":
            # Enable heads where boundary_pressure > threshold
            gate = boundary_pressure > self.threshold

        elif self.policy == "topk":
            # Enable top-k heads per token (across all layers)
            # Reshape to [B, T, n_layer * n_head]
            bp_flat = boundary_pressure.view(B, T, -1)
            k = min(self.budget_cap, n_layer * n_head)

            # Get top-k indices
            _, topk_idx = torch.topk(bp_flat, k, dim=-1)

            # Create gate mask
            gate_flat = torch.zeros(B, T, n_layer * n_head, dtype=torch.bool, device=device)
            gate_flat.scatter_(2, topk_idx, True)
            gate = gate_flat.view(B, T, n_layer, n_head)

        elif self.policy == "hybrid":
            # Always enable min_uniform_budget heads uniformly
            # Plus additional heads based on threshold
            threshold_gate = boundary_pressure > self.threshold

            # Create uniform gate (first min_uniform_budget heads per layer)
            uniform_gate = torch.zeros(B, T, n_layer, n_head, dtype=torch.bool, device=device)
            heads_per_layer = max(1, self.min_uniform_budget // n_layer)
            uniform_gate[:, :, :, :heads_per_layer] = True

            gate = threshold_gate | uniform_gate

        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        # Update statistics
        self.total_tokens += B * T
        self.far_context_enabled += gate.sum().item()
        self.heads_enabled_per_token.append(gate.float().sum(dim=(2, 3)).mean().item())

        return gate

    def get_statistics(self) -> Dict:
        """Get gating statistics."""
        total_heads = self.n_layer * self.n_head
        return {
            "policy": self.policy,
            "total_tokens": self.total_tokens,
            "far_context_fraction": self.far_context_enabled / (self.total_tokens * total_heads + 1e-10),
            "avg_heads_per_token": np.mean(self.heads_enabled_per_token) if self.heads_enabled_per_token else 0,
            "threshold": self.threshold if self.policy in ["threshold", "hybrid"] else None,
            "budget_cap": self.budget_cap if self.policy == "topk" else None,
        }


def evaluate_policy(
    model: GPT2_RGSA,
    policy: ConditionalGatingPolicy,
    n_eval: int = 20,
    seed: int = 1,
) -> Dict:
    """
    Evaluate gating policy on random data.

    For real experiment, would run actual training with gating.
    Here we measure:
    - Gating statistics (what fraction enabled)
    - Whether gating aligns with ΔKL (high ΔKL -> enabled)

    Args:
        model: RGSA model
        policy: Gating policy to evaluate
        n_eval: Number of evaluation samples
        seed: Random seed

    Returns:
        Dict with evaluation results
    """
    torch.manual_seed(seed)
    cfg = model.config
    T = 128

    # Collect alignment statistics
    enabled_delta_kls = []
    disabled_delta_kls = []

    for _ in range(n_eval):
        idx = torch.randint(0, cfg.vocab_size, (2, T))

        # Compute signals
        signals = model.compute_conditional_signals(idx)
        boundary_pressure = signals["boundary_pressure"]

        # Compute gate
        gate = policy.compute_gate(boundary_pressure)

        # Sample some positions for ΔKL measurement
        positions = list(range(cfg.local_window, T, 10))
        heads = [(l, h) for l in range(cfg.n_layer) for h in range(cfg.n_head)][:8]

        delta_kls = model.compute_conditional_impact_kl(idx, positions, heads)

        # Check alignment: high ΔKL should be enabled
        for (pos, l, h), delta_kl in delta_kls.items():
            # Average gate across batch at this position
            gate_value = gate[:, pos, l, h].float().mean().item()
            if gate_value > 0.5:
                enabled_delta_kls.append(delta_kl)
            else:
                disabled_delta_kls.append(delta_kl)

    # Compute alignment metrics
    results = {
        "policy": policy.policy,
        "gating_stats": policy.get_statistics(),
        "alignment": {},
    }

    if enabled_delta_kls and disabled_delta_kls:
        results["alignment"] = {
            "enabled_mean_kl": float(np.mean(enabled_delta_kls)),
            "disabled_mean_kl": float(np.mean(disabled_delta_kls)),
            "kl_ratio": float(np.mean(enabled_delta_kls) / (np.mean(disabled_delta_kls) + 1e-10)),
            "n_enabled": len(enabled_delta_kls),
            "n_disabled": len(disabled_delta_kls),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="RGSA v19 Phase 3: Gating Policies")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for gating")
    parser.add_argument("--budget-cap", type=int, default=8, help="Budget cap for topk")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Seeds to test")
    args = parser.parse_args()

    print("RGSA v19 Phase 3: Dynamic Gating Policies")
    print("=" * 60)

    # Create model
    cfg = RGSAConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        local_window=64,
        chunk_size=32,
        top_b=4,
    )

    model = GPT2_RGSA(cfg)
    model.eval()

    seeds = [int(s) for s in args.seeds.split(",")]

    # Test each policy
    policies = [
        ("uniform", ConditionalGatingPolicy(cfg.n_layer, cfg.n_head, "uniform")),
        ("threshold", ConditionalGatingPolicy(cfg.n_layer, cfg.n_head, "threshold", threshold=args.threshold)),
        ("topk", ConditionalGatingPolicy(cfg.n_layer, cfg.n_head, "topk", budget_cap=args.budget_cap)),
        ("hybrid", ConditionalGatingPolicy(cfg.n_layer, cfg.n_head, "hybrid",
                                            threshold=args.threshold, min_uniform_budget=4)),
    ]

    all_results = []

    for policy_name, policy in policies:
        print(f"\n{'='*60}")
        print(f"Policy: {policy_name}")
        print(f"{'='*60}")

        seed_results = []
        for seed in seeds:
            result = evaluate_policy(model, policy, n_eval=10, seed=seed)
            seed_results.append(result)

            # Reset policy stats for next seed
            policy.total_tokens = 0
            policy.far_context_enabled = 0
            policy.heads_enabled_per_token = []

        # Aggregate
        avg_enabled = np.mean([
            r["gating_stats"]["far_context_fraction"] for r in seed_results
        ])
        avg_heads = np.mean([
            r["gating_stats"]["avg_heads_per_token"] for r in seed_results
        ])

        if seed_results[0]["alignment"]:
            avg_kl_ratio = np.mean([r["alignment"]["kl_ratio"] for r in seed_results])
        else:
            avg_kl_ratio = 1.0

        print(f"  Far-context enabled: {avg_enabled:.2%}")
        print(f"  Avg heads/token: {avg_heads:.1f}")
        print(f"  KL ratio (enabled/disabled): {avg_kl_ratio:.2f}")

        all_results.append({
            "policy": policy_name,
            "avg_enabled_fraction": avg_enabled,
            "avg_heads_per_token": avg_heads,
            "avg_kl_ratio": avg_kl_ratio,
            "seed_results": seed_results,
        })

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Policy':<15} {'Enabled %':>12} {'Heads/Token':>12} {'KL Ratio':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['policy']:<15} {r['avg_enabled_fraction']:>11.1%} "
              f"{r['avg_heads_per_token']:>12.1f} {r['avg_kl_ratio']:>10.2f}")
    print("-" * 60)

    # Determine best policy
    # Good policy: high KL ratio (enables high-ΔKL heads) with reasonable budget
    best = max(all_results, key=lambda r: r["avg_kl_ratio"] if r["avg_enabled_fraction"] < 0.9 else 0)

    print(f"\nBest policy: {best['policy']}")
    print(f"  Enables {best['avg_enabled_fraction']:.1%} of heads")
    print(f"  KL ratio: {best['avg_kl_ratio']:.2f}x")

    if best["avg_kl_ratio"] > 1.5 and best["avg_enabled_fraction"] < 0.8:
        print("\nCONCLUSION: Conditional gating VIABLE")
        print("The gating policy successfully identifies high-impact heads.")
    elif best["avg_kl_ratio"] > 1.0:
        print("\nCONCLUSION: Conditional gating shows MARGINAL benefit")
        print("Some alignment exists but may not justify complexity.")
    else:
        print("\nCONCLUSION: Conditional gating NOT VIABLE")
        print("Gating does not align well with head importance.")

    # Save results
    os.makedirs("rgsa_v19_results", exist_ok=True)
    with open("rgsa_v19_results/phase3_gating.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to rgsa_v19_results/phase3_gating.json")


if __name__ == "__main__":
    main()
