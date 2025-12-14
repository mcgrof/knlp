#!/usr/bin/env python3
"""
Combine FIM traces and weight importance scores for mobile packing.

FIM (Fisher Information Matrix) provides layer-level importance from
optimization geometry. Weight importance (magnitude/Adam) provides
per-weight granularity. This script combines both signals for optimal
page packing prioritization.

No GPU required - pure Python/numpy operations.
"""

import os
import sys
import json
import argparse
import re
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

import numpy as np


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores to [0, 1] range using min-max scaling.

    Args:
        scores: Dictionary of name -> score

    Returns:
        Normalized scores
    """
    if not scores:
        return {}

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return {k: 0.5 for k in scores}

    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def extract_layer_number(param_name: str) -> Optional[int]:
    """
    Extract layer number from parameter name.

    Examples:
        'transformer.h.0.attn.c_attn.weight' -> 0
        'transformer.h.11.mlp.c_fc.weight' -> 11
        'model.layers.5.self_attn.q_proj.weight' -> 5

    Args:
        param_name: Full parameter name

    Returns:
        Layer number or None if not found
    """
    # GPT-2 style: transformer.h.N.xxx
    match = re.search(r"\.h\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))

    # Llama/Mistral style: model.layers.N.xxx
    match = re.search(r"\.layers\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))

    return None


def load_fim_traces(fim_path: str) -> Dict[int, float]:
    """
    Load FIM traces and return per-layer importance.

    Supports multiple FIM file formats:
    1. Per-layer dict: {"layer_0": 0.95, "layer_1": 0.88, ...}
    2. Summary format with layer metrics
    3. W&B-style with nested structure

    Args:
        fim_path: Path to FIM JSON file

    Returns:
        Dictionary mapping layer number to FIM trace
    """
    data = load_json(fim_path)

    layer_traces = {}

    # Try different formats
    if "per_layer_head" in data:
        # FIM config format: per_layer_head with layer/head keys
        for key, info in data["per_layer_head"].items():
            layer_num = int(key.split("/")[0])
            # Use total_variance as trace proxy
            trace = info.get("total_variance", info.get("max_eigenvalue", 1.0))
            if layer_num not in layer_traces:
                layer_traces[layer_num] = []
            layer_traces[layer_num].append(trace)

        # Average across heads
        layer_traces = {k: np.mean(v) for k, v in layer_traces.items()}

    elif "layers" in data:
        # Simple layer format
        for layer_key, trace in data["layers"].items():
            match = re.search(r"(\d+)", layer_key)
            if match:
                layer_traces[int(match.group(1))] = trace

    elif any(k.startswith("layer") for k in data.keys()):
        # Direct layer_N keys
        for key, value in data.items():
            match = re.search(r"layer_?(\d+)", key)
            if match:
                layer_num = int(match.group(1))
                if isinstance(value, dict):
                    layer_traces[layer_num] = value.get(
                        "trace", value.get("eigmax", 1.0)
                    )
                else:
                    layer_traces[layer_num] = value

    else:
        print(f"Warning: Could not parse FIM format from {fim_path}")
        print(f"Keys found: {list(data.keys())[:10]}")

    return layer_traces


def expand_fim_to_weights(
    fim_traces: Dict[int, float],
    weight_names: List[str],
) -> Dict[str, float]:
    """
    Expand layer-level FIM traces to per-weight scores.

    Each weight in a layer gets that layer's FIM trace value.

    Args:
        fim_traces: Layer number -> FIM trace
        weight_names: List of weight parameter names

    Returns:
        Dictionary mapping weight names to FIM-derived importance
    """
    expanded = {}

    # Use mean trace for weights without layer assignment
    mean_trace = np.mean(list(fim_traces.values())) if fim_traces else 1.0

    for name in weight_names:
        layer_num = extract_layer_number(name)
        if layer_num is not None and layer_num in fim_traces:
            expanded[name] = fim_traces[layer_num]
        else:
            # Embedding, final layer norm, lm_head, etc.
            expanded[name] = mean_trace

    return expanded


def combine_importance_scores(
    fim_scores: Dict[str, float],
    weight_scores: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Combine FIM and weight importance scores.

    Combined = (1 - alpha) * FIM + alpha * weight_importance

    Args:
        fim_scores: Per-weight FIM-derived scores
        weight_scores: Per-weight importance scores (magnitude/Adam)
        alpha: Blending factor (0 = pure FIM, 1 = pure weight)

    Returns:
        Combined importance scores
    """
    # Normalize both score sets
    fim_norm = normalize_scores(fim_scores)
    weight_norm = normalize_scores(weight_scores)

    # Get all weight names
    all_names = set(fim_norm.keys()) | set(weight_norm.keys())

    combined = {}
    for name in all_names:
        fim_val = fim_norm.get(name, 0.5)
        weight_val = weight_norm.get(name, 0.5)
        combined[name] = (1 - alpha) * fim_val + alpha * weight_val

    return combined


def rank_for_packing(
    combined_scores: Dict[str, float],
    descending: bool = True,
) -> List[Tuple[str, float, int]]:
    """
    Rank weights by importance for packing order.

    Args:
        combined_scores: Combined importance scores
        descending: If True, highest importance first (for hot pages)

    Returns:
        List of (name, score, rank) tuples sorted by importance
    """
    sorted_items = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=descending,
    )

    return [(name, score, rank) for rank, (name, score) in enumerate(sorted_items)]


def estimate_tier_assignment(
    ranked_weights: List[Tuple[str, float, int]],
    weight_info: Optional[Dict[str, Dict]] = None,
    hbm_fraction: float = 0.3,
    cpu_fraction: float = 0.5,
) -> Dict[str, str]:
    """
    Assign weights to memory tiers based on importance ranking.

    Args:
        ranked_weights: Weights ranked by importance (highest first)
        weight_info: Optional weight metadata for size-based assignment
        hbm_fraction: Fraction of weights for HBM tier (top importance)
        cpu_fraction: Fraction for CPU tier (middle importance)

    Returns:
        Dictionary mapping weight names to tier ("HBM", "CPU", "SSD")
    """
    n_weights = len(ranked_weights)
    hbm_cutoff = int(n_weights * hbm_fraction)
    cpu_cutoff = int(n_weights * (hbm_fraction + cpu_fraction))

    tiers = {}
    for name, score, rank in ranked_weights:
        if rank < hbm_cutoff:
            tiers[name] = "HBM"
        elif rank < cpu_cutoff:
            tiers[name] = "CPU"
        else:
            tiers[name] = "SSD"

    return tiers


def main():
    parser = argparse.ArgumentParser(
        description="Combine FIM and weight importance for mobile packing"
    )
    parser.add_argument(
        "--fim",
        type=str,
        help="Path to FIM traces JSON file",
    )
    parser.add_argument(
        "--weight-importance",
        type=str,
        required=True,
        help="Path to weight importance JSON (from extract_adam_importance.py)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blending factor: 0=pure FIM, 1=pure weight importance (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_importance.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--hbm-fraction",
        type=float,
        default=0.3,
        help="Fraction of weights for HBM tier (default: 0.3)",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=0.5,
        help="Fraction of weights for CPU tier (default: 0.5)",
    )

    args = parser.parse_args()

    # Load weight importance
    print(f"Loading weight importance from {args.weight_importance}")
    weight_data = load_json(args.weight_importance)

    # Get the importance scores (handle different formats)
    if "magnitude" in weight_data:
        weight_scores = weight_data["magnitude"]
    elif "bitter7" in weight_data:
        weight_scores = weight_data["bitter7"]
    elif "gradient" in weight_data:
        weight_scores = weight_data["gradient"]
    else:
        # Assume the file is just scores
        weight_scores = weight_data

    print(f"Loaded {len(weight_scores)} weight importance scores")

    weight_names = list(weight_scores.keys())

    # Load and process FIM traces if provided
    if args.fim:
        print(f"Loading FIM traces from {args.fim}")
        fim_traces = load_fim_traces(args.fim)
        print(f"Loaded FIM traces for {len(fim_traces)} layers")

        # Expand to per-weight scores
        fim_scores = expand_fim_to_weights(fim_traces, weight_names)
        print(f"Expanded FIM to {len(fim_scores)} weights")
    else:
        print("No FIM file provided - using uniform FIM scores")
        fim_scores = {name: 1.0 for name in weight_names}

    # Combine scores
    print(f"\nCombining scores with alpha={args.alpha}")
    print(f"  alpha=0 → pure FIM, alpha=1 → pure weight importance")
    combined = combine_importance_scores(fim_scores, weight_scores, args.alpha)

    # Rank for packing
    ranked = rank_for_packing(combined, descending=True)

    # Assign tiers
    tiers = estimate_tier_assignment(
        ranked,
        hbm_fraction=args.hbm_fraction,
        cpu_fraction=args.cpu_fraction,
    )

    # Build output
    results = {
        "config": {
            "fim_path": args.fim,
            "weight_importance_path": args.weight_importance,
            "alpha": args.alpha,
            "hbm_fraction": args.hbm_fraction,
            "cpu_fraction": args.cpu_fraction,
        },
        "combined_scores": combined,
        "ranking": [
            {"name": name, "score": score, "rank": rank, "tier": tiers[name]}
            for name, score, rank in ranked
        ],
        "tier_counts": {
            "HBM": sum(1 for t in tiers.values() if t == "HBM"),
            "CPU": sum(1 for t in tiers.values() if t == "CPU"),
            "SSD": sum(1 for t in tiers.values() if t == "SSD"),
        },
    }

    # Save results
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total weights: {len(combined)}")
    print(f"Tier distribution:")
    print(f"  HBM (hot):  {results['tier_counts']['HBM']} weights")
    print(f"  CPU (warm): {results['tier_counts']['CPU']} weights")
    print(f"  SSD (cold): {results['tier_counts']['SSD']} weights")

    print(f"\n=== Top 10 Most Important (HBM candidates) ===")
    for item in results["ranking"][:10]:
        print(f"  {item['rank']+1}. {item['name']}: {item['score']:.4f}")

    print(f"\n=== Bottom 10 Least Important (SSD candidates) ===")
    for item in results["ranking"][-10:]:
        print(f"  {item['rank']+1}. {item['name']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()
