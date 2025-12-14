#!/usr/bin/env python3
"""
Convert FIM scores into a quantization plan for llama.cpp.

Policy (from llama.cpp discussion #12741):
1. FFN tensors are most sensitive - upgrade high-FIM ones first
2. Attention Q/K/V are next priority
3. Token embeddings are least sensitive (can stay at base quant)

Plan format matches what emit_llama_quantize_cmd.py expects.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_fim_scores(path: str) -> dict:
    """Load FIM scores from JSON file."""
    with open(path) as f:
        return json.load(f)


def select_top_k_layers(
    layer_scores: dict[str, float],
    tensor_group: str,
    top_k: int | None = None,
    top_pct: float | None = None,
) -> list[int]:
    """
    Select top-K or top-% layers by FIM score for a given tensor group.

    Args:
        layer_scores: Dict of "layer.group" -> FIM score
        tensor_group: Group to filter (e.g., "ffn_down")
        top_k: Select top K layers by score
        top_pct: Select top X% of layers by score

    Returns:
        List of layer indices to upgrade
    """
    # Filter to requested group
    group_scores = {}
    for key, score in layer_scores.items():
        match = re.match(r"(\d+)\.(.+)", key)
        if match:
            layer_idx = int(match.group(1))
            group = match.group(2)
            if group == tensor_group:
                group_scores[layer_idx] = score

    if not group_scores:
        return []

    # Sort by score descending
    sorted_layers = sorted(
        group_scores.keys(), key=lambda x: group_scores[x], reverse=True
    )

    # Select top-K or top-%
    if top_k is not None:
        return sorted_layers[:top_k]
    elif top_pct is not None:
        n = max(1, int(len(sorted_layers) * top_pct / 100))
        return sorted_layers[:n]
    else:
        return sorted_layers


def create_quantization_plan(
    fim_data: dict,
    base_quant: str = "Q4_K_M",
    upgrade_quant: str = "q6_k",
    ffn_top_pct: float = 15.0,
    attn_top_pct: float = 10.0,
    upgrade_ffn: bool = True,
    upgrade_attn: bool = False,
) -> dict:
    """
    Create a quantization plan based on FIM scores.

    Strategy (conservative, per #12741 observations):
    - Base: Q4_K_M for most tensors
    - Upgrade: top-X% of FFN layers to q6_k (most sensitive)
    - Optionally: top-X% of attention layers to q6_k

    Args:
        fim_data: Output from fim_score.py
        base_quant: Base quantization level
        upgrade_quant: Quantization level for high-FIM tensors
        ffn_top_pct: Percentage of FFN layers to upgrade
        attn_top_pct: Percentage of attention layers to upgrade
        upgrade_ffn: Whether to upgrade FFN layers
        upgrade_attn: Whether to upgrade attention layers

    Returns:
        Plan dict with base_quant and list of overrides
    """
    layer_scores = fim_data.get("by_layer_group", {})
    overrides = []

    if upgrade_ffn:
        # FFN tensors (most sensitive per #12741)
        for ffn_group in ["ffn_down", "ffn_up", "ffn_gate"]:
            layers = select_top_k_layers(layer_scores, ffn_group, top_pct=ffn_top_pct)
            if layers:
                overrides.append(
                    {
                        "pattern": ffn_group,
                        "layers": sorted(layers),
                        "qtype": upgrade_quant,
                    }
                )

    if upgrade_attn:
        # Attention tensors
        for attn_group in ["attn_q", "attn_k", "attn_v", "attn_output"]:
            layers = select_top_k_layers(layer_scores, attn_group, top_pct=attn_top_pct)
            if layers:
                overrides.append(
                    {
                        "pattern": attn_group,
                        "layers": sorted(layers),
                        "qtype": upgrade_quant,
                    }
                )

    # Compute estimated impact
    total_layers = len(
        set(int(k.split(".")[0]) for k in layer_scores.keys() if "." in k)
    )

    upgraded_layers = set()
    for override in overrides:
        upgraded_layers.update(override["layers"])

    plan = {
        "base_quant": base_quant,
        "upgrade_quant": upgrade_quant,
        "overrides": overrides,
        "stats": {
            "total_layers": total_layers,
            "upgraded_layer_count": len(upgraded_layers),
            "ffn_top_pct": ffn_top_pct if upgrade_ffn else 0,
            "attn_top_pct": attn_top_pct if upgrade_attn else 0,
        },
    }

    return plan


def create_global_override_plan(
    fim_data: dict,
    base_quant: str = "Q4_K_M",
    upgrade_quant: str = "q6_k",
    top_groups: int = 2,
) -> dict:
    """
    Create a plan with global (all-layer) tensor overrides.

    This is simpler than per-layer selection and targets entire tensor
    groups based on their average FIM score.
    """
    group_scores = fim_data.get("by_group", {})

    # Sort groups by FIM score, skip embeddings (least sensitive)
    skip_groups = {"token_embedding", "output"}
    candidate_groups = {k: v for k, v in group_scores.items() if k not in skip_groups}

    sorted_groups = sorted(
        candidate_groups.keys(), key=lambda g: candidate_groups[g], reverse=True
    )[:top_groups]

    overrides = [
        {"pattern": group, "layers": None, "qtype": upgrade_quant}
        for group in sorted_groups
    ]

    return {
        "base_quant": base_quant,
        "upgrade_quant": upgrade_quant,
        "overrides": overrides,
        "stats": {
            "strategy": "global",
            "upgraded_groups": sorted_groups,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert FIM scores to quantization plan"
    )
    parser.add_argument(
        "--fim-scores",
        type=str,
        required=True,
        help="Path to FIM scores JSON from fim_score.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="quant_plan.json",
        help="Output plan JSON file",
    )
    parser.add_argument(
        "--base-quant",
        type=str,
        default="Q4_K_M",
        help="Base quantization level",
    )
    parser.add_argument(
        "--upgrade-quant",
        type=str,
        default="q6_k",
        help="Quantization level for high-FIM tensors",
    )
    parser.add_argument(
        "--ffn-top-pct",
        type=float,
        default=15.0,
        help="Percentage of FFN layers to upgrade (by FIM)",
    )
    parser.add_argument(
        "--attn-top-pct",
        type=float,
        default=0.0,
        help="Percentage of attention layers to upgrade (0 = none)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["layerwise", "global"],
        default="layerwise",
        help="Selection strategy: per-layer or global tensor groups",
    )
    parser.add_argument(
        "--global-top-groups",
        type=int,
        default=3,
        help="Number of top groups to upgrade (for global strategy)",
    )
    args = parser.parse_args()

    # Load FIM scores
    print(f"Loading FIM scores from {args.fim_scores}")
    fim_data = load_fim_scores(args.fim_scores)

    # Create plan
    if args.strategy == "layerwise":
        plan = create_quantization_plan(
            fim_data,
            base_quant=args.base_quant,
            upgrade_quant=args.upgrade_quant,
            ffn_top_pct=args.ffn_top_pct,
            attn_top_pct=args.attn_top_pct,
            upgrade_ffn=args.ffn_top_pct > 0,
            upgrade_attn=args.attn_top_pct > 0,
        )
    else:
        plan = create_global_override_plan(
            fim_data,
            base_quant=args.base_quant,
            upgrade_quant=args.upgrade_quant,
            top_groups=args.global_top_groups,
        )

    # Save plan
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"Quantization plan saved to {output_path}")

    # Print summary
    print(f"\n=== Quantization Plan ===")
    print(f"Base quant: {plan['base_quant']}")
    print(f"Upgrade quant: {plan['upgrade_quant']}")
    print(f"\nOverrides:")
    for override in plan["overrides"]:
        pattern = override["pattern"]
        qtype = override["qtype"]
        layers = override.get("layers")
        if layers:
            if len(layers) > 6:
                layer_str = f"[{layers[0]}, {layers[1]}, ..., {layers[-1]}] ({len(layers)} layers)"
            else:
                layer_str = str(layers)
            print(f"  {pattern}: {qtype} for layers {layer_str}")
        else:
            print(f"  {pattern}: {qtype} (all layers)")

    if "total_layers" in plan["stats"]:
        total = plan["stats"]["total_layers"]
        upgraded = plan["stats"]["upgraded_layer_count"]
        print(
            f"\nLayer coverage: {upgraded}/{total} layers upgraded ({100*upgraded/total:.1f}%)"
        )


if __name__ == "__main__":
    main()
