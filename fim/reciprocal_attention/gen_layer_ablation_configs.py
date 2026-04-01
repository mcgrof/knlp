#!/usr/bin/env python3
"""Generate layer-selector ablation configs from a FIM summary.

Reads a FIM summary JSON (from the matched harness attention-stats collection)
and produces two surgical selection JSONs:

  Arm A: layer selector = fim_trace (per_layer_traces), head selector = max eigenvalue
  Arm B: layer selector = attn_layer_eigmax (per-layer mean of per-head max eigenvalue),
         head selector = max eigenvalue

Both arms use the same head count and the same eigenvalue-based head ranking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_fim_summary(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: FIM summary not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text())


def select_layers(
    scores_per_layer: List[float],
    keep_top_k: int = 5,
    skip_highest: bool = True,
) -> List[int]:
    """Rank layers by score, skip highest, keep top-k."""
    ranked = sorted(range(len(scores_per_layer)), key=lambda i: scores_per_layer[i], reverse=True)
    if skip_highest and ranked:
        ranked = ranked[1:]
    return ranked[:keep_top_k]


def select_heads(
    candidate_layers: List[int],
    per_head_eigmax: List[List[float]],
    top_n: int,
) -> List[Tuple[float, int, int]]:
    """Select top-N heads by eigenvalue across candidate layers."""
    scored: List[Tuple[float, int, int]] = []
    for layer_idx in candidate_layers:
        for head_idx, score in enumerate(per_head_eigmax[layer_idx]):
            scored.append((float(score), layer_idx, head_idx))
    scored.sort(reverse=True)
    return scored[:top_n]


def build_selection_json(
    chosen_heads: List[Tuple[float, int, int]],
    candidate_layers: List[int],
    layer_scores: Dict[str, float],
    layer_selector: str,
    head_selector: str = "max_eigenvalue",
    model: str = "llama-1b",
) -> Dict[str, Any]:
    layers: Dict[str, List[int]] = {}
    scores: Dict[str, Dict[str, float]] = {}
    for score, layer_idx, head_idx in chosen_heads:
        layers.setdefault(str(layer_idx), []).append(head_idx)
        scores.setdefault(str(layer_idx), {})[str(head_idx)] = score
    for v in layers.values():
        v.sort()

    return {
        "model": model,
        "layer_selector": layer_selector,
        "head_selector": head_selector,
        "selection_method": f"ablation-{layer_selector}-{head_selector}",
        "candidate_layers": candidate_layers,
        "candidate_layer_scores": layer_scores,
        "selected_head_count": len(chosen_heads),
        "layers": layers,
        "scores": scores,
    }


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    print(f"  wrote: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fim-summary", required=True, help="Path to FIM summary JSON")
    parser.add_argument("--top-heads", type=int, default=28, help="Number of heads to select")
    parser.add_argument("--top-layers", type=int, default=5, help="Number of candidate layers")
    parser.add_argument("--output-dir", default="configs", help="Output directory for selection JSONs")
    parser.add_argument("--prefix", default="ra_ablation_llama1b", help="Output filename prefix")
    args = parser.parse_args()

    summary = load_fim_summary(args.fim_summary)
    per_layer_traces = summary["per_layer_traces"]
    per_head_eigmax = summary.get("per_head_max_eigenvalue") or summary.get("per_head_scores")
    if per_head_eigmax is None:
        print("ERROR: FIM summary missing per_head_max_eigenvalue / per_head_scores", file=sys.stderr)
        sys.exit(1)

    # Arm A: layer selection by FIM trace (per_layer_traces)
    print("=== Arm A: layer_selector=fim_trace, head_selector=max_eigenvalue ===")
    arm_a_layers = select_layers(per_layer_traces, keep_top_k=args.top_layers)
    arm_a_heads = select_heads(arm_a_layers, per_head_eigmax, args.top_heads)
    arm_a_layer_scores = {str(i): per_layer_traces[i] for i in arm_a_layers}
    arm_a = build_selection_json(
        arm_a_heads, arm_a_layers, arm_a_layer_scores,
        layer_selector="fim_trace",
    )
    print(f"  candidate layers: {arm_a_layers}")
    print(f"  selected heads: {len(arm_a_heads)}")

    # Arm B: layer selection by per-layer mean eigenvalue
    print("=== Arm B: layer_selector=attn_layer_eigmax, head_selector=max_eigenvalue ===")
    per_layer_eigmax_mean = [
        sum(per_head_eigmax[i]) / max(len(per_head_eigmax[i]), 1)
        for i in range(len(per_head_eigmax))
    ]
    arm_b_layers = select_layers(per_layer_eigmax_mean, keep_top_k=args.top_layers)
    arm_b_heads = select_heads(arm_b_layers, per_head_eigmax, args.top_heads)
    arm_b_layer_scores = {str(i): per_layer_eigmax_mean[i] for i in arm_b_layers}
    arm_b = build_selection_json(
        arm_b_heads, arm_b_layers, arm_b_layer_scores,
        layer_selector="attn_layer_eigmax",
    )
    print(f"  candidate layers: {arm_b_layers}")
    print(f"  selected heads: {len(arm_b_heads)}")

    # Save
    out_dir = Path(args.output_dir)
    save_json(out_dir / f"{args.prefix}_arm_a_fimtrace.json", arm_a)
    save_json(out_dir / f"{args.prefix}_arm_b_eigmax.json", arm_b)

    # Report overlap
    a_set = {(l, h) for _, l, h in arm_a_heads}
    b_set = {(l, h) for _, l, h in arm_b_heads}
    overlap = a_set & b_set
    print(f"\n=== Overlap: {len(overlap)}/{args.top_heads} heads in common ===")
    if arm_a_layers == arm_b_layers:
        print("WARNING: both arms selected identical candidate layers — ablation may not be meaningful")
    else:
        print(f"  Arm A layers: {arm_a_layers}")
        print(f"  Arm B layers: {arm_b_layers}")
        diff_a = set(arm_a_layers) - set(arm_b_layers)
        diff_b = set(arm_b_layers) - set(arm_a_layers)
        if diff_a:
            print(f"  Only in A: {sorted(diff_a)}")
        if diff_b:
            print(f"  Only in B: {sorted(diff_b)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
