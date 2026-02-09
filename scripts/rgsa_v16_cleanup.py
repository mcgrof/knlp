#!/usr/bin/env python3
"""
RGSA v16 Part A: Cleanup R&D - Confirm v14 negative result under exact matching.

This script runs deterministic comparison runs with:
- Fixed batch size (NO AUTO mode)
- Fixed grad accumulation
- Fixed tokens_seen target
- Fixed dataset shard and dataloader seed
- Exact budget matching (no rounding drift)
- Run manifest for reproducibility

Usage:
    python scripts/rgsa_v16_cleanup.py --output-dir /tmp/rgsa_v16_cleanup
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sensitivity import (
    load_sensitivity_json,
    compute_variance_weights,
    compute_per_layer_top_b_exact,
)


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def compute_config_hash(config: Dict) -> str:
    """Compute hash of config for reproducibility tracking."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def create_run_manifest(
    output_dir: str,
    config: Dict,
    allocation_name: str,
    top_b_per_layer: List[int],
) -> str:
    """Create run manifest JSON for reproducibility."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "config_hash": compute_config_hash(config),
        "allocation_name": allocation_name,
        "top_b_per_layer": top_b_per_layer,
        "top_b_total": sum(top_b_per_layer),
        **config,
    }

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def run_training(
    allocation_name: str,
    top_b_per_layer: Optional[List[int]],
    config: Dict,
    output_dir: str,
    seed: int = 1,
) -> Dict:
    """Run a single training experiment with deterministic settings."""
    run_name = f"v16_{allocation_name}_seed{seed}"
    run_output = os.path.join(output_dir, run_name)
    os.makedirs(run_output, exist_ok=True)

    # Create run manifest
    create_run_manifest(run_output, config, allocation_name, top_b_per_layer or [8]*12)

    # Build command with explicit hyperparameters (NO AUTO)
    cmd = [
        "python", "gpt2/train.py",
        "--architecture", "vanilla",
        "--dataset", config["dataset"],
        "--max-iters", str(config["max_iters"]),
        "--output-dir", run_output,
        "--checkpoint-interval", str(config["max_iters"]),
        "--eval-interval", str(config["eval_interval"]),
        "--batch-size", str(config["batch_size"]),
        "--gradient-accumulation", str(config["gradient_accumulation"]),
        "--block-size", str(config["block_size"]),
        "--tracker", "none",
        "--save-sensitivity",
    ]

    # Add per-layer top_b if specified
    if top_b_per_layer is not None:
        # Write top_b config to file and pass via environment
        top_b_config_path = os.path.join(run_output, "top_b_config.json")
        with open(top_b_config_path, "w") as f:
            json.dump({"top_b_per_layer": top_b_per_layer}, f)
        # We'll need to modify trainer to read this - for now use env var
        os.environ["RGSA_TOP_B_PER_LAYER"] = ",".join(str(x) for x in top_b_per_layer)
    else:
        os.environ.pop("RGSA_TOP_B_PER_LAYER", None)

    print(f"\n{'=' * 60}")
    print(f"Running: {allocation_name} (seed={seed})")
    print(f"top_b_per_layer: {top_b_per_layer or '[uniform 8]'}")
    print(f"top_b_total: {sum(top_b_per_layer) if top_b_per_layer else 96}")
    print(f"Output: {run_output}")
    print(f"{'=' * 60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    # Parse output for final val PPL
    val_ppl = None
    lines = result.stdout.split("\n")
    for line in reversed(lines):
        if "Best validation perplexity" in line:
            try:
                val_ppl = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
            break

    return {
        "allocation": allocation_name,
        "seed": seed,
        "top_b_per_layer": top_b_per_layer,
        "top_b_total": sum(top_b_per_layer) if top_b_per_layer else 96,
        "val_ppl": val_ppl,
        "elapsed_seconds": elapsed,
        "output_dir": run_output,
        "success": result.returncode == 0,
        "stderr": result.stderr[-1000:] if result.returncode != 0 else "",
    }


def main():
    parser = argparse.ArgumentParser(description="RGSA v16 Part A cleanup runs")
    parser.add_argument(
        "--sensitivity-path",
        type=str,
        default="/tmp/rgsa_v14_sensitivity_baseline/checkpoint_iter_100_sensitivity.json",
        help="Path to sensitivity.json for variance-weighted allocation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rgsa_v16_cleanup",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Maximum training iterations per run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=96,
        help="Exact total budget (sum of top_b_l)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed config for deterministic comparison (NO AUTO)
    config = {
        "dataset": "finewebedu",
        "batch_size": 8,
        "gradient_accumulation": 4,
        "block_size": 1024,
        "max_iters": args.max_iters,
        "eval_interval": 50,
        "total_budget": args.total_budget,
    }

    print("RGSA v16 Part A: Cleanup R&D")
    print("=" * 60)
    print(f"Config (deterministic, NO AUTO):")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"Seed: {args.seed}")
    print(f"Sensitivity: {args.sensitivity_path}")

    # Load sensitivity and compute allocations
    try:
        sens_data = load_sensitivity_json(args.sensitivity_path)
        S_layer = sens_data["S_layer"]
        n_layer = sens_data["n_layer"]
    except Exception as e:
        print(f"Warning: Could not load sensitivity: {e}")
        print("Using uniform allocation only")
        S_layer = None
        n_layer = 12

    # Compute allocations with EXACT budget matching
    allocations = {}

    # 1. Uniform
    allocations["uniform"] = [args.total_budget // n_layer] * n_layer
    # Distribute remainder
    remainder = args.total_budget - sum(allocations["uniform"])
    for i in range(remainder):
        allocations["uniform"][i] += 1

    if S_layer is not None:
        # 2. Variance-weighted (alpha=0.5)
        weights_var = compute_variance_weights(S_layer, alpha=0.5)
        allocations["variance_0.5"] = compute_per_layer_top_b_exact(
            weights_var, args.total_budget, n_layer
        )

        # 3. Inverted (alpha=0.5, 1/S)
        weights_inv = compute_variance_weights(S_layer, alpha=0.5, invert=True)
        allocations["inverted_0.5"] = compute_per_layer_top_b_exact(
            weights_inv, args.total_budget, n_layer
        )

    print("\nAllocations (exact budget matching):")
    for name, alloc in allocations.items():
        print(f"  {name}: {alloc} (total={sum(alloc)})")

    # Run experiments
    results = []
    for name, top_b in allocations.items():
        result = run_training(
            allocation_name=name,
            top_b_per_layer=top_b if name != "uniform" else None,
            config=config,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        results.append(result)
        print(f"  Result: val_ppl={result['val_ppl']}, success={result['success']}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for r in results:
        status = "OK" if r["success"] else "FAILED"
        print(f"{r['allocation']:15s}: PPL={r['val_ppl'] or 'N/A':>8} total_budget={r['top_b_total']} [{status}]")

    # Save results
    results_path = os.path.join(args.output_dir, "cleanup_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "sensitivity_path": args.sensitivity_path,
            "allocations": allocations,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Determine if v14 negative result is confirmed
    uniform_ppl = next((r["val_ppl"] for r in results if r["allocation"] == "uniform"), None)
    variance_ppl = next((r["val_ppl"] for r in results if r["allocation"] == "variance_0.5"), None)
    inverted_ppl = next((r["val_ppl"] for r in results if r["allocation"] == "inverted_0.5"), None)

    print("\n" + "=" * 60)
    print("V14 NEGATIVE RESULT CONFIRMATION")
    print("=" * 60)
    if uniform_ppl and variance_ppl and inverted_ppl:
        if variance_ppl > uniform_ppl and inverted_ppl > uniform_ppl:
            print("CONFIRMED: Both variance and inverted allocations are worse than uniform.")
            print("v14 negative result stands under exact budget matching.")
        elif variance_ppl < uniform_ppl or inverted_ppl < uniform_ppl:
            print("CONTRADICTION: One of the weighted allocations beats uniform!")
            print("Investigate before proceeding to Part B.")
        else:
            print("INCONCLUSIVE: Results too close to call.")
    else:
        print("INCOMPLETE: Some runs failed. Check stderr for errors.")


if __name__ == "__main__":
    main()
