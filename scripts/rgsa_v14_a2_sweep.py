#!/usr/bin/env python3
"""
RGSA v14 Phase 1 (A2): Variance-weighted allocation sweep.

Runs experiments with different variance_alpha values (0, 0.5, 1.0) to test
whether sensitivity-based budget allocation improves quality.

Usage:
    python scripts/rgsa_v14_a2_sweep.py --sensitivity-path <path> --output-dir <dir>
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_balanced_allocations(sensitivity_path: str, target_total: int = 96):
    """Compute per-layer allocations that approximately match target total budget."""
    from utils.sensitivity import (
        load_sensitivity_json,
        compute_variance_weights,
        compute_per_layer_top_b,
    )

    data = load_sensitivity_json(sensitivity_path)
    S_layer = data["S_layer"]
    n_layer = data["n_layer"]

    results = {}

    for alpha in [0.0, 0.5, 1.0]:
        if alpha == 0.0:
            # Uniform allocation
            results[alpha] = {
                "top_b_base": 8,
                "top_b_per_layer": None,  # Use uniform
                "total": n_layer * 8,
            }
        else:
            # Find top_b_base that gives closest to target_total
            best_base = 8
            best_diff = float("inf")
            best_top_b = None

            for base in range(8, 50):
                weights = compute_variance_weights(S_layer, alpha=alpha)
                top_b = compute_per_layer_top_b(
                    weights,
                    top_b_base=base,
                    n_layer=n_layer,
                    top_b_min=2,
                    top_b_max=16,
                )
                total = sum(top_b)
                diff = abs(total - target_total)
                if diff < best_diff:
                    best_diff = diff
                    best_base = base
                    best_top_b = top_b

            results[alpha] = {
                "top_b_base": best_base,
                "top_b_per_layer": best_top_b,
                "total": sum(best_top_b) if best_top_b else 0,
            }

    return results


def run_training(
    alpha: float,
    sensitivity_path: str,
    output_dir: str,
    max_iters: int = 500,
    seed: int = 1,
    top_b_base: int = 8,
) -> Dict:
    """Run a single training experiment."""
    run_name = f"rgsa_v14_a2_alpha{alpha}_seed{seed}"
    run_output = os.path.join(output_dir, run_name)
    os.makedirs(run_output, exist_ok=True)

    # Build command
    cmd = [
        "python",
        "gpt2/train.py",
        "--architecture",
        "vanilla",
        "--dataset",
        "finewebedu",
        "--max-iters",
        str(max_iters),
        "--output-dir",
        run_output,
        "--checkpoint-interval",
        str(max_iters),  # Only at end
        "--eval-interval",
        "50",
        "--batch-size",
        "8",
        "--gradient-accumulation",
        "4",
        "--tracker",
        "none",
        "--save-sensitivity",
        "--variance-alpha",
        str(alpha),
    ]

    # Add sensitivity path for variance-weighted allocation
    if alpha > 0 and sensitivity_path:
        cmd.extend(["--sensitivity-path", sensitivity_path])

    print(f"\n{'=' * 60}")
    print(f"Running: alpha={alpha}, seed={seed}")
    print(f"Output: {run_output}")
    print(f"{'=' * 60}")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
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
        if "val" in line.lower() and "ppl" in line.lower():
            # Try to parse "val X.XXX, ppl YY.YY" format
            try:
                parts = line.split(",")
                for part in parts:
                    if "ppl" in part.lower():
                        val_ppl = float(part.split()[-1])
                        break
            except (ValueError, IndexError):
                pass

    return {
        "alpha": alpha,
        "seed": seed,
        "val_ppl": val_ppl,
        "elapsed_seconds": elapsed,
        "output_dir": run_output,
        "success": result.returncode == 0,
        "stderr": result.stderr[-1000:] if result.returncode != 0 else "",
    }


def main():
    parser = argparse.ArgumentParser(description="RGSA v14 A2 variance sweep")
    parser.add_argument(
        "--sensitivity-path",
        type=str,
        required=True,
        help="Path to sensitivity.json from baseline training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rgsa_v14_a2_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=500,
        help="Maximum training iterations per run",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated seeds to run",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.5,1.0",
        help="Comma-separated alpha values to test",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"RGSA v14 A2 Variance Sweep")
    print(f"Sensitivity: {args.sensitivity_path}")
    print(f"Alphas: {alphas}")
    print(f"Seeds: {seeds}")
    print(f"Max iters: {args.max_iters}")

    # Compute balanced allocations
    allocations = compute_balanced_allocations(args.sensitivity_path)
    print("\nComputed allocations (matched total budget):")
    for alpha, alloc in allocations.items():
        print(f"  alpha={alpha}: base={alloc['top_b_base']}, total={alloc['total']}")
        if alloc["top_b_per_layer"]:
            print(f"    top_b_per_layer={alloc['top_b_per_layer']}")

    # Run experiments
    results = []
    for alpha in alphas:
        alloc = allocations[alpha]
        for seed in seeds:
            result = run_training(
                alpha=alpha,
                sensitivity_path=args.sensitivity_path,
                output_dir=args.output_dir,
                max_iters=args.max_iters,
                seed=seed,
                top_b_base=alloc["top_b_base"],
            )
            results.append(result)
            print(f"  Result: val_ppl={result['val_ppl']}, success={result['success']}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    summary = {}
    for alpha in alphas:
        alpha_results = [r for r in results if r["alpha"] == alpha and r["success"]]
        if alpha_results:
            ppls = [r["val_ppl"] for r in alpha_results if r["val_ppl"] is not None]
            if ppls:
                import statistics

                summary[alpha] = {
                    "mean_ppl": statistics.mean(ppls),
                    "std_ppl": statistics.stdev(ppls) if len(ppls) > 1 else 0,
                    "min_ppl": min(ppls),
                    "max_ppl": max(ppls),
                    "n_runs": len(ppls),
                }
                print(
                    f"alpha={alpha}: PPL = {summary[alpha]['mean_ppl']:.2f} ± {summary[alpha]['std_ppl']:.2f} "
                    f"(n={summary[alpha]['n_runs']}, min={summary[alpha]['min_ppl']:.2f}, max={summary[alpha]['max_ppl']:.2f})"
                )

    # Save results
    results_path = os.path.join(args.output_dir, "a2_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "sensitivity_path": args.sensitivity_path,
                "allocations": {str(k): v for k, v in allocations.items()},
                "results": results,
                "summary": {str(k): v for k, v in summary.items()},
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
