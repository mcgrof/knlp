#!/usr/bin/env python3
"""
Evaluate and compare PPL for quantized models.

Runs llama-perplexity on multiple GGUF models and produces a comparison
table in the same style as llama.cpp discussion #12741.

Reference: https://github.com/ggml-org/llama.cpp/discussions/12741
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def run_perplexity(
    model_path: str,
    test_file: str,
    llama_perplexity: str = "llama-perplexity",
    ctx_size: int = 512,
    batch_size: int = 512,
) -> dict | None:
    """
    Run llama-perplexity and parse the output.

    Returns dict with 'ppl' and 'ppl_stderr' or None on failure.
    """
    cmd = [
        llama_perplexity,
        "-m",
        model_path,
        "-f",
        test_file,
        "-c",
        str(ctx_size),
        "-b",
        str(batch_size),
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: Timeout evaluating {model_path}")
        return None
    except FileNotFoundError:
        print(f"ERROR: {llama_perplexity} not found. Install llama.cpp or add to PATH.")
        return None

    # Parse output for perplexity
    # Format: "Final estimate: PPL = 12.3456 +/- 0.1234"
    output = result.stdout + result.stderr

    ppl_match = re.search(r"Final estimate: PPL = ([\d.]+)", output)
    stderr_match = re.search(r"\+/- ([\d.]+)", output)

    if ppl_match:
        ppl = float(ppl_match.group(1))
        stderr = float(stderr_match.group(1)) if stderr_match else 0.0
        return {"ppl": ppl, "ppl_stderr": stderr}
    else:
        # Try alternative format
        ppl_match = re.search(r"perplexity = ([\d.]+)", output, re.IGNORECASE)
        if ppl_match:
            return {"ppl": float(ppl_match.group(1)), "ppl_stderr": 0.0}

    print(f"ERROR: Could not parse perplexity from output:\n{output[:500]}")
    return None


def compute_comparison_metrics(
    results: dict[str, dict],
    baseline_key: str = "naive",
) -> dict[str, dict]:
    """
    Compute delta and percentage change vs baseline.

    Returns results with added 'delta_ppl' and 'pct_change' fields.
    """
    if baseline_key not in results:
        print(f"Warning: Baseline '{baseline_key}' not found in results")
        return results

    baseline_ppl = results[baseline_key]["ppl"]
    baseline_size = results[baseline_key]["size_mb"]

    for name, data in results.items():
        data["delta_ppl"] = data["ppl"] - baseline_ppl
        data["pct_ppl_change"] = (data["ppl"] - baseline_ppl) / baseline_ppl * 100
        data["size_delta_mb"] = data["size_mb"] - baseline_size
        data["size_pct_change"] = (
            (data["size_mb"] - baseline_size) / baseline_size * 100
        )

    return results


def format_results_table(results: dict[str, dict]) -> str:
    """Format results as a markdown table (similar to #12741 style)."""
    lines = []
    lines.append("| Model | Size (MB) | Size Δ | PPL | PPL Δ | PPL % |")
    lines.append("|-------|-----------|--------|-----|-------|-------|")

    for name, data in sorted(results.items()):
        size = data["size_mb"]
        size_delta = data.get("size_delta_mb", 0)
        ppl = data["ppl"]
        ppl_delta = data.get("delta_ppl", 0)
        ppl_pct = data.get("pct_ppl_change", 0)

        size_delta_str = f"{size_delta:+.1f}" if size_delta != 0 else "-"
        ppl_delta_str = f"{ppl_delta:+.4f}" if ppl_delta != 0 else "-"
        ppl_pct_str = f"{ppl_pct:+.2f}%" if ppl_pct != 0 else "-"

        lines.append(
            f"| {name} | {size:.1f} | {size_delta_str} | "
            f"{ppl:.4f} | {ppl_delta_str} | {ppl_pct_str} |"
        )

    return "\n".join(lines)


def format_wandb_metrics(results: dict[str, dict]) -> dict:
    """Format results for W&B logging."""
    wandb_data = {}

    for name, data in results.items():
        prefix = f"eval/{name}"
        wandb_data[f"{prefix}/ppl"] = data["ppl"]
        wandb_data[f"{prefix}/size_mb"] = data["size_mb"]
        if "delta_ppl" in data:
            wandb_data[f"{prefix}/delta_ppl"] = data["delta_ppl"]
            wandb_data[f"{prefix}/pct_ppl_change"] = data["pct_ppl_change"]

    return wandb_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPL for quantized models")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to GGUF models to evaluate (name:path format, e.g., naive:model.gguf)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test text file for perplexity evaluation",
    )
    parser.add_argument(
        "--llama-perplexity",
        type=str,
        default="llama-perplexity",
        help="Path to llama-perplexity binary",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=512,
        help="Context size for perplexity evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for perplexity evaluation",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="naive",
        help="Name of baseline model for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--output-wandb",
        type=str,
        default=None,
        help="Output JSON file for W&B metrics",
    )
    args = parser.parse_args()

    # Parse model specifications
    models = {}
    for spec in args.models:
        if ":" in spec:
            name, path = spec.split(":", 1)
        else:
            name = Path(spec).stem
            path = spec
        models[name] = path

    # Evaluate each model
    results = {}
    for name, path in models.items():
        print(f"\n=== Evaluating: {name} ===")
        print(f"Model: {path}")

        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path}")
            continue

        size_mb = get_file_size_mb(path)
        print(f"Size: {size_mb:.1f} MB")

        ppl_result = run_perplexity(
            path,
            args.test_file,
            llama_perplexity=args.llama_perplexity,
            ctx_size=args.ctx_size,
            batch_size=args.batch_size,
        )

        if ppl_result:
            results[name] = {
                "path": path,
                "size_mb": size_mb,
                "ppl": ppl_result["ppl"],
                "ppl_stderr": ppl_result["ppl_stderr"],
            }
            print(f"PPL: {ppl_result['ppl']:.4f} +/- {ppl_result['ppl_stderr']:.4f}")
        else:
            print(f"Failed to evaluate {name}")

    if not results:
        print("\nERROR: No models were successfully evaluated")
        sys.exit(1)

    # Compute comparison metrics
    results = compute_comparison_metrics(results, args.baseline)

    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(format_results_table(results))

    # Save JSON results
    output = {
        "baseline": args.baseline,
        "test_file": args.test_file,
        "ctx_size": args.ctx_size,
        "results": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save W&B metrics if requested
    if args.output_wandb:
        wandb_data = format_wandb_metrics(results)
        wandb_path = Path(args.output_wandb)
        with open(wandb_path, "w") as f:
            json.dump(wandb_data, f, indent=2)
        print(f"W&B metrics saved to {wandb_path}")

    # Print summary
    if len(results) >= 2 and args.baseline in results:
        fim_key = next(k for k in results.keys() if k != args.baseline)
        baseline = results[args.baseline]
        fim = results[fim_key]

        print("\n=== Summary ===")
        print(
            f"Baseline ({args.baseline}): {baseline['ppl']:.4f} PPL, {baseline['size_mb']:.1f} MB"
        )
        print(f"FIM-guided ({fim_key}): {fim['ppl']:.4f} PPL, {fim['size_mb']:.1f} MB")
        print(f"PPL change: {fim['delta_ppl']:+.4f} ({fim['pct_ppl_change']:+.2f}%)")
        print(
            f"Size change: {fim['size_delta_mb']:+.1f} MB ({fim['size_pct_change']:+.2f}%)"
        )


if __name__ == "__main__":
    main()
