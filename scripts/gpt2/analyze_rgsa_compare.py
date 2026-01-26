#!/usr/bin/env python3
"""
analyze_rgsa_compare.py - Generate plots and report from RGSA comparison runs

Usage:
    python scripts/gpt2/analyze_rgsa_compare.py --group <wandb_group>
    python scripts/gpt2/analyze_rgsa_compare.py --baseline <run_id> --rgsa <run_id>
    python scripts/gpt2/analyze_rgsa_compare.py --local <output_dir>

Outputs:
    - compare_val_ppl.png: Validation perplexity vs steps
    - compare_teacher_recall.png: Teacher top-k recall vs steps (RGSA only)
    - compare_entropy_loadbalance.png: Routing entropy and load balance vs steps
    - compare_tokens_per_query.png: Tokens per query distribution vs steps
    - report.md: Summary of final metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available, using local data only")


def fetch_wandb_runs(
    group: Optional[str] = None,
    baseline_id: Optional[str] = None,
    rgsa_id: Optional[str] = None,
    project: str = "gpt2-rgsa-compare",
    entity: str = "mcgrof-citizen",
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Fetch run histories from W&B."""
    if not HAS_WANDB:
        return None, None

    api = wandb.Api()

    baseline_data = None
    rgsa_data = None

    if group:
        # Find runs by group
        runs = api.runs(
            f"{entity}/{project}", filters={"group": group}, order="-created_at"
        )
        for run in runs:
            tags = run.tags or []
            if "baseline" in tags:
                baseline_data = _extract_run_data(run)
            elif "rgsa" in tags:
                rgsa_data = _extract_run_data(run)
    else:
        # Find runs by ID
        if baseline_id:
            try:
                run = api.run(f"{entity}/{project}/{baseline_id}")
                baseline_data = _extract_run_data(run)
            except Exception as e:
                print(f"Warning: Could not fetch baseline run {baseline_id}: {e}")

        if rgsa_id:
            try:
                run = api.run(f"{entity}/{project}/{rgsa_id}")
                rgsa_data = _extract_run_data(run)
            except Exception as e:
                print(f"Warning: Could not fetch RGSA run {rgsa_id}: {e}")

    return baseline_data, rgsa_data


def _extract_run_data(run) -> Dict:
    """Extract relevant metrics from a W&B run."""
    history = run.history(samples=10000)

    data = {
        "name": run.name,
        "id": run.id,
        "config": dict(run.config),
        "summary": dict(run.summary),
        "iterations": [],
        "val_loss": [],
        "val_ppl": [],
        "train_loss": [],
        "train_ppl": [],
        "teacher_recall": [],
        "routing_entropy": [],
        "load_balance": [],
        "tokens_per_query_mean": [],
        "tokens_per_query_p50": [],
        "tokens_per_query_p95": [],
        "tokens_per_query_p99": [],
        "iter_time_sec": [],
        "tokens_per_sec": [],
    }

    # Extract metrics from history
    for _, row in history.iterrows():
        if "iteration" in row and not np.isnan(row.get("iteration", np.nan)):
            data["iterations"].append(int(row["iteration"]))

            # Validation metrics
            if "val_loss" in row and not np.isnan(row.get("val_loss", np.nan)):
                data["val_loss"].append(row["val_loss"])
                data["val_ppl"].append(row.get("val_perplexity", np.nan))

            # Training metrics
            if "train_loss" in row and not np.isnan(row.get("train_loss", np.nan)):
                data["train_loss"].append(row["train_loss"])
                data["train_ppl"].append(row.get("train_perplexity", np.nan))

            # RGSA metrics
            if "rgsa/teacher_topk_recall" in row:
                data["teacher_recall"].append(
                    row.get("rgsa/teacher_topk_recall", np.nan)
                )
                data["routing_entropy"].append(row.get("rgsa/routing_entropy", np.nan))
                data["load_balance"].append(row.get("rgsa/load_balance_score", np.nan))
                data["tokens_per_query_mean"].append(
                    row.get("rgsa/tokens_per_query_mean", np.nan)
                )
                data["tokens_per_query_p50"].append(
                    row.get("rgsa/candidates_p50", np.nan)
                )
                data["tokens_per_query_p95"].append(
                    row.get("rgsa/candidates_p95", np.nan)
                )
                data["tokens_per_query_p99"].append(
                    row.get("rgsa/candidates_p99", np.nan)
                )

            # Performance metrics
            if "perf/iter_time_sec" in row:
                data["iter_time_sec"].append(row.get("perf/iter_time_sec", np.nan))
                data["tokens_per_sec"].append(row.get("perf/tokens_per_sec", np.nan))

    return data


def load_local_data(output_dir: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load data from local training output directories."""
    baseline_data = None
    rgsa_data = None

    output_path = Path(output_dir)

    # Look for baseline and RGSA subdirectories
    for subdir in output_path.iterdir():
        if not subdir.is_dir():
            continue

        # Find training metrics JSON
        metrics_files = list(subdir.glob("training_metrics*.json"))
        if not metrics_files:
            continue

        metrics_file = metrics_files[0]
        with open(metrics_file) as f:
            metrics = json.load(f)

        data = {
            "name": subdir.name,
            "id": subdir.name,
            "config": {},
            "summary": {"final_iteration": metrics.get("final_iteration", 0)},
            "iterations": metrics.get("metrics", {}).get("iterations", []),
            "val_loss": metrics.get("metrics", {}).get("val_losses", []),
            "val_ppl": metrics.get("metrics", {}).get("val_perplexities", []),
            "train_loss": metrics.get("metrics", {}).get("train_losses", []),
            "train_ppl": metrics.get("metrics", {}).get("train_perplexities", []),
            "teacher_recall": [],
            "routing_entropy": [],
            "load_balance": [],
            "tokens_per_query_mean": [],
            "tokens_per_query_p50": [],
            "tokens_per_query_p95": [],
            "tokens_per_query_p99": [],
            "iter_time_sec": [],
            "tokens_per_sec": [],
        }

        if "baseline" in subdir.name.lower():
            baseline_data = data
        elif "rgsa" in subdir.name.lower():
            rgsa_data = data

    return baseline_data, rgsa_data


def plot_val_ppl(
    baseline: Optional[Dict], rgsa: Optional[Dict], output_dir: str
) -> None:
    """Plot validation perplexity vs steps."""
    if not HAS_MATPLOTLIB:
        return

    plt.figure(figsize=(10, 6))

    if baseline and baseline["val_ppl"]:
        # Align iterations with val_ppl (eval happens at intervals)
        n_evals = len(baseline["val_ppl"])
        if baseline["iterations"]:
            # Use actual iteration values if available
            iters = baseline["iterations"][:n_evals]
        else:
            iters = list(range(n_evals))
        plt.plot(iters, baseline["val_ppl"], "b-o", label="Baseline GPT-2", linewidth=2)

    if rgsa and rgsa["val_ppl"]:
        n_evals = len(rgsa["val_ppl"])
        if rgsa["iterations"]:
            iters = rgsa["iterations"][:n_evals]
        else:
            iters = list(range(n_evals))
        plt.plot(iters, rgsa["val_ppl"], "r-s", label="RGSA", linewidth=2)

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Validation Perplexity", fontsize=12)
    plt.title("Validation Perplexity: Baseline vs RGSA", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "compare_val_ppl.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_teacher_recall(rgsa: Optional[Dict], output_dir: str) -> None:
    """Plot teacher top-k recall vs steps (RGSA only)."""
    if not HAS_MATPLOTLIB or not rgsa:
        return

    recall = [r for r in rgsa.get("teacher_recall", []) if not np.isnan(r)]
    if not recall:
        print("No teacher recall data available, skipping plot")
        return

    plt.figure(figsize=(10, 6))

    iters = list(range(len(recall)))
    plt.plot(iters, recall, "g-o", linewidth=2, markersize=8)

    plt.xlabel("Eval Step", fontsize=12)
    plt.ylabel("Teacher Top-k Recall", fontsize=12)
    plt.title("RGSA: Teacher Top-k Recall vs Training Steps", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add horizontal line at target (0.7)
    plt.axhline(y=0.7, color="r", linestyle="--", alpha=0.5, label="Target (0.7)")
    plt.legend(fontsize=11)

    plt.tight_layout()

    output_path = Path(output_dir) / "compare_teacher_recall.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_entropy_loadbalance(rgsa: Optional[Dict], output_dir: str) -> None:
    """Plot routing entropy and load balance vs steps."""
    if not HAS_MATPLOTLIB or not rgsa:
        return

    entropy = [e for e in rgsa.get("routing_entropy", []) if not np.isnan(e)]
    load_bal = [lb for lb in rgsa.get("load_balance", []) if not np.isnan(lb)]

    if not entropy and not load_bal:
        print("No entropy/load balance data available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Entropy plot
    if entropy:
        iters = list(range(len(entropy)))
        ax1.plot(iters, entropy, "b-o", linewidth=2, markersize=8)
        ax1.set_xlabel("Eval Step", fontsize=12)
        ax1.set_ylabel("Routing Entropy (normalized)", fontsize=12)
        ax1.set_title("RGSA: Routing Entropy", fontsize=14)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(
            y=0.5, color="r", linestyle="--", alpha=0.5, label="Collapse threshold"
        )
        ax1.legend()

    # Load balance plot
    if load_bal:
        iters = list(range(len(load_bal)))
        ax2.plot(iters, load_bal, "g-o", linewidth=2, markersize=8)
        ax2.set_xlabel("Eval Step", fontsize=12)
        ax2.set_ylabel("Load Balance Score", fontsize=12)
        ax2.set_title("RGSA: Load Balance", fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(
            y=0.5, color="r", linestyle="--", alpha=0.5, label="Collapse threshold"
        )
        ax2.legend()

    plt.tight_layout()

    output_path = Path(output_dir) / "compare_entropy_loadbalance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_tokens_per_query(rgsa: Optional[Dict], output_dir: str) -> None:
    """Plot tokens per query distribution vs steps."""
    if not HAS_MATPLOTLIB or not rgsa:
        return

    mean_vals = [m for m in rgsa.get("tokens_per_query_mean", []) if not np.isnan(m)]
    p50_vals = [p for p in rgsa.get("tokens_per_query_p50", []) if not np.isnan(p)]
    p95_vals = [p for p in rgsa.get("tokens_per_query_p95", []) if not np.isnan(p)]
    p99_vals = [p for p in rgsa.get("tokens_per_query_p99", []) if not np.isnan(p)]

    if not mean_vals:
        print("No tokens per query data available, skipping plot")
        return

    plt.figure(figsize=(10, 6))

    iters = list(range(len(mean_vals)))
    plt.plot(iters, mean_vals, "b-o", label="Mean", linewidth=2)

    if p50_vals and len(p50_vals) == len(mean_vals):
        plt.plot(iters, p50_vals, "g--", label="p50", linewidth=1.5, alpha=0.7)
    if p95_vals and len(p95_vals) == len(mean_vals):
        plt.plot(
            iters,
            p95_vals,
            "orange",
            linestyle="--",
            label="p95",
            linewidth=1.5,
            alpha=0.7,
        )
    if p99_vals and len(p99_vals) == len(mean_vals):
        plt.plot(iters, p99_vals, "r--", label="p99", linewidth=1.5, alpha=0.7)

    plt.xlabel("Eval Step", fontsize=12)
    plt.ylabel("Tokens per Query", fontsize=12)
    plt.title("RGSA: Candidate Tokens per Query", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "compare_tokens_per_query.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(
    baseline: Optional[Dict], rgsa: Optional[Dict], output_dir: str
) -> None:
    """Generate markdown report summarizing final metrics."""
    report_lines = [
        "# RGSA Comparison Report",
        "",
        "## Summary",
        "",
    ]

    # Baseline metrics
    if baseline:
        report_lines.extend(
            [
                "### Baseline GPT-2",
                "",
                f"- **Run ID**: {baseline.get('id', 'N/A')}",
                f"- **Final Iteration**: {baseline.get('summary', {}).get('final_iteration', 'N/A')}",
            ]
        )

        if baseline["val_ppl"]:
            final_ppl = baseline["val_ppl"][-1]
            report_lines.append(f"- **Final Val PPL**: {final_ppl:.2f}")

        if baseline["val_loss"]:
            final_loss = baseline["val_loss"][-1]
            report_lines.append(f"- **Final Val Loss**: {final_loss:.4f}")

        report_lines.append("")

    # RGSA metrics
    if rgsa:
        report_lines.extend(
            [
                "### RGSA",
                "",
                f"- **Run ID**: {rgsa.get('id', 'N/A')}",
                f"- **Final Iteration**: {rgsa.get('summary', {}).get('final_iteration', 'N/A')}",
            ]
        )

        if rgsa["val_ppl"]:
            final_ppl = rgsa["val_ppl"][-1]
            report_lines.append(f"- **Final Val PPL**: {final_ppl:.2f}")

        if rgsa["val_loss"]:
            final_loss = rgsa["val_loss"][-1]
            report_lines.append(f"- **Final Val Loss**: {final_loss:.4f}")

        # RGSA-specific metrics
        if rgsa["teacher_recall"]:
            valid_recall = [r for r in rgsa["teacher_recall"] if not np.isnan(r)]
            if valid_recall:
                final_recall = valid_recall[-1]
                report_lines.append(f"- **Final Teacher Recall**: {final_recall:.4f}")

        if rgsa["routing_entropy"]:
            valid_entropy = [e for e in rgsa["routing_entropy"] if not np.isnan(e)]
            if valid_entropy:
                final_entropy = valid_entropy[-1]
                report_lines.append(f"- **Final Routing Entropy**: {final_entropy:.4f}")

        if rgsa["load_balance"]:
            valid_lb = [lb for lb in rgsa["load_balance"] if not np.isnan(lb)]
            if valid_lb:
                final_lb = valid_lb[-1]
                report_lines.append(f"- **Final Load Balance**: {final_lb:.4f}")

        if rgsa["tokens_per_query_mean"]:
            valid_tpq = [t for t in rgsa["tokens_per_query_mean"] if not np.isnan(t)]
            if valid_tpq:
                final_tpq = valid_tpq[-1]
                report_lines.append(f"- **Final Tokens/Query (mean)**: {final_tpq:.1f}")

        report_lines.append("")

    # Comparison
    if baseline and rgsa and baseline["val_ppl"] and rgsa["val_ppl"]:
        baseline_final = baseline["val_ppl"][-1]
        rgsa_final = rgsa["val_ppl"][-1]
        ppl_diff = rgsa_final - baseline_final
        ppl_pct = (ppl_diff / baseline_final) * 100

        report_lines.extend(
            [
                "## Comparison",
                "",
                f"- **PPL Difference**: {ppl_diff:+.2f} ({ppl_pct:+.1f}%)",
            ]
        )

        if abs(ppl_pct) < 5:
            report_lines.append("- **Status**: RGSA within 5% of baseline (target met)")
        elif ppl_pct > 0:
            report_lines.append(
                f"- **Status**: RGSA {ppl_pct:.1f}% worse than baseline"
            )
        else:
            report_lines.append(
                f"- **Status**: RGSA {abs(ppl_pct):.1f}% better than baseline"
            )

        report_lines.append("")

    # Collapse indicators
    if rgsa:
        report_lines.extend(
            [
                "## Collapse Indicators",
                "",
            ]
        )

        collapse_detected = False

        if rgsa["routing_entropy"]:
            valid_entropy = [e for e in rgsa["routing_entropy"] if not np.isnan(e)]
            if valid_entropy and valid_entropy[-1] < 0.3:
                report_lines.append(
                    f"- **WARNING**: Low routing entropy ({valid_entropy[-1]:.4f}) "
                    "suggests router may be collapsing"
                )
                collapse_detected = True

        if rgsa["load_balance"]:
            valid_lb = [lb for lb in rgsa["load_balance"] if not np.isnan(lb)]
            if valid_lb and valid_lb[-1] < 0.3:
                report_lines.append(
                    f"- **WARNING**: Low load balance ({valid_lb[-1]:.4f}) "
                    "suggests some chunks dominate selection"
                )
                collapse_detected = True

        if rgsa["teacher_recall"]:
            valid_recall = [r for r in rgsa["teacher_recall"] if not np.isnan(r)]
            if valid_recall and valid_recall[-1] < 0.5:
                report_lines.append(
                    f"- **WARNING**: Low teacher recall ({valid_recall[-1]:.4f}) "
                    "suggests router not finding relevant chunks"
                )
                collapse_detected = True

        if not collapse_detected:
            report_lines.append("- No collapse indicators detected")

        report_lines.append("")

    # Write report
    report_content = "\n".join(report_lines)
    output_path = Path(output_dir) / "report.md"
    with open(output_path, "w") as f:
        f.write(report_content)
    print(f"Saved: {output_path}")

    # Also print to stdout
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and report from RGSA comparison runs"
    )
    parser.add_argument("--group", help="W&B group name for comparison runs")
    parser.add_argument("--baseline", help="W&B run ID for baseline")
    parser.add_argument("--rgsa", help="W&B run ID for RGSA")
    parser.add_argument("--local", help="Local output directory with training results")
    parser.add_argument(
        "--output",
        default="rgsa_analysis",
        help="Output directory for plots and report",
    )
    parser.add_argument(
        "--project",
        default="gpt2-rgsa-compare",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        default="mcgrof-citizen",
        help="W&B entity (user or team)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Fetch or load data
    if args.local:
        print(f"Loading local data from: {args.local}")
        baseline_data, rgsa_data = load_local_data(args.local)
    elif args.group or (args.baseline and args.rgsa):
        print("Fetching data from W&B...")
        baseline_data, rgsa_data = fetch_wandb_runs(
            group=args.group,
            baseline_id=args.baseline,
            rgsa_id=args.rgsa,
            project=args.project,
            entity=args.entity,
        )
    else:
        print("Error: Must specify --group, --baseline/--rgsa, or --local")
        sys.exit(1)

    if not baseline_data and not rgsa_data:
        print("Error: No data found")
        sys.exit(1)

    print(f"\nBaseline data: {'Found' if baseline_data else 'Not found'}")
    print(f"RGSA data: {'Found' if rgsa_data else 'Not found'}")

    # Generate plots
    print("\nGenerating plots...")
    plot_val_ppl(baseline_data, rgsa_data, args.output)
    plot_teacher_recall(rgsa_data, args.output)
    plot_entropy_loadbalance(rgsa_data, args.output)
    plot_tokens_per_query(rgsa_data, args.output)

    # Generate report
    print("\nGenerating report...")
    generate_report(baseline_data, rgsa_data, args.output)

    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
