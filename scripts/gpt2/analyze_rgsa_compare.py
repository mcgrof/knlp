#!/usr/bin/env python3
"""
analyze_rgsa_compare.py - Generate plots and report from RGSA comparison runs

Supports multi-seed runs with mean +/- std aggregation, and ablation
configs (rgsa_dense, rgsa_random).

Usage:
    python scripts/gpt2/analyze_rgsa_compare.py --group <wandb_group>
    python scripts/gpt2/analyze_rgsa_compare.py --local <output_dir>

Outputs:
    - compare_val_ppl.png: Mean +/- std validation perplexity vs steps
    - compare_teacher_recall.png: Teacher top-k recall vs steps
    - compare_entropy_loadbalance.png: Routing entropy and load balance
    - report.md: Summary table with mean/std at each eval checkpoint
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
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


# Model tag to display name mapping
MODEL_DISPLAY = {
    "baseline": "Baseline GPT-2",
    "rgsa": "RGSA",
    "rgsa_dense": "RGSA (dense ablation)",
    "rgsa_random": "RGSA (random routing)",
}

# Colors for each model type
MODEL_COLORS = {
    "baseline": "#3498db",
    "rgsa": "#e74c3c",
    "rgsa_dense": "#2ecc71",
    "rgsa_random": "#9b59b6",
}

MODEL_MARKERS = {
    "baseline": "o",
    "rgsa": "s",
    "rgsa_dense": "^",
    "rgsa_random": "D",
}


def _extract_run_data(run) -> Dict:
    """Extract relevant metrics from a W&B run."""
    history = run.history(samples=10000)

    data = {
        "name": run.name,
        "id": run.id,
        "tags": run.tags or [],
        "config": dict(run.config),
        "summary": dict(run.summary),
        "eval_iterations": [],
        "val_ppl": [],
        "val_loss": [],
    }

    for _, row in history.iterrows():
        if "val_perplexity" in row and not np.isnan(row.get("val_perplexity", np.nan)):
            it = int(row.get("iteration", 0))
            data["eval_iterations"].append(it)
            data["val_ppl"].append(row["val_perplexity"])
            data["val_loss"].append(row.get("val_loss", np.nan))

    return data


def _identify_model_tag(run_data: Dict) -> str:
    """Identify model tag from run tags or name."""
    tags = run_data.get("tags", [])
    name = run_data.get("name", "")

    for tag in ["rgsa_random", "rgsa_dense", "rgsa", "baseline"]:
        if tag in tags or tag in name:
            return tag
    return "unknown"


def _identify_seed(run_data: Dict) -> Optional[int]:
    """Extract seed from run tags or name."""
    tags = run_data.get("tags", [])
    name = run_data.get("name", "")

    for tag in tags:
        m = re.match(r"seed(\d+)", tag)
        if m:
            return int(m.group(1))

    m = re.search(r"seed(\d+)", name)
    if m:
        return int(m.group(1))

    return None


def fetch_wandb_group(
    group: str,
    project: str = "gpt2-rgsa-compare",
    entity: str = "mcgrof-citizen",
) -> Dict[str, List[Dict]]:
    """Fetch all runs in a W&B group, organized by model tag."""
    if not HAS_WANDB:
        print("Error: wandb not available")
        return {}

    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": group},
        order="-created_at",
    )

    by_model = defaultdict(list)
    for run in runs:
        data = _extract_run_data(run)
        tag = _identify_model_tag(data)
        seed = _identify_seed(data)
        data["seed"] = seed
        by_model[tag].append(data)

    return dict(by_model)


def parse_log_file(log_path: Path) -> Dict:
    """Parse a training log file to extract eval metrics."""
    data = {
        "eval_iterations": [],
        "val_ppl": [],
        "val_loss": [],
        "teacher_recall": [],
        "routing_entropy": [],
        "load_balance": [],
        "candidates_mean": [],
    }

    text = log_path.read_text()
    # Strip ANSI escape codes
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    # Parse eval lines: "Eval @ iter N: train X, val Y, ppl Z"
    for m in re.finditer(
        r"Eval @ iter (\d+): train [\d.]+, val ([\d.]+), ppl ([\d.]+)", text
    ):
        data["eval_iterations"].append(int(m.group(1)))
        data["val_loss"].append(float(m.group(2)))
        data["val_ppl"].append(float(m.group(3)))

    # Parse RGSA diagnostics
    for m in re.finditer(r"Teacher top-k recall: ([\d.]+)", text):
        data["teacher_recall"].append(float(m.group(1)))
    for m in re.finditer(r"Routing entropy: ([\d.]+)", text):
        data["routing_entropy"].append(float(m.group(1)))
    for m in re.finditer(r"Load balance: ([\d.]+)", text):
        data["load_balance"].append(float(m.group(1)))
    for m in re.finditer(r"Candidates mean: ([\d.]+)", text):
        data["candidates_mean"].append(float(m.group(1)))

    return data


def load_local_data(output_dir: str) -> Dict[str, List[Dict]]:
    """Load data from local log files, organized by model tag."""
    output_path = Path(output_dir)
    by_model = defaultdict(list)

    # Find all log files matching pattern: <model_tag>_seed<N>.log
    # or <model_tag>.log (single seed)
    for log_file in sorted(output_path.glob("*.log")):
        name = log_file.stem  # e.g. "rgsa_seed1" or "baseline_seed42"

        # Determine model tag and seed
        seed_match = re.search(r"_seed(\d+)$", name)
        if seed_match:
            seed = int(seed_match.group(1))
            model_tag = name[: seed_match.start()]
        else:
            seed = None
            model_tag = name

        data = parse_log_file(log_file)
        data["name"] = name
        data["id"] = name
        data["seed"] = seed
        data["log_file"] = str(log_file)

        if data["eval_iterations"]:
            by_model[model_tag].append(data)

    return dict(by_model)


def aggregate_by_iteration(
    runs: List[Dict], metric_key: str = "val_ppl"
) -> Dict[int, List[float]]:
    """Aggregate a metric across runs at each iteration."""
    by_iter = defaultdict(list)
    for run in runs:
        iters = run.get("eval_iterations", [])
        vals = run.get(metric_key, [])
        for it, val in zip(iters, vals):
            by_iter[it].append(val)
    return dict(by_iter)


def plot_val_ppl_multiseed(by_model: Dict[str, List[Dict]], output_dir: str) -> None:
    """Plot mean +/- std validation perplexity for each model."""
    if not HAS_MATPLOTLIB:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_tag in ["baseline", "rgsa", "rgsa_dense", "rgsa_random"]:
        runs = by_model.get(model_tag, [])
        if not runs:
            continue

        agg = aggregate_by_iteration(runs, "val_ppl")
        if not agg:
            continue

        iters = sorted(agg.keys())
        means = [np.mean(agg[i]) for i in iters]
        stds = [np.std(agg[i]) if len(agg[i]) > 1 else 0 for i in iters]

        color = MODEL_COLORS.get(model_tag, "gray")
        marker = MODEL_MARKERS.get(model_tag, "o")
        label = MODEL_DISPLAY.get(model_tag, model_tag)
        n_seeds = max(len(agg[i]) for i in iters)
        if n_seeds > 1:
            label += f" (n={n_seeds})"

        # Full scale (log)
        ax1.plot(
            iters,
            means,
            color=color,
            marker=marker,
            label=label,
            linewidth=2,
            markersize=6,
        )
        if any(s > 0 for s in stds):
            ax1.fill_between(
                iters,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2,
                color=color,
            )

        # Zoomed (skip iter 0)
        zoom_iters = [i for i in iters if i > 0]
        zoom_means = [np.mean(agg[i]) for i in zoom_iters]
        zoom_stds = [np.std(agg[i]) if len(agg[i]) > 1 else 0 for i in zoom_iters]

        ax2.plot(
            zoom_iters,
            zoom_means,
            color=color,
            marker=marker,
            label=label,
            linewidth=2,
            markersize=6,
        )
        if any(s > 0 for s in zoom_stds):
            ax2.fill_between(
                zoom_iters,
                [m - s for m, s in zip(zoom_means, zoom_stds)],
                [m + s for m, s in zip(zoom_means, zoom_stds)],
                alpha=0.2,
                color=color,
            )

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Validation Perplexity", fontsize=12)
    ax1.set_title("Val PPL (all evals)", fontsize=14)
    ax1.set_yscale("log")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Validation Perplexity", fontsize=12)
    ax2.set_title("Val PPL (zoomed, iter > 0)", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "compare_val_ppl.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def plot_routing_diagnostics(by_model: Dict[str, List[Dict]], output_dir: str) -> None:
    """Plot routing diagnostics for RGSA variants."""
    if not HAS_MATPLOTLIB:
        return

    # Only plot for models that have routing diagnostics
    has_diag = False
    for tag in ["rgsa", "rgsa_random"]:
        runs = by_model.get(tag, [])
        for run in runs:
            if run.get("teacher_recall"):
                has_diag = True
                break

    if not has_diag:
        print("No routing diagnostics data available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for tag in ["rgsa", "rgsa_random"]:
        runs = by_model.get(tag, [])
        if not runs:
            continue

        color = MODEL_COLORS.get(tag, "gray")
        label = MODEL_DISPLAY.get(tag, tag)

        # Teacher recall
        recall_agg = aggregate_by_iteration(runs, "teacher_recall")
        if recall_agg:
            iters = sorted(recall_agg.keys())
            means = [np.mean(recall_agg[i]) for i in iters]
            axes[0].plot(
                iters,
                means,
                color=color,
                marker="o",
                label=label,
                linewidth=2,
                markersize=6,
            )

        # Routing entropy
        entropy_agg = aggregate_by_iteration(runs, "routing_entropy")
        if entropy_agg:
            iters = sorted(entropy_agg.keys())
            means = [np.mean(entropy_agg[i]) for i in iters]
            axes[1].plot(
                iters,
                means,
                color=color,
                marker="o",
                label=label,
                linewidth=2,
                markersize=6,
            )

        # Load balance
        lb_agg = aggregate_by_iteration(runs, "load_balance")
        if lb_agg:
            iters = sorted(lb_agg.keys())
            means = [np.mean(lb_agg[i]) for i in iters]
            axes[2].plot(
                iters,
                means,
                color=color,
                marker="o",
                label=label,
                linewidth=2,
                markersize=6,
            )

    axes[0].axhline(y=0.7, color="r", linestyle="--", alpha=0.5, label="Target")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Teacher Top-k Recall")
    axes[0].set_title("Teacher Recall")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Collapse")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Routing Entropy")
    axes[1].set_title("Routing Entropy")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Collapse")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Load Balance")
    axes[2].set_title("Load Balance")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "compare_rgsa_diagnostics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(by_model: Dict[str, List[Dict]], output_dir: str) -> None:
    """Generate markdown report with mean/std tables."""
    lines = [
        "# RGSA Multi-Seed Comparison Report",
        "",
    ]

    # Determine common eval iterations
    all_iters = set()
    for runs in by_model.values():
        for run in runs:
            all_iters.update(run.get("eval_iterations", []))
    eval_iters = sorted(all_iters)
    if not eval_iters:
        lines.append("No evaluation data found.")
        _write_report(lines, output_dir)
        return

    # Summary table
    lines.extend(
        [
            "## Val PPL Summary (mean +/- std)",
            "",
        ]
    )

    # Header
    model_tags = [
        t for t in ["baseline", "rgsa", "rgsa_dense", "rgsa_random"] if t in by_model
    ]
    header = "| Iteration |"
    separator = "|-----------|"
    for tag in model_tags:
        n = len(by_model[tag])
        display = MODEL_DISPLAY.get(tag, tag)
        header += f" {display} (n={n}) |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    # Rows for each eval iteration
    for it in eval_iters:
        row = f"| {it} |"
        for tag in model_tags:
            agg = aggregate_by_iteration(by_model[tag], "val_ppl")
            vals = agg.get(it, [])
            if vals:
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0
                if std > 0:
                    row += f" {mean:.2f} +/- {std:.2f} |"
                else:
                    row += f" {mean:.2f} |"
            else:
                row += " - |"
        lines.append(row)

    lines.append("")

    # Final comparison at last common iteration
    common_iters = eval_iters
    if len(model_tags) > 1 and common_iters:
        # Find last iteration where all models have data
        last_common = None
        for it in reversed(common_iters):
            all_have = True
            for tag in model_tags:
                agg = aggregate_by_iteration(by_model[tag], "val_ppl")
                if it not in agg:
                    all_have = False
                    break
            if all_have:
                last_common = it
                break

        if last_common is not None:
            lines.extend(
                [
                    f"## Comparison at Iteration {last_common}",
                    "",
                ]
            )

            baseline_agg = aggregate_by_iteration(
                by_model.get("baseline", []), "val_ppl"
            )
            baseline_vals = baseline_agg.get(last_common, [])
            baseline_mean = np.mean(baseline_vals) if baseline_vals else None

            for tag in model_tags:
                agg = aggregate_by_iteration(by_model[tag], "val_ppl")
                vals = agg.get(last_common, [])
                if not vals:
                    continue
                mean = np.mean(vals)
                std = np.std(vals) if len(vals) > 1 else 0
                display = MODEL_DISPLAY.get(tag, tag)

                if std > 0:
                    lines.append(f"- **{display}**: {mean:.2f} +/- {std:.2f}")
                else:
                    lines.append(f"- **{display}**: {mean:.2f}")

                if baseline_mean and tag != "baseline":
                    pct = ((mean - baseline_mean) / baseline_mean) * 100
                    direction = "better" if pct < 0 else "worse"
                    lines.append(f"  - vs baseline: {pct:+.1f}% ({direction})")

            lines.append("")

    # Ablation interpretation
    if "rgsa" in by_model and ("rgsa_dense" in by_model or "rgsa_random" in by_model):
        lines.extend(
            [
                "## Ablation Interpretation",
                "",
            ]
        )

        # Check if rgsa > rgsa_random (retrieval gating matters)
        if "rgsa_random" in by_model and common_iters:
            last_it = common_iters[-1]
            rgsa_agg = aggregate_by_iteration(by_model["rgsa"], "val_ppl")
            random_agg = aggregate_by_iteration(by_model["rgsa_random"], "val_ppl")
            rgsa_vals = rgsa_agg.get(last_it, [])
            random_vals = random_agg.get(last_it, [])

            if rgsa_vals and random_vals:
                rgsa_mean = np.mean(rgsa_vals)
                random_mean = np.mean(random_vals)
                if rgsa_mean < random_mean:
                    lines.append(
                        "- RGSA (learned) outperforms RGSA (random): "
                        "learned routing contributes to quality."
                    )
                else:
                    lines.append(
                        "- RGSA (learned) does NOT outperform RGSA (random): "
                        "learned routing may not be the causal mechanism."
                    )

        if "rgsa_dense" in by_model and common_iters:
            last_it = common_iters[-1]
            rgsa_agg = aggregate_by_iteration(by_model["rgsa"], "val_ppl")
            dense_agg = aggregate_by_iteration(by_model["rgsa_dense"], "val_ppl")
            rgsa_vals = rgsa_agg.get(last_it, [])
            dense_vals = dense_agg.get(last_it, [])

            if rgsa_vals and dense_vals:
                rgsa_mean = np.mean(rgsa_vals)
                dense_mean = np.mean(dense_vals)
                if rgsa_mean < dense_mean:
                    lines.append(
                        "- RGSA (learned) outperforms RGSA (dense): "
                        "routing computation contributes beyond param count."
                    )
                else:
                    lines.append(
                        "- RGSA (learned) does NOT outperform RGSA (dense): "
                        "extra params alone may explain quality gain."
                    )

        lines.append("")

    # Routing diagnostics summary
    for tag in ["rgsa", "rgsa_random"]:
        runs = by_model.get(tag, [])
        if not runs:
            continue

        has_recall = any(r.get("teacher_recall") for r in runs)
        if not has_recall:
            continue

        display = MODEL_DISPLAY.get(tag, tag)
        lines.extend([f"## {display} Routing Diagnostics", ""])

        # Get last eval values
        for metric_name, metric_key in [
            ("Teacher Recall", "teacher_recall"),
            ("Routing Entropy", "routing_entropy"),
            ("Load Balance", "load_balance"),
        ]:
            agg = aggregate_by_iteration(runs, metric_key)
            if not agg:
                continue
            last_it = max(agg.keys())
            vals = agg[last_it]
            mean = np.mean(vals)
            lines.append(f"- {metric_name} @ iter {last_it}: {mean:.4f}")

        lines.append("")

    _write_report(lines, output_dir)


def _write_report(lines: List[str], output_dir: str) -> None:
    """Write report to file and stdout."""
    report_content = "\n".join(lines)
    output_path = Path(output_dir) / "report.md"
    with open(output_path, "w") as f:
        f.write(report_content)
    print(f"Saved: {output_path}")
    print("\n" + "=" * 60)
    print(report_content)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and report from RGSA comparison runs"
    )
    parser.add_argument("--group", help="W&B group name for comparison runs")
    parser.add_argument("--local", help="Local output directory with training logs")
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

    os.makedirs(args.output, exist_ok=True)

    if args.local:
        print(f"Loading local data from: {args.local}")
        by_model = load_local_data(args.local)
    elif args.group:
        print(f"Fetching data from W&B group: {args.group}")
        by_model = fetch_wandb_group(
            group=args.group,
            project=args.project,
            entity=args.entity,
        )
    else:
        print("Error: Must specify --group or --local")
        sys.exit(1)

    if not by_model:
        print("Error: No data found")
        sys.exit(1)

    print(f"\nModels found: {list(by_model.keys())}")
    for tag, runs in by_model.items():
        seeds = [r.get("seed") for r in runs]
        print(f"  {tag}: {len(runs)} runs (seeds: {seeds})")

    print("\nGenerating plots...")
    plot_val_ppl_multiseed(by_model, args.output)
    plot_routing_diagnostics(by_model, args.output)

    print("\nGenerating report...")
    generate_report(by_model, args.output)

    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
