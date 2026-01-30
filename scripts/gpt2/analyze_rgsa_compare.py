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
    "rgsa_static": "RGSA (static)",
    "rgsa_dense": "RGSA (dense ablation)",
    "rgsa_random": "RGSA (random routing)",
    "rgsa_dyn_a04": "RGSA dyn a=0.4",
    "rgsa_dyn_a05": "RGSA dyn a=0.5",
    "rgsa_dyn_a06": "RGSA dyn a=0.6",
    "rgsa_dyn_piecewise": "RGSA dyn piecewise",
    "sweep_t4_w128": "top_b=4, win=128",
    "sweep_t4_w256": "top_b=4, win=256",
    "sweep_t8_w128": "top_b=8, win=128",
    "sweep_t8_w256": "top_b=8, win=256",
    "sweep_t16_w128": "top_b=16, win=128",
    "sweep_t16_w256": "top_b=16, win=256",
}

# Colors for each model type
MODEL_COLORS = {
    "baseline": "#3498db",
    "rgsa": "#e74c3c",
    "rgsa_static": "#e74c3c",
    "rgsa_dense": "#2ecc71",
    "rgsa_random": "#9b59b6",
    "rgsa_dyn_a04": "#f39c12",
    "rgsa_dyn_a05": "#e67e22",
    "rgsa_dyn_a06": "#d35400",
    "rgsa_dyn_piecewise": "#1abc9c",
    "sweep_t4_w128": "#e6194b",
    "sweep_t4_w256": "#3cb44b",
    "sweep_t8_w128": "#4363d8",
    "sweep_t8_w256": "#f58231",
    "sweep_t16_w128": "#911eb4",
    "sweep_t16_w256": "#42d4f4",
}

MODEL_MARKERS = {
    "baseline": "o",
    "rgsa": "s",
    "rgsa_static": "s",
    "rgsa_dense": "^",
    "rgsa_random": "D",
    "rgsa_dyn_a04": "v",
    "rgsa_dyn_a05": "P",
    "rgsa_dyn_a06": "*",
    "rgsa_dyn_piecewise": "X",
    "sweep_t4_w128": "v",
    "sweep_t4_w256": "^",
    "sweep_t8_w128": "<",
    "sweep_t8_w256": ">",
    "sweep_t16_w128": "p",
    "sweep_t16_w256": "h",
}

# Ordered list of all known model tags for consistent plotting
ALL_MODEL_TAGS = [
    "baseline",
    "rgsa",
    "rgsa_static",
    "rgsa_dyn_a04",
    "rgsa_dyn_a05",
    "rgsa_dyn_a06",
    "rgsa_dyn_piecewise",
    "rgsa_dense",
    "rgsa_random",
    "sweep_t4_w128",
    "sweep_t4_w256",
    "sweep_t8_w128",
    "sweep_t8_w256",
    "sweep_t16_w128",
    "sweep_t16_w256",
]

# Sweep configs: map model tag to (top_b, local_window) for Pareto plot
SWEEP_PARAMS = {
    "sweep_t4_w128": (4, 128),
    "sweep_t4_w256": (4, 256),
    "sweep_t8_w128": (8, 128),
    "sweep_t8_w256": (8, 256),
    "sweep_t16_w128": (16, 128),
    "sweep_t16_w256": (16, 256),
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

    # Check sweep tags first (more specific)
    for tag in SWEEP_PARAMS:
        if tag in tags or tag in name:
            return tag

    for tag in [
        "rgsa_random",
        "rgsa_dense",
        "rgsa_dyn_a04",
        "rgsa_dyn_a05",
        "rgsa_dyn_a06",
        "rgsa_dyn_piecewise",
        "rgsa_static",
        "rgsa",
        "baseline",
    ]:
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
        "chunk_size_eff": [],
        "n_chunks_eff": [],
        "tokens_per_query": [],
        "iter_time_ms": [],
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

    # Parse dynamic chunking metrics
    for m in re.finditer(r"Chunk size \(eff\): (\d+), n_chunks: (\d+)", text):
        data["chunk_size_eff"].append(int(m.group(1)))
        data["n_chunks_eff"].append(int(m.group(2)))

    # Parse tokens/query: "Tokens/query: 512.0 (max 768.0)"
    for m in re.finditer(r"Tokens/query: ([\d.]+)", text):
        data["tokens_per_query"].append(float(m.group(1)))

    # Parse iter time: "123.45ms/iter" or "iter time: 123.45ms"
    for m in re.finditer(r"([\d.]+)ms/iter", text):
        data["iter_time_ms"].append(float(m.group(1)))

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

    for model_tag in ALL_MODEL_TAGS:
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


def plot_pareto_curve(by_model: Dict[str, List[Dict]], output_dir: str) -> None:
    """Plot Pareto frontier: val PPL vs tokens/query for sweep configs."""
    if not HAS_MATPLOTLIB:
        return

    sweep_tags = [t for t in SWEEP_PARAMS if t in by_model]
    if not sweep_tags and "baseline" not in by_model:
        print("No sweep data available, skipping Pareto plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    points = []  # (tokens_per_query, ppl_mean, ppl_std, label, color, marker)

    # Baseline point (tokens/query = block_size = 1024 for dense attention)
    if "baseline" in by_model:
        runs = by_model["baseline"]
        best_ppls = []
        for run in runs:
            if run["val_ppl"]:
                best_ppls.append(min(run["val_ppl"]))
        if best_ppls:
            mean_ppl = np.mean(best_ppls)
            std_ppl = np.std(best_ppls) if len(best_ppls) > 1 else 0
            points.append(
                (1024.0, mean_ppl, std_ppl, "Baseline (dense)", "#3498db", "o")
            )

    # Sweep points
    for tag in sweep_tags:
        runs = by_model[tag]
        top_b, local_window = SWEEP_PARAMS[tag]

        best_ppls = []
        tpq_vals = []
        for run in runs:
            if run["val_ppl"]:
                best_ppls.append(min(run["val_ppl"]))
            tpq_vals.extend(run.get("tokens_per_query", []))

        if not best_ppls:
            continue

        mean_ppl = np.mean(best_ppls)
        std_ppl = np.std(best_ppls) if len(best_ppls) > 1 else 0

        # Use measured tokens/query if available, else estimate
        if tpq_vals:
            tpq = np.mean(tpq_vals)
        else:
            # Estimate: top_b * chunk_size + local_window
            tpq = top_b * 64 + local_window

        color = MODEL_COLORS.get(tag, "gray")
        marker = MODEL_MARKERS.get(tag, "o")
        label = MODEL_DISPLAY.get(tag, tag)
        points.append((tpq, mean_ppl, std_ppl, label, color, marker))

    # Also include static/dynamic RGSA if present
    for tag in ["rgsa_static", "rgsa_dyn_a05"]:
        if tag not in by_model:
            continue
        runs = by_model[tag]
        best_ppls = []
        tpq_vals = []
        for run in runs:
            if run["val_ppl"]:
                best_ppls.append(min(run["val_ppl"]))
            tpq_vals.extend(run.get("tokens_per_query", []))
        if not best_ppls:
            continue
        mean_ppl = np.mean(best_ppls)
        std_ppl = np.std(best_ppls) if len(best_ppls) > 1 else 0
        tpq = np.mean(tpq_vals) if tpq_vals else 768.0
        color = MODEL_COLORS.get(tag, "gray")
        marker = MODEL_MARKERS.get(tag, "o")
        label = MODEL_DISPLAY.get(tag, tag)
        points.append((tpq, mean_ppl, std_ppl, label, color, marker))

    if not points:
        print("No Pareto data, skipping plot")
        return

    # Plot each point
    for tpq, ppl, std, label, color, marker in points:
        ax.errorbar(
            tpq,
            ppl,
            yerr=std if std > 0 else None,
            fmt=marker,
            color=color,
            label=label,
            markersize=10,
            capsize=4,
            linewidth=2,
        )
        ax.annotate(
            label,
            (tpq, ppl),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
            color=color,
        )

    # Draw Pareto frontier
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier_x = []
    frontier_y = []
    best_ppl = float("inf")
    for tpq, ppl, _, _, _, _ in sorted_pts:
        if ppl <= best_ppl:
            frontier_x.append(tpq)
            frontier_y.append(ppl)
            best_ppl = ppl
    if len(frontier_x) > 1:
        ax.plot(
            frontier_x,
            frontier_y,
            "k--",
            alpha=0.4,
            linewidth=1.5,
            label="Pareto frontier",
        )

    ax.set_xlabel("Tokens / Query", fontsize=12)
    ax.set_ylabel("Best Val PPL", fontsize=12)
    ax.set_title("Compute-Quality Pareto Frontier", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "pareto_compute_quality.png"
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
    model_tags = [t for t in ALL_MODEL_TAGS if t in by_model]
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

    # Dynamic chunking stats
    has_chunk_stats = False
    for tag in ALL_MODEL_TAGS:
        runs = by_model.get(tag, [])
        for run in runs:
            if run.get("chunk_size_eff"):
                has_chunk_stats = True
                break

    if has_chunk_stats:
        lines.extend(["## Dynamic Chunking Statistics", ""])
        for tag in ALL_MODEL_TAGS:
            runs = by_model.get(tag, [])
            if not runs:
                continue
            all_cs = []
            all_nc = []
            for run in runs:
                all_cs.extend(run.get("chunk_size_eff", []))
                all_nc.extend(run.get("n_chunks_eff", []))
            if all_cs:
                display = MODEL_DISPLAY.get(tag, tag)
                cs_arr = np.array(all_cs)
                nc_arr = np.array(all_nc)
                lines.append(
                    f"- **{display}**: chunk_size_eff "
                    f"mean={cs_arr.mean():.1f}, "
                    f"min={cs_arr.min()}, max={cs_arr.max()}, "
                    f"n_chunks mean={nc_arr.mean():.1f}"
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
    parser.add_argument(
        "--local", "--dir", help="Local output directory with training logs"
    )
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
    plot_pareto_curve(by_model, args.output)

    print("\nGenerating report...")
    generate_report(by_model, args.output)

    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
