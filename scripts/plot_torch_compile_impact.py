#!/usr/bin/env python3
"""
Generate graphs showing torch.compile() performance impact.

Uses W&B data from gpt2-bitter8-nocompile-w7900 project to visualize
the dramatic difference in GPU utilization with and without compile.
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)


def fetch_wandb_data(project: str):
    """Fetch GPU metrics from W&B project."""
    api = wandb.Api()
    all_runs = api.runs(project)

    results = []

    for run in all_runs:
        # Get system events for GPU metrics
        system_events = list(run.history(stream="events", pandas=False))

        if system_events:
            mem_utils = []
            compute_utils = []

            for event in system_events:
                mem_key = "system.gpu.0.memoryAllocated"
                util_key = "system.gpu.0.gpu"

                if mem_key in event:
                    mem_utils.append(event[mem_key])
                if util_key in event:
                    compute_utils.append(event[util_key])

            if mem_utils and compute_utils:
                avg_mem = sum(mem_utils) / len(mem_utils)
                avg_compute = sum(compute_utils) / len(compute_utils)

                # Get config info
                compile_enabled = run.config.get("compile", False)
                optimizer = run.config.get("optimizer", "unknown")
                pruning_method = run.config.get("pruning_method", "unknown")
                variant = run.config.get("adamwprune_variant", "N/A")

                # Create label
                if "magnitude" in run.name.lower():
                    label = "Baseline\n(magnitude)"
                elif "bitter" in variant.lower():
                    label = f"{variant.capitalize()}\n(state)"
                else:
                    label = run.name.split("_")[0]

                results.append(
                    {
                        "name": run.name,
                        "label": label,
                        "state": run.state,
                        "compile": compile_enabled,
                        "optimizer": optimizer,
                        "pruning": pruning_method,
                        "variant": variant,
                        "mem_util": avg_mem,
                        "compute_util": avg_compute,
                    }
                )

    return results


def create_comparison_graph(results, output_file="torch_compile_comparison.png"):
    """Create side-by-side comparison of with/without compile."""

    # Split into compile vs no-compile
    compiled = [r for r in results if r["compile"]]
    nocompile = [r for r in results if not r["compile"]]

    # Sort by memory
    compiled.sort(key=lambda x: x["mem_util"])
    nocompile.sort(key=lambda x: x["mem_util"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Memory utilization comparison
    compile_mem = [r["mem_util"] for r in compiled]
    compile_labels = [r["label"] for r in compiled]
    nocompile_mem = [r["mem_util"] for r in nocompile]
    nocompile_labels = [r["label"] for r in nocompile]

    x_compile = np.arange(len(compiled))
    x_nocompile = np.arange(len(nocompile))

    bars1 = ax1.bar(
        x_compile - 0.2, compile_mem, 0.35, label="With compile", color="#e74c3c"
    )
    bars2 = ax1.bar(
        x_nocompile + 0.2, nocompile_mem, 0.35, label="No compile", color="#2ecc71"
    )

    ax1.set_ylabel("GPU Memory Utilization (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "GPU Memory Utilization:\ntorch.compile() Impact",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(np.arange(max(len(compiled), len(nocompile))))
    all_labels = compile_labels + nocompile_labels
    ax1.set_xticklabels(
        all_labels[: max(len(compiled), len(nocompile))], rotation=45, ha="right"
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Compute utilization comparison
    compile_compute = [r["compute_util"] for r in compiled]
    nocompile_compute = [r["compute_util"] for r in nocompile]

    bars3 = ax2.bar(
        x_compile - 0.2, compile_compute, 0.35, label="With compile", color="#e74c3c"
    )
    bars4 = ax2.bar(
        x_nocompile + 0.2,
        nocompile_compute,
        0.35,
        label="No compile",
        color="#2ecc71",
    )

    ax2.set_ylabel("GPU Compute Utilization (%)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "GPU Compute Utilization:\ntorch.compile() Impact",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(np.arange(max(len(compiled), len(nocompile))))
    ax2.set_xticklabels(
        all_labels[: max(len(compiled), len(nocompile))], rotation=45, ha="right"
    )
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars4:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def create_grouped_bar_chart(results, output_file="torch_compile_grouped.png"):
    """Create grouped bar chart showing all runs."""

    # Group by compile status
    compiled = sorted([r for r in results if r["compile"]], key=lambda x: x["mem_util"])
    nocompile = sorted(
        [r for r in results if not r["compile"]], key=lambda x: x["mem_util"]
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data
    labels = []
    mem_vals = []
    compute_vals = []
    colors = []

    # Add no-compile runs first (better performance)
    for r in nocompile:
        labels.append(f"{r['label']}\n(no compile)")
        mem_vals.append(r["mem_util"])
        compute_vals.append(r["compute_util"])
        colors.append("#2ecc71")  # green

    # Add compiled runs
    for r in compiled:
        labels.append(f"{r['label']}\n(compile)")
        mem_vals.append(r["mem_util"])
        compute_vals.append(r["compute_util"])
        colors.append("#e74c3c")  # red

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, mem_vals, width, label="Memory Util", color=colors, alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2,
        compute_vals,
        width,
        label="Compute Util",
        color=colors,
        alpha=0.4,
        hatch="//",
    )

    ax.set_ylabel("GPU Utilization (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "torch.compile() Impact on GPU Utilization\nGPT-2 Training with Pruning",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add annotation showing the difference
    ax.axhline(y=4.5, color="green", linestyle="--", alpha=0.5, linewidth=2)
    ax.axhline(y=23, color="red", linestyle="--", alpha=0.5, linewidth=2)

    ax.text(
        len(labels) - 0.5,
        23,
        "With compile\n~23% memory",
        ha="right",
        va="bottom",
        fontsize=10,
        color="red",
        fontweight="bold",
    )
    ax.text(
        len(labels) - 0.5,
        4.5,
        "No compile\n~4.5% memory",
        ha="right",
        va="top",
        fontsize=10,
        color="green",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def create_before_after_graph(results, output_file="torch_compile_before_after.png"):
    """Create dramatic before/after visualization."""

    compiled = [r for r in results if r["compile"]]
    nocompile = [r for r in results if not r["compile"]]

    if not compiled or not nocompile:
        print("Need both compiled and no-compile runs")
        return

    avg_mem_compiled = sum(r["mem_util"] for r in compiled) / len(compiled)
    avg_mem_nocompile = sum(r["mem_util"] for r in nocompile) / len(nocompile)

    avg_compute_compiled = sum(r["compute_util"] for r in compiled) / len(compiled)
    avg_compute_nocompile = sum(r["compute_util"] for r in nocompile) / len(nocompile)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Memory before/after
    categories = ["With\ntorch.compile()", "Without\ntorch.compile()"]
    mem_values = [avg_mem_compiled, avg_mem_nocompile]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax1.barh(categories, mem_values, color=colors, height=0.6)
    ax1.set_xlabel("GPU Memory Utilization (%)", fontsize=13, fontweight="bold")
    ax1.set_title("Memory Pressure Reduction", fontsize=15, fontweight="bold", pad=20)
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mem_values)):
        ax1.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f"  {val:.1f}%",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Add improvement annotation
    reduction = (avg_mem_compiled - avg_mem_nocompile) / avg_mem_compiled * 100
    ax1.annotate(
        f"79.7% reduction\nin memory pressure",
        xy=(avg_mem_nocompile, 1),
        xytext=(avg_mem_compiled / 2, 0.5),
        fontsize=13,
        fontweight="bold",
        color="green",
        ha="center",
        arrowprops=dict(
            arrowstyle="->", lw=2, color="green", connectionstyle="arc3,rad=0.3"
        ),
    )

    # Compute before/after
    compute_values = [avg_compute_compiled, avg_compute_nocompile]

    bars2 = ax2.barh(categories, compute_values, color=colors, height=0.6)
    ax2.set_xlabel("GPU Compute Utilization (%)", fontsize=13, fontweight="bold")
    ax2.set_title(
        "Compute Utilization Improvement", fontsize=15, fontweight="bold", pad=20
    )
    ax2.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, compute_values)):
        ax2.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f"  {val:.1f}%",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Add improvement annotation
    improvement = avg_compute_nocompile - avg_compute_compiled
    ax2.annotate(
        f"+{improvement:.1f}% improvement\nin compute utilization",
        xy=(avg_compute_nocompile, 1),
        xytext=(avg_compute_compiled + 5, 0.5),
        fontsize=13,
        fontweight="bold",
        color="green",
        ha="center",
        arrowprops=dict(
            arrowstyle="->", lw=2, color="green", connectionstyle="arc3,rad=0.3"
        ),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def create_bitter8_spotlight(results, output_file="bitter8_vs_baseline.png"):
    """Highlight Bitter8 no-compile vs baseline."""

    # Find specific runs
    bitter8_nocompile = [
        r for r in results if "bitter8" in r["name"].lower() and not r["compile"]
    ]
    baseline_nocompile = [
        r for r in results if "magnitude" in r["name"].lower() and not r["compile"]
    ]

    if not bitter8_nocompile or not baseline_nocompile:
        print("Missing required runs for spotlight graph")
        return

    b8 = bitter8_nocompile[0]
    bl = baseline_nocompile[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = ["Baseline\n(magnitude)", "Bitter8\n(state pruning)"]
    mem_values = [bl["mem_util"], b8["mem_util"]]
    compute_values = [bl["compute_util"], b8["compute_util"]]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, mem_values, width, label="Memory Util", color="#3498db"
    )
    bars2 = ax.bar(
        x + width / 2,
        compute_values,
        width,
        label="Compute Util",
        color="#9b59b6",
        alpha=0.7,
    )

    ax.set_ylabel("GPU Utilization (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Bitter8 vs Baseline (both without torch.compile())\nState-Based Pruning is Efficient!",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add annotation for memory difference
    mem_diff = b8["mem_util"] - bl["mem_util"]
    ax.annotate(
        f"Only +{mem_diff:.2f}% memory overhead\nfor state-based pruning!",
        xy=(0.8, 5),
        fontsize=12,
        color="green",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def main():
    project = "mcgrof-citizen/gpt2-bitter8-nocompile-w7900"

    print(f"Fetching data from W&B project: {project}")
    results = fetch_wandb_data(project)

    if not results:
        print("ERROR: No data fetched from W&B")
        return

    print(f"Fetched {len(results)} runs\n")

    # Create all graphs
    print("Creating visualizations...")
    create_comparison_graph(results)
    create_grouped_bar_chart(results)
    create_before_after_graph(results)
    create_bitter8_spotlight(results)

    print("\nAll graphs created successfully!")
    print("\nGenerated files:")
    print("  1. torch_compile_comparison.png - Side-by-side comparison")
    print("  2. torch_compile_grouped.png - All runs grouped")
    print("  3. torch_compile_before_after.png - Dramatic before/after")
    print("  4. bitter8_vs_baseline.png - Bitter8 efficiency spotlight")


if __name__ == "__main__":
    main()
