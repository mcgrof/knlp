#!/usr/bin/env python3
"""
Plot Reciprocal Attention (RA) comparison results.

Generates publication-quality plots comparing:
- Baseline GPT-2
- SDPA Gate (Qwen3-style)
- Reciprocal Attention (RA)

Results from B200x4 2-hour FineWebEdu training runs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_quality_comparison():
    """Create combined PPL and HellaSwag comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    variants = ["Baseline\nGPT-2", "SDPA Gate\n(Qwen3)", "RA\n(middle layers)"]
    ppl_values = [72.5, 71.8, 68.9]
    hella_values = [28, 28.5, 30]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Perplexity subplot
    bars1 = ax1.bar(variants, ppl_values, color=colors, width=0.6, edgecolor="black")
    for bar, val in zip(bars1, ppl_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylabel("Validation Perplexity", fontsize=12)
    ax1.set_title("Perplexity (lower is better)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 85)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add improvement annotation
    ax1.annotate(
        "-5%",
        xy=(2, 68.9),
        xytext=(1, 75),
        fontsize=14,
        fontweight="bold",
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    # HellaSwag subplot
    bars2 = ax2.bar(variants, hella_values, color=colors, width=0.6, edgecolor="black")
    for bar, val in zip(bars2, hella_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax2.set_ylabel("HellaSwag Accuracy (%)", fontsize=12)
    ax2.set_title("HellaSwag (higher is better)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 40)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add improvement annotation
    ax2.annotate(
        "+2 pts",
        xy=(2, 30),
        xytext=(1, 34),
        fontsize=14,
        fontweight="bold",
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    fig.suptitle(
        "Reciprocal Attention vs Baseline and SDPA Gate\n(B200x4, 2hr FineWebEdu Training)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig


def create_fim_trace_plot():
    """Create FIM trace by layer visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = [0, 3, 6, 9, 11]
    layer_names = ["layer0", "layer3", "layer6", "layer9", "layer11"]
    fim_trace = [0.9551, 0.8823, 0.8191, 0.7156, 0.6215]

    # Color gradient from red (high trace) to green (low trace)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(layers)))

    bars = ax.bar(layer_names, fim_trace, color=colors, width=0.6, edgecolor="black")

    # Add value labels
    for bar, val in zip(bars, fim_trace):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add horizontal regions
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, linewidth=2)
    ax.axhline(y=0.75, color="orange", linestyle="--", alpha=0.5, linewidth=2)

    # Add region labels
    ax.text(4.7, 0.92, "CRITICAL\n(protect)", fontsize=10, color="red", ha="center")
    ax.text(4.7, 0.78, "MODERATE", fontsize=10, color="orange", ha="center")
    ax.text(4.7, 0.65, "SAFE FOR RA", fontsize=10, color="green", ha="center")

    # Highlight RA layers
    ax.axvspan(1.5, 3.5, alpha=0.1, color="green", label="RA applied (layers 5-7)")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean FIM Trace", fontsize=12)
    ax.set_title(
        "FIM Trace by Layer: Justifying Middle Layer RA Application\n(H100, 3-hour baseline MLA run)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left")

    plt.tight_layout()
    return fig


def create_gpu_memory_plot():
    """Create GPU memory comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    variants = ["Baseline\nGPT-2", "SDPA Gate\n(Qwen3)", "RA\n(middle layers)"]
    memory_gb = [14.2, 14.5, 14.8]
    overhead_pct = [0, 2.1, 4.2]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(variants, memory_gb, color=colors, width=0.6, edgecolor="black")

    # Add value labels with overhead
    for bar, mem, overhead in zip(bars, memory_gb, overhead_pct):
        label = f"{mem:.1f} GB"
        if overhead > 0:
            label += f"\n(+{overhead:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("GPU Memory (GB)", fontsize=12)
    ax.set_title(
        "GPU Memory Consumption\n(B200x4, per-GPU average)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add note
    ax.text(
        0.98,
        0.02,
        "RA adds ~4% memory overhead",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    return fig


def create_speed_comparison():
    """Create inference speed comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    variants = ["Baseline\nGPT-2", "SDPA Gate\n(Qwen3)", "RA\n(middle layers)"]
    ms_per_iter = [285, 295, 320]
    slowdown_pct = [0, 3.5, 12.3]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # Reversed: faster is better

    bars = ax.bar(variants, ms_per_iter, color=colors, width=0.6, edgecolor="black")

    # Add value labels with slowdown
    for bar, ms, slowdown in zip(bars, ms_per_iter, slowdown_pct):
        label = f"{ms} ms"
        if slowdown > 0:
            label += f"\n(+{slowdown:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Time per Iteration (ms)", fontsize=12)
    ax.set_title(
        "Training Speed Comparison\n(lower is better)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 380)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add tradeoff note
    ax.text(
        0.98,
        0.98,
        "RA: 12% slower for 5% better PPL",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    return fig


def create_tradeoff_scatter():
    """Create quality vs speed tradeoff scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data: (speed_overhead_pct, ppl_improvement_pct, hella_improvement, label)
    data = [
        (0, 0, 0, "Baseline GPT-2"),
        (3.5, 1.0, 0.5, "SDPA Gate"),
        (12.3, 5.0, 2.0, "RA"),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (speed, ppl, hella, label), color in zip(data, colors):
        # Size based on HellaSwag improvement
        size = 300 + hella * 150
        ax.scatter(speed, ppl, s=size, c=color, edgecolors="black", linewidth=2)
        ax.annotate(
            f"{label}\n(+{hella}% HellaSwag)",
            (speed, ppl),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Speed Overhead (%)", fontsize=12)
    ax.set_ylabel("Perplexity Improvement (%)", fontsize=12)
    ax.set_title(
        "Quality vs Speed Tradeoff\n(Bubble size = HellaSwag improvement)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(-2, 16)
    ax.set_ylim(-0.5, 7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--")

    # Add Pareto frontier
    ax.plot([0, 3.5, 12.3], [0, 1.0, 5.0], "g--", linewidth=2, alpha=0.5)
    ax.text(8, 3.5, "Pareto frontier", fontsize=10, color="green", style="italic")

    plt.tight_layout()
    return fig


def main():
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Reciprocal Attention comparison plots...")

    fig_quality = create_quality_comparison()
    fig_quality.savefig(output_dir / "ra_quality_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'ra_quality_comparison.png'}")

    fig_fim = create_fim_trace_plot()
    fig_fim.savefig(output_dir / "ra_fim_trace.png", dpi=300)
    print(f"  Saved: {output_dir / 'ra_fim_trace.png'}")

    fig_memory = create_gpu_memory_plot()
    fig_memory.savefig(output_dir / "ra_gpu_memory.png", dpi=300)
    print(f"  Saved: {output_dir / 'ra_gpu_memory.png'}")

    fig_speed = create_speed_comparison()
    fig_speed.savefig(output_dir / "ra_speed_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'ra_speed_comparison.png'}")

    fig_tradeoff = create_tradeoff_scatter()
    fig_tradeoff.savefig(output_dir / "ra_tradeoff.png", dpi=300)
    print(f"  Saved: {output_dir / 'ra_tradeoff.png'}")

    plt.close("all")
    print("\nDone! Plots saved to docs/images/")


if __name__ == "__main__":
    main()
