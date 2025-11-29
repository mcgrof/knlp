#!/usr/bin/env python3
"""
Plot KVSplice extreme compression results.

Generates publication-quality visualizations comparing 70% and 90%
compression ratios across different hardware.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from W&B runs
data = {
    "90_h100": {
        "name": "H100: 90% Compression",
        "baseline_loss": 2.1613,
        "baseline_ppl": 8.68,
        "kvsplice_loss": 2.1604,
        "kvsplice_ppl": 8.67,
        "dims": "256→26",
        "compression": 18,
    },
    "70_a100": {
        "name": "A100 40G: 70% Compression",
        "baseline_loss": 3.7476,
        "baseline_ppl": 42.42,
        "kvsplice_loss": 3.7091,
        "kvsplice_ppl": 40.82,
        "dims": "256→77",
        "compression": 20,
    },
}


def plot_perplexity_comparison():
    """Plot perplexity comparison across compression ratios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    configs = ["90_h100", "70_a100"]
    labels = ["H100\n90% (18x)", "A100\n70% (20x)"]
    x = np.arange(len(configs))
    width = 0.35

    # Plot 1: Validation Loss
    baseline_losses = [data[c]["baseline_loss"] for c in configs]
    kvsplice_losses = [data[c]["kvsplice_loss"] for c in configs]

    bars1 = ax1.bar(
        x - width / 2, baseline_losses, width, label="MLA Baseline", color="#e74c3c"
    )
    bars2 = ax1.bar(
        x + width / 2, kvsplice_losses, width, label="MLA + KVSplice", color="#27ae60"
    )

    ax1.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_title(
        "KVSplice Improves Perplexity\nAcross Compression Ratios",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Plot 2: Perplexity
    baseline_ppls = [data[c]["baseline_ppl"] for c in configs]
    kvsplice_ppls = [data[c]["kvsplice_ppl"] for c in configs]

    bars1 = ax2.bar(
        x - width / 2, baseline_ppls, width, label="MLA Baseline", color="#e74c3c"
    )
    bars2 = ax2.bar(
        x + width / 2, kvsplice_ppls, width, label="MLA + KVSplice", color="#27ae60"
    )

    ax2.set_ylabel("Validation Perplexity", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Lower is Better\n(Green bars win)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig("kvsplice_compression_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: kvsplice_compression_comparison.png")


def plot_improvement_delta():
    """Plot improvement delta showing KVSplice gains."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = ["90_h100", "70_a100"]
    labels = [
        "H100: 90% Compression\n(256→26, 18x)",
        "A100: 70% Compression\n(256→77, 20x)",
    ]

    # Calculate improvement percentages
    improvements = []
    for c in configs:
        baseline = data[c]["baseline_ppl"]
        kvsplice = data[c]["kvsplice_ppl"]
        improvement = (baseline - kvsplice) / baseline * 100
        improvements.append(improvement)

    x = np.arange(len(configs))
    bars = ax.barh(x, improvements, color="#27ae60", edgecolor="black", linewidth=1.5)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Perplexity Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "KVSplice Beats Baseline at Extreme Compression\n"
        "Positive = Better than MLA Baseline",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(
            val + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"+{val:.2f}%",
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("kvsplice_improvement_delta.png", dpi=300, bbox_inches="tight")
    print("Saved: kvsplice_improvement_delta.png")


def plot_compression_vs_quality():
    """Plot compression ratio vs quality trade-off."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot baselines
    for i, c in enumerate(["90_h100", "70_a100"]):
        d = data[c]
        color = "#3498db" if "h100" in c else "#9b59b6"
        marker = "o" if "h100" in c else "s"

        # Baseline point
        ax.scatter(
            6,
            d["baseline_ppl"],
            s=200,
            color=color,
            marker=marker,
            alpha=0.5,
            label=f'{d["name"].split(":")[0]} Baseline (MLA only)',
        )

        # KVSplice point
        ax.scatter(
            d["compression"],
            d["kvsplice_ppl"],
            s=200,
            color=color,
            marker=marker,
            label=d["name"],
        )

        # Draw arrow
        ax.annotate(
            "",
            xy=(d["compression"], d["kvsplice_ppl"]),
            xytext=(6, d["baseline_ppl"]),
            arrowprops=dict(
                arrowstyle="->", lw=2, color=color, alpha=0.7, linestyle="--"
            ),
        )

        # Add improvement annotation
        improvement = (d["baseline_ppl"] - d["kvsplice_ppl"]) / d["baseline_ppl"] * 100
        mid_x = (6 + d["compression"]) / 2
        mid_y = (d["baseline_ppl"] + d["kvsplice_ppl"]) / 2
        ax.text(
            mid_x,
            mid_y,
            f'+{improvement:.2f}%\n{d["dims"]}',
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color),
        )

    ax.set_xlabel("Total Compression Ratio (x)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Validation Perplexity (lower is better)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "KVSplice: Higher Compression + Better Quality\n"
        "Both configurations improve perplexity while increasing compression",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4, 22)

    # Invert y-axis so lower perplexity is higher on graph
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("kvsplice_compression_vs_quality.png", dpi=300, bbox_inches="tight")
    print("Saved: kvsplice_compression_vs_quality.png")


def main():
    print("=" * 80)
    print("Generating KVSplice Compression Visualizations")
    print("=" * 80)

    print("\n1. Generating perplexity comparison...")
    plot_perplexity_comparison()

    print("\n2. Generating improvement delta...")
    plot_improvement_delta()

    print("\n3. Generating compression vs quality...")
    plot_compression_vs_quality()

    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - kvsplice_compression_comparison.png")
    print("  - kvsplice_improvement_delta.png")
    print("  - kvsplice_compression_vs_quality.png")


if __name__ == "__main__":
    main()
