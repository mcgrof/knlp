#!/usr/bin/env python3
"""
Generate PNG visualizations for RA fixed pattern results.
Compares A100 and W7900 GPU results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication-quality defaults
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13

# Data from W&B
a100_data = {
    "B0": 964.50,
    "RAEARLY0": 897.60,
    "RALATE0": 880.12,
    "RAALL0": 887.30,
}

w7900_data = {
    "B0": 335.94,
    "RAEARLY0": 313.75,
    "RALATE0": 316.04,
    "RAALL0": 322.88,
}

# Labels
labels = [
    "Baseline\n(B0)",
    "Early Layers\nReciprocal\n(RAEARLY)",
    "Late Layers\nReciprocal\n(RALATE)",
    "All Layers\nReciprocal\n(RAALL)",
]
short_labels = ["B0", "RAEARLY", "RALATE", "RAALL"]

# Colors
color_baseline = "#95a5a6"
color_ra = "#3498db"
color_best = "#27ae60"


def plot_perplexity_comparison():
    """Side-by-side comparison of A100 and W7900 results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # A100
    a100_values = list(a100_data.values())
    colors_a100 = [color_baseline] + [
        color_best if i == 2 else color_ra for i in range(1, 4)
    ]
    bars1 = ax1.bar(
        range(4),
        a100_values,
        color=colors_a100,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, a100_values)):
        height = bar.get_height()
        improvement = ""
        if i > 0:
            delta = ((val - a100_values[0]) / a100_values[0]) * 100
            improvement = f"\n({delta:+.1f}%)"
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}{improvement}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_ylabel("Validation Perplexity", fontsize=11, fontweight="bold")
    ax1.set_title(
        "A100 GPU (40GB)\ngrad_acc=16, eff_batch=512, 201 iters",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_ylim(0, max(a100_values) * 1.15)

    # W7900
    w7900_values = list(w7900_data.values())
    colors_w7900 = [color_baseline] + [
        color_best if i == 0 else color_ra for i in range(3)
    ]
    bars2 = ax2.bar(
        range(4),
        w7900_values,
        color=colors_w7900,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, w7900_values)):
        height = bar.get_height()
        improvement = ""
        if i > 0:
            delta = ((val - w7900_values[0]) / w7900_values[0]) * 100
            improvement = f"\n({delta:+.1f}%)"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}{improvement}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_ylabel("Validation Perplexity", fontsize=11, fontweight="bold")
    ax2.set_title(
        "W7900 GPU (48GB)\ngrad_acc=8, eff_batch=256, 601 iters",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_ylim(0, max(w7900_values) * 1.15)

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_baseline, label="Baseline GPT-2"),
        mpatches.Patch(color=color_ra, label="RA Pattern"),
        mpatches.Patch(color=color_best, label="Best RA Pattern"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(
        "docs/images/gpt2_ra_fixed_patterns_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("✓ Generated: docs/images/gpt2_ra_fixed_patterns_comparison.png")
    plt.close()


def plot_relative_improvements():
    """Show relative improvements vs baseline for both GPUs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate improvements
    a100_improvements = [
        (a100_data[k] - a100_data["B0"]) / a100_data["B0"] * 100
        for k in ["RAEARLY0", "RALATE0", "RAALL0"]
    ]
    w7900_improvements = [
        (w7900_data[k] - w7900_data["B0"]) / w7900_data["B0"] * 100
        for k in ["RAEARLY0", "RALATE0", "RAALL0"]
    ]

    x = np.arange(3)
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        a100_improvements,
        width,
        label="A100 (grad_acc=16)",
        color="#e74c3c",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )
    bars2 = ax.bar(
        x + width / 2,
        w7900_improvements,
        width,
        label="W7900 (grad_acc=8)",
        color="#3498db",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom" if height < 0 else "top",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel(
        "Perplexity Improvement vs Baseline (%)", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        "RA Pattern Improvements: A100 vs W7900\n(Negative = Better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["RAEARLY\n(layers 0-5)", "RALATE\n(layers 6-11)", "RAALL\n(all layers)"]
    )
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(
        "docs/images/gpt2_ra_relative_improvements.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Generated: docs/images/gpt2_ra_relative_improvements.png")
    plt.close()


def plot_pattern_rankings():
    """Show pattern rankings on both GPUs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # A100 ranking (lower is better)
    a100_ranking = [
        ("RALATE\n(layers 6-11)", 880.12, -8.7),
        ("RAALL\n(all layers)", 887.30, -8.0),
        ("RAEARLY\n(layers 0-5)", 897.60, -6.9),
    ]

    y_pos = np.arange(len(a100_ranking))
    perplexities = [x[1] for x in a100_ranking]
    improvements = [x[2] for x in a100_ranking]

    colors_a100 = [color_best, color_ra, color_ra]
    bars1 = ax1.barh(
        y_pos,
        perplexities,
        color=colors_a100,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add labels
    for i, (bar, (name, ppl, imp)) in enumerate(zip(bars1, a100_ranking)):
        ax1.text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{ppl:.1f} ({imp:.1f}%)",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([x[0] for x in a100_ranking])
    ax1.set_xlabel("Validation Perplexity", fontsize=11, fontweight="bold")
    ax1.set_title(
        "A100: Best to Worst\n(Lower is Better)", fontsize=11, fontweight="bold"
    )
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    # W7900 ranking
    w7900_ranking = [
        ("RAEARLY\n(layers 0-5)", 313.75, -6.6),
        ("RALATE\n(layers 6-11)", 316.04, -5.9),
        ("RAALL\n(all layers)", 322.88, -3.9),
    ]

    perplexities_w = [x[1] for x in w7900_ranking]
    colors_w7900 = [color_best, color_ra, color_ra]
    bars2 = ax2.barh(
        y_pos,
        perplexities_w,
        color=colors_w7900,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add labels
    for i, (bar, (name, ppl, imp)) in enumerate(zip(bars2, w7900_ranking)):
        ax2.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{ppl:.1f} ({imp:.1f}%)",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([x[0] for x in w7900_ranking])
    ax2.set_xlabel("Validation Perplexity", fontsize=11, fontweight="bold")
    ax2.set_title(
        "W7900: Best to Worst\n(Lower is Better)", fontsize=11, fontweight="bold"
    )
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(
        "docs/images/gpt2_ra_pattern_rankings.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Generated: docs/images/gpt2_ra_pattern_rankings.png")
    plt.close()


def main():
    print("Generating RA fixed pattern visualization PNGs...")
    print()

    plot_perplexity_comparison()
    plot_relative_improvements()
    plot_pattern_rankings()

    print()
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
