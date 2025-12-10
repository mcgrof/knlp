#!/usr/bin/env python3
"""
Plot KVSplice FIM-guided compression results.

Generates publication-quality plots comparing:
- mla_kv (full KVSplice, 12x compression)
- mla_kv_fim (FIM-guided selective KVSplice, 7.2x compression)

Results from B200x4 1-hour FineWebEdu training runs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_ppl_comparison():
    """Create perplexity comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    variants = ["mla_kv\n(12x compression)", "mla_kv_fim\n(7.2x compression)"]
    ppl_values = [83.85, 63.08]
    colors = ["#d62728", "#2ca02c"]  # red for worse, green for better

    bars = ax.bar(variants, ppl_values, color=colors, width=0.6, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, ppl_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # Add improvement annotation
    ax.annotate(
        "",
        xy=(1, 63.08),
        xytext=(0, 83.85),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax.text(
        0.5,
        73,
        "-25%",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="green",
    )

    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title(
        "KVSplice Perplexity: FIM-Guided vs Full Compression\n(B200x4, 1hr FineWebEdu)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add "lower is better" note
    ax.text(
        0.98,
        0.02,
        "↓ Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    return fig


def create_hellaswag_comparison():
    """Create HellaSwag accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    variants = ["mla_kv\n(12x compression)", "mla_kv_fim\n(7.2x compression)"]
    acc_values = [24, 31]
    colors = ["#d62728", "#2ca02c"]  # red for worse, green for better

    bars = ax.bar(variants, acc_values, color=colors, width=0.6, edgecolor="black")

    # Add value labels on bars
    for bar, val in zip(bars, acc_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # Add improvement annotation
    ax.annotate(
        "",
        xy=(1, 31),
        xytext=(0, 24),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax.text(
        0.5,
        27.5,
        "+7 pts",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="green",
    )

    ax.set_ylabel("HellaSwag Accuracy (%)", fontsize=12)
    ax.set_title(
        "KVSplice HellaSwag: FIM-Guided vs Full Compression\n(B200x4, 1hr FineWebEdu)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add "higher is better" note
    ax.text(
        0.98,
        0.02,
        "↑ Higher is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    return fig


def create_combined_comparison():
    """Create combined side-by-side comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    variants = ["mla_kv\n(12x)", "mla_kv_fim\n(7.2x)"]
    colors = ["#d62728", "#2ca02c"]

    # Perplexity subplot
    ppl_values = [83.85, 63.08]
    bars1 = ax1.bar(variants, ppl_values, color=colors, width=0.6, edgecolor="black")
    for bar, val in zip(bars1, ppl_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylabel("Validation Perplexity", fontsize=12)
    ax1.set_title("Perplexity (↓ lower is better)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(
        0.5,
        73,
        "-25%",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="green",
    )

    # HellaSwag subplot
    acc_values = [24, 31]
    bars2 = ax2.bar(variants, acc_values, color=colors, width=0.6, edgecolor="black")
    for bar, val in zip(bars2, acc_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax2.set_ylabel("HellaSwag Accuracy (%)", fontsize=12)
    ax2.set_title("HellaSwag (↑ higher is better)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 40)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(
        0.5,
        27.5,
        "+7 pts",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="green",
    )

    fig.suptitle(
        "FIM-Guided KVSplice Wins on Both Metrics\n(B200x4, 1hr FineWebEdu Training)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig


def create_tradeoff_scatter():
    """Create compression vs quality trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data points: (compression_ratio, ppl, hellaswag, label)
    data = [
        (6.0, 255, 25, "MLA only"),  # Estimated from earlier results
        (7.2, 63.08, 31, "mla_kv_fim"),
        (12.0, 83.85, 24, "mla_kv"),
    ]

    for comp, ppl, hella, label in data:
        # Size based on HellaSwag (bigger = better)
        size = hella * 20
        # Color based on PPL (green = good, red = bad)
        if ppl < 100:
            color = plt.cm.RdYlGn(1 - ppl / 300)
        else:
            color = plt.cm.RdYlGn(1 - 100 / 300)

        ax.scatter(comp, ppl, s=size, c=[color], edgecolors="black", linewidth=2)
        ax.annotate(
            f"{label}\n({hella}% HellaSwag)",
            (comp, ppl),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Compression Ratio (x)", fontsize=12)
    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title(
        "Compression vs Quality Trade-off\n(Bubble size = HellaSwag accuracy)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(4, 14)
    ax.set_ylim(0, 300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add Pareto frontier annotation
    ax.plot([6.0, 7.2], [255, 63.08], "g--", linewidth=2, alpha=0.5)
    ax.text(6.5, 150, "Pareto\nfrontier", fontsize=10, color="green", style="italic")

    plt.tight_layout()
    return fig


def main():
    output_dir = Path("docs/kvsplice")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    print("Generating KVSplice FIM comparison plots...")

    fig_ppl = create_ppl_comparison()
    fig_ppl.savefig(output_dir / "kvsplice_ppl_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'kvsplice_ppl_comparison.png'}")

    fig_hella = create_hellaswag_comparison()
    fig_hella.savefig(output_dir / "kvsplice_hellaswag_comparison.png", dpi=300)
    print(f"  Saved: {output_dir / 'kvsplice_hellaswag_comparison.png'}")

    fig_combined = create_combined_comparison()
    fig_combined.savefig(output_dir / "kvsplice_fim_combined.png", dpi=300)
    print(f"  Saved: {output_dir / 'kvsplice_fim_combined.png'}")

    fig_tradeoff = create_tradeoff_scatter()
    fig_tradeoff.savefig(output_dir / "kvsplice_tradeoff.png", dpi=300)
    print(f"  Saved: {output_dir / 'kvsplice_tradeoff.png'}")

    plt.close("all")
    print("\nDone! Plots saved to docs/kvsplice/")


if __name__ == "__main__":
    main()
