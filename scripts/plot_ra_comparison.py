#!/usr/bin/env python3
"""
Plot the validated Reciprocal Attention (RA) GPT-2 comparison.

Provenance: GPT-2 small (~124M), FineWebEdu, matched training runs
comparing baseline GPT-2, a Qwen-style SDPA output gate, and RA on
middle layers. The campaign was tracked in Weights & Biases (project
gpt2-ra-sdpa-ablation, W&B era); the quality numbers below (val PPL
72.5 / 71.8 / 68.9, HellaSwag 28 / 28.5 / 30) are the project-owner-
validated record of that experiment. Scaling beyond this setup is
unproven: matched 1B runs were neutral within noise. See
docs/ra-evidence.md.

Earlier versions of this script also drew memory, speed, FIM-trace,
and Pareto-tradeoff plots. Their side-channel numbers could not be
traced to retained artifacts of the same campaign, so those plots
were removed; only the validated quality comparison remains.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def create_quality_comparison():
    """Combined PPL and HellaSwag comparison (GPT-2 small-scale)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    variants = ["Baseline\nGPT-2", "SDPA Gate\n(Qwen-style)", "RA\n(middle layers)"]
    ppl_values = [72.5, 71.8, 68.9]
    hella_values = [28, 28.5, 30]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

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

    ax1.annotate(
        "-5%",
        xy=(2, 68.9),
        xytext=(1, 75),
        fontsize=14,
        fontweight="bold",
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

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
        "Reciprocal Attention vs Baseline and SDPA Gate\n"
        "(GPT-2 small-scale, FineWebEdu; scaling beyond this setup unproven — "
        "matched 1B runs were neutral)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    return fig


def main():
    output_dir = Path("docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating the validated RA quality comparison plot...")
    fig_quality = create_quality_comparison()
    fig_quality.savefig(
        output_dir / "ra_quality_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {output_dir / 'ra_quality_comparison.png'}")
    plt.close("all")


if __name__ == "__main__":
    main()
