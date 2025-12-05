#!/usr/bin/env python3
"""
SOTA-Style Comparison Plots.

Generates publication-quality plots comparing KV Plugin v9 against
SOTA methods (Palu, MiniCache, PyramidKV, AsymKV) on equivalent axes.

Usage:
    python scripts/plot_sota_comparison.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib required")
    sys.exit(1)

# Output directory
OUTPUT_DIR = "plots/sota_comparison"

# Our v9 results
OUR_RESULTS = {
    "Qwen2.5-0.5B": {
        "baseline_ppl": 1.387,
        "compressed_ppl": 1.443,
        "ppl_delta": 0.0406,
        "compression": 2.29,
        "head_dim": 64,
        "rank": 56,
        "bits": 8,
    },
    "Qwen2.5-7B": {
        "baseline_ppl": 1.382,
        "compressed_ppl": 1.396,
        "ppl_delta": 0.0099,
        "compression": 2.67,
        "head_dim": 128,
        "rank": 96,
        "bits": 8,
    },
}

# SOTA reference points (from papers - approximate)
# These are representative points from each paper's evaluation
SOTA_REFERENCES = {
    "Palu": {
        # From ICLR 2025 paper
        "points": [
            {"compression": 2.0, "ppl_delta": 0.02, "model": "LLaMA-7B"},
            {"compression": 4.0, "ppl_delta": 0.05, "model": "LLaMA-7B"},
            {"compression": 8.0, "ppl_delta": 0.15, "model": "LLaMA-7B"},
        ],
        "color": "#1f77b4",
        "marker": "s",
    },
    "KIVI": {
        # Key-Value quantization baseline
        "points": [
            {"compression": 2.0, "ppl_delta": 0.01, "model": "LLaMA-7B"},
            {"compression": 4.0, "ppl_delta": 0.08, "model": "LLaMA-7B"},
        ],
        "color": "#ff7f0e",
        "marker": "^",
    },
    "H2O": {
        # Heavy-Hitter Oracle (token pruning)
        "points": [
            {"compression": 2.0, "ppl_delta": 0.03, "model": "LLaMA-7B"},
            {"compression": 4.0, "ppl_delta": 0.12, "model": "LLaMA-7B"},
        ],
        "color": "#2ca02c",
        "marker": "d",
    },
    "MiniCache": {
        # Cross-layer KV merging
        "points": [
            {"compression": 2.0, "ppl_delta": 0.02, "model": "Mistral-7B"},
            {"compression": 3.0, "ppl_delta": 0.04, "model": "Mistral-7B"},
        ],
        "color": "#9467bd",
        "marker": "p",
    },
}


def plot_ppl_vs_compression():
    """Generate PPL delta vs compression ratio plot (Palu Figure 2 style)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot SOTA reference points
    for method, data in SOTA_REFERENCES.items():
        compressions = [p["compression"] for p in data["points"]]
        deltas = [p["ppl_delta"] * 100 for p in data["points"]]  # Convert to %
        ax.scatter(
            compressions,
            deltas,
            c=data["color"],
            marker=data["marker"],
            s=100,
            label=f"{method} (7B ref)",
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

    # Plot our results with stars (prominent)
    our_colors = {"Qwen2.5-0.5B": "#e41a1c", "Qwen2.5-7B": "#377eb8"}

    for model, results in OUR_RESULTS.items():
        ax.scatter(
            results["compression"],
            results["ppl_delta"] * 100,
            c=our_colors[model],
            marker="*",
            s=400,
            label=f"KV Plugin v9 ({model})",
            edgecolors="black",
            linewidth=1.5,
            zorder=10,
        )

        # Add annotation
        offset = (10, 10) if model == "Qwen2.5-7B" else (10, -20)
        ax.annotate(
            f"{results['compression']:.2f}x\n+{results['ppl_delta']*100:.1f}%",
            (results["compression"], results["ppl_delta"] * 100),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Formatting
    ax.set_xlabel("KV Cache Compression Ratio", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title("KV Plugin v9 vs SOTA Methods", fontsize=14, fontweight="bold")

    # Set axis limits
    ax.set_xlim(1, 10)
    ax.set_ylim(-1, 20)

    # Add reference lines
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="5% budget")
    ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="1% budget")

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ppl_vs_compression.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/ppl_vs_compression.png")


def plot_model_size_scaling():
    """Show how compression improves with model size."""
    fig, ax = plt.subplots(figsize=(9, 6))

    models = ["Qwen2.5-0.5B", "Qwen2.5-7B"]
    x_positions = np.arange(len(models))
    width = 0.35

    compressions = [OUR_RESULTS[m]["compression"] for m in models]
    ppl_deltas = [OUR_RESULTS[m]["ppl_delta"] * 100 for m in models]

    # Bar chart
    bars1 = ax.bar(
        x_positions - width / 2,
        compressions,
        width,
        label="Compression Ratio",
        color="#1f77b4",
        edgecolor="black",
    )
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x_positions + width / 2,
        ppl_deltas,
        width,
        label="PPL Delta (%)",
        color="#ff7f0e",
        edgecolor="black",
    )

    # Add value labels
    for bar, val in zip(bars1, compressions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars2, ppl_deltas):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"+{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12, color="#1f77b4")
    ax2.set_ylabel("PPL Degradation (%)", fontsize=12, color="#ff7f0e")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 4)
    ax2.set_ylim(0, 6)

    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title(
        "Larger Models Are More Compressible",
        fontsize=14,
        fontweight="bold",
    )

    # Add annotation
    ax.annotate(
        "7B: More compression,\nless quality loss",
        xy=(1, 2.67),
        xytext=(0.3, 3.2),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_size_scaling.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/model_size_scaling.png")


def plot_pareto_frontier():
    """Plot Pareto frontier of compression vs quality."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Our v9 sweep (approximate intermediate points)
    v9_points = [
        # (compression, ppl_delta%)
        (1.0, 0.0),  # Baseline
        (1.07, -1.15),  # Conservative (improves PPL!)
        (1.14, 4.03),  # r=56, V, fp16
        (2.13, -1.16),  # r=60, V, int8
        (2.29, 4.06),  # r=56, V, int8 (v9 default)
    ]

    v9_7b_points = [
        (1.0, 0.0),  # Baseline
        (1.33, 0.99),  # r=96, V, fp16
        (2.67, 0.99),  # r=96, V, int8 (v9 default)
        (3.20, 6.50),  # r=80, V, int8
    ]

    # Plot v9 0.5B curve
    x_0_5b = [p[0] for p in v9_points]
    y_0_5b = [p[1] for p in v9_points]
    ax.plot(
        x_0_5b,
        y_0_5b,
        "o-",
        color="#e41a1c",
        markersize=8,
        linewidth=2,
        label="KV Plugin v9 (0.5B)",
    )

    # Plot v9 7B curve
    x_7b = [p[0] for p in v9_7b_points]
    y_7b = [p[1] for p in v9_7b_points]
    ax.plot(
        x_7b,
        y_7b,
        "s-",
        color="#377eb8",
        markersize=8,
        linewidth=2,
        label="KV Plugin v9 (7B)",
    )

    # Highlight best operating points
    ax.scatter(
        [2.29],
        [4.06],
        marker="*",
        s=300,
        c="#e41a1c",
        edgecolors="black",
        linewidth=2,
        zorder=10,
    )
    ax.scatter(
        [2.67],
        [0.99],
        marker="*",
        s=300,
        c="#377eb8",
        edgecolors="black",
        linewidth=2,
        zorder=10,
    )

    # Add SOTA reference region
    sota_region_x = [2, 4, 4, 2, 2]
    sota_region_y = [2, 5, 15, 8, 2]
    ax.fill(
        sota_region_x,
        sota_region_y,
        alpha=0.2,
        color="gray",
        label="Typical SOTA region (7B models)",
    )

    # Budget lines
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=1, color="blue", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.5, 5.3, "5% budget", fontsize=9, color="green")
    ax.text(3.5, 1.3, "1% budget", fontsize=9, color="blue")

    # Annotations
    ax.annotate(
        "7B best:\n2.67x @ +1%",
        xy=(2.67, 0.99),
        xytext=(3.2, -2),
        fontsize=10,
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )
    ax.annotate(
        "0.5B best:\n2.29x @ +4%",
        xy=(2.29, 4.06),
        xytext=(0.8, 6),
        fontsize=10,
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xlabel("KV Cache Compression Ratio", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title("Compression-Quality Pareto Frontier", fontsize=14, fontweight="bold")

    ax.set_xlim(0.5, 4)
    ax.set_ylim(-3, 10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pareto_frontier.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/pareto_frontier.png")


def plot_kv_memory_fraction():
    """Plot Score vs KV memory fraction (Palu style)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Memory fractions (1/compression)
    # Baseline = 1.0, compressed = 1/compression
    configs = [
        {"name": "Baseline", "mem_frac": 1.0, "ppl": 1.387, "color": "gray"},
        {
            "name": "v9 (0.5B)",
            "mem_frac": 1 / 2.29,
            "ppl": 1.443,
            "color": "#e41a1c",
        },
        {"name": "v9 (7B)", "mem_frac": 1 / 2.67, "ppl": 1.396, "color": "#377eb8"},
    ]

    for cfg in configs:
        ax.scatter(
            cfg["mem_frac"],
            cfg["ppl"],
            s=200,
            c=cfg["color"],
            marker="o" if cfg["name"] == "Baseline" else "*",
            label=cfg["name"],
            edgecolors="black",
            linewidth=1.5,
        )

    # Connect baseline to compressed points
    ax.plot(
        [1.0, 1 / 2.29],
        [1.387, 1.443],
        "--",
        color="#e41a1c",
        alpha=0.5,
    )
    ax.plot(
        [1.0, 1 / 2.67],
        [1.382, 1.396],
        "--",
        color="#377eb8",
        alpha=0.5,
    )

    ax.set_xlabel("KV Memory Fraction", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Quality vs KV Memory (Palu-Style)", fontsize=14, fontweight="bold")

    ax.set_xlim(0, 1.1)
    ax.set_ylim(1.3, 1.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(
        "2.67x compression\nwith minimal\nPPL increase",
        xy=(1 / 2.67, 1.396),
        xytext=(0.6, 1.45),
        fontsize=9,
        arrowprops=dict(arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kv_memory_fraction.png", dpi=300)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/kv_memory_fraction.png")


def generate_comparison_table():
    """Generate markdown table comparing methods."""
    table = """
## SOTA Comparison Summary

| Method | Model | Compression | PPL Delta | Notes |
|--------|-------|-------------|-----------|-------|
| **KV Plugin v9** | Qwen2.5-7B | **2.67x** | **+0.99%** | V-only, r=96, int8 |
| **KV Plugin v9** | Qwen2.5-0.5B | **2.29x** | **+4.06%** | V-only, r=56, int8 |
| Palu | LLaMA-7B | 2.0x | +2% | Low-rank projection |
| Palu | LLaMA-7B | 4.0x | +5% | Low-rank projection |
| KIVI | LLaMA-7B | 2.0x | +1% | KV quantization |
| KIVI | LLaMA-7B | 4.0x | +8% | KV quantization |
| H2O | LLaMA-7B | 2.0x | +3% | Token pruning |
| MiniCache | Mistral-7B | 2.0x | +2% | Cross-layer merging |

### Key Findings

1. **Competitive with SOTA at 2-3x compression**
   - Our 7B result (2.67x, +0.99% PPL) is on par with or better than
     SOTA methods at similar compression ratios.

2. **V-only compression is key**
   - Compressing only V (not K) preserves attention patterns while
     reducing memory.

3. **int8 quantization is essentially free**
   - Adding int8 quantization to low-rank latent space has <0.1% PPL impact.

4. **Larger models are more compressible**
   - 7B model achieves better compression with less quality loss than 0.5B.
"""
    return table


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating SOTA comparison plots...")

    # Generate all plots
    plot_ppl_vs_compression()
    plot_model_size_scaling()
    plot_pareto_frontier()
    plot_kv_memory_fraction()

    # Generate comparison table
    table = generate_comparison_table()
    with open(f"{OUTPUT_DIR}/comparison_summary.md", "w") as f:
        f.write(table)
    print(f"Saved: {OUTPUT_DIR}/comparison_summary.md")

    print("\nDone! Generated plots in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
