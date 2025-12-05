#!/usr/bin/env python3
"""
Plot Qwen2.5-0.5B compression vs quality results.

Generates AsymKV-style plots showing perplexity vs KV memory fraction.

Usage:
    python scripts/plot_qwen_compression.py
    python scripts/plot_qwen_compression.py --output-dir plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_compression_vs_ppl(output_dir: Path):
    """Generate compression vs perplexity plot (Palu Figure 2 style)."""
    # Qwen2.5-0.5B data from ablation
    configs = {
        "Baseline": {"memory_fraction": 1.0, "ppl": 1.86, "marker": "o"},
        "Balanced (7x)": {"memory_fraction": 1 / 7, "ppl": 1.86, "marker": "s"},
        "Aggressive (14x)": {"memory_fraction": 1 / 14, "ppl": 1.86, "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each config
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
    for (name, data), color in zip(configs.items(), colors):
        ax.scatter(
            data["memory_fraction"],
            data["ppl"],
            s=150,
            marker=data["marker"],
            color=color,
            label=name,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    # Connect with line to show trend
    x_vals = [configs[k]["memory_fraction"] for k in configs]
    y_vals = [configs[k]["ppl"] for k in configs]
    ax.plot(x_vals, y_vals, "k--", alpha=0.3, zorder=1)

    # Styling
    ax.set_xlabel("KV Memory Fraction", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("Qwen2.5-0.5B: Compression vs Quality", fontsize=14)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 1.5)
    ax.set_ylim(1.0, 3.0)

    # Add annotation for hero result
    ax.annotate(
        "14x compression\n0% PPL loss",
        xy=(1 / 14, 1.86),
        xytext=(0.15, 2.5),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "qwen_compression_vs_ppl.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_kv_ablation(output_dir: Path):
    """Generate K vs V quantization ablation bar chart."""
    # Ablation data
    data = {
        "Balanced": {
            "v_8bit": {"ppl": 1.86, "toks": 72},
            "v_4bit": {"ppl": 1.86, "toks": 72},
            "kv_8bit": {"ppl": 1.86, "toks": 71},
            "kv_4bit": {"ppl": 1.86, "toks": 71},
        },
        "Aggressive": {
            "v_8bit": {"ppl": 1.86, "toks": 70},
            "v_4bit": {"ppl": 1.86, "toks": 71},
            "kv_8bit": {"ppl": 1.86, "toks": 70},
            "kv_4bit": {"ppl": 1.86, "toks": 71},
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # PPL comparison
    x = np.arange(4)
    width = 0.35
    labels = ["V 8-bit", "V 4-bit", "K+V 8-bit", "K+V 4-bit"]

    balanced_ppl = [1.86, 1.86, 1.86, 1.86]
    aggressive_ppl = [1.86, 1.86, 1.86, 1.86]

    ax1.bar(
        x - width / 2, balanced_ppl, width, label="Balanced (7x)", color="steelblue"
    )
    ax1.bar(
        x + width / 2, aggressive_ppl, width, label="Aggressive (14x)", color="coral"
    )
    ax1.axhline(y=1.86, color="green", linestyle="--", alpha=0.7, label="Baseline")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("PPL by Quantization Config")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylim(0, 2.5)
    ax1.legend()

    # Throughput comparison
    balanced_toks = [72, 72, 71, 71]
    aggressive_toks = [70, 71, 70, 71]

    ax2.bar(
        x - width / 2, balanced_toks, width, label="Balanced (7x)", color="steelblue"
    )
    ax2.bar(
        x + width / 2, aggressive_toks, width, label="Aggressive (14x)", color="coral"
    )
    ax2.axhline(y=73, color="green", linestyle="--", alpha=0.7, label="Baseline")
    ax2.set_ylabel("Tokens/sec")
    ax2.set_title("Throughput by Quantization Config")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylim(0, 80)
    ax2.legend()

    plt.suptitle("Qwen2.5-0.5B: K vs V Quantization Ablation", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "qwen_kv_ablation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_memory_breakdown(output_dir: Path):
    """Generate memory breakdown visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = ["Baseline", "Balanced", "Aggressive"]
    kv_memory = [12.0, 1.71, 0.86]
    compression = ["1x", "7x", "14x"]

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    bars = ax.barh(configs, kv_memory, color=colors, edgecolor="black", linewidth=1.5)

    # Add compression labels
    for bar, comp, mem in zip(bars, compression, kv_memory):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{comp}\n({mem:.2f} MB)",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("KV Cache Size (MB)", fontsize=12)
    ax.set_title("Qwen2.5-0.5B: KV Cache Memory at 1024 Context Length", fontsize=14)
    ax.set_xlim(0, 16)
    ax.axvline(x=12.0, color="gray", linestyle=":", alpha=0.5)

    # Add annotation
    ax.annotate(
        "14x smaller!\nSame PPL",
        xy=(0.86, 2),
        xytext=(5, 2.3),
        fontsize=12,
        fontweight="bold",
        color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )

    plt.tight_layout()
    output_path = output_dir / "qwen_memory_breakdown.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Qwen2.5-0.5B compression plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Qwen2.5-0.5B compression plots...")
    plot_compression_vs_ppl(output_dir)
    plot_kv_ablation(output_dir)
    plot_memory_breakdown(output_dir)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
