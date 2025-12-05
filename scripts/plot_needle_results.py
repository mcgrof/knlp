#!/usr/bin/env python3
"""
Plot needle-in-haystack evaluation results.

Generates visualizations showing retrieval accuracy across:
- Different context lengths
- Different needle positions
- Baseline vs compressed comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_needle_heatmap(results: dict, output_path: str, title: str = None):
    """Create heatmap of needle retrieval accuracy."""
    baseline = results.get("baseline", {})
    compressed = results.get("compressed", {})

    # Extract context lengths and positions
    ctx_keys = sorted(baseline.keys(), key=lambda x: int(x.split("_")[1]))
    if not ctx_keys:
        print("No baseline data to plot")
        return

    pos_keys = sorted(baseline[ctx_keys[0]].keys(), key=lambda x: int(x.split("_")[1]))

    ctx_lengths = [int(k.split("_")[1]) for k in ctx_keys]
    positions = [int(k.split("_")[1]) / 100 for k in pos_keys]

    # Create data matrices
    baseline_data = np.array(
        [[baseline[ctx][pos] * 100 for pos in pos_keys] for ctx in ctx_keys]
    )

    has_compressed = bool(compressed and any(compressed.values()))
    if has_compressed:
        compressed_data = np.array(
            [
                [compressed.get(ctx, {}).get(pos, 0) * 100 for pos in pos_keys]
                for ctx in ctx_keys
            ]
        )

    # Create figure
    if has_compressed:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Baseline heatmap
        im1 = axes[0].imshow(
            baseline_data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto"
        )
        axes[0].set_xticks(range(len(positions)))
        axes[0].set_xticklabels([f"{p:.1f}" for p in positions])
        axes[0].set_yticks(range(len(ctx_lengths)))
        axes[0].set_yticklabels(ctx_lengths)
        axes[0].set_xlabel("Needle Position")
        axes[0].set_ylabel("Context Length (tokens)")
        axes[0].set_title("Baseline")

        # Add text annotations
        for i in range(len(ctx_lengths)):
            for j in range(len(positions)):
                axes[0].text(
                    j,
                    i,
                    f"{baseline_data[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if baseline_data[i, j] < 50 else "black",
                )

        # Compressed heatmap
        im2 = axes[1].imshow(
            compressed_data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto"
        )
        axes[1].set_xticks(range(len(positions)))
        axes[1].set_xticklabels([f"{p:.1f}" for p in positions])
        axes[1].set_yticks(range(len(ctx_lengths)))
        axes[1].set_yticklabels(ctx_lengths)
        axes[1].set_xlabel("Needle Position")
        axes[1].set_ylabel("Context Length (tokens)")
        axes[1].set_title("Compressed (2.67x)")

        for i in range(len(ctx_lengths)):
            for j in range(len(positions)):
                axes[1].text(
                    j,
                    i,
                    f"{compressed_data[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if compressed_data[i, j] < 50 else "black",
                )

        fig.colorbar(im2, ax=axes, label="Retrieval Accuracy (%)", shrink=0.8)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        im = ax.imshow(baseline_data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f"{p:.1f}" for p in positions])
        ax.set_yticks(range(len(ctx_lengths)))
        ax.set_yticklabels(ctx_lengths)
        ax.set_xlabel("Needle Position")
        ax.set_ylabel("Context Length (tokens)")
        ax.set_title("Baseline")

        for i in range(len(ctx_lengths)):
            for j in range(len(positions)):
                ax.text(
                    j,
                    i,
                    f"{baseline_data[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if baseline_data[i, j] < 50 else "black",
                )

        fig.colorbar(im, ax=ax, label="Retrieval Accuracy (%)")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_needle_comparison(results_7b: dict, results_05b: dict, output_path: str):
    """Compare 7B vs 0.5B needle retrieval."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    positions = [0.1, 0.5, 0.9]
    x = np.arange(len(positions))
    width = 0.35

    # 7B comparison at 2048 tokens
    baseline_7b = results_7b.get("baseline", {}).get("ctx_2048", {})
    compressed_7b = results_7b.get("compressed", {}).get("ctx_2048", {})

    if baseline_7b:
        b_vals = [baseline_7b.get(f"pos_{int(p*100)}", 0) * 100 for p in positions]
        c_vals = [compressed_7b.get(f"pos_{int(p*100)}", 0) * 100 for p in positions]

        bars1 = axes[0].bar(
            x - width / 2, b_vals, width, label="Baseline", color="steelblue"
        )
        bars2 = axes[0].bar(
            x + width / 2, c_vals, width, label="Compressed", color="coral"
        )

        axes[0].set_xlabel("Needle Position")
        axes[0].set_ylabel("Retrieval Accuracy (%)")
        axes[0].set_title("Qwen2.5-7B @ 2048 tokens")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{p:.1f}" for p in positions])
        axes[0].set_ylim(0, 110)
        axes[0].legend()
        axes[0].axhline(y=100, color="gray", linestyle="--", alpha=0.5)

        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        for bar in bars2:
            height = bar.get_height()
            axes[0].annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    # 0.5B at 1024 tokens (baseline only)
    baseline_05b = results_05b.get("baseline", {}).get("ctx_1024", {})

    if baseline_05b:
        b_vals = [baseline_05b.get(f"pos_{int(p*100)}", 0) * 100 for p in positions]

        bars = axes[1].bar(x, b_vals, width * 1.5, label="Baseline", color="steelblue")

        axes[1].set_xlabel("Needle Position")
        axes[1].set_ylabel("Retrieval Accuracy (%)")
        axes[1].set_title("Qwen2.5-0.5B @ 1024 tokens (Baseline)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{p:.1f}" for p in positions])
        axes[1].set_ylim(0, 110)
        axes[1].axhline(y=100, color="gray", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            axes[1].annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )

    plt.suptitle("Needle-in-Haystack Retrieval: Model Size Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot needle-in-haystack results")
    parser.add_argument(
        "--results-7b", type=str, default="results/needle_7b_compressed.json"
    )
    parser.add_argument(
        "--results-05b", type=str, default="results/needle_05b_baseline.json"
    )
    parser.add_argument("--output-dir", type=str, default="plots/needle")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_7b = {}
    results_05b = {}

    if Path(args.results_7b).exists():
        with open(args.results_7b) as f:
            results_7b = json.load(f)

    if Path(args.results_05b).exists():
        with open(args.results_05b) as f:
            results_05b = json.load(f)

    # Generate plots
    if results_7b:
        plot_needle_heatmap(
            results_7b,
            str(output_dir / "needle_heatmap_7b.png"),
            title="Qwen2.5-7B: Needle-in-Haystack Retrieval",
        )

    if results_05b:
        plot_needle_heatmap(
            results_05b,
            str(output_dir / "needle_heatmap_05b.png"),
            title="Qwen2.5-0.5B: Needle-in-Haystack Retrieval (Baseline)",
        )

    if results_7b and results_05b:
        plot_needle_comparison(
            results_7b, results_05b, str(output_dir / "needle_comparison.png")
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
