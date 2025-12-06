#!/usr/bin/env python3
"""
Generate scaling laws plots showing larger models compress better.

Usage:
    python scripts/plot_scaling_laws_b200.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_quality_results():
    """Load all quality benchmark results."""
    results_dir = Path("key_results")
    all_results = []

    for result_file in results_dir.glob("quality_*.json"):
        with open(result_file) as f:
            data = json.load(f)
            all_results.extend(data.get("results", []))

    return all_results


def get_model_size(model_name: str) -> float:
    """Extract model size in billions from name."""
    name = model_name.lower()
    if "72b" in name:
        return 72.0
    elif "7b" in name:
        return 7.0
    elif "1.5b" in name:
        return 1.5
    elif "0.5b" in name:
        return 0.5
    elif "gpt2" in name:
        return 0.124
    return 0.0


def plot_scaling_law(results, output_dir="docs/kv_plugin"):
    """Plot PPL degradation vs model size at fixed compression."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model
    model_data = {}
    for r in results:
        if r["metric"] == "perplexity":
            model = r["model_name"]
            size = get_model_size(model)
            if size == 0:
                continue

            if model not in model_data:
                model_data[model] = {"size": size, "baseline": None, "compressed": {}}

            if r["compression_type"] == "baseline":
                model_data[model]["baseline"] = r["value"]
            elif r["compression_type"] == "calibrated":
                model_data[model]["compressed"][r["rank"]] = r["value"]

    # Calculate delta PPL for each rank
    ranks_to_plot = [96, 112, 120]
    colors = ["tab:red", "tab:orange", "tab:green"]

    for rank, color in zip(ranks_to_plot, colors):
        sizes = []
        deltas = []

        for model, data in model_data.items():
            if data["baseline"] and rank in data["compressed"]:
                delta = (
                    (data["compressed"][rank] - data["baseline"])
                    / data["baseline"]
                    * 100
                )
                sizes.append(data["size"])
                deltas.append(delta)

        if sizes:
            # Sort by size
            sorted_pairs = sorted(zip(sizes, deltas))
            sizes, deltas = zip(*sorted_pairs)

            ax.plot(
                sizes,
                deltas,
                "o-",
                color=color,
                linewidth=2,
                markersize=10,
                label=f"Rank {rank} (1.{128//rank:.0f}x V-only)",
            )

    ax.set_xlabel("Model Size (Billion Parameters)", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title(
        "KV Compression Scaling Law: Larger Models Compress Better", fontsize=14
    )
    ax.set_xscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(
        "Better compression\nwith larger models",
        xy=(40, 8),
        fontsize=10,
        ha="center",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/b200_scaling_law.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/b200_scaling_law.png")


def plot_compression_curve(results, output_dir="docs/kv_plugin"):
    """Plot PPL vs compression ratio for different model sizes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    head_dim = 128

    # Group by model
    model_data = {}
    for r in results:
        if r["metric"] == "perplexity":
            model = r["model_name"]
            size = get_model_size(model)
            if size == 0:
                continue

            if model not in model_data:
                model_data[model] = {"size": size, "baseline": None, "points": []}

            if r["compression_type"] == "baseline":
                model_data[model]["baseline"] = r["value"]
            elif r["compression_type"] == "calibrated":
                rank = r["rank"]
                effective_ratio = 2 * head_dim / (head_dim + rank)
                model_data[model]["points"].append((effective_ratio, r["value"]))

    # Plot each model
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(model_data)))
    for (model, data), color in zip(
        sorted(model_data.items(), key=lambda x: x[1]["size"]), colors
    ):
        if not data["baseline"] or not data["points"]:
            continue

        # Add baseline point
        points = [(1.0, data["baseline"])] + data["points"]
        points.sort(key=lambda x: x[0])

        ratios, ppls = zip(*points)
        deltas = [(p - data["baseline"]) / data["baseline"] * 100 for p in ppls]

        model_short = model.split("/")[-1]
        ax.plot(
            ratios,
            deltas,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{model_short} ({data['size']:.1f}B)",
        )

    ax.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="10% threshold")
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="5% threshold")

    ax.set_xlabel("Effective Compression Ratio (V-only)", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title(
        "KV Compression: Quality vs Compression by Model Size (B200)", fontsize=14
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.98, 1.20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/b200_compression_curves.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/b200_compression_curves.png")


def main():
    results = load_quality_results()
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found.")
        return

    plot_scaling_law(results)
    plot_compression_curve(results)


if __name__ == "__main__":
    main()
