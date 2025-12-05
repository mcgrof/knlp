#!/usr/bin/env python3
"""
Plot compression failure curves from direct compression test results.

Usage:
    python scripts/plot_compression_failure.py
    python scripts/plot_compression_failure.py --input results/direct_compression_test.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_cosine_similarity_curves(results: list, output_dir: Path):
    """Plot K and V cosine similarity vs compression ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Group by quant_bits
    for bits in [16, 8, 4]:
        data = [r for r in results if r["quant_bits"] == bits]
        if not data:
            continue

        ratios = [r["compression_ratio"] for r in data]
        k_cos = [r["k_cosine_sim"] for r in data]
        v_cos = [r["v_cosine_sim"] for r in data]

        label = f"FP16" if bits == 16 else f"Int{bits}"
        marker = "o" if bits == 16 else "s" if bits == 8 else "^"

        ax1.plot(ratios, k_cos, marker=marker, label=label, linewidth=2, markersize=8)
        ax2.plot(ratios, v_cos, marker=marker, label=label, linewidth=2, markersize=8)

    # K plot
    ax1.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="95% threshold")
    ax1.axhline(
        y=0.90, color="orange", linestyle="--", alpha=0.7, label="90% threshold"
    )
    ax1.axhline(y=0.75, color="red", linestyle="--", alpha=0.7, label="75% threshold")
    ax1.set_xlabel("Compression Ratio", fontsize=12)
    ax1.set_ylabel("Cosine Similarity", fontsize=12)
    ax1.set_title("K Tensor Reconstruction Quality", fontsize=14)
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    # V plot
    ax2.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="95% threshold")
    ax2.axhline(
        y=0.90, color="orange", linestyle="--", alpha=0.7, label="90% threshold"
    )
    ax2.axhline(y=0.75, color="red", linestyle="--", alpha=0.7, label="75% threshold")
    ax2.set_xlabel("Compression Ratio", fontsize=12)
    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_title("V Tensor Reconstruction Quality", fontsize=14)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Qwen2.5-0.5B: KV Cache Compression Quality", fontsize=16, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "compression_cosine_similarity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_k_vs_v_comparison(results: list, output_dir: Path):
    """Plot K vs V sensitivity comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # FP16 only for clarity
    data = [r for r in results if r["quant_bits"] == 16]
    ratios = [r["compression_ratio"] for r in data]
    k_cos = [r["k_cosine_sim"] for r in data]
    v_cos = [r["v_cosine_sim"] for r in data]

    ax.plot(
        ratios, k_cos, "o-", label="K (Keys)", linewidth=2, markersize=10, color="blue"
    )
    ax.plot(
        ratios, v_cos, "s-", label="V (Values)", linewidth=2, markersize=10, color="red"
    )

    # Fill area between
    ax.fill_between(ratios, k_cos, v_cos, alpha=0.2, color="purple")

    # Annotations
    ax.annotate(
        "V degrades 2x faster\nthan K",
        xy=(2.0, 0.82),
        xytext=(4, 0.9),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("K vs V Compression Sensitivity (Qwen2.5-0.5B)", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "k_vs_v_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_safe_zones(results: list, output_dir: Path):
    """Plot safe/moderate/aggressive/extreme zones."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # FP16 data
    data = [r for r in results if r["quant_bits"] == 16]
    ratios = np.array([r["compression_ratio"] for r in data])
    v_cos = np.array([r["v_cosine_sim"] for r in data])

    # Plot V cosine similarity
    ax.plot(
        ratios,
        v_cos,
        "s-",
        linewidth=2,
        markersize=10,
        color="darkblue",
        label="V CosSim",
    )

    # Define zones
    zones = [
        (1.0, 1.3, "Safe", "lightgreen"),
        (1.3, 2.0, "Moderate", "yellow"),
        (2.0, 4.0, "Aggressive", "orange"),
        (4.0, 100, "Extreme", "red"),
    ]

    for x_min, x_max, name, color in zones:
        ax.axvspan(
            x_min,
            min(x_max, ratios.max() * 1.1),
            alpha=0.2,
            color=color,
            label=f"{name} zone",
        )

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("V Cosine Similarity", fontsize=12)
    ax.set_title("KV Cache Compression: Safe Operating Zones", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Add threshold lines
    ax.axhline(y=0.90, color="green", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.axhline(y=0.70, color="orange", linestyle=":", alpha=0.8, linewidth=1.5)
    ax.axhline(y=0.50, color="red", linestyle=":", alpha=0.8, linewidth=1.5)

    plt.tight_layout()

    output_path = output_dir / "compression_safe_zones.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_mse_curves(results: list, output_dir: Path):
    """Plot MSE vs compression ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # FP16 only
    data = [r for r in results if r["quant_bits"] == 16]
    ratios = [r["compression_ratio"] for r in data]
    k_mse = [r["k_mse"] for r in data]
    v_mse = [r["v_mse"] for r in data]

    ax.semilogy(
        ratios, k_mse, "o-", label="K MSE", linewidth=2, markersize=10, color="blue"
    )
    ax.semilogy(
        ratios, v_mse, "s-", label="V MSE", linewidth=2, markersize=10, color="red"
    )

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Mean Squared Error (log scale)", fontsize=12)
    ax.set_title("Reconstruction Error vs Compression (Qwen2.5-0.5B)", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    output_path = output_dir / "compression_mse.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot compression failure curves")
    parser.add_argument(
        "--input",
        type=str,
        default="results/direct_compression_test.json",
        help="Input JSON file with results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/compression_failure_curves",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    # Load results
    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(results)} results from {args.input}")
    print(f"Model: {data.get('model', 'unknown')}")
    print(f"Output: {output_dir}")
    print()

    plot_cosine_similarity_curves(results, output_dir)
    plot_k_vs_v_comparison(results, output_dir)
    plot_safe_zones(results, output_dir)
    plot_mse_curves(results, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
