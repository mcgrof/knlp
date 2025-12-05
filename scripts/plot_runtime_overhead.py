#!/usr/bin/env python3
"""
Plot actual runtime overhead from generation benchmarks.

Uses real data from bench_generate.py runs on H100.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    output_dir = Path("plots/paper")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Actual measured data from bench_generate.py
    models = ["Qwen2.5-7B", "Mistral-7B"]
    baseline_throughput = [62.5, 54.3]  # tok/s
    compressed_throughput = [42.8, 38.7]  # tok/s
    compression_ratio = [2.67, 2.42]  # x

    # Calculate overhead
    throughput_overhead = [
        (1 - c / b) * 100 for b, c in zip(baseline_throughput, compressed_throughput)
    ]
    memory_savings = [(1 - 1 / r) * 100 for r in compression_ratio]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Throughput comparison
    x = np.arange(len(models))
    width = 0.35

    bars1 = axes[0].bar(
        x - width / 2, baseline_throughput, width, label="Baseline", color="steelblue"
    )
    bars2 = axes[0].bar(
        x + width / 2,
        compressed_throughput,
        width,
        label="Compressed",
        color="coral",
    )

    axes[0].set_ylabel("Throughput (tokens/sec)")
    axes[0].set_title("Generation Throughput on H100")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].set_ylim(0, 80)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    # Plot 2: Trade-off visualization
    axes[1].barh(
        models,
        throughput_overhead,
        color="coral",
        alpha=0.8,
        label="Throughput Overhead",
    )
    axes[1].barh(
        models,
        [-s for s in memory_savings],
        color="seagreen",
        alpha=0.8,
        label="Memory Savings",
    )

    axes[1].axvline(x=0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Percentage (%)")
    axes[1].set_title("Compression Trade-off")
    axes[1].set_xlim(-80, 40)
    axes[1].legend(loc="lower right")

    # Add value labels
    for i, (overhead, savings) in enumerate(zip(throughput_overhead, memory_savings)):
        axes[1].annotate(
            f"+{overhead:.0f}%",
            xy=(overhead, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
        axes[1].annotate(
            f"-{savings:.0f}%",
            xy=(-savings, i),
            xytext=(-25, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )

    plt.suptitle(
        "KV Compression: ~31% Throughput Cost for 2.5-2.7x Memory Savings",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    # Save
    output_path = output_dir / "fig8_runtime_overhead.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Also save PDF
    output_path_pdf = output_dir / "fig8_runtime_overhead.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Recreate for PDF
    bars1 = axes[0].bar(
        x - width / 2, baseline_throughput, width, label="Baseline", color="steelblue"
    )
    bars2 = axes[0].bar(
        x + width / 2, compressed_throughput, width, label="Compressed", color="coral"
    )
    axes[0].set_ylabel("Throughput (tokens/sec)")
    axes[0].set_title("Generation Throughput on H100")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].set_ylim(0, 80)
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    axes[1].barh(
        models,
        throughput_overhead,
        color="coral",
        alpha=0.8,
        label="Throughput Overhead",
    )
    axes[1].barh(
        models,
        [-s for s in memory_savings],
        color="seagreen",
        alpha=0.8,
        label="Memory Savings",
    )
    axes[1].axvline(x=0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Percentage (%)")
    axes[1].set_title("Compression Trade-off")
    axes[1].set_xlim(-80, 40)
    axes[1].legend(loc="lower right")
    for i, (overhead, savings) in enumerate(zip(throughput_overhead, memory_savings)):
        axes[1].annotate(
            f"+{overhead:.0f}%",
            xy=(overhead, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )
        axes[1].annotate(
            f"-{savings:.0f}%",
            xy=(-savings, i),
            xytext=(-25, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
        )

    plt.suptitle(
        "KV Compression: ~31% Throughput Cost for 2.5-2.7x Memory Savings",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path_pdf}")


if __name__ == "__main__":
    main()
