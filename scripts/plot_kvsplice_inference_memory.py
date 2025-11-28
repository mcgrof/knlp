#!/usr/bin/env python3
"""Generate visualization plots for KVSplice inference memory verification."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)


def plot_cache_comparison():
    """Plot cache memory comparison across sequence lengths."""
    seq_lengths = [256, 512, 1024]
    mla_cache = [3.00, 6.00, 12.00]
    kvsplice_cache = [1.50, 3.00, 6.00]
    standard_cache = [9.00, 18.00, 36.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Line chart showing cache growth
    x = np.arange(len(seq_lengths))
    width = 0.25

    ax1.plot(
        seq_lengths,
        standard_cache,
        "o-",
        linewidth=2,
        markersize=8,
        label="Standard GPT-2",
        color="#d62728",
    )
    ax1.plot(
        seq_lengths,
        mla_cache,
        "s-",
        linewidth=2,
        markersize=8,
        label="MLA (6x)",
        color="#ff7f0e",
    )
    ax1.plot(
        seq_lengths,
        kvsplice_cache,
        "^-",
        linewidth=2,
        markersize=8,
        label="KVSplice (12x)",
        color="#2ca02c",
    )

    ax1.set_xlabel("Sequence Length (tokens)")
    ax1.set_ylabel("KV Cache Memory (MB)")
    ax1.set_title("KV Cache Memory vs Sequence Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(seq_lengths)

    # Right plot: Bar chart showing savings at seq_len=1024
    categories = ["Standard\nGPT-2", "MLA\n(6x)", "KVSplice\n(12x)"]
    values = [36.00, 12.00, 6.00]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    bars = ax2.bar(
        categories, values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f} MB",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add compression annotations
    ax2.text(
        1.5,
        30,
        "6x compression\n(66.7% reduction)",
        ha="center",
        fontsize=9,
        style="italic",
    )
    ax2.text(
        2,
        20,
        "12x compression\n(83.3% reduction)",
        ha="center",
        fontsize=9,
        style="italic",
    )

    ax2.set_ylabel("KV Cache Memory (MB)")
    ax2.set_title("Cache Memory at 1024 Tokens")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        "docs/kvsplice/inference_memory_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("Saved: docs/kvsplice/inference_memory_comparison.png")
    plt.close()


def plot_compression_breakdown():
    """Plot compression breakdown showing cache structure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data
    architectures = ["Standard GPT-2", "MLA (6x)", "KVSplice (12x)"]
    cache_sizes = [36.0, 12.0, 6.0]
    dimensions = [
        "2 × 12 layers × 1024 × 768 dims",
        "12 layers × 1024 × 256 dims",
        "12 layers × 1024 × 128 dims",
    ]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    # Create horizontal bar chart
    y_pos = np.arange(len(architectures))
    bars = ax.barh(
        y_pos, cache_sizes, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add value labels
    for i, (bar, size, dim) in enumerate(zip(bars, cache_sizes, dimensions)):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2.0,
            f"{size:.1f} MB",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=12,
        )
        ax.text(
            width / 2,
            bar.get_y() + bar.get_height() / 2.0,
            dim,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    # Add compression annotations
    ax.annotate(
        "",
        xy=(12, 1),
        xytext=(36, 1),
        arrowprops=dict(arrowstyle="<->", lw=2, color="black"),
    )
    ax.text(24, 1.3, "6x compression", ha="center", fontsize=10, fontweight="bold")

    ax.annotate(
        "",
        xy=(6, 0),
        xytext=(12, 0),
        arrowprops=dict(arrowstyle="<->", lw=2, color="black"),
    )
    ax.text(9, 0.3, "2x compression", ha="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(architectures)
    ax.set_xlabel("KV Cache Memory (MB)", fontsize=12)
    ax.set_title(
        "KV Cache Compression Breakdown (1024 tokens)", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 42)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("docs/kvsplice/compression_breakdown.png", dpi=300, bbox_inches="tight")
    print("Saved: docs/kvsplice/compression_breakdown.png")
    plt.close()


def plot_cache_shapes():
    """Visualize cache tensor shapes for each architecture."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Standard GPT-2: [n_layers, 2, B, H, T, d_head]
    # MLA: [n_layers, B, T, d_latent]
    # KVSplice: [n_layers, B, T, d_compressed]

    architectures = [
        ("Standard GPT-2", "[12, 2, 1, 12, 1024, 64]", 36.0, "#d62728"),
        ("MLA (6x)", "[12, 1, 1024, 256]", 12.0, "#ff7f0e"),
        ("KVSplice (12x)", "[12, 1, 1024, 128]", 6.0, "#2ca02c"),
    ]

    for ax, (name, shape, size, color) in zip(axes, architectures):
        # Create visualization of tensor
        ax.add_patch(
            mpatches.Rectangle(
                (0.2, 0.3),
                0.6,
                0.4,
                fill=True,
                facecolor=color,
                edgecolor="black",
                linewidth=2,
                alpha=0.7,
            )
        )

        # Add text
        ax.text(
            0.5,
            0.5,
            f"{size:.1f} MB",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="white",
        )
        ax.text(
            0.5,
            0.15,
            shape,
            ha="center",
            va="center",
            fontsize=10,
            fontfamily="monospace",
        )
        ax.set_title(name, fontsize=12, fontweight="bold")

        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.suptitle(
        "KV Cache Tensor Shapes (batch=1, seq_len=1024)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig("docs/kvsplice/cache_shapes.png", dpi=300, bbox_inches="tight")
    print("Saved: docs/kvsplice/cache_shapes.png")
    plt.close()


def plot_savings_percentage():
    """Plot memory savings percentage across sequence lengths."""
    fig, ax = plt.subplots(figsize=(10, 6))

    seq_lengths = [256, 512, 1024]
    x = np.arange(len(seq_lengths))
    width = 0.35

    # Savings vs Standard GPT-2
    mla_savings = [66.7, 66.7, 66.7]
    kvsplice_savings = [83.3, 83.3, 83.3]

    bars1 = ax.bar(
        x - width / 2,
        mla_savings,
        width,
        label="MLA (6x)",
        color="#ff7f0e",
        alpha=0.7,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        kvsplice_savings,
        width,
        label="KVSplice (12x)",
        color="#2ca02c",
        alpha=0.7,
        edgecolor="black",
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
                va="bottom",
                fontweight="bold",
            )

    ax.set_ylabel("Memory Reduction vs Standard GPT-2 (%)", fontsize=12)
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_title("KV Cache Memory Savings", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    # Add reference line at 50%
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2.5, 52, "50% savings", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig("docs/kvsplice/savings_percentage.png", dpi=300, bbox_inches="tight")
    print("Saved: docs/kvsplice/savings_percentage.png")
    plt.close()


def main():
    """Generate all plots."""
    print("Generating KVSplice inference memory verification plots...")

    plot_cache_comparison()
    plot_compression_breakdown()
    plot_cache_shapes()
    plot_savings_percentage()

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
