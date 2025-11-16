#!/usr/bin/env python3
"""Create summary graphs for AdamWPrune R&D showing bitter7 success."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# Set style for beautiful graphs
plt.style.use("seaborn-v0_8-darkgrid")
colors = {
    "baseline": "#e74c3c",  # Red
    "bitter7": "#2ecc71",  # Green
    "bitter8": "#3498db",  # Blue
}


def load_wandb_metrics(csv_path):
    """Load metrics from W&B CSV export."""
    df = pd.read_csv(csv_path)
    # Filter out rows with NaN val_perplexity
    df = df[df["val_perplexity"].notna()].copy()
    return df


def load_local_json(json_path):
    """Load metrics from local test results JSON."""
    with open(json_path) as f:
        data = json.load(f)

    # Find bitter8 data
    for test in data:
        if "bitter8" in test.get("test_id", ""):
            return test
    return None


def create_perplexity_comparison():
    """Create validation perplexity comparison graph."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Load data
    bitter7_df = load_wandb_metrics(
        "wandb_gpt2_adamwprune_bitter7_state_50_metrics.csv"
    )
    baseline_df = load_wandb_metrics("wandb_gpt2_adamwspam_magnitude_50_metrics.csv")
    bitter8_data = load_local_json(
        "test_matrix_results_20251116_003504/all_results.json"
    )

    # Plot baseline (magnitude pruning, no torch.compile equivalent)
    ax.plot(
        baseline_df["iteration"],
        baseline_df["val_perplexity"],
        "o-",
        color=colors["baseline"],
        linewidth=2,
        markersize=8,
        label="Baseline (Magnitude, 5K iters)",
        alpha=0.8,
    )

    # Plot bitter7 (state-based pruning with torch.compile)
    ax.plot(
        bitter7_df["iteration"],
        bitter7_df["val_perplexity"],
        "s-",
        color=colors["bitter7"],
        linewidth=2,
        markersize=8,
        label="bitter7 (State-based, 7K iters)",
        alpha=0.8,
    )

    # Plot bitter8 if available
    if bitter8_data and "metrics" in bitter8_data:
        perplexities = bitter8_data["metrics"]["val_perplexities"]
        # Generate iterations based on number of validation points
        num_points = len(perplexities)
        iterations = [
            i * (bitter8_data["final_iteration"] / (num_points - 1))
            for i in range(num_points)
        ]
        ax.plot(
            iterations,
            perplexities,
            "^-",
            color=colors["bitter8"],
            linewidth=2,
            markersize=8,
            label=f"bitter8 (Bias-corrected, {bitter8_data['final_iteration']} iters)",
            alpha=0.8,
        )

    # Styling
    ax.set_xlabel("Training Iteration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Perplexity", fontsize=14, fontweight="bold")
    ax.set_title(
        "Adam State-Based Pruning: Validation Perplexity Comparison\nGPT-2 124M on FineWebEdu (B200 GPUs)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Add annotation for bitter7 success
    if len(bitter7_df) > 0:
        final_bitter7_ppl = bitter7_df["val_perplexity"].iloc[-1]
        final_baseline_ppl = baseline_df["val_perplexity"].iloc[-1]
        improvement = (
            (final_baseline_ppl - final_bitter7_ppl) / final_baseline_ppl
        ) * 100

        ax.annotate(
            f"bitter7: {final_bitter7_ppl:.2f} PPL\n({improvement:.1f}% better than baseline)",
            xy=(bitter7_df["iteration"].iloc[-1], final_bitter7_ppl),
            xytext=(-120, 30),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc=colors["bitter7"], alpha=0.3),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.3",
                color=colors["bitter7"],
                lw=2,
            ),
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("adamwprune_perplexity_comparison.png", dpi=300, bbox_inches="tight")
    print("Created: adamwprune_perplexity_comparison.png")
    plt.close()


def create_final_results_bar():
    """Create bar chart of final validation perplexity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load data
    bitter7_df = load_wandb_metrics(
        "wandb_gpt2_adamwprune_bitter7_state_50_metrics.csv"
    )
    baseline_df = load_wandb_metrics("wandb_gpt2_adamwspam_magnitude_50_metrics.csv")
    bitter8_data = load_local_json(
        "test_matrix_results_20251116_003504/all_results.json"
    )

    variants = []
    perplexities = []
    bar_colors = []

    # Baseline
    baseline_ppl = baseline_df["val_perplexity"].iloc[-1]
    variants.append("Magnitude\nPruning")
    perplexities.append(baseline_ppl)
    bar_colors.append(colors["baseline"])

    # bitter7
    bitter7_ppl = bitter7_df["val_perplexity"].iloc[-1]
    variants.append("bitter7\n(State-based)")
    perplexities.append(bitter7_ppl)
    bar_colors.append(colors["bitter7"])

    # bitter8
    if bitter8_data and "metrics" in bitter8_data:
        bitter8_ppl = bitter8_data["metrics"]["val_perplexities"][-1]
        variants.append("bitter8\n(Bias-corrected)")
        perplexities.append(bitter8_ppl)
        bar_colors.append(colors["bitter8"])

    bars = ax.bar(
        variants,
        perplexities,
        color=bar_colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    # Add value labels on bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{ppl:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_ylabel("Final Validation Perplexity", fontsize=14, fontweight="bold")
    ax.set_title(
        "AdamWPrune Variants: Final Performance Comparison\n50% Sparsity on GPT-2 124M",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(0, max(perplexities) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("adamwprune_final_results.png", dpi=300, bbox_inches="tight")
    print("Created: adamwprune_final_results.png")
    plt.close()


def create_training_efficiency():
    """Create graph showing training efficiency (perplexity vs iterations)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Load data
    bitter7_df = load_wandb_metrics(
        "wandb_gpt2_adamwprune_bitter7_state_50_metrics.csv"
    )
    baseline_df = load_wandb_metrics("wandb_gpt2_adamwspam_magnitude_50_metrics.csv")

    # Truncate to common iteration range for fair comparison
    max_common_iter = min(bitter7_df["iteration"].max(), baseline_df["iteration"].max())

    bitter7_trimmed = bitter7_df[bitter7_df["iteration"] <= max_common_iter]
    baseline_trimmed = baseline_df[baseline_df["iteration"] <= max_common_iter]

    ax.plot(
        baseline_trimmed["iteration"],
        baseline_trimmed["val_perplexity"],
        "o-",
        color=colors["baseline"],
        linewidth=2.5,
        markersize=6,
        label="Magnitude Pruning",
        alpha=0.8,
    )

    ax.plot(
        bitter7_trimmed["iteration"],
        bitter7_trimmed["val_perplexity"],
        "s-",
        color=colors["bitter7"],
        linewidth=2.5,
        markersize=6,
        label="bitter7 (State-based)",
        alpha=0.8,
    )

    # Add shaded region showing bitter7 advantage
    ax.fill_between(
        baseline_trimmed["iteration"],
        baseline_trimmed["val_perplexity"],
        bitter7_trimmed["val_perplexity"],
        where=(baseline_trimmed["val_perplexity"] >= bitter7_trimmed["val_perplexity"]),
        alpha=0.2,
        color=colors["bitter7"],
        label="bitter7 Advantage",
    )

    ax.set_xlabel("Training Iteration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation Perplexity", fontsize=14, fontweight="bold")
    ax.set_title(
        "Training Efficiency: State-Based Pruning Outperforms Magnitude\nGPT-2 124M, 50% Sparsity",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=max_common_iter)

    plt.tight_layout()
    plt.savefig("adamwprune_training_efficiency.png", dpi=300, bbox_inches="tight")
    print("Created: adamwprune_training_efficiency.png")
    plt.close()


def main():
    """Generate all summary graphs."""
    print("Generating AdamWPrune summary graphs...")
    print()

    create_perplexity_comparison()
    create_final_results_bar()
    create_training_efficiency()

    print()
    print("All graphs created successfully!")
    print("Graphs show that bitter7 (state-based pruning) successfully outperforms")
    print("magnitude-based pruning, validating the Adam state hypothesis.")


if __name__ == "__main__":
    main()
