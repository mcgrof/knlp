#!/usr/bin/env python3
"""
Plot lm-eval benchmark results comparing baseline vs compressed.

Usage:
    python scripts/plot_lm_eval_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load quality benchmark results."""
    results_file = Path("key_results/comprehensive_qwen7b_r120_quality.json")
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file) as f:
        return json.load(f)


def plot_task_comparison(results, output_dir="docs/kv_plugin"):
    """Plot baseline vs compressed task performance."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if "quality" not in results:
        print("No quality results found")
        return

    quality = results["quality"]
    baseline = quality.get("baseline", {})
    compressed = quality.get("compressed", {})

    tasks = list(baseline.keys())
    baseline_scores = [baseline[t] * 100 for t in tasks]
    compressed_scores = [compressed[t] * 100 for t in tasks]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Side-by-side bar chart
    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, compressed_scores, width, label='Compressed (r120)', color='#3498db', alpha=0.8)

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Task Performance: Baseline vs V-only Compression (Rank 120)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace('_', ' ').title() for t in tasks], fontsize=11)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(60, 90)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    # Plot 2: Delta chart
    deltas = [compressed[t] * 100 - baseline[t] * 100 for t in tasks]
    colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in deltas]

    bars3 = ax2.bar(x, deltas, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='-5% threshold')
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='+5% threshold')

    ax2.set_ylabel('Accuracy Change (%)', fontsize=12)
    ax2.set_title('Quality Impact of Compression (Rank 120)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.replace('_', ' ').title() for t in tasks], fontsize=11)
    ax2.set_ylim(-10, 10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, delta in zip(bars3, deltas):
        height = bar.get_height()
        ax2.annotate(f'{delta:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    # Add annotation about PPL
    ax2.annotate(
        'PPL increased +6%\nbut task accuracy unchanged',
        xy=(0.5, 0.15),
        xycoords='axes fraction',
        fontsize=10,
        ha='center',
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    output_path = f"{output_dir}/b200_lm_eval_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tradeoff_summary(output_dir="docs/kv_plugin"):
    """Plot comprehensive tradeoff summary."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Data for rank 120 and rank 96
    metrics = ['Memory\nSavings', 'Throughput\nPenalty', 'PPL\nIncrease', 'Task\nDegradation']
    rank_120 = [3.1, -13, 6, 0]
    rank_96 = [12.5, -14, 35, None]  # None for unknown

    x = np.arange(len(metrics))
    width = 0.35

    # Plot rank 120
    colors_120 = ['#27ae60', '#e74c3c', '#f39c12', '#27ae60']
    bars1 = ax.bar(x - width/2, rank_120, width, label='Rank 120 (conservative)',
                   color=colors_120, alpha=0.8, edgecolor='black', linewidth=1)

    # Plot rank 96 (with handling for None)
    rank_96_plot = [v if v is not None else 0 for v in rank_96]
    colors_96 = ['#27ae60', '#e74c3c', '#e74c3c', '#bdc3c7']
    bars2 = ax.bar(x + width/2, rank_96_plot, width, label='Rank 96 (aggressive)',
                   color=colors_96, alpha=0.8, edgecolor='black', linewidth=1)

    # Mark the unknown value
    ax.annotate('?', xy=(x[3] + width/2, 2), fontsize=20, ha='center', fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('KV Compression Tradeoffs: Memory vs Throughput vs Quality', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(-20, 40)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, rank_120):
        if val is not None:
            ax.annotate(f'{val:+.1f}%' if val != 0 else '0%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3 if bar.get_height() >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                       fontsize=9, fontweight='bold')

    for bar, val in zip(bars2, rank_96):
        if val is not None:
            ax.annotate(f'{val:+.1f}%' if val != 0 else '0%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3 if bar.get_height() >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                       fontsize=9, fontweight='bold')

    # Add verdict box
    ax.annotate(
        'VERDICT: Throughput cost (-13%) exceeds memory benefit (+3-12%)',
        xy=(0.5, 0.02),
        xycoords='axes fraction',
        fontsize=11,
        ha='center',
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8)
    )

    plt.tight_layout()
    output_path = f"{output_dir}/b200_tradeoff_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results = load_results()

    if results:
        plot_task_comparison(results)

    plot_tradeoff_summary()
    print("\nPlots generated in docs/kv_plugin/")


if __name__ == "__main__":
    main()
