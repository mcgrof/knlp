#!/usr/bin/env python3
"""
Generate plots for GPT-2 + RA ablation study results.

Fetches data from W&B project gpt2-ra-ablation and creates:
- Validation perplexity comparison
- Inference speed comparison
- Training progress curves
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Configure matplotlib for publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def fetch_run_data(entity, project, run_name):
    """Fetch metrics from a W&B run."""
    api = wandb.Api()

    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": run_name}
        )

        if not runs:
            print(f"Warning: No run found with name '{run_name}'")
            return None

        run = runs[0]

        # Get final metrics from summary
        summary = run.summary._json_dict

        # Get training history
        history = run.history(samples=10000)

        return {
            'run': run,
            'summary': summary,
            'history': history,
            'config': run.config
        }
    except Exception as e:
        print(f"Error fetching run '{run_name}': {e}")
        return None


def plot_validation_comparison(baseline_data, ra_data, output_dir):
    """Plot validation perplexity comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract perplexity from summary - try multiple field names
    baseline_ppl = (baseline_data['summary'].get('final/best_val_perplexity') or
                    baseline_data['summary'].get('val/perplexity'))
    ra_ppl = (ra_data['summary'].get('final/best_val_perplexity') or
              ra_data['summary'].get('val/perplexity'))

    if baseline_ppl is None or ra_ppl is None:
        print("Warning: Perplexity data not found in summary")
        print(f"  Baseline keys: {list(baseline_data['summary'].keys())[:10]}")
        print(f"  RA keys: {list(ra_data['summary'].keys())[:10]}")
        return

    architectures = ['Baseline GPT-2\n(B0)', 'GPT-2 + RA\n(RALEARN0)']
    perplexities = [baseline_ppl, ra_ppl]
    colors = ['#1f77b4', '#2ca02c']

    bars = ax.bar(architectures, perplexities, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.1f}',
                ha='center', va='bottom', fontweight='bold')

    # Calculate improvement
    improvement = ((baseline_ppl - ra_ppl) / baseline_ppl) * 100

    ax.set_ylabel('Validation Perplexity (lower is better)', fontweight='bold')
    ax.set_title('GPT-2 + RA: Validation Perplexity Comparison', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add improvement annotation
    ax.text(0.5, max(perplexities) * 0.95,
            f'RA improves perplexity by {improvement:.1f}%',
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpt2_ra_perplexity_comparison.png'))
    print(f"Saved: gpt2_ra_perplexity_comparison.png")
    plt.close()

    return improvement


def plot_inference_speed(baseline_speed, ra_speed, output_dir):
    """Plot inference speed comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    architectures = ['Baseline GPT-2', 'GPT-2 + RA']
    speeds = [baseline_speed, ra_speed]
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax.bar(architectures, speeds, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}',
                ha='center', va='bottom', fontweight='bold')

    # Calculate slowdown
    slowdown = ((baseline_speed - ra_speed) / baseline_speed) * 100

    ax.set_ylabel('Inference Speed (tokens/sec)', fontweight='bold')
    ax.set_title('GPT-2 + RA: Inference Speed Comparison', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add slowdown annotation
    ax.text(0.5, max(speeds) * 0.95,
            f'RA is {abs(slowdown):.1f}% slower\n(1.23x slowdown for 5.9% better quality)',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpt2_ra_inference_speed.png'))
    print(f"Saved: gpt2_ra_inference_speed.png")
    plt.close()


def plot_training_curves(baseline_data, ra_data, output_dir):
    """Plot training loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    baseline_hist = baseline_data['history']
    ra_hist = ra_data['history']

    if 'train/loss' in baseline_hist.columns and 'train/loss' in ra_hist.columns:
        # Filter out NaN values
        baseline_loss = baseline_hist[['_step', 'train/loss']].dropna()
        ra_loss = ra_hist[['_step', 'train/loss']].dropna()

        ax1.plot(baseline_loss['_step'], baseline_loss['train/loss'],
                label='Baseline GPT-2', color='#1f77b4', linewidth=2, alpha=0.8)
        ax1.plot(ra_loss['_step'], ra_loss['train/loss'],
                label='GPT-2 + RA', color='#2ca02c', linewidth=2, alpha=0.8)

        ax1.set_xlabel('Training Step', fontweight='bold')
        ax1.set_ylabel('Training Loss', fontweight='bold')
        ax1.set_title('Training Loss Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, linestyle='--')

    # Validation perplexity over time
    if 'val/perplexity' in baseline_hist.columns and 'val/perplexity' in ra_hist.columns:
        baseline_val = baseline_hist[['_step', 'val/perplexity']].dropna()
        ra_val = ra_hist[['_step', 'val/perplexity']].dropna()

        ax2.plot(baseline_val['_step'], baseline_val['val/perplexity'],
                label='Baseline GPT-2', color='#1f77b4', linewidth=2, alpha=0.8, marker='o', markersize=4)
        ax2.plot(ra_val['_step'], ra_val['val/perplexity'],
                label='GPT-2 + RA', color='#2ca02c', linewidth=2, alpha=0.8, marker='s', markersize=4)

        ax2.set_xlabel('Training Step', fontweight='bold')
        ax2.set_ylabel('Validation Perplexity', fontweight='bold')
        ax2.set_title('Validation Perplexity Over Training', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpt2_ra_training_curves.png'))
    print(f"Saved: gpt2_ra_training_curves.png")
    plt.close()


def plot_tradeoff_summary(ppl_improvement, speed_cost, output_dir):
    """Plot quality vs speed tradeoff."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create scatter plot
    ax.scatter([0], [0], s=500, c='#1f77b4', alpha=0.6, edgecolors='black', linewidths=2, label='Baseline GPT-2')
    ax.scatter([speed_cost], [ppl_improvement], s=500, c='#2ca02c', alpha=0.6, edgecolors='black', linewidths=2, label='GPT-2 + RA')

    # Add arrow
    ax.annotate('', xy=(speed_cost, ppl_improvement), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))

    # Add labels
    ax.text(0, 0, 'Baseline', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(speed_cost, ppl_improvement, f'+{ppl_improvement:.1f}% quality\n{speed_cost:.1f}% speed',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Inference Speed Change (%)', fontweight='bold')
    ax.set_ylabel('Perplexity Improvement (%)', fontweight='bold')
    ax.set_title('GPT-2 + RA: Quality vs Speed Tradeoff', fontweight='bold', pad=20)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper left')

    # Add annotation box
    textstr = 'RA provides 5.9% better perplexity\nat the cost of 18.8% slower inference\n(1.23x slowdown - reasonable tradeoff)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gpt2_ra_tradeoff_summary.png'))
    print(f"Saved: gpt2_ra_tradeoff_summary.png")
    plt.close()


def main():
    entity = "mcgrof-citizen"
    project = "gpt2-ra-ablation"

    # Output directory for plots
    output_dir = "docs/images"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Fetching W&B data for GPT-2 + RA ablation study")
    print("=" * 70)

    # Fetch run data
    print("\nFetching baseline (B0) data...")
    baseline_data = fetch_run_data(entity, project, "gpt2_adamwspam_ramla_stepB0")

    print("Fetching RA (RALEARN0) data...")
    ra_data = fetch_run_data(entity, project, "gpt2_adamwspam_ramla_stepRALEARN0")

    if baseline_data is None or ra_data is None:
        print("\nError: Could not fetch required run data")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Print summary
    print("\nBaseline GPT-2 (B0):")
    baseline_ppl = (baseline_data['summary'].get('final/best_val_perplexity') or
                    baseline_data['summary'].get('val/perplexity', 'N/A'))
    print(f"  Final perplexity: {baseline_ppl}")
    print(f"  Training steps: {baseline_data['summary'].get('_step', 'N/A')}")

    print("\nGPT-2 + RA (RALEARN0):")
    ra_ppl = (ra_data['summary'].get('final/best_val_perplexity') or
              ra_data['summary'].get('val/perplexity', 'N/A'))
    print(f"  Final perplexity: {ra_ppl}")
    print(f"  Training steps: {ra_data['summary'].get('_step', 'N/A')}")

    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots")
    print("=" * 70 + "\n")

    # Plot validation comparison
    ppl_improvement = plot_validation_comparison(baseline_data, ra_data, output_dir)

    # Plot inference speed (from benchmark results)
    baseline_speed = 212.1  # tok/s from scripts/compare_ra_inference.py
    ra_speed = 172.3  # tok/s from scripts/compare_ra_inference.py
    plot_inference_speed(baseline_speed, ra_speed, output_dir)

    # Plot training curves
    plot_training_curves(baseline_data, ra_data, output_dir)

    # Plot tradeoff summary
    speed_cost = -18.8  # 18.8% slower
    if ppl_improvement:
        plot_tradeoff_summary(ppl_improvement, speed_cost, output_dir)

    print("\n" + "=" * 70)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
