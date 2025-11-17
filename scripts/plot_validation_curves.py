#!/usr/bin/env python3
"""
Plot validation loss curves from test matrix results.

Extracts validation loss from either:
1. training_metrics.json (if available - new tests after metrics fix)
2. output.log files (fallback - old tests before metrics fix)

Usage:
    python3 scripts/plot_validation_curves.py test_matrix_results_20251110_040616
"""

import argparse
import json
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def extract_from_log(log_file):
    """Extract validation loss from output.log."""
    iterations = []
    train_losses = []
    val_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # Match lines like: "Iter      0 | train loss 10.9683 | val loss 10.9690 | lr 0.00e+00"
            match = re.search(r'Iter\s+(\d+)\s+\|\s+train loss\s+([\d.]+)\s+\|\s+val loss\s+([\d.]+)', line)
            if match:
                iterations.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))

    return iterations, train_losses, val_losses


def extract_from_json(json_file):
    """Extract validation loss from training_metrics.json."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Check if eval metrics exist (new format after metrics fix)
    if 'eval_iteration' in data and data['eval_iteration']:
        return (
            data['eval_iteration'],
            data.get('train_loss', []),
            data.get('val_loss', [])
        )
    else:
        return None, None, None


def load_step_data(results_dir, step_name):
    """Load validation data for a specific step."""
    step_dir = Path(results_dir) / step_name

    if not step_dir.exists():
        return None, None, None

    # Try JSON first (new format)
    json_file = step_dir / "training_metrics.json"
    if json_file.exists():
        iters, train_loss, val_loss = extract_from_json(json_file)
        if iters:
            return iters, train_loss, val_loss

    # Fallback to log extraction (old format)
    log_file = step_dir / "output.log"
    if log_file.exists():
        return extract_from_log(log_file)

    return None, None, None


def plot_validation_curves(results_dir, output_file=None):
    """Plot validation loss curves for all steps."""
    results_dir = Path(results_dir)

    # Find all step directories
    step_dirs = sorted([d for d in results_dir.iterdir()
                       if d.is_dir() and d.name.startswith('gpt2_')])

    if not step_dirs:
        print(f"No test directories found in {results_dir}")
        return

    # Extract step names (V0, V1, etc.)
    step_data = {}
    for step_dir in step_dirs:
        # Extract step from name like "gpt2_adamwspam_ramla_stepV0"
        match = re.search(r'step([VRL]\d+|[0-9]+)$', step_dir.name)
        if not match:
            continue

        step_name = match.group(1)
        iters, train_loss, val_loss = load_step_data(results_dir, step_dir.name)

        if iters and val_loss:
            step_data[step_name] = {
                'iterations': iters,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'name': step_dir.name
            }

    if not step_data:
        print("No validation data found")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Define colors for steps
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot validation loss
    for i, (step, data) in enumerate(sorted(step_data.items())):
        color = colors[i % len(colors)]
        ax1.plot(data['iterations'], data['val_loss'],
                marker='o', label=f"Step {step}", color=color, linewidth=2)

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('Validation Loss over Training', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot training loss
    for i, (step, data) in enumerate(sorted(step_data.items())):
        if data['train_loss']:
            color = colors[i % len(colors)]
            ax2.plot(data['iterations'], data['train_loss'],
                    marker='s', label=f"Step {step}", color=color,
                    linewidth=2, alpha=0.7, markersize=4)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss over Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")
    else:
        output_file = results_dir / "validation_loss_curves.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("Validation Loss Summary (final values)")
    print("="*80)
    for step, data in sorted(step_data.items()):
        final_val = data['val_loss'][-1]
        final_train = data['train_loss'][-1] if data['train_loss'] else None
        n_iters = data['iterations'][-1]
        print(f"  {step:5s}: val_loss={final_val:.4f}  train_loss={final_train:.4f if final_train else 0:.4f}  ({n_iters:3d} iters)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation loss curves from test matrix results"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to test matrix results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results_dir/validation_loss_curves.png)"
    )

    args = parser.parse_args()

    plot_validation_curves(args.results_dir, args.output)


if __name__ == "__main__":
    main()
