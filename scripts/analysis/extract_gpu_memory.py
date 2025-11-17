#!/usr/bin/env python3
"""
Extract GPU memory usage from W&B runs and create comparison visualization.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_gpu_memory_stats(entity, project, run_id):
    """Extract GPU memory statistics from a W&B run."""
    api = wandb.Api()

    run = api.run(f"{entity}/{project}/{run_id}")

    print(f"\n{'='*80}")
    print(f"Run: {run.name}")
    print(f"ID: {run.id}")
    print(f"State: {run.state}")
    print(f"{'='*80}")

    # Get full history with pandas
    df = run.history(pandas=True)

    print(f"\nTotal columns: {len(df.columns)}")

    # Find GPU memory columns
    gpu_mem_cols = [
        c for c in df.columns
        if c.startswith("system/gpu.") and c.endswith("memoryAllocated")
    ]

    print(f"GPU memory columns found: {gpu_mem_cols}")

    if not gpu_mem_cols:
        print("WARNING: No GPU memory metrics found!")
        return None

    # Extract GPU memory data with timestamps
    gpu_mem_df = df[["_step", "_timestamp"] + gpu_mem_cols].copy()

    # Convert bytes to GiB
    for col in gpu_mem_cols:
        gib_col = col.replace("memoryAllocated", "memoryAllocated_GiB")
        gpu_mem_df[gib_col] = gpu_mem_df[col] / (1024**3)

    # Compute statistics
    print(f"\nGPU Memory Statistics (GiB):")
    print(f"{'GPU':<10} {'Mean':<12} {'Peak':<12} {'Min':<12}")
    print("-" * 50)

    stats = {}
    for col in gpu_mem_cols:
        gpu_id = col.split(".")[1]  # Extract GPU number
        gib_col = col.replace("memoryAllocated", "memoryAllocated_GiB")

        mean_mem = gpu_mem_df[gib_col].mean()
        peak_mem = gpu_mem_df[gib_col].max()
        min_mem = gpu_mem_df[gib_col].min()

        print(f"GPU {gpu_id:<6} {mean_mem:<12.2f} {peak_mem:<12.2f} {min_mem:<12.2f}")

        stats[gpu_id] = {
            'mean': mean_mem,
            'peak': peak_mem,
            'min': min_mem,
            'col': col
        }

    return {
        'run': run,
        'df': gpu_mem_df,
        'gpu_cols': gpu_mem_cols,
        'stats': stats
    }

def compare_runs(runs_data):
    """Create comparison visualization for multiple runs."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GPU Memory Comparison: Magnitude vs Bitter7', fontsize=14, fontweight='bold')

    # Plot 1: GPU 0 memory over time
    ax1 = axes[0, 0]
    for run_name, data in runs_data.items():
        if data is None:
            continue
        df = data['df']
        gib_col = data['gpu_cols'][0].replace("memoryAllocated", "memoryAllocated_GiB")
        ax1.plot(df['_step'], df[gib_col],
                label=run_name.replace('gpt2_', ''), linewidth=2, alpha=0.8)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('GPU Memory (GiB)')
    ax1.set_title('GPU 0 Memory Usage Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: All GPUs for first run
    ax2 = axes[0, 1]
    first_run = list(runs_data.values())[0]
    if first_run is not None:
        for col in first_run['gpu_cols']:
            gpu_id = col.split(".")[1]
            gib_col = col.replace("memoryAllocated", "memoryAllocated_GiB")
            ax2.plot(first_run['df']['_step'], first_run['df'][gib_col],
                    label=f'GPU {gpu_id}', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('GPU Memory (GiB)')
    ax2.set_title(f'All GPUs - {list(runs_data.keys())[0].replace("gpt2_", "")}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Peak memory comparison (bar chart)
    ax3 = axes[1, 0]

    run_names = []
    peak_memories = []

    for run_name, data in runs_data.items():
        if data is None:
            continue
        run_names.append(run_name.replace('gpt2_', '').replace('_', '\n'))
        # Average peak across all GPUs
        avg_peak = np.mean([s['peak'] for s in data['stats'].values()])
        peak_memories.append(avg_peak)

    x = np.arange(len(run_names))
    bars = ax3.bar(x, peak_memories, color=['#1f77b4', '#ff7f0e'], alpha=0.8, width=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(run_names)
    ax3.set_ylabel('Peak GPU Memory (GiB)')
    ax3.set_title('Average Peak Memory Per Run')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, peak_memories):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} GiB', ha='center', va='bottom', fontsize=10)

    # Plot 4: Memory distribution (box plot)
    ax4 = axes[1, 1]

    box_data = []
    box_labels = []

    for run_name, data in runs_data.items():
        if data is None:
            continue
        # Collect all GPU memory values for this run
        all_mem = []
        for col in data['gpu_cols']:
            gib_col = col.replace("memoryAllocated", "memoryAllocated_GiB")
            all_mem.extend(data['df'][gib_col].dropna().tolist())

        if all_mem:
            box_data.append(all_mem)
            box_labels.append(run_name.replace('gpt2_', '').replace('_', '\n'))

    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel('GPU Memory (GiB)')
    ax4.set_title('Memory Distribution Across All GPUs')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('gpu_memory_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print("Saved visualization to: gpu_memory_comparison.png")
    print(f"{'='*80}")

def main():
    entity = "mcgrof-citizen"
    project = "gpt2-bitter7-b200x4"

    # Define runs to compare
    runs_config = {
        "gpt2_adamwspam_magnitude_50": "rswzg7b8",      # Magnitude baseline
        "gpt2_adamwprune_bitter7_state_50": "eptbmgdt",  # Bitter7
    }

    runs_data = {}

    for run_name, run_id in runs_config.items():
        data = get_gpu_memory_stats(entity, project, run_id)
        runs_data[run_name] = data

        if data is not None:
            # Save individual run data
            csv_file = f"gpu_memory_{run_name}.csv"
            data['df'].to_csv(csv_file, index=False)
            print(f"\nSaved to: {csv_file}")

    # Create comparison visualization
    if any(d is not None for d in runs_data.values()):
        compare_runs(runs_data)

        # Print summary comparison
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")

        for run_name, data in runs_data.items():
            if data is None:
                continue

            avg_peak = np.mean([s['peak'] for s in data['stats'].values()])
            avg_mean = np.mean([s['mean'] for s in data['stats'].values()])

            print(f"\n{run_name}:")
            print(f"  Average Peak Memory: {avg_peak:.2f} GiB")
            print(f"  Average Mean Memory: {avg_mean:.2f} GiB")

        # Calculate difference
        if len(runs_data) == 2 and all(d is not None for d in runs_data.values()):
            mag_data = list(runs_data.values())[0]
            b7_data = list(runs_data.values())[1]

            mag_peak = np.mean([s['peak'] for s in mag_data['stats'].values()])
            b7_peak = np.mean([s['peak'] for s in b7_data['stats'].values()])

            diff = b7_peak - mag_peak
            pct_diff = (diff / mag_peak) * 100

            print(f"\n{'='*80}")
            print(f"Bitter7 uses {diff:+.2f} GiB ({pct_diff:+.1f}%) compared to Magnitude")
            print(f"{'='*80}")

if __name__ == "__main__":
    main()
