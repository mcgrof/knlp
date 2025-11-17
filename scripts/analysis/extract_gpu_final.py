#!/usr/bin/env python3
"""
Extract GPU memory usage from W&B using scan_history to get system metrics.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_gpu_memory_stats(entity, project, run_id, run_name):
    """Extract GPU memory statistics from a W&B run."""
    api = wandb.Api()

    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)

    print(f"\n{'='*80}")
    print(f"Run: {run.name}")
    print(f"ID: {run.id}")
    print(f"State: {run.state}")
    print(f"{'='*80}")

    # Use scan_history to get ALL data including system metrics
    print("Scanning history (this may take a moment)...")
    history = run.scan_history()

    # Convert to DataFrame
    history_df = pd.DataFrame(history)

    print(f"\nTotal rows: {len(history_df)}")
    print(f"Total columns: {len(history_df.columns)}")

    print("\nAll available columns:")
    for col in sorted(history_df.columns):
        print(f"  - {col}")

    # Find GPU memory columns
    gpu_mem_cols = [
        c for c in history_df.columns
        if ('system' in c and 'gpu' in c and 'memory' in c.lower())
        or (c.startswith("system/gpu.") and "memory" in c.lower())
    ]

    print(f"\nGPU memory columns found: {len(gpu_mem_cols)}")
    for col in gpu_mem_cols:
        print(f"  - {col}")

    if not gpu_mem_cols:
        print("WARNING: No GPU memory metrics found!")

        # Show system-related columns for debugging
        system_cols = [c for c in history_df.columns if 'system' in c.lower()]
        if system_cols:
            print(f"\nFound {len(system_cols)} system-related columns:")
            for col in system_cols[:20]:
                print(f"  - {col}")

        return None

    # Extract GPU memory data with timestamps
    cols_to_extract = ['_step', '_timestamp'] + gpu_mem_cols
    available_cols = [c for c in cols_to_extract if c in history_df.columns]

    gpu_mem_df = history_df[available_cols].copy()

    # Convert to GiB if the values are in bytes
    gib_cols = []
    for col in gpu_mem_cols:
        # Check if values look like bytes (> 1000)
        sample_val = gpu_mem_df[col].dropna().iloc[0] if len(gpu_mem_df[col].dropna()) > 0 else 0

        if sample_val > 1000:  # Likely bytes
            gib_col = col + "_GiB"
            gpu_mem_df[gib_col] = gpu_mem_df[col] / (1024**3)
            gib_cols.append(gib_col)
        else:  # Already in GiB or similar
            gib_cols.append(col)

    # Compute statistics
    print(f"\nGPU Memory Statistics:")
    print(f"{'Metric':<50} {'Mean':<12} {'Peak':<12} {'Min':<12}")
    print("-" * 90)

    stats = {}
    for i, col in enumerate(gpu_mem_cols):
        gib_col = gib_cols[i] if i < len(gib_cols) else col

        data = gpu_mem_df[gib_col].dropna()
        if len(data) == 0:
            continue

        mean_mem = data.mean()
        peak_mem = data.max()
        min_mem = data.min()

        print(f"{col:<50} {mean_mem:<12.2f} {peak_mem:<12.2f} {min_mem:<12.2f}")

        stats[col] = {
            'mean': mean_mem,
            'peak': peak_mem,
            'min': min_mem,
            'gib_col': gib_col
        }

    return {
        'run': run,
        'run_name': run_name,
        'df': gpu_mem_df,
        'gpu_cols': gpu_mem_cols,
        'gib_cols': gib_cols,
        'stats': stats
    }

def compare_runs(runs_data):
    """Create comparison visualization for multiple runs."""

    valid_runs = {k: v for k, v in runs_data.items() if v is not None}

    if len(valid_runs) == 0:
        print("\nNo valid runs with GPU data to compare!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GPU Memory Comparison: Magnitude vs Bitter7', fontsize=14, fontweight='bold')

    # Plot 1: First GPU metric over time for all runs
    ax1 = axes[0, 0]
    for run_name, data in valid_runs.items():
        if len(data['gib_cols']) > 0:
            first_col = data['gib_cols'][0]
            df = data['df']

            if '_step' in df.columns:
                ax1.plot(df['_step'], df[first_col],
                        label=run_name.replace('gpt2_', ''), linewidth=2, alpha=0.8)

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('GPU Memory (GiB)')
    ax1.set_title('GPU Memory Usage Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: All GPU metrics for first run
    ax2 = axes[0, 1]
    first_run = list(valid_runs.values())[0]
    for i, col in enumerate(first_run['gib_cols']):
        if '_step' in first_run['df'].columns:
            ax2.plot(first_run['df']['_step'], first_run['df'][col],
                    label=first_run['gpu_cols'][i].split('/')[-1], linewidth=2, alpha=0.8)

    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('GPU Memory (GiB)')
    ax2.set_title(f'All GPU Metrics - {list(valid_runs.keys())[0].replace("gpt2_", "")}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Peak memory comparison (bar chart)
    ax3 = axes[1, 0]

    run_names = []
    peak_memories = []

    for run_name, data in valid_runs.items():
        run_names.append(run_name.replace('gpt2_', '').replace('_', '\n'))
        # Average peak across all GPU metrics
        avg_peak = np.mean([s['peak'] for s in data['stats'].values()])
        peak_memories.append(avg_peak)

    x = np.arange(len(run_names))
    bars = ax3.bar(x, peak_memories, color=['#1f77b4', '#ff7f0e'][:len(run_names)], alpha=0.8, width=0.6)
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

    for run_name, data in valid_runs.items():
        # Collect all GPU memory values for this run
        all_mem = []
        for gib_col in data['gib_cols']:
            all_mem.extend(data['df'][gib_col].dropna().tolist())

        if all_mem:
            box_data.append(all_mem)
            box_labels.append(run_name.replace('gpt2_', '').replace('_', '\n'))

    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(['#1f77b4', '#ff7f0e'][i % 2])
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

    # Define runs to compare (run_display_name: run_id)
    runs_config = {
        "gpt2_adamwspam_magnitude_50": "rswzg7b8",      # Magnitude baseline
        "gpt2_adamwprune_bitter7_state_50": "eptbmgdt",  # Bitter7
    }

    runs_data = {}

    for run_name, run_id in runs_config.items():
        data = get_gpu_memory_stats(entity, project, run_id, run_name)
        runs_data[run_name] = data

        if data is not None:
            # Save individual run data
            csv_file = f"gpu_memory_{run_name}.csv"
            data['df'].to_csv(csv_file, index=False)
            print(f"\nSaved to: {csv_file}")

    # Create comparison visualization
    valid_runs = {k: v for k, v in runs_data.items() if v is not None}

    if valid_runs:
        compare_runs(runs_data)

        # Print summary comparison
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")

        for run_name, data in valid_runs.items():
            avg_peak = np.mean([s['peak'] for s in data['stats'].values()])
            avg_mean = np.mean([s['mean'] for s in data['stats'].values()])

            print(f"\n{run_name}:")
            print(f"  Average Peak Memory: {avg_peak:.2f} GiB")
            print(f"  Average Mean Memory: {avg_mean:.2f} GiB")

        # Calculate difference
        if len(valid_runs) == 2:
            runs_list = list(valid_runs.values())
            mag_data = runs_list[0]
            b7_data = runs_list[1]

            mag_peak = np.mean([s['peak'] for s in mag_data['stats'].values()])
            b7_peak = np.mean([s['peak'] for s in b7_data['stats'].values()])

            diff = b7_peak - mag_peak
            pct_diff = (diff / mag_peak) * 100

            print(f"\n{'='*80}")
            print(f"Bitter7 uses {diff:+.2f} GiB ({pct_diff:+.1f}%) compared to Magnitude")
            print(f"{'='*80}")

if __name__ == "__main__":
    main()
