#!/usr/bin/env python3
"""
Analyze GPU memory utilization (time spent accessing memory).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_memory_utilization(csv_file, run_name):
    """Analyze GPU memory utilization metrics."""
    df = pd.read_csv(csv_file)

    print(f"\n{'='*80}")
    print(f"{run_name}")
    print(f"{'='*80}")

    # system.gpu.X.memory is memory utilization (percentage of time accessing memory)
    mem_util_cols = [c for c in df.columns if c.endswith('.memory') and 'system.gpu.' in c]

    print(f"\nGPU Memory Utilization (% time spent accessing memory):")
    print(f"{'GPU':<10} {'Mean %':<12} {'Peak %':<12} {'Min %':<12}")
    print("-" * 50)

    stats = {}
    for col in sorted(mem_util_cols):
        gpu_id = col.split('.')[2]
        data = df[col].dropna()

        if len(data) > 0:
            mean_util = data.mean()
            peak_util = data.max()
            min_util = data.min()

            print(f"GPU {gpu_id:<6} {mean_util:<12.2f} {peak_util:<12.2f} {min_util:<12.2f}")

            stats[gpu_id] = {
                'mean_util': mean_util,
                'peak_util': peak_util,
                'min_util': min_util,
                'col': col
            }

    # Also look at GPU compute utilization
    gpu_util_cols = [c for c in df.columns if c.endswith('.gpu') and 'system.gpu.' in c]

    if gpu_util_cols:
        print(f"\nGPU Compute Utilization (% time doing compute):")
        print(f"{'GPU':<10} {'Mean %':<12} {'Peak %':<12} {'Min %':<12}")
        print("-" * 50)

        for col in sorted(gpu_util_cols):
            gpu_id = col.split('.')[2]
            data = df[col].dropna()

            if len(data) > 0:
                mean_util = data.mean()
                peak_util = data.max()
                min_util = data.min()

                print(f"GPU {gpu_id:<6} {mean_util:<12.2f} {peak_util:<12.2f} {min_util:<12.2f}")

                if gpu_id in stats:
                    stats[gpu_id]['compute_mean'] = mean_util
                    stats[gpu_id]['compute_peak'] = peak_util

    return df, stats

def create_utilization_plots(mag_df, mag_stats, b7_df, b7_stats):
    """Create memory utilization comparison visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GPU Memory Access Time: Magnitude vs Bitter7', fontsize=14, fontweight='bold')

    gpus = sorted(mag_stats.keys())

    # Plot 1: Average memory utilization comparison
    ax1 = axes[0, 0]

    x = np.arange(len(gpus))
    width = 0.35

    mag_mem_util = [mag_stats[gpu]['mean_util'] for gpu in gpus]
    b7_mem_util = [b7_stats[gpu]['mean_util'] for gpu in gpus]

    bars1 = ax1.bar(x - width/2, mag_mem_util, width, label='Magnitude', alpha=0.8, color='#1f77b4')
    bars2 = ax1.bar(x + width/2, b7_mem_util, width, label='Bitter7', alpha=0.8, color='#ff7f0e')

    ax1.set_xlabel('GPU')
    ax1.set_ylabel('Memory Utilization (%)')
    ax1.set_title('Average % Time Spent Accessing Memory')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'GPU {gpu}' for gpu in gpus])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: Peak memory utilization comparison
    ax2 = axes[0, 1]

    mag_mem_peak = [mag_stats[gpu]['peak_util'] for gpu in gpus]
    b7_mem_peak = [b7_stats[gpu]['peak_util'] for gpu in gpus]

    bars1 = ax2.bar(x - width/2, mag_mem_peak, width, label='Magnitude', alpha=0.8, color='#1f77b4')
    bars2 = ax2.bar(x + width/2, b7_mem_peak, width, label='Bitter7', alpha=0.8, color='#ff7f0e')

    ax2.set_xlabel('GPU')
    ax2.set_ylabel('Memory Utilization (%)')
    ax2.set_title('Peak % Time Spent Accessing Memory')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'GPU {gpu}' for gpu in gpus])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    # Plot 3: GPU 0 memory utilization over time
    ax3 = axes[1, 0]

    gpu0_mem_col = 'system.gpu.0.memory'
    if gpu0_mem_col in mag_df.columns and gpu0_mem_col in b7_df.columns:
        mag_data = mag_df[gpu0_mem_col].dropna()
        b7_data = b7_df[gpu0_mem_col].dropna()

        ax3.plot(range(len(mag_data)), mag_data, label='Magnitude', linewidth=1.5, alpha=0.8)
        ax3.plot(range(len(b7_data)), b7_data, label='Bitter7', linewidth=1.5, alpha=0.8)

    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Memory Utilization (%)')
    ax3.set_title('GPU 0 Memory Access Time Over Training')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Compute vs Memory utilization
    ax4 = axes[1, 1]

    if 'compute_mean' in mag_stats['0'] and 'compute_mean' in b7_stats['0']:
        categories = ['Compute\nUtil', 'Memory\nUtil']

        mag_avg_compute = np.mean([s['compute_mean'] for s in mag_stats.values()])
        mag_avg_memory = np.mean([s['mean_util'] for s in mag_stats.values()])

        b7_avg_compute = np.mean([s['compute_mean'] for s in b7_stats.values()])
        b7_avg_memory = np.mean([s['mean_util'] for s in b7_stats.values()])

        mag_vals = [mag_avg_compute, mag_avg_memory]
        b7_vals = [b7_avg_compute, b7_avg_memory]

        x_pos = np.arange(len(categories))
        bars1 = ax4.bar(x_pos - width/2, mag_vals, width, label='Magnitude', alpha=0.8, color='#1f77b4')
        bars2 = ax4.bar(x_pos + width/2, b7_vals, width, label='Bitter7', alpha=0.8, color='#ff7f0e')

        ax4.set_ylabel('Utilization (%)')
        ax4.set_title('Compute vs Memory Utilization')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(list(bars1) + list(bars2), mag_vals + b7_vals):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('gpu_memory_access_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print("Saved visualization to: gpu_memory_access_comparison.png")
    print(f"{'='*80}")

    # Print summary
    print(f"\n{'='*80}")
    print("MEMORY ACCESS TIME SUMMARY")
    print(f"{'='*80}")

    mag_avg_util = np.mean([s['mean_util'] for s in mag_stats.values()])
    b7_avg_util = np.mean([s['mean_util'] for s in b7_stats.values()])

    print(f"\nMagnitude Pruning:")
    print(f"  Average Memory Utilization: {mag_avg_util:.2f}%")

    print(f"\nBitter7 State Pruning:")
    print(f"  Average Memory Utilization: {b7_avg_util:.2f}%")

    diff = b7_avg_util - mag_avg_util
    pct_increase = (diff / mag_avg_util) * 100 if mag_avg_util > 0 else 0

    print(f"\nDifference (Bitter7 - Magnitude):")
    print(f"  Memory Utilization: {diff:+.2f}% ({pct_increase:+.1f}% increase)")

    if 'compute_mean' in mag_stats['0'] and 'compute_mean' in b7_stats['0']:
        mag_avg_compute = np.mean([s['compute_mean'] for s in mag_stats.values()])
        b7_avg_compute = np.mean([s['compute_mean'] for s in b7_stats.values()])

        print(f"\nCompute Utilization:")
        print(f"  Magnitude: {mag_avg_compute:.2f}%")
        print(f"  Bitter7:   {b7_avg_compute:.2f}%")
        print(f"  Difference: {b7_avg_compute - mag_avg_compute:+.2f}%")

    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("\nMemory utilization = % of time GPU is waiting on memory access")
    print("(Higher = more memory-bound, spending more time accessing memory)")
    print("\nCompute utilization = % of time GPU is doing actual computation")
    print("(Higher = more compute-bound, better GPU usage)")

    if diff > 5:
        print(f"\n→ Bitter7 is significantly MORE MEMORY-BOUND ({diff:.1f}% more time)")
        print("   This is likely because state-based pruning needs to:")
        print("   - Read optimizer states (momentum, variance) frequently")
        print("   - Compute importance scores from these states")
        print("   - Update pruning masks based on state statistics")
    elif diff > 0:
        print(f"\n→ Bitter7 is slightly more memory-bound (+{diff:.1f}%)")
    else:
        print(f"\n→ Memory access patterns are similar")

    print(f"{'='*80}")

def main():
    # Load and analyze
    mag_df, mag_stats = analyze_memory_utilization(
        "system_metrics_gpt2_adamwspam_magnitude_50.csv",
        "Magnitude Pruning (AdamWSPAM)"
    )

    b7_df, b7_stats = analyze_memory_utilization(
        "system_metrics_gpt2_adamwprune_bitter7_state_50.csv",
        "Bitter7 State Pruning (AdamWPrune)"
    )

    # Create comparison plots
    create_utilization_plots(mag_df, mag_stats, b7_df, b7_stats)

if __name__ == "__main__":
    main()
