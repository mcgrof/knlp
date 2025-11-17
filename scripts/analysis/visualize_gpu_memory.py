#!/usr/bin/env python3
"""
Create GPU memory comparison visualization from system metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_analyze(csv_file, run_name):
    """Load system metrics and analyze GPU memory usage."""
    df = pd.read_csv(csv_file)

    print(f"\n{'='*80}")
    print(f"{run_name}")
    print(f"{'='*80}")

    # Find memoryAllocated columns (percentage)
    mem_alloc_cols = [c for c in df.columns
                      if c.endswith('.memoryAllocated') and not c.endswith('Bytes')]

    # Find memoryAllocatedBytes columns
    mem_bytes_cols = [c for c in df.columns if c.endswith('.memoryAllocatedBytes')]

    print(f"\nGPU Memory Allocated (percentage of total VRAM):")
    print(f"{'GPU':<10} {'Mean %':<12} {'Peak %':<12} {'Min %':<12}")
    print("-" * 50)

    stats = {}
    for col in sorted(mem_alloc_cols):
        gpu_id = col.split('.')[2]  # Extract GPU number
        data = df[col].dropna()

        if len(data) > 0:
            mean_pct = data.mean()
            peak_pct = data.max()
            min_pct = data.min()

            print(f"GPU {gpu_id:<6} {mean_pct:<12.2f} {peak_pct:<12.2f} {min_pct:<12.2f}")

            # Also get bytes for this GPU
            bytes_col = f'system.gpu.{gpu_id}.memoryAllocatedBytes'
            if bytes_col in df.columns:
                bytes_data = df[bytes_col].dropna()
                mean_gib = bytes_data.mean() / (1024**3)
                peak_gib = bytes_data.max() / (1024**3)
                min_gib = bytes_data.min() / (1024**3)

                stats[gpu_id] = {
                    'mean_pct': mean_pct,
                    'peak_pct': peak_pct,
                    'mean_gib': mean_gib,
                    'peak_gib': peak_gib,
                    'min_gib': min_gib,
                    'pct_col': col,
                    'bytes_col': bytes_col
                }

    print(f"\nGPU Memory Allocated (GiB):")
    print(f"{'GPU':<10} {'Mean GiB':<12} {'Peak GiB':<12} {'Min GiB':<12}")
    print("-" * 50)

    for gpu_id in sorted(stats.keys()):
        s = stats[gpu_id]
        print(f"GPU {gpu_id:<6} {s['mean_gib']:<12.2f} {s['peak_gib']:<12.2f} {s['min_gib']:<12.2f}")

    return df, stats

def create_comparison_plots(mag_df, mag_stats, b7_df, b7_stats):
    """Create comparison visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('GPU Memory Comparison: Magnitude Pruning vs Bitter7 State Pruning',
                 fontsize=14, fontweight='bold')

    # Plot 1: Peak memory per GPU (bar chart)
    ax1 = axes[0, 0]

    gpus = sorted(mag_stats.keys())
    x = np.arange(len(gpus))
    width = 0.35

    mag_peaks = [mag_stats[gpu]['peak_gib'] for gpu in gpus]
    b7_peaks = [b7_stats[gpu]['peak_gib'] for gpu in gpus]

    bars1 = ax1.bar(x - width/2, mag_peaks, width, label='Magnitude', alpha=0.8, color='#1f77b4')
    bars2 = ax1.bar(x + width/2, b7_peaks, width, label='Bitter7', alpha=0.8, color='#ff7f0e')

    ax1.set_xlabel('GPU')
    ax1.set_ylabel('Peak Memory (GiB)')
    ax1.set_title('Peak GPU Memory Per GPU')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'GPU {gpu}' for gpu in gpus])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Average memory per GPU (bar chart)
    ax2 = axes[0, 1]

    mag_means = [mag_stats[gpu]['mean_gib'] for gpu in gpus]
    b7_means = [b7_stats[gpu]['mean_gib'] for gpu in gpus]

    bars1 = ax2.bar(x - width/2, mag_means, width, label='Magnitude', alpha=0.8, color='#1f77b4')
    bars2 = ax2.bar(x + width/2, b7_means, width, label='Bitter7', alpha=0.8, color='#ff7f0e')

    ax2.set_xlabel('GPU')
    ax2.set_ylabel('Average Memory (GiB)')
    ax2.set_title('Average GPU Memory Per GPU')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'GPU {gpu}' for gpu in gpus])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 3: GPU 0 memory over time
    ax3 = axes[1, 0]

    gpu0_col = 'system.gpu.0.memoryAllocatedBytes'
    if gpu0_col in mag_df.columns and gpu0_col in b7_df.columns:
        mag_gpu0 = mag_df[gpu0_col].dropna() / (1024**3)
        b7_gpu0 = b7_df[gpu0_col].dropna() / (1024**3)

        ax3.plot(range(len(mag_gpu0)), mag_gpu0, label='Magnitude', linewidth=1.5, alpha=0.8)
        ax3.plot(range(len(b7_gpu0)), b7_gpu0, label='Bitter7', linewidth=1.5, alpha=0.8)

    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Memory Allocated (GiB)')
    ax3.set_title('GPU 0 Memory Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary comparison
    ax4 = axes[1, 1]

    # Calculate overall statistics
    mag_avg_peak = np.mean([s['peak_gib'] for s in mag_stats.values()])
    mag_avg_mean = np.mean([s['mean_gib'] for s in mag_stats.values()])

    b7_avg_peak = np.mean([s['peak_gib'] for s in b7_stats.values()])
    b7_avg_mean = np.mean([s['mean_gib'] for s in b7_stats.values()])

    categories = ['Average\nPeak', 'Average\nMean']
    mag_vals = [mag_avg_peak, mag_avg_mean]
    b7_vals = [b7_avg_peak, b7_avg_mean]

    x_pos = np.arange(len(categories))
    bars1 = ax4.bar(x_pos - width/2, mag_vals, width, label='Magnitude', alpha=0.8, color='#1f77b4')
    bars2 = ax4.bar(x_pos + width/2, b7_vals, width, label='Bitter7', alpha=0.8, color='#ff7f0e')

    ax4.set_ylabel('Memory (GiB)')
    ax4.set_title('Overall Memory Usage Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels and differences
    for i, (bar1, bar2, m, b) in enumerate(zip(bars1, bars2, mag_vals, b7_vals)):
        h1 = bar1.get_height()
        h2 = bar2.get_height()

        ax4.text(bar1.get_x() + bar1.get_width()/2., h1,
                f'{h1:.1f}', ha='center', va='bottom', fontsize=10)
        ax4.text(bar2.get_x() + bar2.get_width()/2., h2,
                f'{h2:.1f}', ha='center', va='bottom', fontsize=10)

        # Show difference
        diff = b - m
        pct_diff = (diff / m) * 100
        mid_x = x_pos[i]
        max_h = max(h1, h2)
        ax4.text(mid_x, max_h + 2,
                f'{diff:+.1f} GiB\n({pct_diff:+.1f}%)',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('gpu_memory_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print("Saved visualization to: gpu_memory_comparison.png")
    print(f"{'='*80}")

    # Print summary
    print(f"\n{'='*80}")
    print("MEMORY USAGE SUMMARY")
    print(f"{'='*80}")

    print(f"\nMagnitude Pruning (AdamWSPAM):")
    print(f"  Average Peak Memory: {mag_avg_peak:.2f} GiB")
    print(f"  Average Mean Memory: {mag_avg_mean:.2f} GiB")

    print(f"\nBitter7 State Pruning (AdamWPrune):")
    print(f"  Average Peak Memory: {b7_avg_peak:.2f} GiB")
    print(f"  Average Mean Memory: {b7_avg_mean:.2f} GiB")

    print(f"\nDifference (Bitter7 - Magnitude):")
    peak_diff = b7_avg_peak - mag_avg_peak
    mean_diff = b7_avg_mean - mag_avg_mean
    peak_pct = (peak_diff / mag_avg_peak) * 100
    mean_pct = (mean_diff / mag_avg_mean) * 100

    print(f"  Peak Memory: {peak_diff:+.2f} GiB ({peak_pct:+.1f}%)")
    print(f"  Mean Memory: {mean_diff:+.2f} GiB ({mean_pct:+.1f}%)")

    if abs(peak_pct) < 5:
        print(f"\n→ Memory usage is essentially IDENTICAL (within 5%)")
    elif peak_diff > 0:
        print(f"\n→ Bitter7 uses MORE memory than Magnitude")
    else:
        print(f"\n→ Bitter7 uses LESS memory than Magnitude")

    print(f"{'='*80}")

def main():
    # Load data
    mag_df, mag_stats = load_and_analyze(
        "system_metrics_gpt2_adamwspam_magnitude_50.csv",
        "Magnitude Pruning (AdamWSPAM)"
    )

    b7_df, b7_stats = load_and_analyze(
        "system_metrics_gpt2_adamwprune_bitter7_state_50.csv",
        "Bitter7 State Pruning (AdamWPrune)"
    )

    # Create comparison plots
    create_comparison_plots(mag_df, mag_stats, b7_df, b7_stats)

if __name__ == "__main__":
    main()
