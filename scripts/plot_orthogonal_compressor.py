#!/usr/bin/env python3
"""
Generate visualizations for OrthogonalCompressor performance.

Creates charts showing:
1. Setup time comparison (calibration vs no calibration)
2. Reconstruction quality comparison
3. Inference speed impact
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs("docs/kvsplice", exist_ok=True)

# ============================================================================
# Chart 1: Setup Time Comparison
# ============================================================================

def plot_setup_time():
    """Bar chart comparing setup time across compressors."""
    fig, ax = plt.subplots(figsize=(10, 6))

    compressors = ['PCA\n(calibrated)', 'SVD\n(calibrated)', 'Orthogonal\n(no calib)', 'Orthogonal\n(calibrated)']
    times = [66.55, 65.79, 1.64, 68.45]  # ms from test results
    colors = ['#3498db', '#3498db', '#27ae60', '#95a5a6']

    bars = ax.bar(compressors, times, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{time:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Highlight the speedup
    ax.annotate('40x faster!',
                xy=(2, 1.64), xytext=(2.5, 25),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
                fontsize=14, fontweight='bold', color='#27ae60')

    ax.set_ylabel('Setup Time (ms)', fontsize=12)
    ax.set_title('Compressor Setup Time Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Requires calibration'),
        Patch(facecolor='#27ae60', edgecolor='black', label='No calibration needed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('docs/kvsplice/orthogonal_setup_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: docs/kvsplice/orthogonal_setup_time.png")


# ============================================================================
# Chart 2: Reconstruction Quality Comparison
# ============================================================================

def plot_reconstruction_quality():
    """Bar chart comparing reconstruction MSE."""
    fig, ax = plt.subplots(figsize=(10, 6))

    compressors = ['SVD', 'Orthogonal\n(calibrated)', 'PCA', 'Orthogonal\n(no calib)']
    mse = [44.91, 44.91, 45.03, 166.78]
    colors = ['#3498db', '#27ae60', '#3498db', '#e74c3c']

    bars = ax.bar(compressors, mse, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, m in zip(bars, mse):
        height = bar.get_height()
        ax.annotate(f'{m:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Annotation for matching quality
    ax.annotate('Exact match\nwhen calibrated!',
                xy=(1, 44.91), xytext=(1.5, 80),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
                fontsize=12, fontweight='bold', color='#27ae60', ha='center')

    ax.set_ylabel('Reconstruction MSE', fontsize=12)
    ax.set_title('Reconstruction Quality Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 200)
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line for baseline
    ax.axhline(y=45, color='gray', linestyle='--', alpha=0.5, label='PCA/SVD baseline')

    plt.tight_layout()
    plt.savefig('docs/kvsplice/orthogonal_reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: docs/kvsplice/orthogonal_reconstruction.png")


# ============================================================================
# Chart 3: Trade-off Summary
# ============================================================================

def plot_tradeoff_summary():
    """Scatter plot showing speed vs quality trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data: (setup_time_ms, mse, label, color, marker_size)
    data = [
        (66.55, 44.91, 'PCA (calibrated)', '#3498db', 300),
        (65.79, 44.91, 'SVD (calibrated)', '#9b59b6', 300),
        (1.64, 166.78, 'Orthogonal (no calib)', '#e74c3c', 400),
        (68.45, 44.91, 'Orthogonal (calibrated)', '#27ae60', 300),
    ]

    for time, mse, label, color, size in data:
        ax.scatter(time, mse, s=size, c=color, edgecolors='black',
                   linewidths=2, label=label, zorder=5)

    # Add arrow showing the trade-off
    ax.annotate('', xy=(1.64, 166.78), xytext=(1.64, 44.91),
                arrowprops=dict(arrowstyle='<->', color='#2ecc71', lw=3))
    ax.annotate('Quality gap\n(fixable with\ncalibration)',
                xy=(1.64, 100), xytext=(15, 100),
                fontsize=10, ha='left', va='center')

    ax.set_xlabel('Setup Time (ms)', fontsize=12)
    ax.set_ylabel('Reconstruction MSE', fontsize=12)
    ax.set_title('Speed vs Quality Trade-off\n(Bottom-left is optimal)', fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 80)
    ax.set_ylim(0, 200)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Highlight optimal region
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 10, 60, linewidth=2, edgecolor='#27ae60',
                      facecolor='#27ae60', alpha=0.1)
    ax.add_patch(rect)
    ax.annotate('Optimal\nregion', xy=(5, 30), fontsize=10, ha='center',
                color='#27ae60', fontweight='bold')

    plt.tight_layout()
    plt.savefig('docs/kvsplice/orthogonal_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: docs/kvsplice/orthogonal_tradeoff.png")


if __name__ == "__main__":
    print("Generating OrthogonalCompressor visualizations...")
    plot_setup_time()
    plot_reconstruction_quality()
    plot_tradeoff_summary()
    print("\nAll charts generated!")
