#!/usr/bin/env python3
"""
Generate publication-quality visualizations for KV Plugin v3 documentation.

Creates:
- Compression vs Quality tradeoff chart
- K vs V variance explained comparison
- Memory savings by preset
- Pareto frontier (compression ratio vs quality)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


# Empirical data from compression quality test
COMPRESSION_DATA = {
    "PCA 384 (2x)": {"k_var": 98.0, "v_var": 87.0, "ratio": 2},
    "PCA 256 (3x)": {"k_var": 96.0, "v_var": 75.0, "ratio": 3},
    "PCA 128 (6x)": {"k_var": 93.0, "v_var": 57.0, "ratio": 6},
    "PCA 64 (12x)": {"k_var": 89.0, "v_var": 42.0, "ratio": 12},
    "PCA 32 (24x)": {"k_var": 83.0, "v_var": 30.0, "ratio": 24},
}

PRESET_DATA = {
    "none": {"compression": 1, "k_var": 100, "v_var": 100, "est_ppl": 0, "memory_mb": 144},
    "conservative": {"compression": 6, "k_var": 93, "v_var": 57, "est_ppl": 2, "memory_mb": 24},
    "balanced": {"compression": 12, "k_var": 89, "v_var": 42, "est_ppl": 4.5, "memory_mb": 12},
    "aggressive": {"compression": 18, "k_var": 86, "v_var": 35, "est_ppl": 7.5, "memory_mb": 8},
    "extreme": {"compression": 24, "k_var": 83, "v_var": 30, "est_ppl": 11.5, "memory_mb": 6},
}


def plot_kv_comparison():
    """Bar chart comparing K vs V compressibility."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = list(COMPRESSION_DATA.keys())
    k_vars = [COMPRESSION_DATA[c]["k_var"] for c in configs]
    v_vars = [COMPRESSION_DATA[c]["v_var"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    bars_k = ax.bar(x - width/2, k_vars, width, label='K (Keys)', color='#2ecc71', edgecolor='white')
    bars_v = ax.bar(x + width/2, v_vars, width, label='V (Values)', color='#e74c3c', edgecolor='white')

    # Add value labels
    for bar in bars_k:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars_v:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Variance Explained (%)')
    ax.set_xlabel('Compression Configuration')
    ax.set_title('K vs V Compressibility: Keys Compress Better Than Values')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    # Add annotation about key insight
    ax.annotate('V is the bottleneck:\nonly 42% at 12x',
                xy=(3, 42), xytext=(3.5, 65),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
                fontsize=10, color='#c0392b', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#c0392b'))

    plt.tight_layout()
    return fig


def plot_compression_tradeoff():
    """Line chart showing compression ratio vs variance explained."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ratios = [COMPRESSION_DATA[c]["ratio"] for c in COMPRESSION_DATA]
    k_vars = [COMPRESSION_DATA[c]["k_var"] for c in COMPRESSION_DATA]
    v_vars = [COMPRESSION_DATA[c]["v_var"] for c in COMPRESSION_DATA]

    ax.plot(ratios, k_vars, 'o-', color='#2ecc71', linewidth=2.5, markersize=10,
            label='K Variance Explained', markeredgecolor='white', markeredgewidth=2)
    ax.plot(ratios, v_vars, 's-', color='#e74c3c', linewidth=2.5, markersize=10,
            label='V Variance Explained', markeredgecolor='white', markeredgewidth=2)

    # Fill area between
    ax.fill_between(ratios, k_vars, v_vars, alpha=0.2, color='#95a5a6')

    # Add preset markers
    preset_ratios = [6, 12, 24]
    preset_names = ['conservative', 'balanced', 'extreme']
    for r, name in zip(preset_ratios, preset_names):
        ax.axvline(x=r, color='#3498db', linestyle='--', alpha=0.5)
        ax.annotate(name, xy=(r, 25), ha='center', fontsize=9, color='#2980b9')

    ax.set_xlabel('Compression Ratio (x)')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Compression-Quality Tradeoff: How Much Can We Compress?')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 26)
    ax.set_ylim(20, 105)
    ax.set_xticks([2, 6, 12, 18, 24])

    # Add "sweet spot" annotation
    ax.annotate('Sweet spot:\n12x with 89% K, 42% V',
                xy=(12, 65), xytext=(16, 80),
                arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2),
                fontsize=10, color='#2980b9', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d6eaf8', edgecolor='#2980b9'))

    plt.tight_layout()
    return fig


def plot_memory_savings():
    """Bar chart showing memory savings by preset."""
    fig, ax = plt.subplots(figsize=(10, 6))

    presets = list(PRESET_DATA.keys())
    memories = [PRESET_DATA[p]["memory_mb"] for p in presets]
    compressions = [PRESET_DATA[p]["compression"] for p in presets]

    # Color gradient from red (high memory) to green (low memory)
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#27ae60', '#16a085']

    bars = ax.bar(presets, memories, color=colors, edgecolor='white', linewidth=2)

    # Add compression ratio labels
    for bar, comp in zip(bars, compressions):
        height = bar.get_height()
        ax.annotate(f'{comp}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('KV Cache Size (MB)')
    ax.set_xlabel('Preset')
    ax.set_title('Memory Savings by Preset (GPT-2 at 4K context)')
    ax.set_ylim(0, 170)

    # Add savings annotation
    ax.annotate('24x smaller!',
                xy=(4, 6), xytext=(3.2, 50),
                arrowprops=dict(arrowstyle='->', color='#16a085', lw=2),
                fontsize=12, color='#16a085', fontweight='bold')

    # Add baseline reference line
    ax.axhline(y=144, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2)
    ax.annotate('Baseline: 144 MB', xy=(0.5, 148), fontsize=10, color='#c0392b')

    plt.tight_layout()
    return fig


def plot_pareto_frontier():
    """Scatter plot with Pareto frontier showing optimal presets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    presets = list(PRESET_DATA.keys())
    compressions = [PRESET_DATA[p]["compression"] for p in presets]
    v_vars = [PRESET_DATA[p]["v_var"] for p in presets]
    est_ppls = [PRESET_DATA[p]["est_ppl"] for p in presets]

    # Size by PPL impact (larger = worse)
    sizes = [100 + ppl * 20 for ppl in est_ppls]

    # Color by quality (green = good, red = bad)
    colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    scatter = ax.scatter(compressions, v_vars, s=sizes, c=colors,
                         edgecolors='white', linewidth=2, alpha=0.8)

    # Add labels
    for i, preset in enumerate(presets):
        ax.annotate(preset,
                    xy=(compressions[i], v_vars[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    # Draw Pareto frontier
    ax.plot(compressions, v_vars, '--', color='#3498db', alpha=0.5, linewidth=2)

    ax.set_xlabel('Compression Ratio (x)')
    ax.set_ylabel('V Variance Explained (%)')
    ax.set_title('Compression-Quality Pareto Frontier')
    ax.set_xlim(0, 28)
    ax.set_ylim(20, 110)

    # Legend for size
    legend_elements = [
        plt.scatter([], [], s=100, c='gray', label='Low PPL impact'),
        plt.scatter([], [], s=300, c='gray', label='High PPL impact'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def plot_architecture_overview():
    """Create a visual architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.5, 'KV Plugin v3.0 Architecture',
            ha='center', fontsize=16, fontweight='bold')

    # Model box
    model_box = mpatches.FancyBboxPatch((0.5, 3), 2, 3,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#3498db', alpha=0.3,
                                         edgecolor='#2980b9', linewidth=2)
    ax.add_patch(model_box)
    ax.text(1.5, 5.5, 'HuggingFace\nModel', ha='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 4.5, 'Q_proj', ha='center', fontsize=10)
    ax.text(1.5, 4.0, 'K_proj', ha='center', fontsize=10, color='#e74c3c')
    ax.text(1.5, 3.5, 'V_proj', ha='center', fontsize=10, color='#e74c3c')

    # Arrows to compressor
    ax.annotate('', xy=(4, 4.0), xytext=(2.7, 4.0),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.annotate('', xy=(4, 3.5), xytext=(2.7, 3.5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    # Compressor box
    comp_box = mpatches.FancyBboxPatch((4, 2.5), 2.5, 2.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#e74c3c', alpha=0.3,
                                        edgecolor='#c0392b', linewidth=2)
    ax.add_patch(comp_box)
    ax.text(5.25, 4.5, 'KVCompressor', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.25, 3.9, 'compress()', ha='center', fontsize=10, fontfamily='monospace')
    ax.text(5.25, 3.3, 'expand()', ha='center', fontsize=10, fontfamily='monospace')

    # Cache box
    cache_box = mpatches.FancyBboxPatch((7.5, 3), 2.5, 2,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#27ae60', alpha=0.3,
                                         edgecolor='#1e8449', linewidth=2)
    ax.add_patch(cache_box)
    ax.text(8.75, 4.5, 'Compressed\nKV Cache', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 3.5, '6-24x smaller', ha='center', fontsize=10, color='#1e8449')

    # Arrows
    ax.annotate('', xy=(7.3, 3.8), xytext=(6.7, 3.8),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    # Attention box
    attn_box = mpatches.FancyBboxPatch((7.5, 0.5), 2.5, 2,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#9b59b6', alpha=0.3,
                                        edgecolor='#8e44ad', linewidth=2)
    ax.add_patch(attn_box)
    ax.text(8.75, 2, 'SDPA/FA', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 1.2, 'Full-dim attention', ha='center', fontsize=10)

    # Arrow from cache to attention (expand)
    ax.annotate('', xy=(8.75, 2.7), xytext=(8.75, 2.9),
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2))
    ax.text(9.5, 2.8, 'expand()', fontsize=9, fontfamily='monospace', color='#8e44ad')

    # Compressor types legend
    types_box = mpatches.FancyBboxPatch((0.5, 0.3), 5.5, 2,
                                         boxstyle="round,pad=0.1",
                                         facecolor='#f8f9fa',
                                         edgecolor='#bdc3c7', linewidth=1)
    ax.add_patch(types_box)
    ax.text(3.25, 2.0, 'Compressor Types', ha='center', fontsize=11, fontweight='bold')

    compressors = ['Identity', 'PCA', 'TopK', 'SVD', 'Hybrid']
    for i, comp in enumerate(compressors):
        ax.text(1.0 + i * 1.1, 1.3, comp, ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#bdc3c7'))
    ax.text(3.25, 0.6, 'All implement: compress() / expand() / calibrate()',
            ha='center', fontsize=9, fontstyle='italic')

    plt.tight_layout()
    return fig


def plot_speed_impact():
    """Bar chart showing speed impact is negligible."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    presets = ['none\n(baseline)', 'conservative', 'balanced', 'aggressive', 'extreme']
    ttft = [5.26, 5.33, 5.30, 5.28, 5.26]
    throughput = [371.7, 370.2, 363.0, 364.6, 365.3]

    # TTFT chart
    colors_ttft = ['#3498db'] + ['#2ecc71'] * 4  # Blue baseline, green others
    bars1 = ax1.bar(presets, ttft, color=colors_ttft, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Time to First Token (ms)')
    ax1.set_xlabel('Preset')
    ax1.set_title('TTFT: Negligible Impact')
    ax1.set_ylim(5.0, 5.5)
    ax1.axhline(y=5.26, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)

    # Add delta labels
    deltas_ttft = [0, +1.2, +0.7, +0.4, 0]
    for bar, delta in zip(bars1, deltas_ttft):
        height = bar.get_height()
        label = f'{delta:+.1f}%' if delta != 0 else 'baseline'
        color = '#27ae60' if delta <= 1 else '#e74c3c'
        ax1.annotate(label,
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color=color if delta != 0 else '#3498db')

    # Throughput chart
    colors_tp = ['#3498db'] + ['#2ecc71'] * 4
    bars2 = ax2.bar(presets, throughput, color=colors_tp, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Tokens per Second')
    ax2.set_xlabel('Preset')
    ax2.set_title('Throughput: ~2% Overhead')
    ax2.set_ylim(350, 380)
    ax2.axhline(y=371.7, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)

    # Add delta labels
    deltas_tp = [0, -0.4, -2.3, -1.9, -1.7]
    for bar, delta in zip(bars2, deltas_tp):
        height = bar.get_height()
        label = f'{delta:+.1f}%' if delta != 0 else 'baseline'
        color = '#27ae60' if abs(delta) <= 2 else '#e74c3c'
        ax2.annotate(label,
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color=color if delta != 0 else '#3498db')

    # Add note
    fig.text(0.5, 0.02,
             'Note: Current "pre-expand" mode shows minimal overhead. Custom Triton kernels would provide speed gains.',
             ha='center', fontsize=10, fontstyle='italic', color='#7f8c8d')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig


def main():
    output_dir = Path("docs/kvsplice")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating KV Plugin v3 visualizations...")

    # Generate all charts
    charts = [
        ("kv_comparison.png", plot_kv_comparison, "K vs V compressibility"),
        ("compression_tradeoff.png", plot_compression_tradeoff, "Compression tradeoff"),
        ("memory_savings.png", plot_memory_savings, "Memory savings"),
        ("pareto_frontier.png", plot_pareto_frontier, "Pareto frontier"),
        ("architecture_overview.png", plot_architecture_overview, "Architecture"),
        ("speed_impact.png", plot_speed_impact, "Speed impact"),
    ]

    for filename, plot_func, desc in charts:
        print(f"  Creating {desc}...")
        fig = plot_func()
        fig.savefig(output_dir / filename, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {output_dir / filename}")

    print("\nAll visualizations complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
