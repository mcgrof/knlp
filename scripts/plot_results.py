#!/usr/bin/env python3
"""
Result Visualization Script for KV Plugin

Generates publication-quality plots and tables comparing:
- Full KV cache (FP16 baseline)
- KV int8-only (no rank reduction)
- Orthogonal low-rank compression
- Orthogonal + int8
- Orthogonal + int4 (24x total)

Follows visualization style from:
- Palu (ICLR 2025) - Table 1, 2
- MiniCache (NeurIPS 2024) - Table 1
- PyramidKV (NeurIPS 2024) - Figure 3-5
- AsymKV (NeurIPS 2025) - Figure 2, 3

Usage:
    python scripts/plot_results.py --results-dir results/
    python scripts/plot_results.py --generate-sample
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    print("Warning: matplotlib not available")


# Paper-style color scheme
COLORS = {
    "baseline": "#1f77b4",  # Blue
    "int8_only": "#ff7f0e",  # Orange
    "orthogonal": "#2ca02c",  # Green
    "orthogonal_int8": "#d62728",  # Red
    "orthogonal_int4": "#9467bd",  # Purple
    "palu": "#8c564b",  # Brown
    "minicache": "#e377c2",  # Pink
    "pyramidkv": "#7f7f7f",  # Gray
    "asymkv": "#bcbd22",  # Yellow-green
}


def generate_sample_data() -> Dict:
    """Generate sample benchmark data for demonstration."""
    return {
        "model": "Qwen2.5-7B-Instruct",
        "methods": {
            "FP16 Baseline": {
                "compression": 1.0,
                "ppl_wikitext": 5.72,
                "ppl_c4": 8.14,
                "gsm8k": 0.72,
                "winogrande": 0.81,
                "memory_mb": 1024,
                "ttft_ms": 45.2,
                "tokens_per_sec": 125,
            },
            "Int8 Only": {
                "compression": 2.0,
                "ppl_wikitext": 5.74,
                "ppl_c4": 8.16,
                "gsm8k": 0.71,
                "winogrande": 0.80,
                "memory_mb": 512,
                "ttft_ms": 43.1,
                "tokens_per_sec": 132,
            },
            "Orthogonal 6x": {
                "compression": 6.0,
                "ppl_wikitext": 5.78,
                "ppl_c4": 8.22,
                "gsm8k": 0.70,
                "winogrande": 0.79,
                "memory_mb": 171,
                "ttft_ms": 38.5,
                "tokens_per_sec": 148,
            },
            "Orthogonal+Int8 12x": {
                "compression": 12.0,
                "ppl_wikitext": 5.82,
                "ppl_c4": 8.28,
                "gsm8k": 0.69,
                "winogrande": 0.78,
                "memory_mb": 85,
                "ttft_ms": 35.2,
                "tokens_per_sec": 162,
            },
            "Orthogonal+Int4 24x": {
                "compression": 24.0,
                "ppl_wikitext": 5.95,
                "ppl_c4": 8.45,
                "gsm8k": 0.67,
                "winogrande": 0.76,
                "memory_mb": 43,
                "ttft_ms": 32.1,
                "tokens_per_sec": 178,
            },
        },
        "context_scaling": {
            "context_lengths": [1024, 2048, 4096, 8192, 16384],
            "baseline_memory": [256, 512, 1024, 2048, 4096],
            "int8_memory": [128, 256, 512, 1024, 2048],
            "orthogonal_memory": [43, 85, 171, 341, 683],
            "orthogonal_int8_memory": [21, 43, 85, 171, 341],
            "orthogonal_int4_memory": [11, 21, 43, 85, 171],
        },
    }


def plot_compression_quality(data: Dict, output_path: str):
    """
    Plot compression ratio vs quality (PPL).

    Similar to Palu Figure 2 and PyramidKV Figure 4.
    """
    if not PLT_AVAILABLE:
        print("matplotlib required for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = data["methods"]
    compressions = [m["compression"] for m in methods.values()]
    ppls = [m["ppl_wikitext"] for m in methods.values()]
    names = list(methods.keys())

    colors = [
        COLORS["baseline"],
        COLORS["int8_only"],
        COLORS["orthogonal"],
        COLORS["orthogonal_int8"],
        COLORS["orthogonal_int4"],
    ]

    scatter = ax.scatter(compressions, ppls, c=colors[: len(names)], s=150, zorder=5)

    # Add labels
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (compressions[i], ppls[i]),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Compression Ratio (×)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title(f"Quality vs Compression - {data['model']}", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Add Pareto frontier line
    sorted_idx = np.argsort(compressions)
    pareto_x = [compressions[sorted_idx[0]]]
    pareto_y = [ppls[sorted_idx[0]]]
    for i in sorted_idx[1:]:
        if ppls[i] <= pareto_y[-1] * 1.1:  # Allow 10% degradation
            pareto_x.append(compressions[i])
            pareto_y.append(ppls[i])
    ax.plot(pareto_x, pareto_y, "k--", alpha=0.5, label="Pareto frontier")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_compression_speed(data: Dict, output_path: str):
    """
    Plot compression ratio vs throughput.

    Similar to Palu Figure 3 and AsymKV Figure 3.
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = data["methods"]
    compressions = [m["compression"] for m in methods.values()]
    speeds = [m["tokens_per_sec"] for m in methods.values()]
    names = list(methods.keys())

    colors = [
        COLORS["baseline"],
        COLORS["int8_only"],
        COLORS["orthogonal"],
        COLORS["orthogonal_int8"],
        COLORS["orthogonal_int4"],
    ]

    ax.bar(range(len(names)), speeds, color=colors[: len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")

    ax.set_ylabel("Tokens/sec", fontsize=12)
    ax.set_title(f"Throughput Comparison - {data['model']}", fontsize=14)

    # Add compression ratio annotations
    for i, (c, s) in enumerate(zip(compressions, speeds)):
        ax.annotate(
            f"{c}×",
            (i, s),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_scaling(data: Dict, output_path: str):
    """
    Plot memory usage vs context length.

    Similar to PyramidKV Figure 5 and AsymKV Figure 2.
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ctx = data["context_scaling"]["context_lengths"]

    lines = [
        ("FP16 Baseline", "baseline_memory", COLORS["baseline"], "-"),
        ("Int8 Only", "int8_memory", COLORS["int8_only"], "--"),
        ("Orthogonal 6x", "orthogonal_memory", COLORS["orthogonal"], "-."),
        ("Orth+Int8 12x", "orthogonal_int8_memory", COLORS["orthogonal_int8"], ":"),
        ("Orth+Int4 24x", "orthogonal_int4_memory", COLORS["orthogonal_int4"], "-"),
    ]

    for name, key, color, style in lines:
        ax.plot(
            ctx,
            data["context_scaling"][key],
            label=name,
            color=color,
            linestyle=style,
            linewidth=2,
            marker="o",
        )

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("KV Cache Memory (MB)", fontsize=12)
    ax.set_title(f"Memory Scaling - {data['model']}", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_task_accuracy(data: Dict, output_path: str):
    """
    Plot task accuracy comparison.

    Similar to Palu Table 1 and MiniCache Table 1.
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = data["methods"]
    names = list(methods.keys())
    x = np.arange(len(names))
    width = 0.35

    gsm8k = [m["gsm8k"] * 100 for m in methods.values()]
    winogrande = [m["winogrande"] * 100 for m in methods.values()]

    ax.bar(x - width / 2, gsm8k, width, label="GSM8K", color=COLORS["baseline"])
    ax.bar(
        x + width / 2, winogrande, width, label="Winogrande", color=COLORS["orthogonal"]
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Task Accuracy Comparison - {data['model']}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(data: Dict) -> str:
    """
    Generate LaTeX table in Palu/MiniCache style.

    Similar to Palu Table 1, 2 and MiniCache Table 1.
    """
    methods = data["methods"]

    # Header
    table = (
        r"""
\begin{table}[t]
\centering
\caption{KV Cache Compression Results on """
        + data["model"]
        + r"""}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Ratio} & \textbf{PPL}$\downarrow$ & \textbf{GSM8K}$\uparrow$ & \textbf{WG}$\uparrow$ & \textbf{Tok/s}$\uparrow$ \\
\midrule
"""
    )

    # Rows
    for name, m in methods.items():
        ppl_delta = (
            (m["ppl_wikitext"] / list(methods.values())[0]["ppl_wikitext"]) - 1
        ) * 100
        table += f"{name} & {m['compression']:.0f}$\\times$ & {m['ppl_wikitext']:.2f} "
        table += f"& {m['gsm8k']*100:.1f} & {m['winogrande']*100:.1f} & {m['tokens_per_sec']:.0f} \\\\\n"

    # Footer
    table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return table


def generate_markdown_table(data: Dict) -> str:
    """Generate Markdown table for documentation."""
    methods = data["methods"]

    table = f"## KV Cache Compression Results - {data['model']}\n\n"
    table += "| Method | Ratio | PPL | GSM8K | Winogrande | Tok/s |\n"
    table += "|--------|-------|-----|-------|------------|-------|\n"

    baseline_ppl = list(methods.values())[0]["ppl_wikitext"]
    for name, m in methods.items():
        ppl_delta = ((m["ppl_wikitext"] / baseline_ppl) - 1) * 100
        delta_str = f" (+{ppl_delta:.1f}%)" if ppl_delta > 0.1 else ""
        table += (
            f"| {name} | {m['compression']:.0f}x | {m['ppl_wikitext']:.2f}{delta_str} "
        )
        table += f"| {m['gsm8k']*100:.1f}% | {m['winogrande']*100:.1f}% | {m['tokens_per_sec']:.0f} |\n"

    return table


# =============================================================================
# Literature-Aligned Plots (Palu, MiniCache, PyramidKV, AsymKV style)
# =============================================================================


def plot_ppl_vs_memory_fraction(data: Dict, output_path: str):
    """
    Plot PPL vs KV memory fraction (Palu Figure 2 style).

    X-axis: KV memory as fraction of full cache (0 to 1)
    Y-axis: Perplexity
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = data["methods"]

    # Convert compression to memory fraction (1/compression)
    fractions = [1.0 / m["compression"] for m in methods.values()]
    ppls = [m["ppl_wikitext"] for m in methods.values()]
    names = list(methods.keys())

    colors = [
        COLORS["baseline"],
        COLORS["int8_only"],
        COLORS["orthogonal"],
        COLORS["orthogonal_int8"],
        COLORS["orthogonal_int4"],
    ]

    # Our method points
    ax.scatter(
        fractions, ppls, c=colors[: len(names)], s=200, zorder=5, edgecolors="black"
    )

    # Add labels
    for i, name in enumerate(names):
        offset = (10, 5) if fractions[i] > 0.1 else (-50, 10)
        ax.annotate(
            name,
            (fractions[i], ppls[i]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )

    # Placeholder for literature reference points (uncomment when data available)
    # Literature reference lines (placeholder - add actual values from papers)
    # ax.axhline(y=5.8, color=COLORS["palu"], linestyle="--", alpha=0.5, label="Palu 12x")
    # ax.axhline(y=5.9, color=COLORS["minicache"], linestyle=":", alpha=0.5, label="MiniCache")

    ax.set_xlabel("KV Memory Fraction (1 = full cache)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title(f"Quality vs Memory Fraction - {data['model']}", fontsize=14)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Add "better" annotation
    ax.annotate(
        "Better →",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="gray",
    )
    ax.annotate(
        "↓ Better",
        xy=(0.85, 0.05),
        xycoords="axes fraction",
        fontsize=10,
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_heatmap(output_path: str):
    """
    Plot ablation heatmap (rank × bits).

    Shows PPL delta vs baseline for each combination.
    """
    if not PLT_AVAILABLE:
        return

    # Sample ablation data (placeholder - replace with actual results)
    ranks = ["Full", "256", "128", "64", "32"]
    bits_configs = ["FP16", "Int8 V", "Int8 K+V", "Int4 V", "Int4 K+V"]

    # PPL delta matrix (placeholder values showing expected pattern)
    ppl_deltas = np.array(
        [
            [0.0, 0.3, 0.4, 1.2, 1.5],  # Full rank
            [0.5, 0.8, 0.9, 1.8, 2.1],  # Rank 256
            [1.0, 1.3, 1.5, 2.5, 2.8],  # Rank 128
            [2.0, 2.4, 2.7, 3.8, 4.2],  # Rank 64
            [4.0, 4.5, 5.0, 6.5, 7.2],  # Rank 32
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(ppl_deltas, cmap="RdYlGn_r", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PPL Delta (%)", fontsize=11)

    # Set ticks
    ax.set_xticks(range(len(bits_configs)))
    ax.set_xticklabels(bits_configs, rotation=45, ha="right")
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(ranks)

    # Add text annotations
    for i in range(len(ranks)):
        for j in range(len(bits_configs)):
            color = "white" if ppl_deltas[i, j] > 3 else "black"
            ax.text(
                j, i, f"{ppl_deltas[i, j]:.1f}%", ha="center", va="center", color=color
            )

    ax.set_xlabel("Quantization Configuration", fontsize=12)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_title("PPL Degradation Heatmap (Rank × Quantization)", fontsize=14)

    # Mark safe region
    rect = plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="green", linewidth=3)
    ax.add_patch(rect)
    ax.annotate(
        "Safe Region\n(<2% PPL loss)",
        xy=(1, 1),
        fontsize=10,
        color="green",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_safe_region(data: Dict, output_path: str):
    """
    Visualize the 'safe region' where quality is preserved.

    Similar to AsymKV Figure 3 showing K/V budget tradeoffs.
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = data["methods"]
    compressions = [m["compression"] for m in methods.values()]
    ppls = [m["ppl_wikitext"] for m in methods.values()]
    names = list(methods.keys())

    baseline_ppl = ppls[0]
    ppl_deltas = [(p - baseline_ppl) / baseline_ppl * 100 for p in ppls]

    colors = [
        COLORS["baseline"],
        COLORS["int8_only"],
        COLORS["orthogonal"],
        COLORS["orthogonal_int8"],
        COLORS["orthogonal_int4"],
    ]

    # Plot points
    ax.scatter(
        compressions,
        ppl_deltas,
        c=colors[: len(names)],
        s=200,
        zorder=5,
        edgecolors="black",
    )

    # Add labels
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (compressions[i], ppl_deltas[i]),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # Safe region shading
    ax.axhspan(-1, 5, alpha=0.2, color="green", label="Safe region (<5% PPL)")
    ax.axhspan(5, 10, alpha=0.2, color="yellow", label="Moderate (5-10% PPL)")
    ax.axhspan(10, 20, alpha=0.2, color="red", label="Degraded (>10% PPL)")

    ax.set_xlabel("Compression Ratio (×)", fontsize=12)
    ax.set_ylabel("PPL Degradation (%)", fontsize=12)
    ax.set_title(f"Safe Compression Region - {data['model']}", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_literature_comparison(data: Dict, output_path: str):
    """
    Compare our results with literature baselines.

    Shows our Pareto frontier vs reported points from other papers.
    """
    if not PLT_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Our results
    methods = data["methods"]
    our_compressions = [m["compression"] for m in methods.values()]
    our_ppls = [m["ppl_wikitext"] for m in methods.values()]

    ax.plot(
        our_compressions,
        our_ppls,
        "o-",
        color=COLORS["orthogonal"],
        linewidth=2,
        markersize=10,
        label="Ours (KV Plugin)",
    )

    # Literature reference points (placeholder - replace with actual reported values)
    # Format: (compression, ppl, label, color)
    literature_points = [
        # Palu (ICLR 2025) - placeholder values
        (4, 5.85, "Palu 4x", COLORS["palu"]),
        (8, 5.95, "Palu 8x", COLORS["palu"]),
        # MiniCache (NeurIPS 2024) - placeholder values
        (4, 5.90, "MiniCache 4x", COLORS["minicache"]),
        (8, 6.05, "MiniCache 8x", COLORS["minicache"]),
        # PyramidKV (NeurIPS 2024) - placeholder values
        (6, 5.88, "PyramidKV 6x", COLORS["pyramidkv"]),
        (12, 6.15, "PyramidKV 12x", COLORS["pyramidkv"]),
        # AsymKV (NeurIPS 2025) - placeholder values
        (8, 5.82, "AsymKV 8x", COLORS["asymkv"]),
        (16, 6.00, "AsymKV 16x", COLORS["asymkv"]),
    ]

    # Plot literature points (as hollow markers)
    for comp, ppl, label, color in literature_points:
        ax.scatter(comp, ppl, s=100, facecolors="none", edgecolors=color, linewidths=2)
        ax.annotate(
            label,
            (comp, ppl),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    # Add legend entries for literature
    for name, color in [
        ("Palu", COLORS["palu"]),
        ("MiniCache", COLORS["minicache"]),
        ("PyramidKV", COLORS["pyramidkv"]),
        ("AsymKV", COLORS["asymkv"]),
    ]:
        ax.scatter(
            [], [], s=100, facecolors="none", edgecolors=color, linewidths=2, label=name
        )

    ax.set_xlabel("Compression Ratio (×)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=12)
    ax.set_title("Literature Comparison", fontsize=14)
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Note about placeholder data
    ax.text(
        0.98,
        0.02,
        "Note: Literature points are placeholders\nReplace with actual reported values",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate result plots and tables")
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Directory with result JSON files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="plots", help="Output directory for plots"
    )
    parser.add_argument(
        "--generate-sample", action="store_true", help="Generate sample data and plots"
    )
    args = parser.parse_args()

    if args.generate_sample:
        data = generate_sample_data()
        os.makedirs(args.output_dir, exist_ok=True)

        # Generate basic plots
        plot_compression_quality(data, f"{args.output_dir}/compression_quality.png")
        plot_compression_speed(data, f"{args.output_dir}/compression_speed.png")
        plot_memory_scaling(data, f"{args.output_dir}/memory_scaling.png")
        plot_task_accuracy(data, f"{args.output_dir}/task_accuracy.png")

        # Generate literature-aligned plots
        plot_ppl_vs_memory_fraction(
            data, f"{args.output_dir}/ppl_vs_memory_fraction.png"
        )
        plot_ablation_heatmap(f"{args.output_dir}/ablation_heatmap.png")
        plot_safe_region(data, f"{args.output_dir}/safe_region.png")
        plot_literature_comparison(data, f"{args.output_dir}/literature_comparison.png")

        # Generate tables
        latex = generate_latex_table(data)
        print("\n" + "=" * 60)
        print("LaTeX Table")
        print("=" * 60)
        print(latex)

        markdown = generate_markdown_table(data)
        print("\n" + "=" * 60)
        print("Markdown Table")
        print("=" * 60)
        print(markdown)

        # Save tables
        with open(f"{args.output_dir}/table.tex", "w") as f:
            f.write(latex)
        with open(f"{args.output_dir}/table.md", "w") as f:
            f.write(markdown)

        print(f"\nAll outputs saved to {args.output_dir}/")

    elif args.results_dir:
        # Load results from directory
        results = {}
        for fname in os.listdir(args.results_dir):
            if fname.endswith(".json"):
                with open(os.path.join(args.results_dir, fname)) as f:
                    results[fname] = json.load(f)

        # TODO: Merge and plot results
        print(f"Loaded {len(results)} result files")
        print("Plotting not yet implemented for real results")

    else:
        print("Specify --results-dir or --generate-sample")
        sys.exit(1)


if __name__ == "__main__":
    main()
