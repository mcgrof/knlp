#!/usr/bin/env python3
"""Create complete AdamWPrune comparison including bitter7 WITHOUT torch.compile.

This reveals the true memory cost of torch.compile vs the algorithm itself.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 9)
plt.rcParams["font.size"] = 11

# Complete data
variants = [
    "Movement\n(WITH compile)\nBaseline",
    "bitter8\n(WITHOUT compile)\nIncomplete",
    "bitter7\n(WITHOUT compile)\nIncomplete",
    "bitter7\n(WITH compile)\nBest",
]

ppls = [44.15, 40.94, 38.41, 37.28]
gpu_mems = [33306, None, 13945, 44168]  # MiB
colors = ["#E74C3C", "#3498DB", "#9B59B6", "#27AE60"]

print("=" * 80)
print("COMPLETE ADAMWPRUNE COMPARISON")
print("=" * 80)
print()
print(f"{'Variant':<30} {'compile':<8} {'PPL':<8} {'GPU Mem':<12} {'vs Baseline'}")
print("-" * 80)
for i, (var, ppl, mem) in enumerate(zip(variants, ppls, gpu_mems)):
    var_name = var.split("\n")[0]
    compile_status = "YES" if "WITH" in var else "NO"
    ppl_diff = ((44.15 - ppl) / 44.15) * 100
    mem_str = f"{mem:.0f} MiB" if mem else "N/A"
    mem_diff = f"{((mem - 33306) / 33306) * 100:+.1f}%" if mem else "N/A"

    print(
        f"{var_name:<30} {compile_status:<8} {ppl:<8.2f} {mem_str:<12} PPL: {ppl_diff:+.1f}%, Mem: {mem_diff}"
    )

print()
print("KEY FINDINGS:")
print()
print("1. Algorithm > Optimization:")
print("   bitter8 WITHOUT compile: 40.94 PPL (-7.3% vs baseline WITH compile)")
print("   bitter7 WITHOUT compile: 38.41 PPL (-13.0% vs baseline WITH compile)")
print()
print("2. torch.compile Memory Cost:")
print("   bitter7 WITHOUT compile:  13945 MiB avg")
print("   bitter7 WITH compile:     44168 MiB avg")
print("   torch.compile adds +216.7% memory!")
print()
print("3. Memory Efficiency Winner:")
print("   bitter7 WITHOUT compile uses 58.1% LESS memory than baseline WITH compile")
print("   (13945 vs 33306 MiB) while achieving 13.0% better perplexity!")
print()
print("4. Best Perplexity:")
print("   bitter7 WITH compile: 37.28 PPL (15.6% better than baseline)")
print("   But costs +32.6% memory vs baseline")
print("=" * 80)

# === Graph 1: Perplexity vs GPU Memory Trade-off ===
print("\nCreating scatter plot: Perplexity vs GPU Memory...")
fig, ax = plt.subplots(figsize=(12, 8))

# Plot points (skip bitter8 which has no GPU data)
for i, (var, ppl, mem, color) in enumerate(zip(variants, ppls, gpu_mems, colors)):
    if mem is not None:
        ax.scatter(
            mem,
            ppl,
            s=500,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
            zorder=10,
        )

        # Labels
        var_short = var.split("\n")[0]
        compile_str = "(WITH compile)" if "WITH compile" in var else "(NO compile)"
        ax.annotate(
            f"{var_short}\n{compile_str}\n{ppl:.2f} PPL\n{mem:.0f} MiB",
            xy=(mem, ppl),
            xytext=(15, 0),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=color, alpha=0.3, edgecolor="black"),
        )

# Add quadrant lines
ax.axhline(
    y=44.15,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=1.5,
    label="Baseline PPL",
)
ax.axvline(
    x=33306,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=1.5,
    label="Baseline Memory",
)

# Highlight the winner zone (lower-left = better PPL, less memory)
ax.fill_between(
    [0, 33306],
    [0, 0],
    [44.15, 44.15],
    alpha=0.1,
    color="green",
    label="Winner zone\n(better PPL, less memory)",
)

ax.set_xlabel("Average GPU Memory (MiB)", fontsize=13, weight="bold")
ax.set_ylabel("Validation Perplexity (lower is better)", fontsize=13, weight="bold")
ax.set_title(
    "AdamWPrune: Complete Comparison - Perplexity vs GPU Memory\n"
    + "GPT-2 124M, 50% Sparsity, NVIDIA B200 GPUs",
    fontsize=15,
    weight="bold",
    pad=20,
)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="black", fancybox=True)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim([10000, 50000])
ax.set_ylim([36, 46])

# Add key finding annotation
ax.text(
    0.5,
    0.02,
    "bitter7 WITHOUT compile: Best efficiency (58% less memory, 13% better PPL vs baseline)\n"
    + "torch.compile adds +217% memory to bitter7!",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.8", fc="yellow", alpha=0.3, edgecolor="black", linewidth=2
    ),
    fontsize=11,
    weight="bold",
)

plt.tight_layout()
plt.savefig("adamwprune_complete_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: adamwprune_complete_comparison.png")
plt.close()

# === Graph 2: Bar chart with all variants ===
print("\nCreating bar chart comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Perplexity comparison
bars1 = ax1.bar(
    range(len(variants)), ppls, color=colors, alpha=0.8, edgecolor="black", linewidth=2
)
ax1.set_ylabel("Validation Perplexity (lower is better)", fontsize=12, weight="bold")
ax1.set_title("Final Validation Perplexity", fontsize=14, weight="bold")
ax1.set_xticks(range(len(variants)))
ax1.set_xticklabels(
    [v.replace("\n", " ") for v in variants], rotation=15, ha="right", fontsize=9
)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim([35, 46])

# Add value labels
for bar, ppl in zip(bars1, ppls):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.3,
        f"{ppl:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

# GPU Memory comparison (skip bitter8)
mem_variants = []
mem_values = []
mem_colors = []
for var, mem, color in zip(variants, gpu_mems, colors):
    if mem is not None:
        mem_variants.append(var)
        mem_values.append(mem)
        mem_colors.append(color)

bars2 = ax2.bar(
    range(len(mem_variants)),
    mem_values,
    color=mem_colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=2,
)
ax2.set_ylabel("Average GPU Memory (MiB)", fontsize=12, weight="bold")
ax2.set_title("GPU Memory Consumption", fontsize=14, weight="bold")
ax2.set_xticks(range(len(mem_variants)))
ax2.set_xticklabels(
    [v.replace("\n", " ") for v in mem_variants], rotation=15, ha="right", fontsize=9
)
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar, mem in zip(bars2, mem_values):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1000,
        f"{mem:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

fig.suptitle(
    "AdamWPrune Complete Comparison: Perplexity vs Memory\n50% Sparsity on GPT-2 124M (NVIDIA B200)",
    fontsize=16,
    weight="bold",
)

plt.tight_layout()
plt.savefig("adamwprune_complete_bars.png", dpi=300, bbox_inches="tight")
print("Saved: adamwprune_complete_bars.png")
plt.close()

print("\nDone!")
