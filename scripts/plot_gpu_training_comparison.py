#!/usr/bin/env python3
"""Generate GPU training comparison plot for kvsplice.md"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
    }
)

# Data from GPU comparison results
gpus = ["H100", "W7900", "A100-40G"]
mla_ppl = [8.68, 10.31, 16.66]
kvsplice_ppl = [8.78, 10.65, 16.94]
mla_loss = [2.161, 2.333, 2.813]
kvsplice_loss = [2.173, 2.366, 2.830]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Validation Perplexity
x = np.arange(len(gpus))
width = 0.35

bars1 = ax1.bar(
    x - width / 2,
    mla_ppl,
    width,
    label="MLA (6x)",
    color="#2ca02c",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)
bars2 = ax1.bar(
    x + width / 2,
    kvsplice_ppl,
    width,
    label="MLA+KVSplice (12x)",
    color="#ff7f0e",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

ax1.set_ylabel("Validation Perplexity", fontsize=12)
ax1.set_xlabel("GPU", fontsize=12)
ax1.set_title(
    "Validation Perplexity by GPU (2-4h training)", fontsize=13, fontweight="bold"
)
ax1.set_xticks(x)
ax1.set_xticklabels(gpus)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Right plot: Validation Loss
bars3 = ax2.bar(
    x - width / 2,
    mla_loss,
    width,
    label="MLA (6x)",
    color="#2ca02c",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)
bars4 = ax2.bar(
    x + width / 2,
    kvsplice_loss,
    width,
    label="MLA+KVSplice (12x)",
    color="#ff7f0e",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

ax2.set_ylabel("Validation Loss", fontsize=12)
ax2.set_xlabel("GPU", fontsize=12)
ax2.set_title("Validation Loss by GPU (2-4h training)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(gpus)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("docs/kvsplice/gpu_training_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: docs/kvsplice/gpu_training_comparison.png")

# Also create a degradation comparison plot
fig2, ax = plt.subplots(figsize=(10, 6))

degradation_pct = [(kvsplice_ppl[i] / mla_ppl[i] - 1) * 100 for i in range(len(gpus))]

colors = ["#2ca02c" if d < 2 else "#ff7f0e" for d in degradation_pct]
bars = ax.bar(
    gpus, degradation_pct, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
)

# Add value labels
for bar, val in zip(bars, degradation_pct):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"+{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

ax.set_ylabel("Quality Degradation (%)", fontsize=12)
ax.set_xlabel("GPU", fontsize=12)
ax.set_title(
    "KVSplice Quality Impact by GPU (Perplexity Increase)",
    fontsize=14,
    fontweight="bold",
)
ax.axhline(
    y=2, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label="2% threshold"
)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(degradation_pct) * 1.2)

plt.tight_layout()
plt.savefig(
    "docs/kvsplice/gpu_degradation_comparison.png", dpi=300, bbox_inches="tight"
)
print("Saved: docs/kvsplice/gpu_degradation_comparison.png")

print("\nGenerated GPU training comparison plots!")
