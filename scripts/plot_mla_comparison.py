#!/usr/bin/env python3
"""
Plot MLA variant comparison from W&B results.

Shows that base MLA (MLA0) significantly outperforms KVSplice variant (MLAKV0).
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from https://wandb.ai/mcgrof-citizen/gpt2-mla-ra-ablation-a100-40g/
variants = ["MLA0\n(Base MLA)", "MLAKV0\n(+KVSplice)"]
perplexity = [760.15, 950.49]
colors = ["#2ecc71", "#e74c3c"]  # Green for best, red for worse

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    variants, perplexity, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
)

# Add value labels on bars
for bar, ppl in zip(bars, perplexity):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{ppl:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

# Add percentage difference annotation
diff_pct = ((perplexity[1] - perplexity[0]) / perplexity[0]) * 100
ax.text(
    0.5,
    max(perplexity) * 0.95,
    f"KVSplice: +{diff_pct:.1f}% worse",
    ha="center",
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

ax.set_ylabel("Validation Perplexity (lower is better)", fontsize=12, fontweight="bold")
ax.set_title(
    "MLA Variant Comparison on GPT-2 124M (FineWebEdu, A100 40G)",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.set_ylim(0, max(perplexity) * 1.15)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")

# Add conclusion text
conclusion = "Conclusion: Learned KVSplice compression (MLAKV0) degrades quality by 25%.\nBase MLA (MLA0) is superior for production use."
fig.text(
    0.5,
    0.02,
    conclusion,
    ha="center",
    fontsize=10,
    style="italic",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("docs/images/mla_quality_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: docs/images/mla_quality_comparison.png")
