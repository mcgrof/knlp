#!/usr/bin/env python3
"""
Generate visualization of double-pass elimination CPU benchmark results.
"""
import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
configs = ["12.58M params\n(dim=1024, layers=12)", "50.33M params\n(dim=2048, layers=12)"]
old_times = [118.425, 535.023]  # ms
new_times = [64.251, 300.751]  # ms
speedups = [1.84, 1.78]
improvements = [45.7, 43.8]  # %

# Memory footprint (MB)
old_memory = 497  # MB for all_scores concat (GPT-2 124M params)
new_memory = 5  # MB for 1% sample

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Bitter7 Double-Pass Elimination: CPU Benchmark Results",
    fontsize=16,
    fontweight="bold",
)

# Plot 1: Timing comparison
ax1 = axes[0, 0]
x = np.arange(len(configs))
width = 0.35

bars1 = ax1.bar(
    x - width / 2, old_times, width, label="OLD (double-pass)", alpha=0.8, color="#e74c3c"
)
bars2 = ax1.bar(
    x + width / 2, new_times, width, label="NEW (single-pass)", alpha=0.8, color="#27ae60"
)

ax1.set_ylabel("Pruning Update Time (ms)", fontsize=11)
ax1.set_title("Pruning Update Time Comparison", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Plot 2: Speedup factors
ax2 = axes[0, 1]
bars = ax2.bar(x, speedups, alpha=0.8, color="#3498db", width=0.5)
ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="No speedup")
ax2.set_ylabel("Speedup Factor", fontsize=11)
ax2.set_title("Speedup (OLD / NEW)", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.set_ylim(0, 2.5)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for i, (bar, speedup, improvement) in enumerate(zip(bars, speedups, improvements)):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"{speedup:.2f}x\n({improvement:.1f}% faster)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Plot 3: Memory footprint reduction
ax3 = axes[1, 0]
memory_configs = ["OLD\n(100% concat)", "NEW\n(1% sample)"]
memory_values = [old_memory, new_memory]
colors = ["#e74c3c", "#27ae60"]

bars = ax3.bar(range(len(memory_configs)), memory_values, alpha=0.8, color=colors, width=0.5)
ax3.set_ylabel("Memory (MB)", fontsize=11)
ax3.set_title(
    "Concatenation Memory Footprint\n(GPT-2 124M params)",
    fontsize=12,
    fontweight="bold",
)
ax3.set_xticks(range(len(memory_configs)))
ax3.set_xticklabels(memory_configs)
ax3.set_yscale("log")
ax3.grid(True, alpha=0.3, axis="y")

# Add value labels
for bar, val in zip(bars, memory_values):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val} MB",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Add reduction annotation
ax3.text(
    0.5,
    0.5,
    "99% reduction",
    transform=ax3.transAxes,
    fontsize=14,
    fontweight="bold",
    color="#27ae60",
    ha="center",
    va="center",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#27ae60", linewidth=2),
)

# Plot 4: Key optimizations summary
ax4 = axes[1, 1]
ax4.axis("off")

summary_text = """
KEY OPTIMIZATIONS

1. Single-Pass Importance Computation
   • OLD: Read exp_avg_sq twice per pruning step
     - Pass 1: Build all_scores for threshold
     - Pass 2: Recompute for masking
   • NEW: Read exp_avg_sq once (cached)
   • Result: ~50% reduction in Adam state reads

2. Sampled Threshold Estimation
   • OLD: torch.cat(all_scores) - 124M params
   • NEW: torch.cat(samples) - 1.24M params (1%)
   • Result: 99% reduction in concat size

3. Memory Bandwidth Impact
   • OLD: 2x memory traffic per pruning step
   • NEW: 1x memory traffic per pruning step
   • GPU Impact: Predicted ~25% reduction in
     memory access time (18.59% → ~14.5%)

CONSISTENT SPEEDUP
• 12.58M params: 1.84x speedup
• 50.33M params: 1.78x speedup
• Scales well to GPT-2 124M parameters
"""

ax4.text(
    0.05,
    0.95,
    summary_text,
    transform=ax4.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=1", facecolor="#f8f9fa", edgecolor="#dee2e6", linewidth=1.5),
)

plt.tight_layout()
plt.savefig("docs/images/bitter7_double_pass_speedup.png", dpi=150, bbox_inches="tight")
print("\n" + "=" * 60)
print("Visualization saved to: docs/images/bitter7_double_pass_speedup.png")
print("=" * 60)
print("\nBenchmark Summary:")
print(f"  12.58M params: {old_times[0]:.1f}ms → {new_times[0]:.1f}ms ({speedups[0]:.2f}x)")
print(f"  50.33M params: {old_times[1]:.1f}ms → {new_times[1]:.1f}ms ({speedups[1]:.2f}x)")
print(f"\n  Memory reduction: {old_memory}MB → {new_memory}MB (99%)")
print("=" * 60)
