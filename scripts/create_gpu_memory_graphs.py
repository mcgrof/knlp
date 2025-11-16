#!/usr/bin/env python3
"""Create GPU memory consumption comparison graphs for AdamWPrune variants.

Compares GPU memory usage between:
1. Movement Pruning WITH torch.compile (baseline)
2. bitter7 WITH torch.compile (best performance)
3. bitter8 WITHOUT torch.compile (no test matrix GPU data available)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 16

print("Loading GPU memory data...")

# Load Movement Pruning (WITH torch.compile) system metrics
mag_sys = pd.read_csv("system_metrics_gpt2_adamwspam_magnitude_50.csv")

# Load bitter7 (WITH torch.compile) system metrics
b7_sys = pd.read_csv("system_metrics_gpt2_adamwprune_bitter7_state_50.csv")

# Extract GPU memory allocated (average across 4 GPUs)
# memoryAllocated is percentage, memoryAllocatedBytes is actual bytes


def get_gpu_memory_mb(df):
    """Extract average GPU memory usage across all GPUs in MB."""
    gpu_mem_cols = [c for c in df.columns if "memoryAllocatedBytes" in c]
    gpu_mem_data = df[gpu_mem_cols]

    # Convert bytes to MiB and take average across GPUs
    gpu_mem_mb = gpu_mem_data / (1024 * 1024)
    avg_mem_mb = gpu_mem_mb.mean(axis=1)

    return avg_mem_mb


def get_gpu_memory_percent(df):
    """Extract average GPU memory percentage across all GPUs."""
    gpu_mem_cols = [
        c for c in df.columns if "memoryAllocated" in c and "Bytes" not in c
    ]
    gpu_mem_data = df[gpu_mem_cols]

    # Take average across GPUs
    avg_mem_pct = gpu_mem_data.mean(axis=1)

    return avg_mem_pct


mag_mem_mb = get_gpu_memory_mb(mag_sys)
b7_mem_mb = get_gpu_memory_mb(b7_sys)

mag_mem_pct = get_gpu_memory_percent(mag_sys)
b7_mem_pct = get_gpu_memory_percent(b7_sys)

# Get runtime for x-axis (in seconds)
mag_runtime = mag_sys["_runtime"]
b7_runtime = b7_sys["_runtime"]

print(f"Movement Pruning: {len(mag_mem_mb)} memory samples")
print(f"bitter7: {len(b7_mem_mb)} memory samples")
print()

# Calculate statistics
mag_mean_mb = mag_mem_mb.mean()
mag_max_mb = mag_mem_mb.max()
b7_mean_mb = b7_mem_mb.mean()
b7_max_mb = b7_mem_mb.max()

mag_mean_pct = mag_mem_pct.mean()
b7_mean_pct = b7_mem_pct.mean()

print(f"Movement Pruning (WITH compile):")
print(f"  Mean GPU memory: {mag_mean_mb:.0f} MiB ({mag_mean_pct:.1f}%)")
print(f"  Peak GPU memory: {mag_max_mb:.0f} MiB")
print()
print(f"bitter7 (WITH compile):")
print(f"  Mean GPU memory: {b7_mean_mb:.0f} MiB ({b7_mean_pct:.1f}%)")
print(f"  Peak GPU memory: {b7_max_mb:.0f} MiB")
print()

memory_diff = ((b7_mean_mb - mag_mean_mb) / mag_mean_mb) * 100
print(f"bitter7 uses {memory_diff:+.1f}% memory vs baseline")
print()

# === Graph 1: GPU Memory Over Time (Absolute MiB) ===
print("Creating Graph 1: GPU memory over time (MiB)...")
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(
    mag_runtime,
    mag_mem_mb,
    "-",
    linewidth=2,
    color="#E74C3C",
    label=f"Movement Pruning (WITH compile)\nMean: {mag_mean_mb:.0f} MiB",
    alpha=0.8,
)

ax.plot(
    b7_runtime,
    b7_mem_mb,
    "-",
    linewidth=2,
    color="#27AE60",
    label=f"bitter7 (WITH compile)\nMean: {b7_mean_mb:.0f} MiB",
    alpha=0.8,
)

# Add mean lines
ax.axhline(
    y=mag_mean_mb,
    color="#E74C3C",
    linestyle="--",
    linewidth=1.5,
    alpha=0.5,
)

ax.axhline(
    y=b7_mean_mb,
    color="#27AE60",
    linestyle="--",
    linewidth=1.5,
    alpha=0.5,
)

ax.set_xlabel("Training Time (seconds)", fontsize=13, weight="bold")
ax.set_ylabel("GPU Memory Allocated (MiB)", fontsize=13, weight="bold")
ax.set_title(
    "AdamWPrune: GPU Memory Consumption Comparison\n"
    + "GPT-2 124M, 50% Sparsity, NVIDIA B200 GPUs (4x)",
    fontsize=15,
    weight="bold",
    pad=20,
)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="black", fancybox=True)
ax.grid(True, alpha=0.3, linestyle="--")

# Add annotation showing memory difference
ax.text(
    0.5,
    0.05,
    f"bitter7 uses {memory_diff:+.1f}% memory vs baseline "
    + f"({b7_mean_mb:.0f} vs {mag_mean_mb:.0f} MiB avg)",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.6",
        fc="yellow" if abs(memory_diff) > 5 else "lightblue",
        alpha=0.3,
        edgecolor="black",
        linewidth=2,
    ),
    fontsize=11,
    weight="bold",
)

plt.tight_layout()
plt.savefig("adamwprune_gpu_memory_time.png", dpi=300, bbox_inches="tight")
print("Saved: adamwprune_gpu_memory_time.png")
plt.close()

# === Graph 2: GPU Memory Percentage Over Time ===
print("\nCreating Graph 2: GPU memory percentage over time...")
fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(
    mag_runtime,
    mag_mem_pct,
    "-",
    linewidth=2,
    color="#E74C3C",
    label=f"Movement Pruning (WITH compile)\nMean: {mag_mean_pct:.1f}%",
    alpha=0.8,
)

ax.plot(
    b7_runtime,
    b7_mem_pct,
    "-",
    linewidth=2,
    color="#27AE60",
    label=f"bitter7 (WITH compile)\nMean: {b7_mean_pct:.1f}%",
    alpha=0.8,
)

ax.set_xlabel("Training Time (seconds)", fontsize=13, weight="bold")
ax.set_ylabel("GPU Memory Usage (%)", fontsize=13, weight="bold")
ax.set_title(
    "AdamWPrune: GPU Memory Usage Percentage\n"
    + "GPT-2 124M, 50% Sparsity, NVIDIA B200 GPUs (4x)",
    fontsize=15,
    weight="bold",
    pad=20,
)
ax.legend(loc="upper right", framealpha=0.95, edgecolor="black", fancybox=True)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig("adamwprune_gpu_memory_percent.png", dpi=300, bbox_inches="tight")
print("Saved: adamwprune_gpu_memory_percent.png")
plt.close()

# === Graph 3: Average GPU Memory Bar Chart ===
print("\nCreating Graph 3: Average GPU memory comparison...")
fig, ax = plt.subplots(figsize=(10, 7))

variants = [
    "Movement Pruning\n(WITH compile)\nBaseline",
    "bitter7\n(WITH compile)\nBest",
]
mem_means = [mag_mean_mb, b7_mean_mb]
colors = ["#E74C3C", "#27AE60"]

bars = ax.bar(
    variants, mem_means, color=colors, alpha=0.8, edgecolor="black", linewidth=2
)

# Add value labels on bars
for bar, mem in zip(bars, mem_means):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 100,
        f"{mem:.0f} MiB",
        ha="center",
        va="bottom",
        fontsize=12,
        weight="bold",
    )

ax.set_ylabel("Average GPU Memory (MiB)", fontsize=13, weight="bold")
ax.set_title(
    "AdamWPrune: Average GPU Memory Consumption\n"
    + "GPT-2 124M, 50% Sparsity, NVIDIA B200 GPUs (4x)",
    fontsize=15,
    weight="bold",
    pad=20,
)
ax.grid(True, alpha=0.3, axis="y", linestyle="--")

# Add annotation showing difference
mem_diff_text = (
    f"bitter7 uses {memory_diff:+.1f}% memory vs baseline\n"
    + f"({abs(b7_mean_mb - mag_mean_mb):.0f} MiB {'more' if memory_diff > 0 else 'less'})"
)
ax.text(
    0.5,
    0.85,
    mem_diff_text,
    transform=ax.transAxes,
    ha="center",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.8",
        fc="yellow" if abs(memory_diff) > 5 else "lightgreen",
        alpha=0.3,
        edgecolor="black",
        linewidth=2,
    ),
    fontsize=11,
    weight="bold",
)

plt.tight_layout()
plt.savefig("adamwprune_gpu_memory_avg.png", dpi=300, bbox_inches="tight")
print("Saved: adamwprune_gpu_memory_avg.png")
plt.close()

print("\n" + "=" * 80)
print("GPU MEMORY SUMMARY")
print("=" * 80)
print()
print(f"Movement Pruning (WITH torch.compile):")
print(f"  Average: {mag_mean_mb:.0f} MiB ({mag_mean_pct:.1f}%)")
print(f"  Peak:    {mag_max_mb:.0f} MiB")
print()
print(f"bitter7 (WITH torch.compile):")
print(f"  Average: {b7_mean_mb:.0f} MiB ({b7_mean_pct:.1f}%)")
print(f"  Peak:    {b7_max_mb:.0f} MiB")
print()
print(f"Memory difference: {memory_diff:+.1f}%")
print()
print("Note: bitter8 WITHOUT torch.compile GPU memory data not available")
print("      (test matrix runs don't log system metrics)")
print("=" * 80)
