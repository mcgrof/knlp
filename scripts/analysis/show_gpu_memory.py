#!/usr/bin/env python3
import json
from pathlib import Path

results_dir = Path("test_matrix_results_20250827_231931")
print("Actual GPU Memory Usage (from monitoring):")
print("=" * 50)
print(f"{'Optimizer':<12} {'Average':>10} {'Peak':>10} {'Min':>10}")
print("-" * 42)

memory_data = {}
for gpu_file in sorted(results_dir.glob("*/gpu_stats*.json")):
    optimizer = gpu_file.parent.name.split("_")[1]
    with open(gpu_file) as f:
        data = json.load(f)
        mem_mb = [d["memory_used"] for d in data if "memory_used" in d]
        if mem_mb:
            mem_gb = [m / 1024 for m in mem_mb]
            memory_data[optimizer] = {
                "avg": sum(mem_gb) / len(mem_gb),
                "peak": max(mem_gb),
                "min": min(mem_gb),
            }

# Sort by average memory
for opt in sorted(memory_data.keys(), key=lambda x: memory_data[x]["avg"]):
    stats = memory_data[opt]
    print(
        f"{opt.upper():<12} {stats['avg']:>8.2f} GB {stats['peak']:>8.2f} GB {stats['min']:>8.2f} GB"
    )

if "adamwprune" in memory_data and "adam" in memory_data:
    savings = (
        (memory_data["adam"]["avg"] - memory_data["adamwprune"]["avg"])
        / memory_data["adam"]["avg"]
    ) * 100
    print(f"\nAdamWPrune saves {savings:.1f}% GPU memory vs Adam!")
