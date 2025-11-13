#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compare GPU memory usage between two WandB runs.
"""

import wandb
import argparse
import sys
import pandas as pd


def get_gpu_stats(run):
    """Extract GPU stats from a run."""
    system_events = list(run.history(stream="events", pandas=False))

    if not system_events:
        return None

    gpu_data = {i: [] for i in range(8)}

    for event in system_events:
        for gpu_idx in range(8):
            memory_key = f"system.gpu.{gpu_idx}.memoryAllocatedBytes"
            if memory_key in event:
                gpu_data[gpu_idx].append({
                    'memory_bytes': event.get(memory_key, 0),
                    'memory_pct': event.get(f"system.gpu.{gpu_idx}.memoryAllocated", 0),
                    'utilization': event.get(f"system.gpu.{gpu_idx}.gpu", 0),
                    'power_watts': event.get(f"system.gpu.{gpu_idx}.powerWatts", 0),
                })

    active_gpus = {idx: data for idx, data in gpu_data.items() if data}

    if not active_gpus:
        return None

    # Calculate statistics
    stats = {}
    total_current = 0
    total_avg = 0
    total_max = 0

    for gpu_idx in sorted(active_gpus.keys()):
        data = active_gpus[gpu_idx]
        df = pd.DataFrame(data)

        memory_gb = df['memory_bytes'] / (1024 * 1024 * 1024)

        stats[f'gpu_{gpu_idx}'] = {
            'current_gb': memory_gb.iloc[-1],
            'avg_gb': memory_gb.mean(),
            'max_gb': memory_gb.max(),
            'min_gb': memory_gb.min(),
            'avg_util': df['utilization'].mean(),
            'avg_power': df['power_watts'].mean(),
        }

        total_current += memory_gb.iloc[-1]
        total_avg += memory_gb.mean()
        total_max += memory_gb.max()

    stats['total'] = {
        'current_gb': total_current,
        'avg_gb': total_avg,
        'max_gb': total_max,
        'gpu_count': len(active_gpus),
    }

    return stats


def compare_runs(project, run1_name, run2_name):
    """Compare GPU memory usage between two runs."""
    api = wandb.Api()

    # Get run 1
    runs1 = api.runs(project, filters={"display_name": run1_name})
    if not runs1:
        print(f"Run not found: {run1_name}")
        return None
    run1 = runs1[0]

    # Get run 2
    runs2 = api.runs(project, filters={"display_name": run2_name})
    if not runs2:
        print(f"Run not found: {run2_name}")
        return None
    run2 = runs2[0]

    print("=" * 80)
    print("GPU MEMORY COMPARISON")
    print("=" * 80)
    print(f"\nRun 1: {run1.name}")
    print(f"  URL: {run1.url}")
    print(f"\nRun 2: {run2.name}")
    print(f"  URL: {run2.url}")
    print()

    # Get stats
    print("Fetching GPU stats for Run 1...")
    stats1 = get_gpu_stats(run1)

    print("Fetching GPU stats for Run 2...")
    stats2 = get_gpu_stats(run2)

    if not stats1 or not stats2:
        print("Error: Could not fetch GPU stats for one or both runs")
        return None

    # Compare totals
    print("\n" + "=" * 80)
    print("AGGREGATE COMPARISON (All GPUs)")
    print("=" * 80)

    total1 = stats1['total']
    total2 = stats2['total']

    print(f"\n{'Metric':<30} {'Run 1':<20} {'Run 2':<20} {'Diff':<15}")
    print("-" * 85)

    metrics = [
        ('Current Total Memory', 'current_gb', 'GB'),
        ('Average Total Memory', 'avg_gb', 'GB'),
        ('Maximum Total Memory', 'max_gb', 'GB'),
    ]

    for label, key, unit in metrics:
        val1 = total1[key]
        val2 = total2[key]
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 > 0 else 0

        print(f"{label:<30} {val1:>8.2f} {unit:<11} {val2:>8.2f} {unit:<11} {diff:>+7.2f} {unit} ({diff_pct:>+6.1f}%)")

    # Per-GPU comparison
    print("\n" + "=" * 80)
    print("PER-GPU COMPARISON (Average Memory)")
    print("=" * 80)
    print()

    gpu_count = max(total1['gpu_count'], total2['gpu_count'])

    for gpu_idx in range(gpu_count):
        gpu_key = f'gpu_{gpu_idx}'

        if gpu_key in stats1 and gpu_key in stats2:
            gpu1 = stats1[gpu_key]
            gpu2 = stats2[gpu_key]

            avg1 = gpu1['avg_gb']
            avg2 = gpu2['avg_gb']
            max1 = gpu1['max_gb']
            max2 = gpu2['max_gb']

            diff_avg = avg2 - avg1
            diff_avg_pct = (diff_avg / avg1 * 100) if avg1 > 0 else 0

            diff_max = max2 - max1
            diff_max_pct = (diff_max / max1 * 100) if max1 > 0 else 0

            print(f"GPU {gpu_idx}:")
            print(f"  Average: {avg1:.2f} GB → {avg2:.2f} GB  ({diff_avg:+.2f} GB, {diff_avg_pct:+.1f}%)")
            print(f"  Maximum: {max1:.2f} GB → {max2:.2f} GB  ({diff_max:+.2f} GB, {diff_max_pct:+.1f}%)")
            print(f"  Avg Util: {gpu1['avg_util']:.1f}% → {gpu2['avg_util']:.1f}%")
            print(f"  Avg Power: {gpu1['avg_power']:.1f}W → {gpu2['avg_power']:.1f}W")
            print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_diff = total2['avg_gb'] - total1['avg_gb']
    total_diff_pct = (total_diff / total1['avg_gb'] * 100) if total1['avg_gb'] > 0 else 0

    if total_diff > 0:
        print(f"\n⚠️  Run 2 uses MORE memory: +{total_diff:.2f} GB ({total_diff_pct:+.1f}%)")
    elif total_diff < 0:
        print(f"\n✅ Run 2 uses LESS memory: {total_diff:.2f} GB ({total_diff_pct:+.1f}%)")
    else:
        print(f"\n➖ Memory usage is equivalent")

    print(f"\nAverage across all GPUs:")
    print(f"  Run 1: {total1['avg_gb']:.2f} GB")
    print(f"  Run 2: {total2['avg_gb']:.2f} GB")
    print()

    return {
        'run1': stats1,
        'run2': stats2,
        'total_diff_gb': total_diff,
        'total_diff_pct': total_diff_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare GPU memory between two WandB runs")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run1", required=True, help="First run name")
    parser.add_argument("--run2", required=True, help="Second run name")
    parser.add_argument("--entity", help="WandB entity (username/team)")

    args = parser.parse_args()

    project = f"{args.entity}/{args.project}" if args.entity else args.project
    result = compare_runs(project, args.run1, args.run2)

    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
