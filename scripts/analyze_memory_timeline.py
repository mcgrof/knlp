#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Analyze GPU memory timeline to understand memory usage patterns.
"""

import wandb
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def analyze_timeline(project, run_name):
    """Analyze GPU memory timeline for a run."""
    api = wandb.Api()

    runs = api.runs(project, filters={"display_name": run_name})
    if not runs:
        print(f"Run not found: {run_name}")
        return None
    run = runs[0]

    print(f"Run: {run.name}")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    print("-" * 80)

    # Get system events
    system_events = list(run.history(stream="events", pandas=False))
    print(f"Fetched {len(system_events)} system events\n")

    # Extract GPU 0 memory over time (representative)
    timeline = []
    for event in system_events:
        if "system.gpu.0.memoryAllocatedBytes" in event:
            timeline.append({
                'timestamp': event.get('_timestamp', 0),
                'gpu0_memory_gb': event['system.gpu.0.memoryAllocatedBytes'] / (1024**3),
                'gpu0_util': event.get('system.gpu.0.gpu', 0),
            })

    if not timeline:
        print("No GPU memory data found")
        return None

    df = pd.DataFrame(timeline)
    df['time_hours'] = (df['timestamp'] - df['timestamp'].min()) / 3600

    # Statistics
    print("GPU 0 Memory Analysis:")
    print(f"  Samples: {len(df)}")
    print(f"  Duration: {df['time_hours'].max():.2f} hours")
    print(f"  Current: {df['gpu0_memory_gb'].iloc[-1]:.2f} GB")
    print(f"  Average: {df['gpu0_memory_gb'].mean():.2f} GB")
    print(f"  Maximum: {df['gpu0_memory_gb'].max():.2f} GB")
    print(f"  Minimum: {df['gpu0_memory_gb'].min():.2f} GB")
    print(f"  Std Dev: {df['gpu0_memory_gb'].std():.2f} GB")
    print()

    # Check for patterns
    print("Memory Pattern Analysis:")

    # First 10% vs last 10%
    cutoff = len(df) // 10
    early_avg = df.iloc[:cutoff]['gpu0_memory_gb'].mean()
    late_avg = df.iloc[-cutoff:]['gpu0_memory_gb'].mean()

    print(f"  Early average (first 10%): {early_avg:.2f} GB")
    print(f"  Late average (last 10%): {late_avg:.2f} GB")
    print(f"  Difference: {late_avg - early_avg:+.2f} GB ({(late_avg - early_avg) / early_avg * 100:+.1f}%)")
    print()

    # Check if memory is stable or growing
    if late_avg > early_avg * 1.1:
        print("  ⚠️  Memory usage INCREASED over time (+10%)")
    elif late_avg < early_avg * 0.9:
        print("  ⬇️  Memory usage DECREASED over time (-10%)")
    else:
        print("  ➖ Memory usage STABLE over time")
    print()

    # Quartile analysis
    q1 = df['gpu0_memory_gb'].quantile(0.25)
    q2 = df['gpu0_memory_gb'].quantile(0.50)
    q3 = df['gpu0_memory_gb'].quantile(0.75)

    print("Memory Distribution (Quartiles):")
    print(f"  25th percentile: {q1:.2f} GB")
    print(f"  50th percentile: {q2:.2f} GB")
    print(f"  75th percentile: {q3:.2f} GB")
    print()

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze GPU memory timeline")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run", required=True, help="Run name")
    parser.add_argument("--entity", help="WandB entity")

    args = parser.parse_args()

    project = f"{args.entity}/{args.project}" if args.entity else args.project
    df = analyze_timeline(project, args.run)


if __name__ == "__main__":
    main()
