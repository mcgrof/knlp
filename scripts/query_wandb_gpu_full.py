#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Query WandB GPU memory stats from system events stream.
"""

import wandb
import argparse
import sys
import pandas as pd


def query_gpu_stats(project, run_name=None, run_id=None):
    """Query GPU memory stats from WandB system events."""

    api = wandb.Api()

    # Get the run
    if run_id:
        run = api.run(f"{project}/{run_id}")
    elif run_name:
        runs = api.runs(project, filters={"display_name": run_name})
        if not runs:
            print(f"No run found with name: {run_name}")
            return None
        run = runs[0]
    else:
        runs = api.runs(project, order="-created_at")
        if not runs:
            print(f"No runs found in project: {project}")
            return None
        run = runs[0]

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    print("-" * 80)

    # Get system metrics from events stream
    print("\nFetching GPU metrics from system events...")
    system_events = list(run.history(stream="events", pandas=False))

    if not system_events:
        print("No system events found")
        return None

    print(f"Found {len(system_events)} system event samples")

    # Extract GPU metrics
    gpu_data = {i: [] for i in range(8)}  # Support up to 8 GPUs

    for event in system_events:
        for gpu_idx in range(8):
            memory_key = f"system.gpu.{gpu_idx}.memoryAllocatedBytes"
            if memory_key in event:
                gpu_data[gpu_idx].append({
                    'memory_bytes': event.get(memory_key, 0),
                    'memory_pct': event.get(f"system.gpu.{gpu_idx}.memoryAllocated", 0),
                    'utilization': event.get(f"system.gpu.{gpu_idx}.gpu", 0),
                    'power_watts': event.get(f"system.gpu.{gpu_idx}.powerWatts", 0),
                    'temp': event.get(f"system.gpu.{gpu_idx}.temp", 0),
                })

    # Filter to only GPUs with data
    active_gpus = {idx: data for idx, data in gpu_data.items() if data}

    if not active_gpus:
        print("No GPU data found in system events")
        return None

    print(f"\nDetected {len(active_gpus)} active GPUs")
    print("\nGPU Memory Statistics:")
    print("=" * 80)

    total_current = 0
    total_avg = 0
    total_max = 0

    for gpu_idx in sorted(active_gpus.keys()):
        data = active_gpus[gpu_idx]
        df = pd.DataFrame(data)

        memory_mb = df['memory_bytes'] / (1024 * 1024)
        memory_gb = memory_mb / 1024

        current_gb = memory_gb.iloc[-1]
        avg_gb = memory_gb.mean()
        max_gb = memory_gb.max()
        min_gb = memory_gb.min()

        total_current += current_gb
        total_avg += avg_gb
        total_max += max_gb

        print(f"\nGPU {gpu_idx}:")
        print(f"  Memory Allocated:")
        print(f"    Current:  {current_gb:.2f} GB ({df['memory_pct'].iloc[-1]:.1f}%)")
        print(f"    Average:  {avg_gb:.2f} GB ({df['memory_pct'].mean():.1f}%)")
        print(f"    Maximum:  {max_gb:.2f} GB ({df['memory_pct'].max():.1f}%)")
        print(f"    Minimum:  {min_gb:.2f} GB ({df['memory_pct'].min():.1f}%)")

        print(f"  GPU Utilization:")
        print(f"    Current:  {df['utilization'].iloc[-1]:.1f}%")
        print(f"    Average:  {df['utilization'].mean():.1f}%")
        print(f"    Maximum:  {df['utilization'].max():.1f}%")

        print(f"  Power:")
        print(f"    Current:  {df['power_watts'].iloc[-1]:.1f} W")
        print(f"    Average:  {df['power_watts'].mean():.1f} W")
        print(f"    Maximum:  {df['power_watts'].max():.1f} W")

        print(f"  Temperature:")
        print(f"    Current:  {df['temp'].iloc[-1]:.0f}°C")
        print(f"    Average:  {df['temp'].mean():.0f}°C")
        print(f"    Maximum:  {df['temp'].max():.0f}°C")

    print(f"\n{'='*80}")
    print(f"TOTAL ACROSS ALL {len(active_gpus)} GPUs:")
    print(f"  Current Total Memory:  {total_current:.2f} GB")
    print(f"  Average Total Memory:  {total_avg:.2f} GB")
    print(f"  Maximum Total Memory:  {total_max:.2f} GB")

    print(f"\n{'='*80}")
    print(f"Total samples: {len(system_events)}")

    return active_gpus


def main():
    parser = argparse.ArgumentParser(description="Query WandB GPU memory stats")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run-name", help="Run name (display name)")
    parser.add_argument("--run-id", help="Run ID")
    parser.add_argument("--entity", help="WandB entity (username/team)")

    args = parser.parse_args()

    project = f"{args.entity}/{args.project}" if args.entity else args.project
    gpu_data = query_gpu_stats(project, args.run_name, args.run_id)

    if not gpu_data:
        sys.exit(1)


if __name__ == "__main__":
    main()
