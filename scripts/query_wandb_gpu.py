#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Query WandB for GPU memory stats from a specific run.
"""

import wandb
import argparse
import sys
from pprint import pprint


def query_gpu_stats(project, run_name=None, run_id=None):
    """Query GPU memory stats from WandB."""

    api = wandb.Api()

    # Get the run
    if run_id:
        run = api.run(f"{project}/{run_id}")
    elif run_name:
        # Search for run by name
        runs = api.runs(project, filters={"display_name": run_name})
        if not runs:
            print(f"No run found with name: {run_name}")
            return None
        run = runs[0]
    else:
        # Get latest run
        runs = api.runs(project, order="-created_at")
        if not runs:
            print(f"No runs found in project: {project}")
            return None
        run = runs[0]

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    print("-" * 80)

    # Get GPU metrics - try all possible key formats
    all_keys = []
    for i in range(8):  # Support up to 8 GPUs
        all_keys.extend([
            f"system.gpu.{i}.memory",
            f"system.gpu.{i}.memoryAllocated",
            f"system.gpu.{i}.memoryAllocatedBytes",
            f"system.gpu.{i}.gpu",
            f"system.gpu.{i}.temp",
            f"system.gpu.{i}.powerWatts",
            f"GPU Memory Allocated (%) - GPU {i}",
            f"GPU Memory Allocated (Bytes) - GPU {i}",
        ])
    all_keys.append("_step")

    # Get available keys first
    sample_history = run.scan_history()
    available_keys = set()
    for item in sample_history:
        available_keys.update(item.keys())
        if len(available_keys) > 100:  # Sample enough to find GPU keys
            break

    # Filter to only GPU-related keys
    gpu_keys = [k for k in available_keys if 'gpu' in k.lower() or 'GPU' in k]
    gpu_keys.append("_step")

    print(f"Found GPU-related keys: {gpu_keys[:20]}")
    print()

    # Fetch history with discovered keys
    history = run.history(keys=gpu_keys, pandas=True)

    if history.empty:
        print("No GPU metrics found in this run")
        return None

    # Calculate stats - dynamically detect GPU metrics
    print("\nGPU Memory Statistics (across all logged steps):")
    print("=" * 80)

    # Find all GPU indices in the data
    gpu_indices = set()
    for col in history.columns:
        for i in range(16):  # Check up to 16 GPUs
            if f"GPU {i}" in col or f"gpu.{i}" in col:
                gpu_indices.add(i)

    for gpu_idx in sorted(gpu_indices):
        # Try various key formats
        memory_keys = [
            f"system.gpu.{gpu_idx}.memory",
            f"GPU Memory Allocated (Bytes) - GPU {gpu_idx}",
            f"system.gpu.{gpu_idx}.memoryAllocatedBytes"
        ]
        allocated_pct_keys = [
            f"system.gpu.{gpu_idx}.memoryAllocated",
            f"GPU Memory Allocated (%) - GPU {gpu_idx}"
        ]
        util_keys = [
            f"system.gpu.{gpu_idx}.gpu",
            f"GPU Utilization (%) - GPU {gpu_idx}"
        ]

        # Find which keys exist
        memory_key = next((k for k in memory_keys if k in history.columns), None)
        allocated_key = next((k for k in allocated_pct_keys if k in history.columns), None)
        util_key = next((k for k in util_keys if k in history.columns), None)

        if memory_key or allocated_key:
            print(f"\nGPU {gpu_idx}:")

            if memory_key:
                memory_data = history[memory_key].dropna()
                if not memory_data.empty:
                    # Convert to MB if in bytes
                    if "Bytes" in memory_key:
                        memory_data = memory_data / (1024 * 1024)
                    print(f"  Memory Used:")
                    print(f"    Current:  {memory_data.iloc[-1]:.1f} MB ({memory_data.iloc[-1]/1024:.2f} GB)")
                    print(f"    Average:  {memory_data.mean():.1f} MB ({memory_data.mean()/1024:.2f} GB)")
                    print(f"    Maximum:  {memory_data.max():.1f} MB ({memory_data.max()/1024:.2f} GB)")
                    print(f"    Minimum:  {memory_data.min():.1f} MB ({memory_data.min()/1024:.2f} GB)")

            if allocated_key:
                allocated_data = history[allocated_key].dropna()
                if not allocated_data.empty:
                    print(f"  Memory Allocated (%):")
                    print(f"    Current:  {allocated_data.iloc[-1]:.1f}%")
                    print(f"    Average:  {allocated_data.mean():.1f}%")
                    print(f"    Maximum:  {allocated_data.max():.1f}%")

            if util_key:
                util_data = history[util_key].dropna()
                if not util_data.empty:
                    print(f"  GPU Utilization (%):")
                    print(f"    Current:  {util_data.iloc[-1]:.1f}%")
                    print(f"    Average:  {util_data.mean():.1f}%")
                    print(f"    Maximum:  {util_data.max():.1f}%")

    # Total across all GPUs
    memory_cols = [col for col in history.columns if "Memory" in col and "Bytes" in col]

    if memory_cols:
        total_memory = history[memory_cols].sum(axis=1) / (1024 * 1024)  # Convert to MB
        print(f"\n{'='*80}")
        print(f"TOTAL ACROSS ALL {len(memory_cols)} GPUs:")
        print(f"  Current Total Memory:  {total_memory.iloc[-1]:.1f} MB ({total_memory.iloc[-1]/1024:.2f} GB)")
        print(f"  Average Total Memory:  {total_memory.mean():.1f} MB ({total_memory.mean()/1024:.2f} GB)")
        print(f"  Maximum Total Memory:  {total_memory.max():.1f} MB ({total_memory.max()/1024:.2f} GB)")
        print(f"  Minimum Total Memory:  {total_memory.min():.1f} MB ({total_memory.min()/1024:.2f} GB)")

    print(f"\n{'='*80}")
    print(f"Total logged steps: {len(history)}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Query WandB GPU memory stats")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run-name", help="Run name (display name)")
    parser.add_argument("--run-id", help="Run ID")
    parser.add_argument("--entity", help="WandB entity (username/team)")

    args = parser.parse_args()

    # Build full project path with entity if provided
    project = f"{args.entity}/{args.project}" if args.entity else args.project

    history = query_gpu_stats(project, args.run_name, args.run_id)

    if history is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
