#!/usr/bin/env python3
"""
Analyze bitter7 optimization validation runs from W&B.

Compares three runs:
1. Baseline (magnitude pruning)
2. Old bitter7 (unoptimized, with double-pass)
3. New bitter7 (optimized, single-pass + sampling)
"""
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize W&B API
api = wandb.Api()

# Project path
project = "mcgrof-citizen/gpt2-bitter7-optimized-b200x4"

print(f"Fetching runs from project: {project}")
print("=" * 80)

# Get all runs
runs = api.runs(project)

print(f"\nFound {len(runs)} run(s) in project:\n")

# List all runs
for i, run in enumerate(runs, 1):
    print(f"{i}. {run.name}")
    print(f"   ID: {run.id}")
    print(f"   State: {run.state}")
    print(f"   Created: {run.created_at}")

    # Try to get some basic metrics
    summary = run.summary._json_dict
    if summary:
        if 'val/loss' in summary:
            print(f"   Val Loss: {summary['val/loss']:.4f}")
        if 'train/iter' in summary:
            print(f"   Iterations: {summary['train/iter']}")
    print()

print("=" * 80)
print("\nFetching GPU system metrics (this may take a moment)...\n")

# Fetch system metrics for each run
run_data = {}

for run in runs:
    run_name = run.name
    print(f"Processing: {run_name}")

    try:
        # Get system metrics with stream="events"
        history = run.history(stream="events", samples=10000)

        if history.empty:
            print(f"  WARNING: No system metrics found")
            continue

        # Look for GPU metrics
        gpu_cols = [c for c in history.columns if 'system.gpu.' in c]

        if not gpu_cols:
            print(f"  WARNING: No GPU metrics in system stream")
            continue

        print(f"  Found {len(gpu_cols)} GPU metric columns")

        # Extract memory and compute utilization for each GPU
        mem_util_cols = [c for c in gpu_cols if c.endswith('.memory')]
        compute_cols = [c for c in gpu_cols if c.endswith('.gpu')]

        stats = {}

        # Memory utilization (% time accessing memory)
        if mem_util_cols:
            for col in mem_util_cols:
                gpu_id = col.split('.')[2]
                data = history[col].dropna()
                if len(data) > 0:
                    stats[f'gpu{gpu_id}_memory_mean'] = data.mean()
                    stats[f'gpu{gpu_id}_memory_peak'] = data.max()

        # Compute utilization
        if compute_cols:
            for col in compute_cols:
                gpu_id = col.split('.')[2]
                data = history[col].dropna()
                if len(data) > 0:
                    stats[f'gpu{gpu_id}_compute_mean'] = data.mean()
                    stats[f'gpu{gpu_id}_compute_peak'] = data.max()

        # Average across GPUs
        memory_means = [v for k, v in stats.items() if 'memory_mean' in k]
        compute_means = [v for k, v in stats.items() if 'compute_mean' in k]

        if memory_means:
            stats['avg_memory_util'] = np.mean(memory_means)
        if compute_means:
            stats['avg_compute_util'] = np.mean(compute_means)

        run_data[run_name] = stats

        print(f"  Avg Memory Access: {stats.get('avg_memory_util', 0):.2f}%")
        print(f"  Avg Compute Util:  {stats.get('avg_compute_util', 0):.2f}%")

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

if len(run_data) >= 2:
    # Find baseline and optimized runs
    baseline_key = None
    old_key = None
    new_key = None

    for key in run_data.keys():
        key_lower = key.lower()
        if 'magnitude' in key_lower or 'baseline' in key_lower:
            baseline_key = key
        elif 'optimized' in key_lower or 'new' in key_lower:
            new_key = key
        else:
            old_key = key

    print("\nDetected runs:")
    print(f"  Baseline: {baseline_key or 'Not found'}")
    print(f"  Old Bitter7: {old_key or 'Not found'}")
    print(f"  New Bitter7: {new_key or 'Not found'}")

    if baseline_key and new_key:
        print("\n" + "-" * 80)
        print("GPU Memory Access Time (% time waiting on memory)")
        print("-" * 80)

        baseline_mem = run_data[baseline_key].get('avg_memory_util', 0)
        new_mem = run_data[new_key].get('avg_memory_util', 0)

        print(f"Baseline (magnitude):   {baseline_mem:.2f}%")
        if old_key:
            old_mem = run_data[old_key].get('avg_memory_util', 0)
            print(f"Old Bitter7 (before):   {old_mem:.2f}%")
        print(f"New Bitter7 (after):    {new_mem:.2f}%")

        improvement = baseline_mem - new_mem
        pct_improvement = (improvement / baseline_mem) * 100 if baseline_mem > 0 else 0

        print(f"\nImprovement vs baseline: {improvement:+.2f}% ({pct_improvement:+.1f}%)")

        if old_key:
            old_improvement = old_mem - new_mem
            old_pct = (old_improvement / old_mem) * 100 if old_mem > 0 else 0
            print(f"Improvement vs old:      {old_improvement:+.2f}% ({old_pct:+.1f}%)")

        print("\n" + "-" * 80)
        print("GPU Compute Utilization (% time doing compute)")
        print("-" * 80)

        baseline_compute = run_data[baseline_key].get('avg_compute_util', 0)
        new_compute = run_data[new_key].get('avg_compute_util', 0)

        print(f"Baseline (magnitude):   {baseline_compute:.2f}%")
        if old_key:
            old_compute = run_data[old_key].get('avg_compute_util', 0)
            print(f"Old Bitter7 (before):   {old_compute:.2f}%")
        print(f"New Bitter7 (after):    {new_compute:.2f}%")

        compute_improvement = new_compute - baseline_compute
        compute_pct = (compute_improvement / baseline_compute) * 100 if baseline_compute > 0 else 0

        print(f"\nChange vs baseline: {compute_improvement:+.2f}% ({compute_pct:+.1f}%)")

        if old_key:
            old_compute_diff = new_compute - old_compute
            old_compute_pct = (old_compute_diff / old_compute) * 100 if old_compute > 0 else 0
            print(f"Change vs old:      {old_compute_diff:+.2f}% ({old_compute_pct:+.1f}%)")

        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)

        # Check if we hit our targets
        target_mem = 14.5  # Target memory access time
        target_compute_gain = 10  # Target compute utilization gain

        if new_mem <= target_mem:
            print(f"✓ Memory access time: {new_mem:.2f}% (target: ≤{target_mem}%)")
        else:
            print(f"✗ Memory access time: {new_mem:.2f}% (target: ≤{target_mem}%)")

        if compute_improvement >= target_compute_gain:
            print(f"✓ Compute utilization: +{compute_improvement:.2f}% (target: ≥+{target_compute_gain}%)")
        else:
            print(f"~ Compute utilization: +{compute_improvement:.2f}% (target: ≥+{target_compute_gain}%)")

        if old_key and old_improvement > 0:
            print(f"✓ Improved over old bitter7: -{old_improvement:.2f}% memory access")

else:
    print("\nNot enough runs to compare. Need at least 2 runs.")

print("\n" + "=" * 80)
