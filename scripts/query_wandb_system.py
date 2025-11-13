#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Query WandB system metrics (GPU, CPU, etc.) from a specific run.
"""

import wandb
import argparse
import sys
from pprint import pprint


def query_system_metrics(project, run_name=None, run_id=None):
    """Query system metrics from WandB."""

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

    # Try to get system metrics from run.summary
    print("\nChecking run.summary for GPU metrics:")
    summary = dict(run.summary)
    gpu_summary = {k: v for k, v in summary.items() if 'gpu' in k.lower() or 'GPU' in k}

    if gpu_summary:
        print("\nGPU metrics in summary:")
        for key, value in sorted(gpu_summary.items()):
            print(f"  {key}: {value}")
    else:
        print("  No GPU metrics in summary")

    # Try system_metrics API endpoint if available
    print("\nChecking for system_metrics data:")
    try:
        # Use the underlying API to get system metrics
        system_metrics = run.history(stream="events", pandas=False)
        if system_metrics:
            print(f"  Found {len(list(system_metrics))} system metric events")
            # Sample first few
            for i, event in enumerate(system_metrics):
                if i >= 3:
                    break
                print(f"\n  Event {i}:")
                gpu_keys = {k: v for k, v in event.items() if 'gpu' in k.lower() or 'GPU' in k}
                if gpu_keys:
                    pprint(gpu_keys, indent=4)
        else:
            print("  No system metrics found")
    except Exception as e:
        print(f"  Error accessing system metrics: {e}")

    # Try getting full history with all keys
    print("\nFetching full history to find all keys...")
    history = run.scan_history()
    all_keys = set()
    sample_count = 0
    for entry in history:
        all_keys.update(entry.keys())
        sample_count += 1
        if sample_count >= 100:
            break

    gpu_keys = sorted([k for k in all_keys if 'gpu' in k.lower() or 'GPU' in k])
    print(f"\nAll GPU-related keys found ({len(gpu_keys)}):")
    for key in gpu_keys:
        print(f"  - {key}")

    if not gpu_keys:
        print("\n❌ No GPU metrics found in this run via API")
        print("\nNote: WandB may show GPU metrics in the UI but not expose them via API.")
        print("This can happen if:")
        print("  1. System metrics monitoring is enabled but logged separately")
        print("  2. Metrics are visible in UI but not in history API")
        print("  3. DDP mode interferes with system metrics collection")
        return None

    return gpu_keys


def main():
    parser = argparse.ArgumentParser(description="Query WandB system/GPU metrics")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run-name", help="Run name (display name)")
    parser.add_argument("--run-id", help="Run ID")
    parser.add_argument("--entity", help="WandB entity (username/team)")

    args = parser.parse_args()

    project = f"{args.entity}/{args.project}" if args.entity else args.project
    gpu_keys = query_system_metrics(project, args.run_name, args.run_id)

    if not gpu_keys:
        sys.exit(1)


if __name__ == "__main__":
    main()
