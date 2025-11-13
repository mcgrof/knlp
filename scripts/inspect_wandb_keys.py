#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Inspect what keys are available in a WandB run.
"""

import wandb
import argparse


def inspect_run(project, run_name=None, run_id=None):
    """Inspect available keys in a WandB run."""

    api = wandb.Api()

    # Get the run
    if run_id:
        run = api.run(f"{project}/{run_id}")
    elif run_name:
        runs = api.runs(project, filters={"display_name": run_name})
        if not runs:
            print(f"No run found with name: {run_name}")
            return
        run = runs[0]
    else:
        runs = api.runs(project, order="-created_at")
        if not runs:
            print(f"No runs found in project: {project}")
            return
        run = runs[0]

    print(f"Run: {run.name} ({run.id})")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")
    print("-" * 80)

    # Get all available keys
    print("\nFetching run history to discover keys...")
    history = run.history(pandas=False)

    if not history:
        print("No history data found")
        return

    # Get all unique keys
    all_keys = set()
    count = 0
    for entry in history:
        all_keys.update(entry.keys())
        count += 1
        if count >= 1000:  # Sample more entries
            break

    print(f"\nTotal unique keys: {len(all_keys)}")
    print("\nAvailable keys:")
    print("=" * 80)

    # Categorize keys
    gpu_keys = [k for k in all_keys if 'gpu' in k.lower()]
    system_keys = [k for k in all_keys if 'system' in k.lower()]
    training_keys = [k for k in all_keys if k not in gpu_keys and k not in system_keys]

    if gpu_keys:
        print("\nGPU-related keys:")
        for key in sorted(gpu_keys):
            print(f"  - {key}")

    if system_keys:
        print("\nSystem-related keys:")
        for key in sorted(system_keys):
            print(f"  - {key}")

    print("\nTraining metrics keys:")
    for key in sorted(training_keys)[:30]:  # Show first 30
        print(f"  - {key}")

    if len(training_keys) > 30:
        print(f"  ... and {len(training_keys) - 30} more")


def main():
    parser = argparse.ArgumentParser(description="Inspect WandB run keys")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--run-name", help="Run name (display name)")
    parser.add_argument("--run-id", help="Run ID")
    parser.add_argument("--entity", help="WandB entity (username/team)")

    args = parser.parse_args()

    project = f"{args.entity}/{args.project}" if args.entity else args.project
    inspect_run(project, args.run_name, args.run_id)


if __name__ == "__main__":
    main()
