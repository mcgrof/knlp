#!/usr/bin/env python3
"""
Download system metrics from W&B runs using stream="events".
"""

import wandb
import pandas as pd

def get_system_metrics(entity, project, run_id, run_name):
    """Download system metrics for a specific run."""
    api = wandb.Api()

    run_path = f"{entity}/{project}/{run_id}"
    run = api.run(run_path)

    print(f"\n{'='*80}")
    print(f"Run: {run.name}")
    print(f"ID: {run.id}")
    print(f"State: {run.state}")
    print(f"{'='*80}")

    # Get system metrics using stream="events"
    print("Fetching system metrics (stream='events')...")
    system_metrics = run.history(stream="events")

    print(f"System metrics shape: {system_metrics.shape}")
    print(f"System metrics columns: {len(system_metrics.columns)}")

    # Show column names
    print("\nAvailable system metric columns:")
    for col in sorted(system_metrics.columns):
        print(f"  - {col}")

    # Save to CSV
    csv_file = f"system_metrics_{run_name}.csv"
    system_metrics.to_csv(csv_file, index=False)
    print(f"\nSaved system metrics to: {csv_file}")

    return system_metrics

def main():
    entity = "mcgrof-citizen"
    project = "gpt2-bitter7-b200x4"

    # Define runs to fetch
    runs_config = {
        "gpt2_adamwspam_magnitude_50": "rswzg7b8",      # Magnitude baseline
        "gpt2_adamwprune_bitter7_state_50": "eptbmgdt",  # Bitter7
    }

    all_system_metrics = {}

    for run_name, run_id in runs_config.items():
        try:
            system_metrics = get_system_metrics(entity, project, run_id, run_name)
            all_system_metrics[run_name] = system_metrics

            # Show GPU memory columns if they exist
            gpu_cols = [c for c in system_metrics.columns
                       if 'gpu' in c.lower() and 'memory' in c.lower()]

            if gpu_cols:
                print(f"\nGPU memory columns found: {len(gpu_cols)}")
                for col in gpu_cols:
                    data = system_metrics[col].dropna()
                    if len(data) > 0:
                        print(f"\n  {col}:")
                        print(f"    Count: {len(data)}")
                        print(f"    Mean:  {data.mean():.2f}")
                        print(f"    Max:   {data.max():.2f}")
                        print(f"    Min:   {data.min():.2f}")

        except Exception as e:
            print(f"\nError fetching system metrics for {run_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Fetched system metrics for {len(all_system_metrics)} runs")
    for run_name, metrics_df in all_system_metrics.items():
        print(f"  {run_name}: {len(metrics_df)} rows, {len(metrics_df.columns)} columns")

if __name__ == "__main__":
    main()
