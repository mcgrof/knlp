#!/usr/bin/env python3
"""
Extract GPU memory usage statistics from Weights & Biases runs.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def fetch_run_data(entity, project, run_name):
    """Fetch run data from W&B."""
    api = wandb.Api()

    # Get all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Find the specific run
    target_run = None
    for run in runs:
        if run.name == run_name:
            target_run = run
            break

    if not target_run:
        print(f"Run '{run_name}' not found in {entity}/{project}")
        print("Available runs:")
        for run in runs:
            print(f"  - {run.name} (id: {run.id})")
        return None

    print(f"Found run: {target_run.name} (id: {target_run.id})")
    print(f"  State: {target_run.state}")
    print(f"  Config: {dict(list(target_run.config.items())[:5])}...")

    # Get the run history (metrics over time)
    history = target_run.history(samples=10000)

    print(f"\nCustom metrics columns ({len(history.columns)}):")
    for col in sorted(history.columns):
        print(f"  - {col}")

    # IMPORTANT: Get system metrics (GPU, CPU, memory)
    # Use scan_history to get all available keys including system metrics
    print("\nFetching system metrics...")
    try:
        # Try to get system metrics by scanning all history
        all_keys = set()
        for row in target_run.scan_history():
            all_keys.update(row.keys())
            if len(all_keys) > 100:  # Sample enough to find system keys
                break

        system_keys = [k for k in all_keys if k.startswith('system.') or
                       ('gpu' in k.lower() and not k.startswith('_'))]

        print(f"\nFound {len(system_keys)} system metric keys:")
        for key in sorted(system_keys)[:20]:  # Show first 20
            print(f"  - {key}")

        if system_keys:
            # Fetch history with system keys
            system_history = target_run.history(samples=10000, keys=system_keys, pandas=True)
        else:
            print("No system metrics found, using empty dataframe")
            system_history = pd.DataFrame()

    except Exception as e:
        print(f"Error fetching system metrics: {e}")
        system_history = pd.DataFrame()

    return target_run, history, system_history

def extract_gpu_metrics(system_history):
    """Extract GPU-related metrics from system history."""
    gpu_cols = [col for col in system_history.columns
                if 'gpu' in col.lower() or
                (col.startswith('system.') and ('memory' in col.lower() or 'gpu' in col.lower()))]

    print(f"\nFound {len(gpu_cols)} GPU/memory columns:")
    for col in gpu_cols:
        print(f"  - {col}")

    return gpu_cols

def main():
    entity = "mcgrof-citizen"
    project = "gpt2-bitter7-b200x4"

    # Define the runs to compare
    runs_to_fetch = [
        "gpt2_adamwspam_magnitude_50",  # Baseline magnitude pruning
        "gpt2_adamwprune_bitter7_state_50",  # Bitter7 state pruning
    ]

    all_data = {}

    for run_name in runs_to_fetch:
        print("\n" + "="*80)
        print(f"Fetching: {run_name}")
        print("="*80)

        result = fetch_run_data(entity, project, run_name)
        if result:
            run, history, system_history = result
            gpu_cols = extract_gpu_metrics(system_history)

            # Store relevant columns from both custom and system metrics
            custom_cols = ['_step', '_timestamp', 'iteration', 'sparsity', 'val_loss', 'val_perplexity']
            available_custom = [col for col in custom_cols if col in history.columns]

            all_data[run_name] = {
                'run': run,
                'history': history[available_custom] if available_custom else history,
                'system_history': system_history,
                'gpu_cols': gpu_cols
            }

            # Print summary statistics
            if gpu_cols:
                print("\nGPU Memory Statistics:")
                for col in gpu_cols:
                    if col in system_history.columns:
                        data = system_history[col].dropna()
                        if len(data) > 0:
                            print(f"\n  {col}:")
                            print(f"    Min:  {data.min():.2f}")
                            print(f"    Max:  {data.max():.2f}")
                            print(f"    Mean: {data.mean():.2f}")
                            print(f"    Samples: {len(data)}")

    # Create comparison visualization if we have data
    if len(all_data) >= 2:
        print("\n" + "="*80)
        print("Creating GPU Memory Comparison Visualization")
        print("="*80)

        # Find common GPU memory columns
        common_cols = None
        for run_name, data in all_data.items():
            if common_cols is None:
                common_cols = set(data['gpu_cols'])
            else:
                common_cols &= set(data['gpu_cols'])

        print(f"\nCommon GPU metrics: {common_cols}")

        if common_cols:
            # Plot comparison
            n_metrics = len(common_cols)
            fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=(16, 4*((n_metrics+1)//2)))
            axes = axes.flatten() if n_metrics > 1 else [axes]

            for idx, metric in enumerate(sorted(common_cols)):
                ax = axes[idx]

                for run_name, data in all_data.items():
                    sys_hist = data['system_history']
                    if metric in sys_hist.columns:
                        plot_data = sys_hist[['_step', metric]].dropna()
                        if len(plot_data) > 0:
                            ax.plot(plot_data['_step'], plot_data[metric],
                                   label=run_name.replace('gpt2_', ''),
                                   linewidth=2, alpha=0.8)

                ax.set_xlabel('Training Step')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric}')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(n_metrics, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig('wandb_gpu_comparison.png', dpi=150, bbox_inches='tight')
            print("\nSaved visualization to: wandb_gpu_comparison.png")

    # Save data to CSV for further analysis
    for run_name, data in all_data.items():
        # Save custom metrics
        csv_file = f"wandb_{run_name}_metrics.csv"
        data['history'].to_csv(csv_file, index=False)
        print(f"\nSaved metrics to: {csv_file}")

        # Save system metrics (GPU, etc.)
        system_csv_file = f"wandb_{run_name}_system.csv"
        data['system_history'].to_csv(system_csv_file, index=False)
        print(f"Saved system metrics to: {system_csv_file}")

if __name__ == "__main__":
    main()
