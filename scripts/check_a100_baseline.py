#!/usr/bin/env python3
"""
Check baseline results on A100 to see if high losses are A100-specific.
"""

import wandb
import pandas as pd
import math

def check_a100_results():
    """Fetch results from A100 baseline project."""
    api = wandb.Api()
    project = "mcgrof-citizen/gpt2-baseline-bitter7-bitter8-a100-40g"

    runs = api.runs(project)

    results = []
    for run in runs:
        summary = run.summary._json_dict
        config = run.config

        result = {
            'run_name': run.name,
            'state': run.state,
            'train_loss': summary.get('train_loss'),
            'val_loss': summary.get('val_loss'),
            'architecture': config.get('architecture', 'unknown'),
            'optimizer': config.get('optimizer', 'unknown'),
            'duration': summary.get('_runtime', 0),
        }

        # Calculate perplexity
        if result['val_loss'] is not None:
            result['perplexity'] = math.exp(result['val_loss'])

        results.append(result)

    df = pd.DataFrame(results)

    print("=" * 80)
    print("A100 Baseline Results")
    print("=" * 80)
    print()

    # Filter completed runs
    completed = df[df['state'] == 'finished'].copy()

    if completed.empty:
        print("No completed runs found.")
        print("\nAll runs:")
        print(df[['run_name', 'state']].to_string(index=False))
        return

    print(f"Total completed runs: {len(completed)}")
    print()

    # Sort by val loss
    completed = completed.sort_values('val_loss')

    # Display key results
    display_cols = ['run_name', 'val_loss', 'perplexity', 'train_loss', 'duration']
    print(completed[display_cols].to_string(index=False))
    print()

    # Show if baseline GPT-2 also has high loss
    baseline_runs = completed[completed['run_name'].str.contains('baseline|gpt2', case=False, na=False)]
    if not baseline_runs.empty:
        print("=" * 80)
        print("Baseline GPT-2 Results on A100:")
        print()
        for _, row in baseline_runs.iterrows():
            print(f"  {row['run_name']}")
            print(f"    Val Loss: {row['val_loss']:.4f}")
            print(f"    Perplexity: {row['perplexity']:.2f}")
            print(f"    Train Loss: {row['train_loss']:.4f}")
            print(f"    Duration: {row['duration']:.0f}s ({row['duration']/3600:.1f}h)")
            print()

if __name__ == '__main__':
    check_a100_results()
