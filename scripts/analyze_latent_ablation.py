#!/usr/bin/env python3
"""
Analyze latent architecture ablation results from W&B.
"""

import wandb
import pandas as pd

def fetch_ablation_results():
    """Fetch all runs from the latent architecture ablation project."""
    api = wandb.Api()
    project = "mcgrof-citizen/gpt2-latent-architecture-ablation-a100-40g"

    runs = api.runs(project)

    results = []
    for run in runs:
        # Get summary metrics
        summary = run.summary._json_dict
        config = run.config

        result = {
            'run_name': run.name,
            'state': run.state,
            'val_loss': summary.get('val_loss'),
            'train_loss': summary.get('train_loss'),
            'perplexity': summary.get('perplexity'),
            # Get architecture info from config
            'architecture': config.get('architecture', 'unknown'),
            'learning_rate': config.get('learning_rate'),
            'optimizer': config.get('optimizer'),
            'created_at': run.created_at,
        }

        # Calculate perplexity if not stored
        if result['perplexity'] is None and result['val_loss'] is not None:
            import math
            result['perplexity'] = math.exp(result['val_loss'])

        results.append(result)

    return pd.DataFrame(results)

def analyze_results(df):
    """Analyze and display results."""
    print("=" * 80)
    print("Latent Architecture Ablation Study Results")
    print("=" * 80)
    print()

    # Filter to completed runs
    completed = df[df['state'] == 'finished'].copy()

    if completed.empty:
        print("No completed runs found.")
        print("\nAll runs:")
        print(df[['run_name', 'state', 'val_loss', 'perplexity']].to_string(index=False))
        return

    # Sort by validation loss (lower is better)
    completed = completed.sort_values('val_loss')

    print(f"Completed runs: {len(completed)}")
    print()

    # Display results table
    display_cols = ['run_name', 'architecture', 'val_loss', 'perplexity', 'learning_rate']
    print(completed[display_cols].to_string(index=False))
    print()

    # Find best result
    best = completed.iloc[0]
    print("=" * 80)
    print("Best Result:")
    print(f"  Run: {best['run_name']}")
    print(f"  Architecture: {best['architecture']}")
    print(f"  Validation Loss: {best['val_loss']:.4f}")
    print(f"  Perplexity: {best['perplexity']:.2f}")
    print(f"  Learning Rate: {best['learning_rate']}")
    print(f"  Optimizer: {best['optimizer']}")
    print()

    # Compare architectures
    if 'architecture' in completed.columns:
        print("=" * 80)
        print("Architecture Comparison:")
        print()

        arch_stats = completed.groupby('architecture').agg({
            'val_loss': ['mean', 'min', 'count'],
            'perplexity': ['mean', 'min']
        }).round(4)

        print(arch_stats)
        print()

    # Show incomplete runs if any
    incomplete = df[df['state'] != 'finished']
    if not incomplete.empty:
        print("=" * 80)
        print(f"Incomplete runs: {len(incomplete)}")
        print(incomplete[['run_name', 'state']].to_string(index=False))

if __name__ == '__main__':
    print("Fetching results from W&B...")
    df = fetch_ablation_results()
    print(f"Found {len(df)} total runs\n")
    analyze_results(df)
