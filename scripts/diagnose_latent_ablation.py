#!/usr/bin/env python3
"""
Diagnose why latent architecture ablation runs have such high perplexity.
"""

import wandb

def diagnose_runs():
    """Fetch detailed info about each run to diagnose the issue."""
    api = wandb.Api()
    project = "mcgrof-citizen/gpt2-latent-architecture-ablation-a100-40g"

    runs = api.runs(project)

    for run in runs:
        print("=" * 80)
        print(f"Run: {run.name}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")
        print(f"Duration: {run.summary.get('_runtime', 'unknown')} seconds")
        print()

        # Get config
        config = run.config
        print("Configuration:")
        for key in ['architecture', 'learning_rate', 'optimizer', 'max_iters',
                    'max_time', 'batch_size', 'gradient_accumulation']:
            if key in config:
                print(f"  {key}: {config[key]}")
        print()

        # Get summary metrics
        summary = run.summary._json_dict
        print("Summary Metrics:")
        for key in ['iter', 'train_loss', 'val_loss', 'perplexity',
                    'tokens_seen', 'best_val_loss']:
            if key in summary:
                print(f"  {key}: {summary[key]}")
        print()

        # Try to get training history sample
        try:
            history = run.history(samples=10, keys=['train_loss', 'val_loss', 'iter'])
            if not history.empty:
                print("Training History (first/last few samples):")
                print(history.to_string(index=False))
                print()
        except Exception as e:
            print(f"Could not fetch history: {e}")
            print()

if __name__ == '__main__':
    diagnose_runs()
