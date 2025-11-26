#!/usr/bin/env python3
"""
Get full training history to see if models actually trained.
"""

import wandb
import matplotlib.pyplot as plt

def get_training_history():
    """Fetch training history for each run."""
    api = wandb.Api()
    project = "mcgrof-citizen/gpt2-latent-architecture-ablation-a100-40g"

    runs = api.runs(project)

    for run in runs:
        print("=" * 80)
        print(f"Run: {run.name}")
        print(f"State: {run.state}")
        print()

        # Get full training history
        history = run.history(samples=10000, keys=['train_loss', 'val_loss', 'iter'])

        if history.empty:
            print("No history data available")
            continue

        print(f"Total samples: {len(history)}")
        print()

        # Show first few
        print("First 5 samples:")
        print(history.head(5).to_string(index=False))
        print()

        # Show last few
        print("Last 5 samples:")
        print(history.tail(5).to_string(index=False))
        print()

        # Show min losses
        if 'train_loss' in history.columns:
            min_train = history['train_loss'].min()
            print(f"Min train loss: {min_train:.4f}")
        if 'val_loss' in history.columns:
            min_val = history['val_loss'].min()
            print(f"Min val loss: {min_val:.4f}")
        print()

if __name__ == '__main__':
    get_training_history()
