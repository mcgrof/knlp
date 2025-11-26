#!/usr/bin/env python3
"""
Compare successful W7900 runs with failed A100 runs to find config differences.
"""

import wandb

def compare_configs():
    """Compare configs between successful and failed runs."""
    api = wandb.Api()

    # Get a successful W7900 run
    w7900_project = "mcgrof-citizen/gpt2-kvsplice-ablation-w7900-mla-fixed"
    w7900_runs = api.runs(w7900_project, filters={"state": "finished"})

    # Get a failed A100 run
    a100_project = "mcgrof-citizen/gpt2-baseline-bitter7-bitter8-a100-40g"
    a100_runs = api.runs(a100_project, filters={"state": "finished"})

    if not w7900_runs or not a100_runs:
        print("Could not find runs to compare")
        return

    w7900_run = w7900_runs[0]
    a100_run = a100_runs[0]

    print("=" * 80)
    print("W7900 (Successful) vs A100 (Failed) Configuration Comparison")
    print("=" * 80)
    print()

    print(f"W7900 Run: {w7900_run.name}")
    print(f"  Val Loss: {w7900_run.summary.get('val_loss', 'N/A')}")
    print(f"  State: {w7900_run.state}")
    print()

    print(f"A100 Run: {a100_run.name}")
    print(f"  Val Loss: {a100_run.summary.get('val_loss', 'N/A')}")
    print(f"  State: {a100_run.state}")
    print()

    print("=" * 80)
    print("Configuration Differences:")
    print("=" * 80)

    w7900_config = w7900_run.config
    a100_config = a100_run.config

    # Key training parameters to compare
    keys_to_check = [
        'batch_size',
        'gradient_accumulation',
        'learning_rate',
        'weight_decay',
        'warmup_steps',
        'max_iters',
        'max_time',
        'block_size',
        'optimizer',
        'dataset',
        'compile_model',
        'mixed_precision',
        'amp_dtype',
        'flash_attention',
    ]

    print()
    print(f"{'Parameter':<25} {'W7900 (Good)':<20} {'A100 (Bad)':<20} {'Match':<10}")
    print("-" * 80)

    for key in keys_to_check:
        w7900_val = w7900_config.get(key, 'N/A')
        a100_val = a100_config.get(key, 'N/A')
        match = "✓" if w7900_val == a100_val else "✗ DIFF"

        print(f"{key:<25} {str(w7900_val):<20} {str(a100_val):<20} {match:<10}")

    print()
    print("=" * 80)
    print("All A100 Config Keys:")
    print("=" * 80)
    for key in sorted(a100_config.keys()):
        print(f"  {key}: {a100_config[key]}")

if __name__ == '__main__':
    compare_configs()
