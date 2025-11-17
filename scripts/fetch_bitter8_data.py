#!/usr/bin/env python3
"""Fetch bitter8 B200 run data from W&B for documentation graphs."""

import wandb
import json
import pandas as pd

# Initialize W&B API
api = wandb.Api()

# Fetch runs from the project
project = "mcgrof-citizen/gpt2-bitter8-nocompile-b200x4"
runs = api.runs(project)

print(f"Found {len(runs)} runs in {project}")
print()

# Collect data for all runs
run_data = {}

for run in runs:
    print(f"Run: {run.name}")
    print(f"  ID: {run.id}")
    print(f"  State: {run.state}")
    print(f"  Config: {run.config}")

    # Get history (metrics over time)
    history = run.history(samples=10000)

    if len(history) > 0:
        print(f"  Metrics collected: {len(history)} samples")
        print(f"  Columns: {list(history.columns)}")

        # Save relevant columns
        relevant_cols = [col for col in history.columns if 'val' in col.lower() or 'loss' in col.lower() or 'perplexity' in col.lower() or 'memory' in col.lower() or 'gpu' in col.lower() or 'iter' in col.lower() or 'step' in col.lower()]

        if relevant_cols:
            print(f"  Relevant columns: {relevant_cols}")

            run_data[run.name] = {
                'id': run.id,
                'config': dict(run.config),
                'state': run.state,
                'history': history[relevant_cols].to_dict('records')
            }

    print()

# Save to JSON
output_file = 'bitter8_b200_wandb_data.json'
with open(output_file, 'w') as f:
    json.dump(run_data, f, indent=2, default=str)

print(f"Saved data to {output_file}")

# Print summary statistics
for run_name, data in run_data.items():
    print(f"\n{run_name}:")
    if data['history']:
        df = pd.DataFrame(data['history'])
        print(df.describe())
