#!/usr/bin/env python3
"""List all W&B runs in a project."""

import wandb

entity = "mcgrof-citizen"
project = "gpt2-bitter7-b200x4"

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

print(f"All runs in {entity}/{project}:")
print("=" * 100)
print(f"{'Name':<50} {'State':<15} {'ID':<15} {'Created':<20}")
print("=" * 100)

for run in runs:
    created = run.created_at.split('T')[0] if hasattr(run, 'created_at') else 'N/A'
    print(f"{run.name:<50} {run.state:<15} {run.id:<15} {created:<20}")
