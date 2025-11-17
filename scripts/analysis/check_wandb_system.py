#!/usr/bin/env python3
"""Check W&B run for system metrics in different locations."""

import wandb

entity = "mcgrof-citizen"
project = "gpt2-bitter7-b200x4"
run_id = "eptbmgdt"  # Current bitter7 run

api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")

print(f"Run: {run.name} (id: {run.id}, state: {run.state})")

# Check summary
print("\n" + "="*80)
print("Run Summary:")
print("="*80)
for key in sorted(run.summary.keys()):
    if 'system' in key.lower() or 'gpu' in key.lower() or 'memory' in key.lower():
        print(f"  {key}: {run.summary[key]}")

# Check config
print("\n" + "="*80)
print("System-related config:")
print("="*80)
for key in sorted(run.config.keys()):
    if 'system' in key.lower() or 'gpu' in key.lower() or 'device' in key.lower():
        print(f"  {key}: {run.config[key]}")

# Try to access system metrics directly
print("\n" + "="*80)
print("Attempting to fetch system metrics...")
print("="*80)

try:
    # Try fetching with stream='system'
    system_metrics = list(run.scan_history(keys=None, page_size=100, min_step=0, max_step=100))
    if system_metrics:
        first_row = system_metrics[0]
        print(f"First system row has {len(first_row)} keys")
        for k, v in list(first_row.items())[:10]:
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")

# Check files
print("\n" + "="*80)
print("Run files:")
print("="*80)
for file in run.files():
    print(f"  - {file.name}")
