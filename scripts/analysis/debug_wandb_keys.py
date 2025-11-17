#!/usr/bin/env python3
"""Debug: Print ALL keys available in W&B run."""

import wandb

entity = "mcgrof-citizen"
project = "gpt2-bitter7-b200x4"
run_id = "eptbmgdt"  # Current bitter7 run

api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")

print(f"Run: {run.name} (id: {run.id}, state: {run.state})")
print("\nScanning all history keys...")

all_keys = set()
row_count = 0

for row in run.scan_history():
    all_keys.update(row.keys())
    row_count += 1

    if row_count == 1:
        print(f"\nFirst row keys ({len(row.keys())}):")
        for key in sorted(row.keys()):
            print(f"  {key}: {row[key]}")

    if row_count >= 1000:  # Sample first 1000 rows
        break

print(f"\nScanned {row_count} rows")
print(f"\nAll unique keys found ({len(all_keys)}):")
for key in sorted(all_keys):
    print(f"  - {key}")

# Check for system keys
system_keys = [k for k in all_keys if 'system' in k.lower() or 'gpu' in k.lower()]
print(f"\nSystem/GPU keys ({len(system_keys)}):")
for key in sorted(system_keys):
    print(f"  - {key}")
