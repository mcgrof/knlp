#!/usr/bin/env python3
"""Debug: Show ALL columns from W&B run."""

import wandb

entity = "mcgrof-citizen"
project = "gpt2-bitter7-b200x4"
run_id = "eptbmgdt"

api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")

print(f"Run: {run.name} (state: {run.state})")

# Get history with pandas
df = run.history(pandas=True)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nALL COLUMNS:")
print("="*80)
for i, col in enumerate(sorted(df.columns)):
    print(f"{i+1:3}. {col}")

# Check for any system-related columns
system_cols = [c for c in df.columns if 'system' in c.lower() or 'gpu' in c.lower()]
print(f"\n\nSystem/GPU columns: {len(system_cols)}")
if system_cols:
    for col in system_cols:
        print(f"  - {col}")
else:
    print("  (none found)")

# Show a sample row
print("\n\nFirst row sample:")
print("="*80)
for col in df.columns[:20]:
    print(f"{col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
