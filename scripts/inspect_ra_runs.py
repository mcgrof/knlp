#!/usr/bin/env python3
"""Inspect W&B run to see what metrics are available."""

import wandb
import sys

project = "mcgrof-citizen/gpt2-ra-ablation-w7900"
run_name = "gpt2_adamwspam_ramla_stepB0"

api = wandb.Api()
runs = api.runs(project, filters={"display_name": run_name})

if not runs:
    print(f"Run not found: {run_name}")
    sys.exit(1)

run = runs[0]

print("=" * 80)
print(f"Run: {run_name}")
print(f"State: {run.state}")
print("=" * 80)

print("\nSummary keys:")
for key in sorted(run.summary.keys()):
    value = run.summary[key]
    print(f"  {key}: {value}")

print("\nConfig keys:")
for key in sorted(run.config.keys()):
    value = run.config[key]
    print(f"  {key}: {value}")
