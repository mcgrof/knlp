#!/usr/bin/env python3
"""
Compare RA ablation results between A100 and W7900 GPUs.

Fetches data from W&B and performs apples-to-apples comparison.
"""

import wandb
import pandas as pd
import sys

# W&B projects
A100_PROJECT = "mcgrof-citizen/gpt2-ra-ablation-a100-40g"
W7900_PROJECT = "mcgrof-citizen/gpt2-ra-ablation-w7900"

# Expected run names (based on defconfig)
EXPECTED_RUNS = [
    "gpt2_adamwspam_ramla_stepB0",
    "gpt2_adamwspam_ramla_stepRAEARLY0",
    "gpt2_adamwspam_ramla_stepRALATE0",
    "gpt2_adamwspam_ramla_stepRAALL0",
    "gpt2_adamwspam_ramla_stepRALEARN0",
]


def fetch_run_data(project, run_name):
    """Fetch key metrics from a W&B run."""
    api = wandb.Api()

    try:
        runs = api.runs(project, filters={"display_name": run_name})
        if not runs:
            print(f"  ✗ Run not found: {run_name}")
            return None

        run = runs[0]

        # Get final validation metrics
        summary = run.summary

        # Get hyperparameters
        config = run.config

        data = {
            "run_name": run_name,
            "state": run.state,
            "val_loss": summary.get("val_loss"),
            "val_perplexity": summary.get("val_perplexity"),
            "best_val_loss": summary.get("final/best_val_loss"),
            "best_val_perplexity": summary.get("final/best_val_perplexity"),
            "train_loss_final": summary.get("train_loss"),
            "train_perplexity_final": summary.get("train_perplexity"),
            "iteration": summary.get("iteration"),
            "hellaswag_acc": summary.get("lm_eval/hellaswag_acc,none"),
            "hellaswag_acc_norm": summary.get("lm_eval/hellaswag_acc_norm,none"),
            "batch_size": config.get("batch_size"),
            "gradient_accumulation": config.get("gradient_accumulation"),
            "effective_batch": config.get("batch_size", 0)
            * config.get("gradient_accumulation", 0),
            "learning_rate": config.get("learning_rate"),
        }

        return data

    except Exception as e:
        print(f"  ✗ Error fetching {run_name}: {e}")
        return None


def main():
    print("=" * 80)
    print("RA Ablation Results Comparison: A100 vs W7900")
    print("=" * 80)
    print()

    # Fetch A100 data
    print("Fetching A100 results...")
    a100_data = []
    for run_name in EXPECTED_RUNS:
        data = fetch_run_data(A100_PROJECT, run_name)
        if data:
            a100_data.append(data)
            status = "✓" if data["state"] == "finished" else "✗"
            print(f"  {status} {run_name}: {data['state']}")
    print()

    # Fetch W7900 data
    print("Fetching W7900 results...")
    w7900_data = []
    for run_name in EXPECTED_RUNS:
        data = fetch_run_data(W7900_PROJECT, run_name)
        if data:
            w7900_data.append(data)
            status = "✓" if data["state"] == "finished" else "✗"
            print(f"  {status} {run_name}: {data['state']}")
    print()

    # Create DataFrames
    df_a100 = pd.DataFrame(a100_data)
    df_w7900 = pd.DataFrame(w7900_data)

    # Filter to successful runs only
    df_a100_success = df_a100[df_a100["state"] == "finished"].copy()
    df_w7900_success = df_w7900[df_w7900["state"] == "finished"].copy()

    print("=" * 80)
    print("A100 Results (Successful Runs)")
    print("=" * 80)
    print(
        f"Gradient Accumulation: {df_a100_success['gradient_accumulation'].iloc[0] if len(df_a100_success) > 0 else 'N/A'}"
    )
    print(
        f"Effective Batch Size: {df_a100_success['effective_batch'].iloc[0] if len(df_a100_success) > 0 else 'N/A'}"
    )
    print()
    print(f"{'Run':<15} {'Best Val PPL':<15} {'HellaSwag Acc':<15} {'Iterations':<12}")
    print("-" * 80)
    for _, row in df_a100_success.iterrows():
        run_short = row["run_name"].replace("gpt2_adamwspam_ramla_step", "")
        ppl = (
            f"{row['best_val_perplexity']:.2f}"
            if row["best_val_perplexity"] is not None
            else "N/A"
        )
        acc = (
            f"{row['hellaswag_acc_norm']*100:.1f}%"
            if row["hellaswag_acc_norm"] is not None
            else "N/A"
        )
        iters = f"{row['iteration']:.0f}" if row["iteration"] is not None else "N/A"
        print(f"{run_short:<15} {ppl:<15} {acc:<15} {iters:<12}")
    print()

    print("=" * 80)
    print("W7900 Results (Successful Runs)")
    print("=" * 80)
    print(
        f"Gradient Accumulation: {df_w7900_success['gradient_accumulation'].iloc[0] if len(df_w7900_success) > 0 else 'N/A'}"
    )
    print(
        f"Effective Batch Size: {df_w7900_success['effective_batch'].iloc[0] if len(df_w7900_success) > 0 else 'N/A'}"
    )
    print()
    print(f"{'Run':<15} {'Best Val PPL':<15} {'HellaSwag Acc':<15} {'Iterations':<12}")
    print("-" * 80)
    for _, row in df_w7900_success.iterrows():
        run_short = row["run_name"].replace("gpt2_adamwspam_ramla_step", "")
        ppl = (
            f"{row['best_val_perplexity']:.2f}"
            if row["best_val_perplexity"] is not None
            else "N/A"
        )
        acc = (
            f"{row['hellaswag_acc_norm']*100:.1f}%"
            if row["hellaswag_acc_norm"] is not None
            else "N/A"
        )
        iters = f"{row['iteration']:.0f}" if row["iteration"] is not None else "N/A"
        print(f"{run_short:<15} {ppl:<15} {acc:<15} {iters:<12}")
    print()

    # Compare common successful runs
    print("=" * 80)
    print("Apples-to-Apples Comparison (Common Successful Runs)")
    print("=" * 80)
    print()

    a100_runs = set(df_a100_success["run_name"])
    w7900_runs = set(df_w7900_success["run_name"])
    common_runs = sorted(a100_runs & w7900_runs)

    if not common_runs:
        print("No common successful runs found!")
        return

    print(f"Found {len(common_runs)} common runs")
    print()

    print(
        f"{'Run':<40} {'A100 Perplexity':<18} {'W7900 Perplexity':<18} {'Difference':<12}"
    )
    print("-" * 90)

    for run_name in common_runs:
        a100_row = df_a100_success[df_a100_success["run_name"] == run_name].iloc[0]
        w7900_row = df_w7900_success[df_w7900_success["run_name"] == run_name].iloc[0]

        run_short = run_name.replace("gpt2_adamwspam_ramla_step", "")
        a100_ppl = a100_row["best_val_perplexity"]
        w7900_ppl = w7900_row["best_val_perplexity"]

        if a100_ppl is not None and w7900_ppl is not None:
            diff = w7900_ppl - a100_ppl
            print(f"{run_short:<40} {a100_ppl:<18.2f} {w7900_ppl:<18.2f} {diff:+12.2f}")
        else:
            a100_str = f"{a100_ppl:.2f}" if a100_ppl is not None else "N/A"
            w7900_str = f"{w7900_ppl:.2f}" if w7900_ppl is not None else "N/A"
            print(f"{run_short:<40} {a100_str:<18} {w7900_str:<18} {'N/A':<12}")

    print()
    print("=" * 80)
    print("Key Observations")
    print("=" * 80)

    # Find baseline and RA runs
    baseline_a100 = df_a100_success[df_a100_success["run_name"].str.contains("stepB0")]
    baseline_w7900 = df_w7900_success[
        df_w7900_success["run_name"].str.contains("stepB0")
    ]

    ra_a100 = df_a100_success[~df_a100_success["run_name"].str.contains("stepB0")]
    ra_w7900 = df_w7900_success[~df_w7900_success["run_name"].str.contains("stepB0")]

    if len(baseline_a100) > 0 and len(ra_a100) > 0:
        baseline_ppl_a100 = baseline_a100["best_val_perplexity"].iloc[0]
        print(
            f"\nA100 (grad_acc={df_a100_success['gradient_accumulation'].iloc[0]}, eff_batch=512):"
        )
        print(f"  Baseline (B0): {baseline_ppl_a100:.2f}")
        print(f"  RA variants:")
        for _, row in ra_a100.iterrows():
            run_short = row["run_name"].replace("gpt2_adamwspam_ramla_step", "")
            delta = row["best_val_perplexity"] - baseline_ppl_a100
            better = "✓ BETTER" if delta < 0 else "✗ WORSE"
            print(
                f"    {better} {run_short}: {row['best_val_perplexity']:.2f} ({delta:+.2f} vs baseline)"
            )

    if len(baseline_w7900) > 0 and len(ra_w7900) > 0:
        baseline_ppl_w7900 = baseline_w7900["best_val_perplexity"].iloc[0]
        print(
            f"\nW7900 (grad_acc={df_w7900_success['gradient_accumulation'].iloc[0]}, eff_batch=256):"
        )
        print(f"  Baseline (B0): {baseline_ppl_w7900:.2f}")
        print(f"  RA variants:")
        for _, row in ra_w7900.iterrows():
            run_short = row["run_name"].replace("gpt2_adamwspam_ramla_step", "")
            delta = row["best_val_perplexity"] - baseline_ppl_w7900
            better = "✓ BETTER" if delta < 0 else "✗ WORSE"
            print(
                f"    {better} {run_short}: {row['best_val_perplexity']:.2f} ({delta:+.2f} vs baseline)"
            )

    print()


if __name__ == "__main__":
    main()
