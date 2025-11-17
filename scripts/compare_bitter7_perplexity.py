#!/usr/bin/env python3
"""
Compare validation perplexity across bitter7 optimization runs.

Analyzes:
1. Baseline (magnitude pruning)
2. Old bitter7 (unoptimized)
3. New bitter7 (optimized)
"""
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize W&B API
api = wandb.Api()

# Project path
project = "mcgrof-citizen/gpt2-bitter7-optimized-b200x4"

print(f"Fetching runs from project: {project}")
print("=" * 80)

# Get all runs
runs = api.runs(project)

print(f"\nFound {len(runs)} run(s) in project\n")

# Fetch validation metrics for each run
run_data = {}

for run in runs:
    run_name = run.name
    run_state = run.state
    print(f"Processing: {run_name} ({run_state})")

    try:
        # Get training history (validation metrics)
        history = run.history(samples=10000)

        if history.empty:
            print(f"  WARNING: No training history found")
            continue

        # Debug: print all available columns
        all_cols = list(history.columns)
        print(f"  Available columns ({len(all_cols)}): {', '.join(all_cols[:10])}...")

        # Look for validation loss/perplexity
        val_cols = [
            c
            for c in history.columns
            if "val/" in c or "eval/" in c or "loss" in c.lower()
        ]

        if not val_cols:
            print(f"  WARNING: No validation metrics found")
            # Check summary for final metrics
            summary = run.summary._json_dict
            if summary:
                print(f"  Summary keys: {list(summary.keys())[:10]}")
            continue

        print(f"  Found {len(val_cols)} validation metric columns: {val_cols}")

        # Extract validation loss and compute perplexity
        stats = {}

        # Try both naming conventions
        val_loss_col = None
        if "val/loss" in history.columns:
            val_loss_col = "val/loss"
        elif "val_loss" in history.columns:
            val_loss_col = "val_loss"

        if val_loss_col:
            val_loss = history[val_loss_col].dropna()
            if len(val_loss) > 0:
                # Perplexity = exp(loss)
                perplexities = np.exp(val_loss)
                stats["final_val_loss"] = val_loss.iloc[-1]
                stats["final_perplexity"] = perplexities.iloc[-1]
                stats["best_val_loss"] = val_loss.min()
                stats["best_perplexity"] = np.exp(val_loss.min())
                stats["num_val_points"] = len(val_loss)

                print(f"  Final val loss: {stats['final_val_loss']:.4f}")
                print(f"  Final perplexity: {stats['final_perplexity']:.2f}")
                print(f"  Best perplexity: {stats['best_perplexity']:.2f}")
            else:
                print(f"  WARNING: val_loss column empty")
        else:
            print(f"  WARNING: No val_loss column found")

        # Get iteration count
        iter_col = None
        if "train/iter" in history.columns:
            iter_col = "train/iter"
        elif "iteration" in history.columns:
            iter_col = "iteration"

        if iter_col:
            iters = history[iter_col].dropna()
            if len(iters) > 0:
                stats["final_iteration"] = int(iters.iloc[-1])
                print(f"  Iterations: {stats['final_iteration']}")

        # Store full history for plotting
        run_data[run_name] = {"stats": stats, "history": history, "state": run_state}

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    print()

print("=" * 80)
print("PERPLEXITY COMPARISON")
print("=" * 80)

if len(run_data) >= 2:
    # Find baseline and optimized runs
    baseline_key = None
    old_key = None
    new_key = None

    for key in run_data.keys():
        key_lower = key.lower()
        if "magnitude" in key_lower or "baseline" in key_lower:
            baseline_key = key
        elif "optimized" in key_lower or "new" in key_lower:
            new_key = key
        else:
            old_key = key

    print("\nDetected runs:")
    print(f"  Baseline: {baseline_key or 'Not found'}")
    print(f"  Old Bitter7: {old_key or 'Not found'}")
    print(f"  New Bitter7: {new_key or 'Not found'}")

    if baseline_key:
        baseline_stats = run_data[baseline_key]["stats"]

        print("\n" + "-" * 80)
        print("Final Validation Perplexity")
        print("-" * 80)

        baseline_ppl = baseline_stats.get("final_perplexity", 0)
        print(f"Baseline (magnitude):   {baseline_ppl:.2f}")

        if old_key:
            old_stats = run_data[old_key]["stats"]
            old_ppl = old_stats.get("final_perplexity", 0)
            old_diff = old_ppl - baseline_ppl
            old_pct = (old_diff / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            print(
                f"Old Bitter7:            {old_ppl:.2f} ({old_diff:+.2f}, {old_pct:+.1f}%)"
            )

        if new_key:
            new_stats = run_data[new_key]["stats"]
            new_ppl = new_stats.get("final_perplexity", 0)
            new_diff = new_ppl - baseline_ppl
            new_pct = (new_diff / baseline_ppl) * 100 if baseline_ppl > 0 else 0
            print(
                f"New Bitter7 (optimized): {new_ppl:.2f} ({new_diff:+.2f}, {new_pct:+.1f}%)"
            )

            # Check if running
            if run_data[new_key]["state"] == "running":
                new_iter = new_stats.get("final_iteration", 0)
                print(f"  NOTE: Run still in progress (iter {new_iter})")

        print("\n" + "-" * 80)
        print("Best Validation Perplexity (across all checkpoints)")
        print("-" * 80)

        baseline_best = baseline_stats.get("best_perplexity", 0)
        print(f"Baseline (magnitude):   {baseline_best:.2f}")

        if old_key:
            old_best = run_data[old_key]["stats"].get("best_perplexity", 0)
            old_best_diff = old_best - baseline_best
            old_best_pct = (
                (old_best_diff / baseline_best) * 100 if baseline_best > 0 else 0
            )
            print(
                f"Old Bitter7:            {old_best:.2f} ({old_best_diff:+.2f}, {old_best_pct:+.1f}%)"
            )

        if new_key:
            new_best = run_data[new_key]["stats"].get("best_perplexity", 0)
            new_best_diff = new_best - baseline_best
            new_best_pct = (
                (new_best_diff / baseline_best) * 100 if baseline_best > 0 else 0
            )
            print(
                f"New Bitter7 (optimized): {new_best:.2f} ({new_best_diff:+.2f}, {new_best_pct:+.1f}%)"
            )

        # Create plot if we have at least 2 runs with data
        plot_runs = {}
        val_loss_cols = {}
        iter_cols = {}

        for key, data in run_data.items():
            hist = data["history"]
            # Find val_loss column
            val_loss_col = (
                "val/loss"
                if "val/loss" in hist.columns
                else "val_loss" if "val_loss" in hist.columns else None
            )
            # Find iteration column
            iter_col = (
                "train/iter"
                if "train/iter" in hist.columns
                else "iteration" if "iteration" in hist.columns else None
            )

            if val_loss_col and iter_col:
                if key == baseline_key:
                    plot_runs["Baseline (magnitude)"] = hist
                    val_loss_cols["Baseline (magnitude)"] = val_loss_col
                    iter_cols["Baseline (magnitude)"] = iter_col
                elif key == old_key:
                    plot_runs["Old Bitter7"] = hist
                    val_loss_cols["Old Bitter7"] = val_loss_col
                    iter_cols["Old Bitter7"] = iter_col
                elif key == new_key:
                    plot_runs["New Bitter7 (optimized)"] = hist
                    val_loss_cols["New Bitter7 (optimized)"] = val_loss_col
                    iter_cols["New Bitter7 (optimized)"] = iter_col

        if len(plot_runs) >= 2:
            print("\n" + "=" * 80)
            print("Creating comparison plot...")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            colors = {
                "Baseline (magnitude)": "#3498db",
                "Old Bitter7": "#e74c3c",
                "New Bitter7 (optimized)": "#27ae60",
            }

            # Plot 1: Validation Loss
            for name, hist in plot_runs.items():
                val_col = val_loss_cols[name]
                iter_col = iter_cols[name]
                val_loss = hist[val_col].dropna()
                iterations = hist[iter_col].dropna()[: len(val_loss)]
                ax1.plot(
                    iterations,
                    val_loss,
                    label=name,
                    color=colors.get(name, "gray"),
                    linewidth=2,
                    alpha=0.8,
                )

            ax1.set_xlabel("Iteration", fontsize=11)
            ax1.set_ylabel("Validation Loss", fontsize=11)
            ax1.set_title(
                "Validation Loss Over Training", fontsize=12, fontweight="bold"
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Perplexity
            for name, hist in plot_runs.items():
                val_col = val_loss_cols[name]
                iter_col = iter_cols[name]
                val_loss = hist[val_col].dropna()
                perplexity = np.exp(val_loss)
                iterations = hist[iter_col].dropna()[: len(val_loss)]
                ax2.plot(
                    iterations,
                    perplexity,
                    label=name,
                    color=colors.get(name, "gray"),
                    linewidth=2,
                    alpha=0.8,
                )

            ax2.set_xlabel("Iteration", fontsize=11)
            ax2.set_ylabel("Perplexity", fontsize=11)
            ax2.set_title(
                "Validation Perplexity Over Training", fontsize=12, fontweight="bold"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "bitter7_perplexity_comparison.png", dpi=150, bbox_inches="tight"
            )
            print(f"Saved plot to: bitter7_perplexity_comparison.png")

        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)

        # Check if perplexity is comparable
        if new_key and baseline_key:
            new_final = run_data[new_key]["stats"].get("final_perplexity", float("inf"))
            baseline_final = run_data[baseline_key]["stats"].get("final_perplexity", 0)

            ppl_diff_pct = (
                abs((new_final - baseline_final) / baseline_final) * 100
                if baseline_final > 0
                else 0
            )

            if ppl_diff_pct < 5:
                print(
                    f"✓ Perplexity comparable to baseline (within 5%): {ppl_diff_pct:.1f}%"
                )
            elif ppl_diff_pct < 10:
                print(f"~ Perplexity acceptable (within 10%): {ppl_diff_pct:.1f}%")
            else:
                print(f"✗ Perplexity worse than baseline: {ppl_diff_pct:.1f}%")

            if run_data[new_key]["state"] == "running":
                print(
                    "\n  NOTE: Optimized run still in progress - final metrics may change"
                )

else:
    print("\nNot enough runs to compare. Need at least 2 runs.")

print("\n" + "=" * 80)
