#!/usr/bin/env python3
"""
Backfill final metrics to existing W&B runs for visualization.

W&B summary values don't appear in charts - only logged history points do.
This script reads final/* and test/* summary values from completed runs
and re-logs them as history points so they appear in charts.

Usage:
    python scripts/wandb_backfill_final_metrics.py --project neighborloader-vs-pageaware
    python scripts/wandb_backfill_final_metrics.py --project neighborloader-vs-pageaware --dry-run
"""

import argparse
import sys

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)


def backfill_run(run, dry_run: bool = False) -> bool:
    """Backfill final metrics for a single run.

    Returns True if metrics were backfilled, False if skipped.
    """
    # Check if run has final metrics in summary
    summary = run.summary
    final_keys = [k for k in summary.keys() if k.startswith(("final/", "test/"))]

    if not final_keys:
        print(f"  Skip {run.name}: no final/* or test/* metrics in summary")
        return False

    # Check if already backfilled (has history point for test/f1)
    history = run.history(keys=["test/f1"], pandas=False)
    if history and len(history) > 0:
        # Check if test/f1 was ever logged (not just present in summary)
        has_test_f1_logged = any(row.get("test/f1") is not None for row in history)
        if has_test_f1_logged:
            print(f"  Skip {run.name}: already has test/f1 in history")
            return False

    # Prepare metrics to backfill
    metrics = {}
    for key in final_keys:
        value = summary.get(key)
        if value is not None:
            metrics[key] = value

    if not metrics:
        print(f"  Skip {run.name}: no valid metric values")
        return False

    print(f"  Backfill {run.name}: {len(metrics)} metrics")
    for key, value in sorted(metrics.items()):
        print(
            f"    {key}: {value:.4f}"
            if isinstance(value, float)
            else f"    {key}: {value}"
        )

    if dry_run:
        print("    (dry-run, not writing)")
        return True

    # Resume the run and log metrics
    try:
        resumed = wandb.init(
            project=run.project,
            entity=run.entity,
            id=run.id,
            resume="allow",
        )

        # Log metrics as final history point
        wandb.log(metrics)

        wandb.finish()
        print(f"    Done!")
        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Backfill final metrics to W&B runs for visualization"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="neighborloader-vs-pageaware",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username or team). Uses default if not specified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Only process specific run ID",
    )
    args = parser.parse_args()

    # Initialize W&B API
    api = wandb.Api()

    # Build project path
    if args.entity:
        project_path = f"{args.entity}/{args.project}"
    else:
        project_path = args.project

    print(f"Backfilling final metrics for project: {project_path}")
    if args.dry_run:
        print("(DRY RUN - no changes will be made)")
    print()

    # Get runs
    try:
        if args.run_id:
            runs = [api.run(f"{project_path}/{args.run_id}")]
        else:
            runs = api.runs(project_path)
    except Exception as e:
        print(f"ERROR: Could not access project: {e}")
        sys.exit(1)

    # Process each run
    backfilled = 0
    skipped = 0

    for run in runs:
        if run.state != "finished":
            print(f"  Skip {run.name}: state={run.state} (not finished)")
            skipped += 1
            continue

        if backfill_run(run, args.dry_run):
            backfilled += 1
        else:
            skipped += 1

    print()
    print(f"Summary: {backfilled} runs backfilled, {skipped} runs skipped")

    if args.dry_run and backfilled > 0:
        print()
        print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
