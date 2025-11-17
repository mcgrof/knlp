#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Copy a wandb run from one project to another.

Usage:
    python3 scripts/copy_wandb_run.py \
        --source entity/project/run_id \
        --dest-project my-project \
        --dest-entity my-entity \
        --new-name new_run_name

This is useful for:
- Copying baseline runs to comparison projects
- Moving runs between organizations
- Creating run backups
- Reorganizing experiment results
"""

import argparse
import sys
import os

try:
    import wandb
except ImportError:
    print("Error: wandb not installed. Install with: pip install wandb")
    sys.exit(1)


def parse_run_path(run_path):
    """
    Parse wandb run path in format: entity/project/run_id

    Args:
        run_path: String in format "entity/project/run_id"

    Returns:
        Tuple of (entity, project, run_id)

    Raises:
        ValueError if format is invalid
    """
    parts = run_path.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid run path format: {run_path}\n"
            f"Expected: entity/project/run_id\n"
            f"Example: mcgrof/lenet5-experiments/abc123def"
        )
    return tuple(parts)


def copy_wandb_run(
    src_entity,
    src_project,
    src_run_id,
    dst_entity,
    dst_project,
    new_name=None,
    verbose=True,
):
    """
    Copy a wandb run from source project to destination project.

    Args:
        src_entity: Source wandb entity (username/org)
        src_project: Source project name
        src_run_id: Source run ID
        dst_entity: Destination wandb entity
        dst_project: Destination project name
        new_name: Optional new name for the run (defaults to source name)
        verbose: Print progress messages

    Returns:
        New run ID in destination project
    """
    # Initialize wandb API
    api = wandb.Api()

    # Get the source run
    src_path = f"{src_entity}/{src_project}/{src_run_id}"
    if verbose:
        print(f"Fetching source run: {src_path}")

    try:
        src_run = api.run(src_path)
    except Exception as e:
        print(f"Error fetching source run: {e}")
        sys.exit(1)

    # Get run metadata
    run_name = new_name if new_name else src_run.name
    config = src_run.config
    summary = src_run.summary._json_dict

    if verbose:
        print(f"Source run: {src_run.name}")
        print(f"  State: {src_run.state}")
        print(f"  Created: {src_run.created_at}")
        print(f"  Config: {len(config)} items")
        print(f"  Summary: {len(summary)} metrics")

    # Get run history (all logged metrics)
    if verbose:
        print(f"Fetching run history...")
    history = src_run.history()

    if verbose:
        print(f"  History: {len(history)} rows")

    # Get run files
    if verbose:
        print(f"Fetching run files...")
    files = src_run.files()
    file_list = list(files)

    if verbose:
        print(f"  Files: {len(file_list)} files")

    # Create new run in destination project
    if verbose:
        print(f"\nCreating new run in {dst_entity}/{dst_project}")
        print(f"  Name: {run_name}")

    new_run = wandb.init(
        project=dst_project,
        entity=dst_entity,
        config=config,
        name=run_name,
        resume="allow",
        tags=src_run.tags + ["copied-run"],
        notes=f"Copied from {src_path}",
    )

    # Log history to new run
    if verbose:
        print(f"Copying history ({len(history)} rows)...")

    for index, row in history.iterrows():
        # Filter out internal wandb columns that start with '_'
        row_dict = {k: v for k, v in row.to_dict().items() if not k.startswith("_")}
        new_run.log(row_dict)

    # Update summary metrics
    if verbose:
        print(f"Updating summary metrics...")
    for key, value in summary.items():
        if not key.startswith("_"):
            new_run.summary[key] = value

    # Copy files to new run
    if verbose and file_list:
        print(f"Copying files ({len(file_list)} files)...")

    for file in file_list:
        try:
            # Download file to temp location
            file.download(replace=True, root="/tmp/wandb_copy")
            file_path = os.path.join("/tmp/wandb_copy", file.name)

            # Upload to new run
            new_run.save(file_path, policy="now")

            if verbose:
                print(f"  Copied: {file.name}")
        except Exception as e:
            print(f"  Warning: Failed to copy {file.name}: {e}")

    # Finish the new run
    new_run.finish()

    new_run_id = new_run.id
    new_path = f"{dst_entity}/{dst_project}/{new_run_id}"

    if verbose:
        print(f"\nRun copied successfully!")
        print(f"Source: {src_path}")
        print(f"Destination: {new_path}")
        print(f"View at: https://wandb.ai/{dst_entity}/{dst_project}/runs/{new_run_id}")

    return new_run_id


def main():
    parser = argparse.ArgumentParser(
        description="Copy a wandb run from one project to another",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy run to same entity, different project
  python3 scripts/copy_wandb_run.py \\
      --source mcgrof/old-project/abc123 \\
      --dest-project new-project

  # Copy run to different entity and rename
  python3 scripts/copy_wandb_run.py \\
      --source mcgrof/experiments/abc123 \\
      --dest-project baselines \\
      --dest-entity my-org \\
      --new-name baseline_v1

  # Use with baseline feature
  make BASELINE=mcgrof/old-project/abc123 defconfig-foo
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source run in format: entity/project/run_id",
    )

    parser.add_argument(
        "--dest-project", type=str, required=True, help="Destination project name"
    )

    parser.add_argument(
        "--dest-entity",
        type=str,
        default=None,
        help="Destination entity (defaults to source entity)",
    )

    parser.add_argument(
        "--new-name",
        type=str,
        default=None,
        help="New name for the run (defaults to source name)",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Parse source run path
    try:
        src_entity, src_project, src_run_id = parse_run_path(args.source)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Default destination entity to source entity
    dst_entity = args.dest_entity if args.dest_entity else src_entity

    # Copy the run
    try:
        new_run_id = copy_wandb_run(
            src_entity=src_entity,
            src_project=src_project,
            src_run_id=src_run_id,
            dst_entity=dst_entity,
            dst_project=args.dest_project,
            new_name=args.new_name,
            verbose=not args.quiet,
        )

        # Output just the new run ID for scripting
        if args.quiet:
            print(new_run_id)

    except Exception as e:
        print(f"Error copying run: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
