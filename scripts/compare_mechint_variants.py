#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Compare mechint analysis results across multiple variants.

Usage:
    python scripts/compare_mechint_variants.py \\
        --v0 mechint_analysis_kv_tinystories_final_model_stepV0 \\
        --v1 mechint_analysis_kv_tinystories_final_model_stepV1 \\
        --output mechint_comparison \\
        --project gpt2-mechint-kv-tinystories
"""

import argparse
import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from lib.mechint import compare_variants


def main():
    parser = argparse.ArgumentParser(description="Compare mechint analysis variants")
    parser.add_argument(
        "--v0",
        type=str,
        required=True,
        help="Path to V0 (baseline) analysis directory",
    )
    parser.add_argument(
        "--v1",
        type=str,
        required=True,
        help="Path to V1 (variant) analysis directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mechint_comparison",
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="W&B project name (if not specified, reads from config)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )

    args = parser.parse_args()

    # Get W&B project name
    if args.project:
        project_name = args.project
    else:
        try:
            from config import config

            project_name = getattr(config, "TRACKER_PROJECT", "mechint-analysis")
        except ImportError:
            project_name = "mechint-analysis"

    # Create variant mapping
    variant_dirs = {
        "V0": args.v0,
        "V1": args.v1,
    }

    print("=" * 80)
    print("Mechint Variant Comparison")
    print("=" * 80)
    print(f"V0 (baseline): {args.v0}")
    print(f"V1 (variant):  {args.v1}")
    print(f"Output:        {args.output}")
    print(f"W&B project:   {project_name}")
    print("=" * 80)

    # Run comparison
    compare_variants(
        variant_dirs=variant_dirs,
        output_dir=args.output,
        project_name=project_name,
        use_wandb=not args.no_wandb,
    )

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
