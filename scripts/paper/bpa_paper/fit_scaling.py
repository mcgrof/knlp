#!/usr/bin/env python3
"""Create derived fit placeholders from manifest-backed experiment results.

This script is intentionally conservative. It validates the presence of the
results tree and records which fit products are expected; wiring to the real
numerical fitting routines should happen on top of this interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import write_json
from manifest import load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan BPA scaling fits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(error)
        return 1

    payload = {
        "mode": "dry-run" if args.dry_run else "plan-only",
        "derived_products": [
            "hill_fit_T4096.json",
            "hill_fit_T16384.json",
            "context_linearity.json",
            "bootstrap_ci.json",
            "residual_summary.json",
            "cross_gpu/comparison_table.json",
        ],
        "note": "Numerical fitting is not fabricated here; this is the expected output contract.",
    }
    out = Path(args.results_root) / "reports" / "fit_plan.json"
    write_json(out, payload)
    print(f"wrote fit plan to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
