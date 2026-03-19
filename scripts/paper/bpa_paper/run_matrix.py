#!/usr/bin/env python3
"""Plan or orchestrate a BPA paper matrix run.

This script does not fake measurement. It records the intended matrix and rerun
filters so real GPU execution can be wired in cleanly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_dirs, iter_points, load_config, now_utc, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan BPA paper matrix run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--rerun-status", default="")
    parser.add_argument("--only-batches", default="")
    parser.add_argument("--only-contexts", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    results_root = Path(args.results_root)
    ensure_dirs(results_root)

    only_batches = {int(x) for x in args.only_batches.split(",") if x}
    only_contexts = {int(x) for x in args.only_contexts.split(",") if x}
    points = []
    for point in iter_points(config):
        if only_batches and point.batch not in only_batches:
            continue
        if only_contexts and point.context not in only_contexts:
            continue
        points.append(
            {"batch": point.batch, "context": point.context, "status": "pending"}
        )

    payload = {
        "timestamp": now_utc(),
        "mode": "dry-run" if args.dry_run else "plan-only",
        "gpu": config.gpu,
        "lane": args.lane or config.lane,
        "rerun_status": [x for x in args.rerun_status.split(",") if x],
        "points": points,
        "notes": config.notes,
    }
    out = results_root / "reports" / f"matrix_{config.gpu}_{config.lane}.json"
    write_json(out, payload)
    print(f"wrote matrix plan to {out}")
    print(f"planned {len(points)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
