#!/usr/bin/env python3
"""Run cheap smoke tests before expensive BPA paper sweeps.

This script is orchestration scaffolding. It prints and records the smoke plan;
it does not pretend to execute kernels inside this repository by itself.
"""

from __future__ import annotations

from pathlib import Path

from common import (
    ensure_dirs,
    iter_points,
    load_config,
    now_utc,
    parse_args,
    write_json,
)


def main() -> int:
    parser = parse_args("Run BPA paper smoke tests")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    results_root = Path(args.results_root)
    ensure_dirs(results_root)
    smoke_points = []
    for point in iter_points(config):
        if point.batch in {1, 8} and point.context in {1024, 4096}:
            smoke_points.append({"batch": point.batch, "context": point.context})
    payload = {
        "timestamp": now_utc(),
        "mode": "dry-run" if args.dry_run else "plan-only",
        "gpu": config.gpu,
        "lane": config.lane,
        "smoke_stages": [
            "environment_capture",
            "kernel_sanity",
            "mini_matrix",
        ],
        "points": smoke_points,
    }
    out = results_root / "reports" / f"smoke_{config.gpu}_{config.lane}.json"
    write_json(out, payload)
    print(f"wrote smoke plan to {out}")
    for point in smoke_points:
        print(f"SMOKE {config.gpu} {config.lane} B={point[batch]} T={point[context]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
