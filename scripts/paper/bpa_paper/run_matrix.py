#!/usr/bin/env python3
"""Plan or orchestrate a BPA paper matrix run.

This script now supports two honest modes:
- plan-only: write the matrix plan without executing any collection command
- subprocess runner: execute a configured per-point command template, capture
  stdout/stderr, and update manifest point status

It still does not fabricate measurements. If the configured runner is not
point-capable yet, the config should remain in plan-only mode.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

from common import ensure_dirs, iter_points, load_config, now_utc, write_json
from manifest import empty_manifest, load_manifest, save_manifest


def _parse_csv_ints(raw: str) -> set[int]:
    return {int(x) for x in raw.split(",") if x}


def _point_id(batch: int, context: int) -> str:
    return f"B{batch}_T{context}"


def _manifest_path(results_root: Path) -> Path:
    return results_root / "manifests" / "run_manifest.json"


def _load_or_init_manifest(path: Path) -> Dict:
    if path.exists():
        return load_manifest(path)
    manifest = empty_manifest()
    save_manifest(path, manifest)
    return manifest


def _upsert_point(manifest: Dict, payload: Dict) -> None:
    for idx, point in enumerate(manifest["points"]):
        if (
            point.get("gpu") == payload.get("gpu")
            and point.get("lane") == payload.get("lane")
            and point.get("batch") == payload.get("batch")
            and point.get("context") == payload.get("context")
        ):
            manifest["points"][idx] = payload
            return
    manifest["points"].append(payload)


def _build_points(
    config, only_batches: set[int], only_contexts: set[int]
) -> List[Dict]:
    points: List[Dict] = []
    for point in iter_points(config):
        if only_batches and point.batch not in only_batches:
            continue
        if only_contexts and point.context not in only_contexts:
            continue
        points.append(
            {"batch": point.batch, "context": point.context, "status": "pending"}
        )
    return points


def _render_command(template: str, substitutions: Dict[str, str]) -> str:
    return template.format(**substitutions)


def _run_point(
    config, results_root: Path, lane: str, batch: int, context: int, dry_run: bool
) -> Dict:
    point_name = _point_id(batch, context)
    point_dir = results_root / "raw" / config.gpu / lane / point_name
    point_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = point_dir / "stdout.log"
    stderr_path = point_dir / "stderr.log"

    runner = config.runner
    if runner is None or runner.kind == "plan-only" or not runner.command_template:
        return {
            "gpu": config.gpu,
            "lane": lane,
            "batch": batch,
            "context": context,
            "status": "skipped",
            "reason": "no point-capable runner configured",
            "point_dir": str(point_dir),
        }

    substitutions = {
        "gpu": config.gpu,
        "lane": lane,
        "model": config.model,
        "batch": str(batch),
        "context": str(context),
        "point_dir": str(point_dir),
        "results_root": str(results_root),
        "warmup_iters": str(config.warmup_iters),
        "measure_iters": str(config.measure_iters),
    }
    env = os.environ.copy()
    for key, value in (runner.env or {}).items():
        env[key] = value.format(**substitutions)

    if dry_run:
        return {
            "gpu": config.gpu,
            "lane": lane,
            "batch": batch,
            "context": context,
            "status": "pending",
            "command": _render_command(runner.command_template, substitutions),
            "point_dir": str(point_dir),
        }

    if runner.setup_command:
        setup_cmd = _render_command(runner.setup_command, substitutions)
        setup_proc = subprocess.run(
            setup_cmd,
            shell=True,
            env=env,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(setup_proc.stdout or "")
        stderr_path.write_text(setup_proc.stderr or "")
        if setup_proc.returncode != 0:
            return {
                "gpu": config.gpu,
                "lane": lane,
                "batch": batch,
                "context": context,
                "status": "failed",
                "phase": "setup",
                "returncode": setup_proc.returncode,
                "command": setup_cmd,
                "point_dir": str(point_dir),
            }

    cmd = _render_command(runner.command_template, substitutions)
    proc = subprocess.run(
        cmd, shell=True, env=env, cwd=Path.cwd(), capture_output=True, text=True
    )
    with stdout_path.open("a", encoding="utf-8") as f:
        if proc.stdout:
            f.write(proc.stdout)
    with stderr_path.open("a", encoding="utf-8") as f:
        if proc.stderr:
            f.write(proc.stderr)

    return {
        "gpu": config.gpu,
        "lane": lane,
        "batch": batch,
        "context": context,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": cmd,
        "point_dir": str(point_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan BPA paper matrix run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--rerun-status", default="")
    parser.add_argument("--only-batches", default="")
    parser.add_argument("--only-contexts", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--execute", action="store_true", help="run configured point commands"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    results_root = Path(args.results_root)
    ensure_dirs(results_root)
    lane = args.lane or config.lane

    only_batches = _parse_csv_ints(args.only_batches)
    only_contexts = _parse_csv_ints(args.only_contexts)
    points = _build_points(config, only_batches, only_contexts)
    if args.limit > 0:
        points = points[: args.limit]

    manifest_path = _manifest_path(results_root)
    manifest = _load_or_init_manifest(manifest_path)
    run_record = {
        "timestamp": now_utc(),
        "gpu": config.gpu,
        "lane": lane,
        "mode": "execute" if args.execute and not args.dry_run else "plan-only",
        "config": str(Path(args.config)),
        "rerun_status": [x for x in args.rerun_status.split(",") if x],
        "limit": args.limit,
    }
    manifest["runs"].append(run_record)

    if not args.execute:
        payload = {
            "timestamp": run_record["timestamp"],
            "mode": "dry-run" if args.dry_run else "plan-only",
            "gpu": config.gpu,
            "lane": lane,
            "rerun_status": run_record["rerun_status"],
            "points": points,
            "notes": config.notes,
            "runner_kind": config.runner.kind if config.runner else "plan-only",
        }
        out = results_root / "reports" / f"matrix_{config.gpu}_{lane}.json"
        write_json(out, payload)
        save_manifest(manifest_path, manifest)
        print(f"wrote matrix plan to {out}")
        print(f"planned {len(points)} points")
        return 0

    executed = 0
    for point in points:
        record = _run_point(
            config=config,
            results_root=results_root,
            lane=lane,
            batch=point["batch"],
            context=point["context"],
            dry_run=args.dry_run,
        )
        _upsert_point(manifest, record)
        executed += 1

    save_manifest(manifest_path, manifest)
    summary = {
        "timestamp": now_utc(),
        "gpu": config.gpu,
        "lane": lane,
        "executed_points": executed,
        "runner_kind": config.runner.kind if config.runner else "plan-only",
        "dry_run": args.dry_run,
    }
    out = results_root / "reports" / f"matrix_execute_{config.gpu}_{lane}.json"
    write_json(out, summary)
    print(f"wrote execution summary to {out}")
    print(f"processed {executed} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
