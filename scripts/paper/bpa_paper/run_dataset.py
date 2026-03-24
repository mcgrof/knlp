#!/usr/bin/env python3
"""Unified public runner for the BPA multi-GPU decode dataset.

This script wraps the existing paper-facing helpers so users do not need to
remember four separate entrypoints or private result-tree conventions.

Example:
  python scripts/paper/bpa_paper/run_dataset.py \
      --results-root ./results/bpa-multi-gpu \
      --gpu a100 --stage smoke

  python scripts/paper/bpa_paper/run_dataset.py \
      --results-root ./results/bpa-multi-gpu \
      --gpu all --stage full-dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_EXPORT_ROOT_NAME = "knlp-paper-memory-decode"
GPU_TO_CONFIG = {
    "a100": CONFIG_DIR / "a100.yaml",
    "h100": CONFIG_DIR / "h100.yaml",
    "b200": CONFIG_DIR / "b200.yaml",
    "w7900": CONFIG_DIR / "w7900.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or plan the BPA multi-GPU decode dataset workflow"
    )
    parser.add_argument(
        "--gpu",
        default="all",
        choices=["all", "a100", "h100", "b200", "w7900"],
        help="which lane config to run",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "smoke",
            "matrix-plan",
            "matrix-exec",
            "fit",
            "package",
            "full-plan",
            "full-dry-run",
        ],
        help="which workflow stage to run",
    )
    parser.add_argument(
        "--results-root",
        required=True,
        help="where to write manifests/reports/raw outputs",
    )
    parser.add_argument(
        "--export-root",
        default=None,
        help="export directory for package stage (default: <results-root>/knlp-paper-memory-decode)",
    )
    parser.add_argument(
        "--only-batches",
        default="",
        help="CSV filter passed through to run_matrix.py",
    )
    parser.add_argument(
        "--only-contexts",
        default="",
        help="CSV filter passed through to run_matrix.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit points passed through to run_matrix.py",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def config_paths(gpu: str) -> list[Path]:
    if gpu == "all":
        return [GPU_TO_CONFIG[name] for name in ["w7900", "a100", "h100", "b200"]]
    return [GPU_TO_CONFIG[gpu]]


def manifest_path(results_root: Path) -> Path:
    return results_root / "manifests" / "run_manifest.json"


def export_root(results_root: Path, requested: str | None) -> Path:
    if requested:
        return Path(requested)
    return results_root / DEFAULT_EXPORT_ROOT_NAME


def run_smoke(config: Path, results_root: Path, dry_run: bool = False) -> int:
    cmd = [
        sys.executable,
        "scripts/paper/bpa_paper/run_smoke.py",
        "--config",
        str(config),
        "--results-root",
        str(results_root),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd)


def run_matrix(
    config: Path,
    results_root: Path,
    execute: bool,
    dry_run: bool,
    only_batches: str,
    only_contexts: str,
    limit: int,
) -> int:
    cmd = [
        sys.executable,
        "scripts/paper/bpa_paper/run_matrix.py",
        "--config",
        str(config),
        "--results-root",
        str(results_root),
    ]
    if execute:
        cmd.append("--execute")
    if dry_run:
        cmd.append("--dry-run")
    if only_batches:
        cmd.extend(["--only-batches", only_batches])
    if only_contexts:
        cmd.extend(["--only-contexts", only_contexts])
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    return run_cmd(cmd)


def run_fit(results_root: Path, dry_run: bool = False) -> int:
    cmd = [
        sys.executable,
        "scripts/paper/bpa_paper/fit_scaling.py",
        "--manifest",
        str(manifest_path(results_root)),
        "--results-root",
        str(results_root),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd)


def run_package(results_root: Path, export_root_path: Path, dry_run: bool = False) -> int:
    cmd = [
        sys.executable,
        "scripts/paper/bpa_paper/package_results.py",
        "--manifest",
        str(manifest_path(results_root)),
        "--source-root",
        str(results_root),
        "--export-root",
        str(export_root_path),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run_cmd(cmd)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    export_root_path = export_root(results_root, args.export_root)
    configs = config_paths(args.gpu)

    if args.stage == "smoke":
        for config in configs:
            rc = run_smoke(config, results_root)
            if rc != 0:
                return rc
        return 0

    if args.stage == "matrix-plan":
        for config in configs:
            rc = run_matrix(
                config,
                results_root,
                execute=False,
                dry_run=False,
                only_batches=args.only_batches,
                only_contexts=args.only_contexts,
                limit=args.limit,
            )
            if rc != 0:
                return rc
        return 0

    if args.stage == "matrix-exec":
        for config in configs:
            rc = run_matrix(
                config,
                results_root,
                execute=True,
                dry_run=False,
                only_batches=args.only_batches,
                only_contexts=args.only_contexts,
                limit=args.limit,
            )
            if rc != 0:
                return rc
        return 0

    if args.stage == "fit":
        return run_fit(results_root)

    if args.stage == "package":
        return run_package(results_root, export_root_path)

    if args.stage == "full-plan":
        for config in configs:
            rc = run_smoke(config, results_root)
            if rc != 0:
                return rc
            rc = run_matrix(
                config,
                results_root,
                execute=False,
                dry_run=False,
                only_batches=args.only_batches,
                only_contexts=args.only_contexts,
                limit=args.limit,
            )
            if rc != 0:
                return rc
        rc = run_fit(results_root)
        if rc != 0:
            return rc
        return run_package(results_root, export_root_path)

    if args.stage == "full-dry-run":
        for config in configs:
            rc = run_smoke(config, results_root, dry_run=True)
            if rc != 0:
                return rc
            rc = run_matrix(
                config,
                results_root,
                execute=True,
                dry_run=True,
                only_batches=args.only_batches,
                only_contexts=args.only_contexts,
                limit=args.limit,
            )
            if rc != 0:
                return rc
        rc = run_fit(results_root, dry_run=True)
        if rc != 0:
            return rc
        return run_package(results_root, export_root_path, dry_run=True)

    raise ValueError(f"unsupported stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
