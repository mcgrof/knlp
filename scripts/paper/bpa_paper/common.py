#!/usr/bin/env python3
"""Common helpers for BPA paper experiment orchestration."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclasses.dataclass
class Point:
    batch: int
    context: int


@dataclasses.dataclass
class RunnerConfig:
    kind: str
    command_template: Optional[str] = None
    setup_command: Optional[str] = None
    env: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class LaneConfig:
    gpu: str
    lane: str
    model: str
    batches: List[int]
    contexts: List[int]
    warmup_iters: int
    measure_iters: int
    lane_type: str
    notes: str
    runner: Optional[RunnerConfig] = None


@dataclasses.dataclass
class RunContext:
    config_path: Path
    results_root: Path
    timestamp: str
    dry_run: bool = False


def now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_config(path: Path) -> LaneConfig:
    data = load_yaml(path)
    return LaneConfig(
        gpu=str(data["gpu"]),
        lane=str(data["lane"]),
        model=str(data["model"]),
        batches=[int(x) for x in data["batches"]],
        contexts=[int(x) for x in data["contexts"]],
        warmup_iters=int(data.get("warmup_iters", 10)),
        measure_iters=int(data.get("measure_iters", 30)),
        lane_type=str(data.get("lane_type", "core")),
        notes=str(data.get("notes", "")),
        runner=(
            RunnerConfig(
                kind=str(data["runner"].get("kind", "plan-only")),
                command_template=data["runner"].get("command_template"),
                setup_command=data["runner"].get("setup_command"),
                env={str(k): str(v) for k, v in data["runner"].get("env", {}).items()},
            )
            if isinstance(data.get("runner"), dict)
            else None
        ),
    )


def iter_points(config: LaneConfig) -> Iterable[Point]:
    for context in config.contexts:
        for batch in config.batches:
            yield Point(batch=batch, context=context)


def ensure_dirs(results_root: Path) -> None:
    for sub in [
        "manifests",
        "reports",
        "raw",
        "derived",
        "figures",
        "logs",
        "system",
    ]:
        (results_root / sub).mkdir(parents=True, exist_ok=True)


def parse_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML lane config")
    parser.add_argument("--results-root", required=True, help="results root")
    parser.add_argument("--dry-run", action="store_true", help="print only")
    return parser


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
