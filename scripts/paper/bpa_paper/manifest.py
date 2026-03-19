#!/usr/bin/env python3
"""Manifest schema helpers for BPA paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

VALID_STATUSES = {"pending", "ok", "failed", "invalid", "skipped"}


def empty_manifest() -> Dict[str, Any]:
    return {
        "version": 1,
        "runs": [],
        "points": [],
        "exports": [],
    }


def validate_manifest(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["manifest must be a dict"]
    for key in ["version", "runs", "points", "exports"]:
        if key not in data:
            errors.append(f"missing key: {key}")
    for point in data.get("points", []):
        status = point.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"invalid point status: {status}")
        for key in ["gpu", "lane", "batch", "context"]:
            if key not in point:
                errors.append(f"point missing key: {key}")
    return errors


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BPA paper manifest")
    parser.add_argument("manifest", help="manifest path")
    args = parser.parse_args()
    data = load_manifest(Path(args.manifest))
    errors = validate_manifest(data)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("manifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
