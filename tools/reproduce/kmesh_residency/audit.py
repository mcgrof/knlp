# SPDX-License-Identifier: MIT
"""Audit whether recorded requests can support a family-residency experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

TRACE_REVISION = "780bcc2979715150d8b9fd4737e026e625444cc9"
SCHEMA = "knlp.kmesh.invocation.v1"
REQUIRED = (
    "request_id",
    "family_id",
    "parent_request_id",
    "engine_id",
    "clock_id",
    "start_ns",
    "end_ns",
    "input_token_ids",
    "output_token_ids",
    "status",
    "cache_observations",
)


def prefix_length(left, right):
    """Count an exact prefix in the supplied unit, without estimating tokens."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def timestamp_seconds(value):
    """Normalize plausible epoch units for descriptive record spacing only."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    for scale in (1, 1000, 1000000, 1000000000):
        seconds = value / scale
        if 946684800 <= seconds < 4102444800:
            return seconds
    return None


def rows(path):
    """Read JSONL without interpreting prompt contents or dropping bad lines."""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
            except (json.JSONDecodeError, ValueError):
                yield number, None
            else:
                yield number, value


def check_invocation(row):
    """Reject ambiguous identities, invented interval types and invalid tokens."""
    if row.get("schema") != SCHEMA or any(key not in row for key in REQUIRED):
        raise ValueError("incomplete invocation schema")
    for key in ("request_id", "family_id", "engine_id", "clock_id"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError("invalid " + key)
    if row["parent_request_id"] is not None and (
        not isinstance(row["parent_request_id"], str) or not row["parent_request_id"]
    ):
        raise ValueError("invalid parent identity")
    if any(type(row[key]) is not int or row[key] < 0 for key in ("start_ns", "end_ns")):
        raise ValueError("invalid interval")
    if row["end_ns"] < row["start_ns"]:
        raise ValueError("negative invocation duration")
    for key in ("input_token_ids", "output_token_ids"):
        if not isinstance(row[key], list) or any(
            type(x) is not int or x < 0 for x in row[key]
        ):
            raise ValueError("invalid token sequence")
    if row["status"] not in ("complete", "cancelled", "error"):
        raise ValueError("invalid terminal status")
    if not isinstance(row["cache_observations"], list):
        raise ValueError("invalid cache observations")


def lineage(records):
    """Measure signed parent-to-child gaps only within an observed clock domain."""
    indexed = {}
    for row in records:
        check_invocation(row)
        key = row["request_id"]
        if key in indexed:
            raise ValueError("duplicate request identity")
        indexed[key] = row
    gaps = []
    unknown = cross_clock = cross_engine = incompatible = 0
    for child in records:
        parent_id = child["parent_request_id"]
        if parent_id is None:
            continue
        seen = {child["request_id"]}
        ancestor = parent_id
        while ancestor in indexed:
            if ancestor in seen:
                raise ValueError("cyclic lineage")
            seen.add(ancestor)
            ancestor = indexed[ancestor]["parent_request_id"]
        parent = indexed.get(parent_id)
        if parent is None:
            unknown += 1
            continue
        if parent["family_id"] != child["family_id"]:
            raise ValueError("parent and child have different families")
        if parent["clock_id"] != child["clock_id"]:
            cross_clock += 1
            continue
        if child["start_ns"] < parent["start_ns"]:
            raise ValueError("child starts before its parent invocation")
        cross_engine += parent["engine_id"] != child["engine_id"]
        ancestor_tokens = parent["input_token_ids"] + parent["output_token_ids"]
        shared = prefix_length(ancestor_tokens, child["input_token_ids"])
        if shared != len(ancestor_tokens):
            incompatible += 1
        gaps.append((child["start_ns"] - parent["end_ns"]) / 1e6)
    return {
        "observed_edges": len(gaps),
        "unknown_parents": unknown,
        "cross_clock_edges": cross_clock,
        "cross_engine_edges": cross_engine,
        "non_exact_parent_prefix_edges": incompatible,
        "signed_fork_gap_ms": gaps,
        "live_forks": sum(gap < 0 for gap in gaps),
    }


def audit(root, mode="lmcache"):
    """Keep text reuse, invocation timing and physical cache evidence separate."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("trace root is not a directory")
    if mode not in ("lmcache", "invocations"):
        raise ValueError("unknown trace format")
    revision = None
    if mode == "lmcache":
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != TRACE_REVISION:
            raise ValueError("trace checkout does not match the pinned revision")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]):
            raise ValueError("trace checkout is dirty")
        paths = [
            root / name
            for name in subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "-z"], text=True
            ).split("\0")
            if name.endswith(".jsonl")
        ]
    else:
        paths = root.rglob("*.jsonl")
    groups = defaultdict(
        lambda: {
            "files": 0,
            "records": 0,
            "parse_errors": 0,
            "fields": Counter(),
            "sessions": defaultdict(list),
        }
    )
    manifest = []
    invocations = []
    duplicates = 0
    seen = set()
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(
            part.startswith(".") or part.endswith("_result")
            for part in relative.parts[:-1]
        ):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("trace symlink is not supported")
        name = relative.parts[0] if len(relative.parts) > 1 else "invocations"
        group = groups[name]
        group["files"] += 1
        manifest.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        for number, row in rows(path):
            if row is None:
                group["parse_errors"] += 1
                continue
            if mode == "invocations":
                check_invocation(row)
                invocations.append(row)
            digest = hashlib.sha256(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
            ).digest()
            if (name, digest) in seen:
                duplicates += 1
                continue
            seen.add((name, digest))
            group["records"] += 1
            group["fields"].update(row.keys())
            session = row.get("session_id")
            stamp = timestamp_seconds(row.get("timestamp"))
            text = row.get("input")
            if (
                type(session) in (str, int)
                and stamp is not None
                and isinstance(text, str)
            ):
                group["sessions"][(type(session).__name__, str(session))].append(
                    (stamp, text)
                )
    if not manifest:
        raise ValueError("no JSONL files found")
    summaries = {}
    for name, group in sorted(groups.items()):
        spacings, fractions = [], []
        session_count = len(group["sessions"])
        for session in group.pop("sessions").values():
            session.sort(key=lambda item: item[0])
            for (start, a), (end, b) in zip(session, session[1:]):
                spacings.append(end - start)
                if b:
                    fractions.append(prefix_length(a, b) / len(b))
        summaries[name] = {
            **group,
            "fields": dict(sorted(group["fields"].items())),
            "session_keys": session_count,
            "record_spacing_seconds_median": (
                statistics.median(spacings) if spacings else None
            ),
            "input_character_prefix_fraction_median": (
                statistics.median(fractions) if fractions else None
            ),
            "temporal_pairs": len(fractions),
        }
    edges = lineage(invocations)
    complete = bool(invocations) and not any(
        g["parse_errors"] for g in summaries.values()
    )
    timed = (
        complete
        and edges["observed_edges"] > 0
        and edges["unknown_parents"] == edges["cross_clock_edges"] == 0
    )
    return {
        "schema": "knlp.kmesh.trace-audit.v1",
        "source_revision": revision,
        "mode": mode,
        "files": manifest,
        "frameworks": summaries,
        "exact_duplicate_records": duplicates,
        "invocation_records": len(invocations),
        "lineage": edges,
        "timed_lineage_complete": timed,
        "pressure_replay_qualified": False,
        "decision": (
            "physical-cache-capture-required"
            if timed
            else "invocation-capture-required"
        ),
        "limits": [
            "Record spacing is not a measured idle or fork gap.",
            "Character-prefix reuse is not token identity or physical KV sharing.",
            "Session identifiers do not establish related families or overlapping inference.",
            "Pressure qualification additionally requires observed pool capacity, block lifetimes, evictions and competing requests.",
            "This audit does not estimate eviction probability or predict a latency gain.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("lmcache", "invocations"))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = {}
    if args.config.exists():
        for line in args.config.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_TRACE_"):
                key, value = line.split("=", 1)
                config[key.removeprefix("CONFIG_KNLP_KMESH_TRACE_")] = json.loads(value)
    root = args.root or os.environ.get("KMESH_TRACE_ROOT") or config.get("ROOT")
    output = (
        args.output
        or os.environ.get("KMESH_TRACE_OUTPUT")
        or config.get("OUTPUT", "results/kmesh-trace-audit")
    )
    mode = (
        args.mode or os.environ.get("KMESH_TRACE_MODE") or config.get("MODE", "lmcache")
    )
    if mode not in ("lmcache", "invocations"):
        parser.error("unknown trace format")
    if args.plan:
        print(
            json.dumps(
                {
                    "root": str(root or ""),
                    "output": str(output),
                    "mode": mode,
                    "source_revision": TRACE_REVISION,
                    "gpu_execution": False,
                },
                indent=2,
            )
        )
        return
    if not root:
        parser.error("set KMESH_TRACE_ROOT or --root")
    output = Path(output)
    if output.exists():
        parser.error("output must be a fresh directory")
    result = audit(root, mode)
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"decision": result["decision"], "output": str(output / "audit.json")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
