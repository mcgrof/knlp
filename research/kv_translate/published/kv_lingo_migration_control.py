#!/usr/bin/env python3
"""Prepare and verify the five-conversation KV-Lingo migration control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import coqa
from .kv_lingo_data import write_json

FILES = ("RAW.jsonl", "NATIVE_TRAJECTORY.jsonl", "OWNERSHIP.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(name: str, row: dict) -> tuple:
    base = (row["conversation_id"], row["starting_model"])
    if name == "OWNERSHIP.jsonl":
        return base
    return (*base, row["turn"])


def comparable_row(name: str, row: dict) -> dict:
    """Return the deterministic migration-control payload for one row."""
    value = dict(row)
    if name == "RAW.jsonl" and "catch_up" in value:
        # Runtime is hardware-dependent.  The span count and token count are
        # semantic control fields; elapsed seconds are not.
        value["catch_up"] = {
            key: item for key, item in value["catch_up"].items() if key != "seconds"
        }
    return value


def prepare(
    conversations_path: Path,
    raw_path: Path,
    native_path: Path,
    ownership_path: Path,
    out_dir: Path,
) -> dict:
    conversations = json.loads(conversations_path.read_text(encoding="utf-8"))
    by_domain = {}
    for domain in coqa.DOMAINS:
        candidates = [row for row in conversations if row["source"] == domain]
        if not candidates:
            raise ValueError(f"no development conversation for domain {domain}")
        by_domain[domain] = min(candidates, key=lambda row: row["id"])
    selected = [by_domain[domain] for domain in coqa.DOMAINS]
    ids = {row["id"] for row in selected}

    out_dir.mkdir(parents=True, exist_ok=False)
    conversations_out = out_dir / "DEVELOPMENT_CONTROL_CONVERSATIONS.json"
    write_json(conversations_out, selected)
    inputs = {
        "RAW.jsonl": raw_path,
        "NATIVE_TRAJECTORY.jsonl": native_path,
        "OWNERSHIP.jsonl": ownership_path,
    }
    artifacts = {
        "DEVELOPMENT_CONTROL_CONVERSATIONS.json": {
            "rows": len(selected),
            "sha256": sha256(conversations_out),
        }
    }
    for name, path in inputs.items():
        rows = [row for row in read_jsonl(path) if row["conversation_id"] in ids]
        expected = len(selected) * 2 * (1 if name == "OWNERSHIP.jsonl" else 10)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} rows, found {len(rows)}")
        target = out_dir / name
        write_jsonl(target, rows)
        artifacts[name] = {"rows": len(rows), "sha256": sha256(target)}
    manifest = {
        "schema": "kv_lingo_provider_migration_control_v1",
        "selection": "smallest conversation id within each frozen CoQA domain",
        "domains": {domain: by_domain[domain]["id"] for domain in coqa.DOMAINS},
        "turns": 10,
        "starting_models": ["4B", "8B"],
        "checkpoint_pair": {
            "forward_4b_to_8b_training_steps": 5000,
            "reverse_8b_to_4b_training_steps": 5000,
        },
        "artifacts": artifacts,
        "Generated-by": "OpenAI Codex",
    }
    write_json(out_dir / "CONTROL_MANIFEST.json", manifest)
    return manifest


def verify(expected_dir: Path, replay_dir: Path) -> dict:
    comparisons = {}
    passed = True
    for name in FILES:
        expected_rows = read_jsonl(expected_dir / name)
        replay_rows = read_jsonl(replay_dir / name)
        expected = {row_key(name, row): row for row in expected_rows}
        replay = {row_key(name, row): row for row in replay_rows}
        if len(expected) != len(expected_rows) or len(replay) != len(replay_rows):
            raise ValueError(f"{name}: duplicate row identity")
        missing = sorted(set(expected) - set(replay))
        extra = sorted(set(replay) - set(expected))
        changed = sorted(
            key
            for key in set(expected) & set(replay)
            if comparable_row(name, expected[key]) != comparable_row(name, replay[key])
        )
        file_passed = not missing and not extra and not changed
        passed = passed and file_passed
        comparisons[name] = {
            "expected_rows": len(expected),
            "replay_rows": len(replay),
            "missing": [list(key) for key in missing],
            "extra": [list(key) for key in extra],
            "changed": [list(key) for key in changed],
            "passed": file_passed,
        }
    return {
        "schema": "kv_lingo_provider_migration_control_result_v1",
        "passed": passed,
        "comparisons": comparisons,
        "Generated-by": "OpenAI Codex",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--conversations", type=Path, required=True)
    prepare_parser.add_argument("--raw", type=Path, required=True)
    prepare_parser.add_argument("--native", type=Path, required=True)
    prepare_parser.add_argument("--ownership", type=Path, required=True)
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--expected-dir", type=Path, required=True)
    verify_parser.add_argument("--replay-dir", type=Path, required=True)
    verify_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.conversations,
            args.raw,
            args.native,
            args.ownership,
            args.out_dir,
        )
    else:
        write_json(args.out, verify(args.expected_dir, args.replay_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
