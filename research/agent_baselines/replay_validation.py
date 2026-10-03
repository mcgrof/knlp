# SPDX-License-Identifier: MIT
"""Strict, dependency-free validation for EfficientAgent replay receipts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{label}: expected an integer >= {minimum}",
    )
    return value


def _expected_task_steps(manifest: dict[str, Any]) -> dict[str, int]:
    _require(isinstance(manifest, dict), "trace manifest must be an object")
    files = manifest.get("files")
    _require(isinstance(files, list) and files, "trace manifest has no files")
    expected: dict[str, int] = {}
    for index, entry in enumerate(files):
        _require(isinstance(entry, dict), f"manifest file {index} is not an object")
        instance_id = entry.get("instance_id")
        _require(
            isinstance(instance_id, str) and bool(instance_id),
            f"manifest file {index} has no instance_id",
        )
        _require(instance_id not in expected, f"duplicate manifest task {instance_id}")
        expected[instance_id] = _integer(
            entry.get("steps"), f"manifest task {instance_id} steps", minimum=1
        )
    _require(
        _integer(manifest.get("tasks"), "manifest tasks") == len(expected),
        "manifest task count mismatch",
    )
    _require(
        _integer(manifest.get("steps"), "manifest steps") == sum(expected.values()),
        "manifest request count mismatch",
    )
    return expected


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[Any]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_replay_records(
    *,
    return_code: int,
    run: dict[str, Any],
    replay: dict[str, Any],
    manifest: dict[str, Any],
    requests: list[Any],
    tasks: list[Any],
) -> dict[str, Any]:
    """Reject partial, interrupted, mismatched, or misidentified replays."""
    expected = _expected_task_steps(manifest)
    expected_requests = sum(expected.values())

    _require(
        type(return_code) is int and return_code == 0, "launcher did not return zero"
    )
    _require(isinstance(run, dict), "run receipt must be an object")
    _require(run.get("status") == "passed", "upstream replay did not pass")
    _require(isinstance(replay, dict), "replay summary must be an object")
    _require(run.get("replay") == replay, "run and replay summaries differ")

    exact_summary = {
        "requests": expected_requests,
        "forced_match": expected_requests,
        "forced_mismatch": 0,
        "errors": 0,
        "prompt_len_mismatch": 0,
        "tasks": len(expected),
    }
    for field, wanted in exact_summary.items():
        _require(
            _integer(replay.get(field), f"replay {field}") == wanted,
            f"replay {field} mismatch: expected {wanted}",
        )
    _require(
        replay.get("all_outputs_forced_exactly") is True,
        "forced-output verification did not pass",
    )
    _require(replay.get("interrupted") is False, "replay was interrupted")

    _require(isinstance(requests, list), "request receipts must be a list")
    _require(
        len(requests) == expected_requests,
        "request receipt count differs from the manifest",
    )
    request_counts: Counter[str] = Counter()
    request_sequences: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(requests):
        _require(isinstance(row, dict), f"request receipt {index} is not an object")
        instance_id = row.get("instance_id")
        _require(instance_id in expected, f"unexpected request task {instance_id!r}")
        sequence = _integer(
            row.get("seq"), f"request {instance_id} sequence", minimum=1
        )
        _require(
            sequence not in request_sequences[instance_id],
            f"duplicate request sequence for {instance_id}: {sequence}",
        )
        _require(
            row.get("forced_match") is True,
            f"request {instance_id}/{sequence} did not force-match",
        )
        _require(
            row.get("prompt_ok") is True,
            f"request {instance_id}/{sequence} has a prompt mismatch",
        )
        request_sequences[instance_id].add(sequence)
        request_counts[instance_id] += 1
    _require(
        dict(request_counts) == expected,
        "per-task request counts differ from the manifest",
    )

    _require(isinstance(tasks, list), "task receipts must be a list")
    _require(
        len(tasks) == len(expected), "task receipt count differs from the manifest"
    )
    completed: dict[str, int] = {}
    for index, row in enumerate(tasks):
        _require(isinstance(row, dict), f"task receipt {index} is not an object")
        instance_id = row.get("instance_id")
        _require(instance_id in expected, f"unexpected completed task {instance_id!r}")
        _require(
            instance_id not in completed, f"duplicate completed task {instance_id}"
        )
        _require(row.get("ok") is True, f"task {instance_id} did not complete cleanly")
        completed[instance_id] = _integer(
            row.get("steps"), f"completed task {instance_id} steps", minimum=1
        )
    _require(completed == expected, "completed task steps differ from the manifest")

    checks = {
        "launcher_returned_zero": True,
        "replay_passed": True,
        "summary_matches_run": True,
        "not_interrupted": True,
        "zero_replay_errors": True,
        "zero_forced_mismatches": True,
        "zero_prompt_length_mismatches": True,
        "forced_outputs_exact": True,
        "exact_request_ids_and_counts": True,
        "exact_task_ids_and_counts": True,
        "request_prompts_verified": True,
    }
    return {
        "status": "complete-token-replay",
        "checks": checks,
        "expected_tasks": len(expected),
        "expected_requests": expected_requests,
    }


def validate_replay_artifacts(
    run_dir: str | Path,
    manifest: dict[str, Any],
    *,
    return_code: int,
) -> dict[str, Any]:
    """Load and strictly validate an EfficientAgent replay output directory."""
    root = Path(run_dir)
    return validate_replay_records(
        return_code=return_code,
        run=_read_json(root / "run.json"),
        replay=_read_json(root / "replay/replay_summary.json"),
        manifest=manifest,
        requests=_read_jsonl(root / "replay/requests.jsonl"),
        tasks=_read_jsonl(root / "replay/tasks.jsonl"),
    )
