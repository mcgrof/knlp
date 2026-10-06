#!/usr/bin/env python3
"""Run one explicit command and always write a terminal-state receipt.

This helper is intentionally small.  It keeps process state in one Python
process, so a failing pipeline, command substitution, or shell trap cannot
leave a receipt marked ``RUNNING`` after the command exits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_with_receipt(
    command: Sequence[str],
    receipt_path: Path,
    *,
    label: str = "kv-lingo-stage",
    deadline_seconds: float | None = None,
) -> dict:
    """Run ``command`` and atomically record COMPLETE, FAILED, or TIMED_OUT."""
    if not command:
        raise ValueError("command must not be empty")
    if deadline_seconds is not None and deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")

    started_at = utc_now()
    started = time.monotonic()
    state = "RUNNING"
    exit_code = 1
    error = None
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            timeout=deadline_seconds,
        )
        exit_code = completed.returncode
        state = "COMPLETE" if exit_code == 0 else "FAILED"
    except subprocess.TimeoutExpired as exception:
        state = "TIMED_OUT"
        exit_code = 124
        error = str(exception)
    except OSError as exception:
        state = "FAILED"
        exit_code = 127
        error = f"{type(exception).__name__}: {exception}"
    finally:
        receipt = {
            "schema": "kv_lingo_command_receipt_v1",
            "label": label,
            "command": list(command),
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "state": state,
            "exit_code": exit_code,
            "error": error,
        }
        write_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--label", default="kv-lingo-stage")
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    receipt = run_with_receipt(
        command,
        args.receipt,
        label=args.label,
        deadline_seconds=args.deadline_seconds,
    )
    return int(receipt["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
