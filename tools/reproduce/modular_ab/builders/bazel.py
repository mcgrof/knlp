#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference Modular adapter using the repository's Bazel wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_bazel(source_dir: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() and path.is_file():
        return str(path)
    source_relative = source_dir / path
    if source_relative.is_file():
        return str(source_relative.resolve())
    located = shutil.which(value)
    if located:
        return located
    raise RuntimeError(f"Bazel executable not found: {value}")


def command_output(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    common = request["common_request"]
    build = common["build"]
    source_dir = Path(common["source"]["directory"])
    output_dir = Path(request["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "bazel-output-base"
    if output_base.exists():
        shutil.rmtree(output_base)

    build_log = output_dir / "build.log"
    profile = output_dir / "build.profile.gz"
    command: list[str] = []
    started_at = utc_now()
    started = time.perf_counter()
    returncode = 1
    failure = ""
    bazel = ""
    try:
        bazel = resolve_bazel(source_dir, build["bazel_executable"])
        command = [bazel, f"--output_base={output_base}", "build"]
        if build["jobs"]:
            command.append(f"--jobs={build['jobs']}")
        if build["capture_profile"]:
            command.append(f"--profile={profile}")
        command.extend(build["arguments"])
        command.append(build["target"])
        with build_log.open("wb") as stream:
            completed = subprocess.run(
                command,
                cwd=source_dir,
                check=False,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        returncode = completed.returncode
    except OSError as error:
        failure = str(error)
        build_log.write_text(failure + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started

    artifacts: list[dict[str, Any]] = []
    if returncode == 0:
        info_command = [
            bazel,
            f"--output_base={output_base}",
            "info",
            "bazel-bin",
        ]
        bazel_bin_output = command_output(info_command, source_dir)
        bazel_bin = Path(bazel_bin_output.splitlines()[-1])
        artifact = (bazel_bin / build["artifact_relpath"]).resolve()
        if not artifact.is_file():
            returncode = 1
            failure = f"expected artifact not found: {artifact}"
        else:
            artifacts.append(
                {
                    "path": str(artifact),
                    "sha256": sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            )

    version = command_output([bazel, "--version"], source_dir) if bazel else ""
    result = {
        "schema_version": 1,
        "variant": request["variant"],
        "common_request_sha256": request["common_request_sha256"],
        "status": "succeeded" if returncode == 0 else "failed",
        "returncode": returncode,
        "failure": failure,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": elapsed,
        "engine": {"name": "bazel", "executable": bazel, "version": version},
        "command": command,
        "build_log": str(build_log),
        "profile": str(profile) if profile.is_file() else "",
        "artifacts": artifacts,
    }
    write_json(result_path, result)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
