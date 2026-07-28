#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference TensorFlow builder adapter using Bazel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def request_environment(request: dict[str, object]) -> dict[str, str]:
    requested = request.get("environment", {})
    if not isinstance(requested, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in requested.items()
    ):
        raise ValueError("request environment keys and values must be strings")
    environment = os.environ.copy()
    environment.update(requested)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    build = request["build"]
    source = Path(request["source"]["directory"])
    output = Path(request["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    output_base = output / "output-base"
    profile = output / "build.profile.gz"
    log_path = output / "build.log"
    bazel = build["bazel_executable"]
    build_arguments = shlex.split(build["arguments"])
    environment = request_environment(request)
    cuda_capabilities = environment.get("TF_CUDA_COMPUTE_CAPABILITIES")
    capability_prefix = "--repo_env=TF_CUDA_COMPUTE_CAPABILITIES="
    capability_arguments = [
        argument.removeprefix(capability_prefix)
        for argument in build_arguments
        if argument.startswith(capability_prefix)
    ]
    if cuda_capabilities:
        if capability_arguments and any(
            value != cuda_capabilities for value in capability_arguments
        ):
            raise ValueError(
                "TF_CUDA_COMPUTE_CAPABILITIES conflicts with build arguments"
            )
        if not capability_arguments:
            build_arguments.append(f"{capability_prefix}{cuda_capabilities}")
    command = [
        bazel,
        f"--output_base={output_base}",
        "build",
        f"--jobs={build['jobs']}",
    ]
    if build["capture_profile"]:
        command.append(f"--profile={profile}")
    command.extend(build_arguments)
    command.append(build["target"])

    started_utc = utc_now()
    started = time.monotonic()
    status = 1
    error = None
    artifacts = []
    version = None
    binary_directory = None
    with log_path.open("w") as log:
        print(f"$ {shlex.join(command)}", file=log, flush=True)
        try:
            version_result = subprocess.run(
                [bazel, "--version"],
                cwd=source,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            version = version_result.stdout.strip()
            status = subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
            if status == 0:
                info_command = [
                    bazel,
                    f"--output_base={output_base}",
                    "info",
                    *build_arguments,
                    "bazel-bin",
                ]
                info = subprocess.run(
                    info_command,
                    cwd=source,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    check=False,
                )
                if info.returncode:
                    status = info.returncode
                    error = "bazel info bazel-bin failed"
                else:
                    binary_directory = Path(info.stdout.strip().splitlines()[-1])
                    artifact = binary_directory / build["artifact_relpath"]
                    if not artifact.is_file():
                        status = 1
                        error = f"expected artifact is missing: {artifact}"
                    else:
                        artifacts.append(
                            {
                                "path": str(artifact),
                                "sha256": sha256_file(artifact),
                                "size_bytes": artifact.stat().st_size,
                            }
                        )
        except OSError as exception:
            error = str(exception)
            print(error, file=log)

    result = {
        "schema_version": request["schema_version"],
        "variant": request["variant"],
        "common_request_sha256": request["common_request_sha256"],
        "status": status,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "engine": {"name": "bazel", "version": version},
        "command": command,
        "log": str(log_path),
        "profile": (
            str(profile) if build["capture_profile"] and profile.is_file() else None
        ),
        "binary_directory": (str(binary_directory) if binary_directory else None),
        "artifacts": artifacts,
        "error": error,
    }
    write_json(args.result, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
