# SPDX-License-Identifier: MIT
"""TensorFlow build-system reproduction orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .tf_config import TensorFlowConfig

SCHEMA_VERSION = 1
STAGED_PACKAGE = "knlp_tensorflow"
STAGED_FILES = ("BUILD", "gemma_savedmodel_runner.cc")
MARKER = ".knlp-managed.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_text(command: list[str], cwd: Path, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def source_provenance(config: TensorFlowConfig) -> dict[str, object]:
    source = config.source_dir
    if not (source / ".git").exists():
        return {
            "git_commit": None,
            "git_remote": None,
            "git_status": [],
        }
    commit = run_text(["git", "rev-parse", "HEAD"], source)
    remote = run_text(
        ["git", "config", "--get", "remote.origin.url"],
        source,
        check=False,
    )
    status_text = run_text(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        source,
    )
    status = []
    for line in status_text.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        if path == STAGED_PACKAGE or path.startswith(f"{STAGED_PACKAGE}/"):
            continue
        status.append(line)
    configure = source / ".tf_configure.bazelrc"
    return {
        "git_commit": commit,
        "git_remote": remote or None,
        "git_status": status,
        "tf_configure_bazelrc_sha256": (
            sha256_file(configure) if configure.is_file() else None
        ),
    }


def adapter_command(config: TensorFlowConfig, adapter: str) -> list[str]:
    if not adapter:
        raise ValueError(
            "alternative builder is empty; provide " "TF_ALT_BUILDER=/path/to/adapter"
        )
    path = config.resolve_repo_path(adapter)
    if not path.is_file():
        raise ValueError(f"builder adapter does not exist: {path}")
    if path.suffix == ".py":
        return [sys.executable, str(path)]
    if not os.access(path, os.X_OK):
        raise ValueError(f"builder adapter is not executable: {path}")
    return [str(path)]


def workload_source(config: TensorFlowConfig) -> Path:
    return config.top / "tools/reproduce/tensorflow_ab/workload"


def staged_hashes(config: TensorFlowConfig) -> dict[str, str]:
    package = config.source_dir / STAGED_PACKAGE
    hashes = {}
    for name in STAGED_FILES:
        path = package / name
        if not path.is_file():
            raise ValueError(
                f"staged workload file is missing: {path}; "
                "run `make tensorflow-stage`"
            )
        hashes[name] = sha256_file(path)
    return hashes


def validate(config: TensorFlowConfig) -> dict[str, object]:
    if not config.source_dir.is_dir():
        raise ValueError(f"TensorFlow source does not exist: {config.source_dir}")
    if not any(
        (config.source_dir / name).exists()
        for name in ("WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel")
    ):
        raise ValueError(f"{config.source_dir} does not look like a Bazel workspace")
    for variant in config.variants():
        adapter_command(config, variant.adapter)
    provenance = source_provenance(config)
    if config.fail_on_dirty_source and provenance["git_status"]:
        details = "\n".join(str(x) for x in provenance["git_status"])
        raise ValueError(
            "TensorFlow source has unrelated changes:\n"
            f"{details}\nUse TF_ALLOW_DIRTY=1 only when intentional."
        )
    return provenance


def cmd_doctor(config: TensorFlowConfig) -> None:
    provenance = validate(config)
    print("=== TensorFlow build-system doctor ===")
    print(f"  mode: {config.mode}")
    print(f"  source: {config.source_dir}")
    print(f"  source commit: {provenance['git_commit'] or '(not a Git checkout)'}")
    print(f"  target: {config.target}")
    print(f"  jobs: {config.jobs or (os.cpu_count() or 1)}")
    print(f"  results: {config.results_root}")
    for variant in config.variants():
        command = adapter_command(config, variant.adapter)
        print(f"  {variant.label} adapter: {' '.join(command)}")


def cmd_stage(config: TensorFlowConfig) -> None:
    validate(config)
    source = workload_source(config)
    destination = config.source_dir / STAGED_PACKAGE
    marker = destination / MARKER
    if destination.exists() and not marker.is_file():
        raise ValueError(f"refusing to overwrite unmanaged package: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in STAGED_FILES:
        source_file = source / name
        destination_file = destination / name
        shutil.copy2(source_file, destination_file)
        hashes[name] = sha256_file(destination_file)
    write_json(
        marker,
        {
            "schema_version": SCHEMA_VERSION,
            "managed_by": "knlp tensorflow_ab",
            "files": hashes,
        },
    )
    print(f"staged workload package: {destination}")


def common_request(config: TensorFlowConfig) -> dict[str, object]:
    provenance = source_provenance(config)
    request = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "directory": str(config.source_dir),
            **provenance,
        },
        "workload": {
            "name": "gemma2b-savedmodel",
            "staged_package": STAGED_PACKAGE,
            "files": staged_hashes(config),
        },
        "build": {
            "target": config.target,
            "artifact_relpath": config.artifact_relpath,
            "arguments": config.build_args,
            "jobs": config.jobs or (os.cpu_count() or 1),
            "capture_profile": config.profile,
            "bazel_executable": config.bazel_bin,
        },
        "environment": {
            key: os.environ[key]
            for key in (
                "CC",
                "CXX",
                "CUDA_HOME",
                "TF_CUDA_VERSION",
                "TF_CUDNN_VERSION",
                "TF_CUDA_COMPUTE_CAPABILITIES",
            )
            if key in os.environ
        },
    }
    request["common_request_sha256"] = canonical_sha256(request)
    return request


def synthesize_failure(
    result_path: Path,
    variant: str,
    request_hash: str,
    returncode: int,
) -> None:
    write_json(
        result_path,
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "common_request_sha256": request_hash,
            "status": returncode,
            "error": "adapter exited without writing a result",
            "artifacts": [],
        },
    )


def cmd_build(config: TensorFlowConfig) -> None:
    validate(config)
    common = common_request(config)
    config.results_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for variant in config.variants():
        variant_root = config.results_root / "build" / variant.label
        request_path = config.results_root / "requests" / f"{variant.label}.json"
        result_path = variant_root / "result.json"
        request = {
            **common,
            "variant": variant.label,
            "output_directory": str(variant_root),
            "requested_utc": utc_now(),
        }
        write_json(request_path, request)
        print(f"{variant.label}: request " f"{common['common_request_sha256']}")
        if config.dry_run:
            print(f"{variant.label}: dry run, adapter not invoked")
            continue
        command = adapter_command(config, variant.adapter) + [
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        completed = subprocess.run(command, cwd=config.top, check=False)
        if not result_path.is_file():
            synthesize_failure(
                result_path,
                variant.label,
                str(common["common_request_sha256"]),
                completed.returncode or 1,
            )
        result = json.loads(result_path.read_text())
        status = int(result.get("status", completed.returncode))
        if status:
            failures.append((variant.label, status))
            print(f"{variant.label}: FAILED ({status})")
        else:
            print(f"{variant.label}: complete")
    if failures:
        rendered = ", ".join(f"{name}={status}" for name, status in failures)
        raise RuntimeError(f"builder failures: {rendered}")


def cmd_report(config: TensorFlowConfig) -> None:
    request_dir = config.results_root / "requests"
    results = {}
    hashes = set()
    for variant in config.variants():
        request_path = request_dir / f"{variant.label}.json"
        result_path = config.results_root / "build" / variant.label / "result.json"
        if not request_path.is_file():
            raise ValueError(f"missing request: {request_path}")
        request = json.loads(request_path.read_text())
        request_hash = request["common_request_sha256"]
        hashes.add(request_hash)
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("common_request_sha256") != request_hash:
                raise ValueError(f"{variant.label} result used a different request")
            results[variant.label] = result
        else:
            results[variant.label] = {
                "status": None,
                "note": "request emitted; no result",
            }
    if len(hashes) != 1:
        raise ValueError("selected builders did not receive one common request")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": config.mode,
        "common_request_sha256": next(iter(hashes)),
        "generated_utc": utc_now(),
        "variants": results,
    }
    if len(results) == 2:
        artifact_hashes = {}
        for label, result in results.items():
            artifact_hashes[label] = [
                artifact.get("sha256") for artifact in result.get("artifacts", [])
            ]
        summary["artifact_hashes"] = artifact_hashes
    output = config.results_root / "build-summary.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report: {output}")


def cmd_clean(config: TensorFlowConfig) -> None:
    destination = config.source_dir / STAGED_PACKAGE
    marker = destination / MARKER
    if not destination.exists():
        print("staged workload is already absent")
        return
    if not marker.is_file():
        raise ValueError(f"refusing to remove unmanaged package: {destination}")
    allowed = set(STAGED_FILES) | {MARKER}
    unexpected = {
        path.name for path in destination.iterdir() if path.name not in allowed
    }
    if unexpected:
        raise ValueError(
            f"refusing to remove package with unexpected files: "
            f"{sorted(unexpected)}"
        )
    shutil.rmtree(destination)
    print(f"removed staged workload package: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("doctor", "stage", "build", "report", "clean")
    )
    parser.add_argument("--config", default=".config")
    args = parser.parse_args()
    try:
        config = TensorFlowConfig.load(args.config)
        {
            "doctor": cmd_doctor,
            "stage": cmd_stage,
            "build": cmd_build,
            "report": cmd_report,
            "clean": cmd_clean,
        }[args.command](config)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"tensorflow-ab: ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
