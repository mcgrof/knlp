# SPDX-License-Identifier: MIT
"""Build and run one Modular target through interchangeable builders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .modular_config import ModularConfig


RESULTS_MARKER = ".knlp-modular-ab-results"
RECORDED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "RUSH_BAZEL_PATH",
    "RUSH_MOJO",
    "RUSH_MOJO_CLANG_RESOURCE_DIR",
    "RUSH_MOJO_CXX",
    "RUSH_MOJO_SYSROOT",
    "RUSH_MOJO_TARGET_ACCELERATOR",
    "RUSH_MOJO_TARGET_CPU",
    "RUSH_MOJO_TARGET_TRIPLE",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def run_text(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def source_provenance(source_dir: Path) -> dict[str, Any]:
    status = run_text(["git", "status", "--porcelain"], cwd=source_dir)
    return {
        "directory": str(source_dir),
        "git_commit": run_text(["git", "rev-parse", "HEAD"], cwd=source_dir),
        "git_remote": run_text(
            ["git", "config", "--get", "remote.origin.url"], cwd=source_dir
        ),
        "git_status": status.splitlines() if status else [],
    }


def ensure_results_root(config: ModularConfig) -> Path:
    root = config.results_root
    root.mkdir(parents=True, exist_ok=True)
    marker = root / RESULTS_MARKER
    if not marker.exists():
        entries = [entry for entry in root.iterdir() if entry != marker]
        if entries:
            raise RuntimeError(
                f"refusing unmanaged non-empty results directory: {root}"
            )
        marker.write_text("managed by knlp modular_ab\n", encoding="utf-8")
    return root


def resolve_adapter(config: ModularConfig, value: str, variant: str) -> list[str]:
    if not value:
        raise RuntimeError(f"{variant} builder adapter is not configured")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config.top / path
    if not path.is_file():
        raise RuntimeError(f"missing {variant} builder adapter: {path}")
    if path.suffix == ".py":
        return [sys.executable, str(path.resolve())]
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"builder adapter is not executable: {path}")
    return [str(path.resolve())]


def resolve_bazel(config: ModularConfig) -> str:
    value = config.bazel_bin
    path = Path(value).expanduser()
    if path.is_absolute() and path.is_file():
        return str(path)
    source_relative = config.source_dir / path
    if source_relative.is_file():
        return str(source_relative.resolve())
    located = shutil.which(value)
    if located:
        return located
    raise RuntimeError(f"reference Bazel executable not found: {value}")


def doctor(config: ModularConfig) -> dict[str, Any]:
    if not config.source_dir.is_dir():
        raise RuntimeError(f"missing Modular source directory: {config.source_dir}")
    if not (config.source_dir / "MODULE.bazel").is_file():
        raise RuntimeError(f"not a Modular checkout: {config.source_dir}")
    if not config.target:
        raise RuntimeError("Modular target is empty")
    if not config.artifact_relpath:
        raise RuntimeError("Modular artifact relative path is empty")
    if config.runtime_runs < 1 or config.runtime_warmups < 0:
        raise RuntimeError("runtime run counts are invalid")
    if not config.runtime_marker:
        raise RuntimeError("runtime validation marker is empty")

    provenance = source_provenance(config.source_dir)
    if not provenance["git_commit"]:
        raise RuntimeError("Modular source is not a Git checkout")
    if (
        config.source_commit
        and provenance["git_commit"] != config.source_commit
    ):
        raise RuntimeError(
            "Modular source commit mismatch: "
            f"expected {config.source_commit}, got {provenance['git_commit']}"
        )
    if config.fail_on_dirty_source and provenance["git_status"]:
        raise RuntimeError(
            "Modular source is dirty; use MODULAR_ALLOW_DIRTY=1 only "
            "for an intentional experiment"
        )

    adapters: dict[str, list[str]] = {}
    for variant in config.variants:
        value = (
            config.reference_builder
            if variant == "reference"
            else config.alternative_builder
        )
        adapters[variant] = resolve_adapter(config, value, variant)
    bazel = resolve_bazel(config) if "reference" in config.variants else ""

    gpu_names: list[str] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        output = run_text(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"]
        )
        gpu_names = output.splitlines() if output else []

    return {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "mode": config.mode,
        "source": provenance,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "gpus": gpu_names,
        },
        "adapters": adapters,
        "reference_bazel": bazel,
    }


def common_request(config: ModularConfig) -> dict[str, Any]:
    environment = {
        key: os.environ[key]
        for key in RECORDED_ENVIRONMENT
        if key in os.environ
    }
    return {
        "schema_version": 1,
        "source": source_provenance(config.source_dir),
        "workload": {
            "name": "modular-tiled-matmul",
            "runtime_marker": config.runtime_marker,
        },
        "build": {
            "target": config.target,
            "artifact_relpath": config.artifact_relpath,
            "arguments": config.build_args,
            "jobs": config.jobs,
            "capture_profile": config.capture_profile,
            "clean_build": True,
            "bazel_executable": config.bazel_bin,
        },
        "runtime": {
            "arguments": config.runtime_args,
            "warmups": config.runtime_warmups,
            "runs": config.runtime_runs,
            "timeout_seconds": config.runtime_timeout,
        },
        "environment": environment,
    }


def cmd_doctor(config: ModularConfig, _args: argparse.Namespace) -> int:
    root = ensure_results_root(config)
    result = doctor(config)
    path = root / "doctor.json"
    write_json(path, result)
    print(path)
    return 0


def failed_build_result(
    variant: str,
    request_hash: str,
    returncode: int,
    adapter_log: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "variant": variant,
        "common_request_sha256": request_hash,
        "status": "failed",
        "returncode": returncode,
        "adapter_log": str(adapter_log),
        "artifacts": [],
    }


def cmd_build(config: ModularConfig, _args: argparse.Namespace) -> int:
    doctor(config)
    root = ensure_results_root(config)
    common = common_request(config)
    request_hash = canonical_sha256(common)
    write_json(root / "common-request.json", common)

    failures = 0
    for variant in config.variants:
        output_dir = root / "build" / variant
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        request_path = output_dir / "request.json"
        result_path = output_dir / "result.json"
        adapter_log = output_dir / "adapter.log"
        request = {
            "schema_version": 1,
            "variant": variant,
            "common_request_sha256": request_hash,
            "common_request": common,
            "output_directory": str(output_dir.resolve()),
        }
        write_json(request_path, request)
        if config.dry_run:
            continue

        adapter_value = (
            config.reference_builder
            if variant == "reference"
            else config.alternative_builder
        )
        command = resolve_adapter(config, adapter_value, variant)
        command += ["--request", str(request_path), "--result", str(result_path)]
        completed = subprocess.run(
            command,
            cwd=config.top,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        adapter_log.write_text(completed.stdout, encoding="utf-8")
        if not result_path.is_file():
            write_json(
                result_path,
                failed_build_result(
                    variant, request_hash, completed.returncode, adapter_log
                ),
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or result.get("status") != "succeeded":
            failures += 1

    if config.dry_run:
        print(root / "common-request.json")
        return 0
    return 1 if failures else 0


def load_artifact(
    config: ModularConfig,
    variant: str,
    override: str | None,
) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
    else:
        result_path = config.results_root / "build" / variant / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing {variant} build result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "succeeded":
            raise RuntimeError(f"{variant} build did not succeed")
        artifacts = result.get("artifacts", [])
        if not artifacts or not artifacts[0].get("path"):
            raise RuntimeError(f"{variant} build returned no executable artifact")
        path = Path(artifacts[0]["path"]).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {variant} runtime artifact: {path}")
    if not os.access(path, os.X_OK):
        raise RuntimeError(f"runtime artifact is not executable: {path}")
    return path


def execute_artifact(
    config: ModularConfig,
    variant: str,
    artifact: Path,
    phase: str,
    pair: int,
    order: int,
    output_dir: Path,
) -> dict[str, Any]:
    stem = f"{phase}-{pair:03d}-{order:02d}-{variant}"
    stdout_path = output_dir / f"{stem}.stdout"
    stderr_path = output_dir / f"{stem}.stderr"
    command = [str(artifact), *config.runtime_args]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=config.source_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.runtime_timeout,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        returncode = 124
        timed_out = True
    elapsed = time.perf_counter() - started
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    marker_found = config.runtime_marker.encode("utf-8") in stdout
    return {
        "variant": variant,
        "phase": phase,
        "pair": pair,
        "order": order,
        "command": command,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "validation_marker_found": marker_found,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def timing_stats(values: list[float]) -> dict[str, float]:
    return {
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
        "geomean_seconds": math.exp(
            statistics.mean([math.log(value) for value in values])
        ),
        "maximum_seconds": max(values),
    }


def cmd_runtime(config: ModularConfig, args: argparse.Namespace) -> int:
    doctor(config)
    root = ensure_results_root(config)
    overrides = {
        "reference": args.reference_artifact,
        "alternative": args.alternative_artifact,
    }
    artifacts = {
        variant: load_artifact(config, variant, overrides[variant])
        for variant in config.variants
    }
    runtime_dir = root / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    for warmup in range(1, config.runtime_warmups + 1):
        for order, variant in enumerate(config.variants, start=1):
            records.append(
                execute_artifact(
                    config,
                    variant,
                    artifacts[variant],
                    "warmup",
                    warmup,
                    order,
                    runtime_dir,
                )
            )

    for pair in range(1, config.runtime_runs + 1):
        order_variants = list(config.variants)
        if pair % 2 == 0:
            order_variants.reverse()
        for order, variant in enumerate(order_variants, start=1):
            records.append(
                execute_artifact(
                    config,
                    variant,
                    artifacts[variant],
                    "measure",
                    pair,
                    order,
                    runtime_dir,
                )
            )

    measured = [record for record in records if record["phase"] == "measure"]
    valid = all(
        record["returncode"] == 0 and record["validation_marker_found"]
        for record in records
    )
    stdout_equal = len(
        {record["stdout_sha256"] for record in measured}
    ) == 1
    stats = {
        variant: timing_stats(
            [
                record["elapsed_seconds"]
                for record in measured
                if record["variant"] == variant
            ]
        )
        for variant in config.variants
    }

    comparison: dict[str, Any] = {}
    if config.mode == "ab":
        pair_ratios = []
        for pair in range(1, config.runtime_runs + 1):
            by_variant = {
                record["variant"]: record["elapsed_seconds"]
                for record in measured
                if record["pair"] == pair
            }
            pair_ratios.append(
                by_variant["alternative"] / by_variant["reference"]
            )
        ratio = math.exp(statistics.mean([math.log(x) for x in pair_ratios]))
        comparison = {
            "alternative_over_reference_pair_ratios": pair_ratios,
            "alternative_over_reference_geomean": ratio,
            "alternative_delta_percent": (ratio - 1.0) * 100.0,
        }

    summary = {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "status": "passed" if valid and stdout_equal else "failed",
        "mode": config.mode,
        "runtime_marker": config.runtime_marker,
        "stdout_byte_identical": stdout_equal,
        "artifacts": {
            variant: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for variant, path in artifacts.items()
        },
        "statistics": stats,
        "comparison": comparison,
        "runs": records,
    }
    write_json(root / "runtime-summary.json", summary)
    print(root / "runtime-summary.json")
    return 0 if summary["status"] == "passed" else 1


def cmd_report(config: ModularConfig, _args: argparse.Namespace) -> int:
    root = ensure_results_root(config)
    common_path = root / "common-request.json"
    if not common_path.is_file():
        raise RuntimeError(f"missing common build request: {common_path}")
    common = json.loads(common_path.read_text(encoding="utf-8"))
    request_hash = canonical_sha256(common)

    builds: dict[str, Any] = {}
    for variant in config.variants:
        result_path = root / "build" / variant / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing {variant} build result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["request_hash_matches"] = (
            result.get("common_request_sha256") == request_hash
        )
        builds[variant] = result

    runtime_path = root / "runtime-summary.json"
    runtime = (
        json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime_path.is_file()
        else None
    )
    builds_passed = all(
        result.get("status") == "succeeded"
        and result["request_hash_matches"]
        for result in builds.values()
    )
    runtime_passed = runtime is None or runtime.get("status") == "passed"
    summary = {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "status": "passed" if builds_passed and runtime_passed else "failed",
        "common_request_sha256": request_hash,
        "builds": builds,
        "runtime": runtime,
    }
    path = root / "summary.json"
    write_json(path, summary)
    print(path)
    return 0 if summary["status"] == "passed" else 1


def cmd_clean(config: ModularConfig, _args: argparse.Namespace) -> int:
    root = config.results_root
    if not root.exists():
        return 0
    if not (root / RESULTS_MARKER).is_file():
        raise RuntimeError(f"refusing to remove unmanaged directory: {root}")
    shutil.rmtree(root)
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=".config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "build", "report", "clean"):
        command_parser = subparsers.add_parser(name)
        add_common_arguments(command_parser)
    runtime_parser = subparsers.add_parser("runtime")
    add_common_arguments(runtime_parser)
    runtime_parser.add_argument("--reference-artifact")
    runtime_parser.add_argument("--alternative-artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = ModularConfig.load(args.config)
        commands = {
            "doctor": cmd_doctor,
            "build": cmd_build,
            "runtime": cmd_runtime,
            "report": cmd_report,
            "clean": cmd_clean,
        }
        return commands[args.command](config, args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
