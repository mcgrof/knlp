#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise EfficientAgent conditioned admission under real host-tier pressure."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
import traceback
from typing import Any

import modal

APP_NAME = "knlp-prefixplan-conditioned-pressure"
EFFICIENTAGENT_REVISION = "b21b12a07a3ccf473c82acf00820b1aa2073044b"
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.13.0@"
    "sha256:1ae45c63e0578cf8ea690299c20b6a32a5c3ddf659e92ea15c0ae6b8040ba741"
)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REMOTE_ROOT = "/runs/agent-cache-baselines-20261002/prefixplan-conditioned-pressure"
REPLAY_VALIDATION_NAME = "replay_validation.py"
REPLAY_VALIDATION_REMOTE = f"/opt/knlp/{REPLAY_VALIDATION_NAME}"

app = modal.App(APP_NAME)
image = modal.Image.from_registry(VLLM_IMAGE).apt_install("git")
if modal.is_local():
    validation_path = pathlib.Path(__file__).with_name(REPLAY_VALIDATION_NAME)
    image = image.add_local_file(validation_path, REPLAY_VALIDATION_REMOTE, copy=True)
image = (
    image.run_commands(
        "python3 -m pip install lmcache==0.3.12",
        "git clone --filter=blob:none "
        "https://github.com/KunmingSHAO/efficientagent_release.git /opt/efficientagent",
        f"git -C /opt/efficientagent checkout {EFFICIENTAGENT_REVISION}",
        "python3 -m pip install --no-deps --no-build-isolation -e /opt/efficientagent",
        "ln -sf $(command -v python3) /usr/local/bin/python",
    )
    .env(
        {
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .entrypoint([])
)
hf = modal.Volume.from_name("knlp-hf", create_if_missing=False)
runs = modal.Volume.from_name("knlp-runs", create_if_missing=False)


def _write_json(path: str, value: Any) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _copy_tree(source: str, destination: str) -> None:
    os.makedirs(destination, exist_ok=True)
    for entry in os.scandir(source):
        target = os.path.join(destination, entry.name)
        if entry.is_dir():
            _copy_tree(entry.path, target)
        elif not entry.name.endswith(".tmp"):
            shutil.copy2(entry.path, target)


def _commit(local_dir: str, run_name: str) -> None:
    _copy_tree(local_dir, os.path.join(REMOTE_ROOT, run_name))
    runs.commit()


def _first_hit_events(server_log: pathlib.Path) -> dict[str, dict[str, int]]:
    """Recover the connector's durable first-hit receipts from its server log."""
    events: dict[str, dict[str, int]] = {}
    pattern = re.compile(r"EA_KVTIER first-hit (\S+).*?counts=(\{.*?\})")
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        counts = ast.literal_eval(match.group(2))
        if not isinstance(counts, dict):
            raise TypeError(f"Expected counter dictionary in: {line}")
        events[match.group(1)] = {str(key): int(value) for key, value in counts.items()}
    return events


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800, volumes={"/hf": hf})
def fetch_model() -> dict:
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(repo_id=MODEL, revision=MODEL_REVISION)
    hf.commit()
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "path": path,
        "seconds": time.time() - started,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=6.0,
    memory=32768,
    timeout=90 * 60,
    volumes={"/hf": hf, "/runs": runs},
)
def qualify(run_name: str, knlp_commit: str) -> dict:
    import sys

    import torch
    import vllm
    from efficientagent.replay import launch, synthetic_trace
    from huggingface_hub import snapshot_download

    sys.path.insert(0, "/opt/knlp")
    from replay_validation import validate_replay_artifacts

    if not run_name or "/" in run_name or run_name.startswith("."):
        raise ValueError("run_name must be one safe path component")
    if not re.fullmatch(r"[0-9a-f]{40}", knlp_commit):
        raise ValueError("knlp_commit must be a full 40-character Git object ID")
    source_revision = subprocess.check_output(
        ["git", "-C", "/opt/efficientagent", "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != EFFICIENTAGENT_REVISION:
        raise RuntimeError(
            f"Expected EfficientAgent {EFFICIENTAGENT_REVISION}, got {source_revision}"
        )

    local_dir = os.path.join("/tmp", run_name)
    remote_result = os.path.join(REMOTE_ROOT, run_name, "result.json")
    os.makedirs(local_dir, exist_ok=True)
    runs.reload()
    if os.path.exists(remote_result):
        prior = json.loads(pathlib.Path(remote_result).read_text())
        if prior.get("status") == "passed":
            return prior

    model_path = snapshot_download(
        repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True
    )
    initial = {
        "status": "running",
        "run_name": run_name,
        "started_at": time.time(),
        "source": {
            "knlp_harness_commit": knlp_commit,
            "harness_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
            ).hexdigest(),
            "replay_validation_sha256": hashlib.sha256(
                pathlib.Path(REPLAY_VALIDATION_REMOTE).read_bytes()
            ).hexdigest(),
            "efficientagent_revision": EFFICIENTAGENT_REVISION,
            "efficientagent_checkout_revision": source_revision,
            "base_image": VLLM_IMAGE,
            "vllm_version": vllm.__version__,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
    }
    _write_json(os.path.join(local_dir, "result.json"), initial)
    _commit(local_dir, run_name)

    trace_dir = os.path.join(local_dir, "trace")
    replay_dir = os.path.join(local_dir, "run")
    try:
        synthetic_trace.generate(
            pathlib.Path(trace_dir),
            tasks=4,
            steps=(3, 3),
            system_tokens=2048,
            task_tokens=2048,
            observation_tokens=(256, 768),
            output_tokens=(8, 16),
            gap_s=(1.0, 1.5),
            token_range=(1000, 20000),
        )
        arguments = [
            "--admission",
            "conditioned",
            "--host-gib",
            "0.1",
            "--model",
            model_path,
            "--served-model-name",
            MODEL,
            "--trace",
            trace_dir,
            "--out",
            replay_dir,
            "--workers",
            "4",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.25",
            "--write-threshold",
            "2",
            "--occupancy-threshold",
            "0.95",
            "--chunk-tokens",
            "256",
            "--window-s",
            "60",
            "--telemetry-max-age-s",
            "5",
            "--max-seconds",
            "300",
            "--server-arg=--enforce-eager",
            "--execute",
        ]
        return_code = launch.main(arguments)
        run = json.loads(pathlib.Path(replay_dir, "run.json").read_text())
        manifest = json.loads(
            pathlib.Path(trace_dir, "trace_manifest.json").read_text()
        )
        replay_validation = validate_replay_artifacts(
            replay_dir, manifest, return_code=return_code
        )
        counters = run.get("kvtier_counters") or {}
        first_hit_events = _first_hit_events(pathlib.Path(replay_dir, "server.log"))
        pressure_counts = first_hit_events.get("PRESSURE_ON") or {}
        telemetry = run.get("telemetry_last") or {}
        capacity = int(telemetry.get("capacity") or 0)
        registered = int(telemetry.get("registered") or 0)
        checks = {
            **replay_validation["checks"],
            "pressure_event_observed": "PRESSURE_ON" in first_hit_events,
            "skip_write_event_observed": "SKIP_WRITE" in first_hit_events,
            "pressure_requests_positive": pressure_counts.get("pressure_requests", 0)
            > 0,
            "skipped_requests_positive": pressure_counts.get("skipped_requests", 0) > 0,
            "estimate_above_capacity_positive": pressure_counts.get(
                "estimate_above_capacity", 0
            )
            > 0,
            "telemetry_full_evicting_positive": pressure_counts.get(
                "telemetry_full_evicting", 0
            )
            > 0,
            "host_evictions_positive": int(telemetry.get("evicted_in_window") or 0) > 0,
            "host_at_admission_threshold": capacity > 0
            and registered >= 0.95 * capacity,
        }
        result = {
            **initial,
            "status": "passed" if all(checks.values()) else "failed",
            "finished_at": time.time(),
            "launcher_arguments": arguments,
            "checks": checks,
            "replay_validation": replay_validation,
            "kvtier_counters": counters,
            "first_hit_events": first_hit_events,
            "telemetry_last": telemetry,
            "reporting_note": (
                "The aggregate kvtier counter files retained only connector_init at "
                "shutdown. Qualification therefore uses the connector's first-hit "
                "event receipts plus final telemetry; both are preserved in run.json "
                "and server.log."
            ),
            "replay_status": run.get("status"),
            "gpu_kv_tokens": run.get("gpu_kv_tokens"),
        }
        _write_json(os.path.join(local_dir, "result.json"), result)
        _commit(local_dir, run_name)
        if result["status"] != "passed":
            raise RuntimeError(f"Conditioned-pressure checks failed: {checks}")
        return result
    except BaseException as error:
        failure = {
            **initial,
            "status": "error",
            "finished_at": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(os.path.join(local_dir, "failure.json"), failure)
        _commit(local_dir, run_name)
        raise


@app.local_entrypoint()
def main(run_name: str, knlp_commit: str):
    fetched = fetch_model.remote()
    print("MODEL_FETCH " + json.dumps(fetched, sort_keys=True), flush=True)
    result = qualify.remote(run_name, knlp_commit)
    print("QUALIFICATION_RESULT " + json.dumps(result, sort_keys=True), flush=True)
