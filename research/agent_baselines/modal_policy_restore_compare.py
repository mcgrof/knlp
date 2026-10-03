#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare host-write policies while forcing a later physical restoration.

Every arm uses the EfficientAgent connector, including its copy-time
deduplication.  A small GPU KV pool and a 0.16 GiB host tier make the sequence
``sacrificial filler, target, pressure fillers, target revisit`` exercise
admission, eviction, and later reuse in one vLLM/LMCache serving stack.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import statistics
import subprocess
import time
import traceback
from typing import Any

import modal

APP_NAME = "knlp-prefixplan-policy-restore"
EFFICIENTAGENT_REVISION = "b21b12a07a3ccf473c82acf00820b1aa2073044b"
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.13.0@"
    "sha256:1ae45c63e0578cf8ea690299c20b6a32a5c3ddf659e92ea15c0ae6b8040ba741"
)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REMOTE_ROOT = "/runs/agent-cache-baselines-20261002/prefixplan-policy-restore"
PATCH_NAME = "efficientagent_stats_instances.patch"
PATCH_REMOTE = f"/opt/knlp/{PATCH_NAME}"
REPLAY_VALIDATION_NAME = "replay_validation.py"
REPLAY_VALIDATION_REMOTE = f"/opt/knlp/{REPLAY_VALIDATION_NAME}"
HOST_GIB = 0.16
CHUNK_TOKENS = 256
KV_BYTES_PER_TOKEN = 28672
PROMPT_TOKENS = 3072
GPU_BLOCKS = 256
FILLERS = 4
NEW_CHUNK_THRESHOLD = 2
UNFILTERED_THRESHOLD = 1_000_000
ARM_CONFIGS = {
    "unfiltered": {"admission": "fixed", "write_threshold": UNFILTERED_THRESHOLD},
    "fixed": {"admission": "fixed", "write_threshold": NEW_CHUNK_THRESHOLD},
    "conditioned": {
        "admission": "conditioned",
        "write_threshold": NEW_CHUNK_THRESHOLD,
    },
}
ARM_ORDERS = (
    ("unfiltered", "fixed", "conditioned"),
    ("conditioned", "unfiltered", "fixed"),
    ("fixed", "conditioned", "unfiltered"),
)

app = modal.App(APP_NAME)
image = modal.Image.from_registry(VLLM_IMAGE).apt_install("git")
if modal.is_local():
    patch_path = pathlib.Path(__file__).with_name(PATCH_NAME)
    image = image.add_local_file(patch_path, PATCH_REMOTE, copy=True)
    validation_path = pathlib.Path(__file__).with_name(REPLAY_VALIDATION_NAME)
    image = image.add_local_file(validation_path, REPLAY_VALIDATION_REMOTE, copy=True)
image = (
    image.run_commands(
        "python3 -m pip install lmcache==0.3.12 pytest",
        "git clone --filter=blob:none "
        "https://github.com/KunmingSHAO/efficientagent_release.git /opt/efficientagent",
        f"git -C /opt/efficientagent checkout {EFFICIENTAGENT_REVISION}",
        f"git -C /opt/efficientagent apply --check {PATCH_REMOTE}",
        f"git -C /opt/efficientagent apply {PATCH_REMOTE}",
        "python3 -m pip install --no-deps --no-build-isolation -e /opt/efficientagent",
        "python3 -m pytest -q /opt/efficientagent/tests/test_connector.py -k stats",
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


def _write_json(path: str | pathlib.Path, value: Any) -> None:
    path = os.fspath(path)
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
        elif not entry.name.endswith((".tmp", ".partial")):
            shutil.copy2(entry.path, target)


def _commit(local_dir: str, remote_dir: str) -> None:
    _copy_tree(local_dir, remote_dir)
    runs.commit()


def _sha256(path: str | pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _safe_component(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _validate_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("knlp_commit must be a full 40-character Git object ID")
    return value


def _trace(out: pathlib.Path, seed: int) -> dict[str, Any]:
    """Write the same ordered, one-request-per-task workload for every arm."""
    rng = random.Random(seed)

    def tokens(count: int) -> list[int]:
        return [rng.randrange(1000, 20000) for _ in range(count)]

    target = tokens(PROMPT_TOKENS)
    fillers = [tokens(PROMPT_TOKENS) for _ in range(FILLERS)]
    # The first two conditioned writes exceed the host tier by one chunk.  Put
    # the sacrificial filler first so LRU evicts it rather than the beginning
    # of the target; LMCache prefix lookup needs the target's leading chunk.
    prompts = [("sacrificial-filler", fillers[0]), ("warm-target", target)]
    prompts += [
        (f"pressure-filler-{index}", filler)
        for index, filler in enumerate(fillers[1:], start=1)
    ]
    prompts += [("revisit-target", target)]
    out.mkdir(parents=True)
    files = []
    for order, (instance_id, prompt) in enumerate(prompts):
        output = tokens(8)
        row = {
            "seq": 1,
            "gap_before_s": 0.0,
            "prompt_shared": 0,
            "prompt_suffix": prompt,
            "prompt_len": len(prompt),
            "output": output,
            "output_len": len(output),
            "recorded_client_elapsed_s": 0.0,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        }
        head = {
            "instance_id": instance_id,
            "order": order,
            "tail_gap_s": 1.3,
            "recorded_jct_s": 0.0,
            "calls_total": 1,
            "calls_replayed": 1,
        }
        filename = f"{order:03d}_{instance_id}.jsonl.gz"
        with gzip.open(out / filename, "wt") as stream:
            stream.write(json.dumps(head) + "\n")
            stream.write(json.dumps(row) + "\n")
        files.append(
            {"order": order, "instance_id": instance_id, "file": filename, "steps": 1}
        )
    manifest = {
        "schema": "efficientagent.replay_trace.v1",
        "synthetic": True,
        "seed": seed,
        "tasks": len(files),
        "steps": len(files),
        "files": files,
        "workload": {
            "prompt_tokens": PROMPT_TOKENS,
            "filler_count": FILLERS,
            "sequence": [name for name, _ in prompts],
            "purpose": "connect pressure admission to a later target restoration",
        },
    }
    _write_json(out / "trace_manifest.json", manifest)
    return manifest


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _target_receipt(run_dir: pathlib.Path) -> dict[str, Any]:
    """Join the final target's client window to LMCache lookup/retrieval logs."""
    from efficientagent.analysis import common

    requests = _jsonl(run_dir / "replay" / "requests.jsonl")
    target = next(row for row in requests if row["instance_id"] == "revisit-target")
    lookups, _storing, _stored, retrieved = common.parse_lmcache_log(
        run_dir / "server.log"
    )
    candidates = [
        row
        for row in lookups
        if row[2] == target["prompt_len"]
        and target["start"] - 0.5 <= row[0] <= target["end"] + 0.5
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one final-target lookup, found {len(candidates)}")
    lookup = candidates[0]
    physical = [
        row
        for row in retrieved
        if target["start"] - 0.2 <= row[0] <= target["end"] + 0.5
    ]
    return {
        "client": target,
        "lookup": {
            "request_id": lookup[1],
            "total_tokens": lookup[2],
            "host_hit_tokens": lookup[3],
            "need_to_load": lookup[4],
        },
        "physical_retrieval_events": [
            {"tokens": row[1], "required_tokens": row[2], "cost_ms": row[3]}
            for row in physical
        ],
        "physical_retrieved_tokens": sum(row[1] for row in physical),
    }


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800, volumes={"/hf": hf})
def fetch_model() -> dict[str, Any]:
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
    timeout=45 * 60,
    volumes={"/hf": hf, "/runs": runs},
)
def run_arm(
    campaign: str, repetition: int, arm: str, knlp_commit: str
) -> dict[str, Any]:
    import sys

    import torch
    import vllm
    from efficientagent.replay import launch
    from huggingface_hub import snapshot_download

    sys.path.insert(0, "/opt/knlp")
    from replay_validation import validate_replay_artifacts

    _safe_component(campaign, "campaign")
    _validate_commit(knlp_commit)
    if repetition < 1:
        raise ValueError("repetition must be positive")
    if arm not in ARM_CONFIGS:
        raise ValueError(f"Unknown arm: {arm}")
    source_revision = subprocess.check_output(
        ["git", "-C", "/opt/efficientagent", "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != EFFICIENTAGENT_REVISION:
        raise RuntimeError(
            f"Expected EfficientAgent {EFFICIENTAGENT_REVISION}, got {source_revision}"
        )
    patch_check = subprocess.run(
        ["git", "-C", "/opt/efficientagent", "diff", "--check"],
        text=True,
        capture_output=True,
    )
    if patch_check.returncode:
        raise RuntimeError(
            f"Patched source failed git diff --check: {patch_check.stderr}"
        )

    run_id = f"r{repetition:02d}-{arm}"
    local_dir = os.path.join("/tmp", campaign, run_id)
    remote_dir = os.path.join(REMOTE_ROOT, campaign, run_id)
    remote_result = os.path.join(remote_dir, "result.json")
    runs.reload()
    if os.path.exists(remote_result):
        prior = json.loads(pathlib.Path(remote_result).read_text())
        if prior.get("status") == "passed":
            return prior
        raise RuntimeError(
            f"Existing non-passing result refuses overwrite: {remote_result}"
        )
    if os.path.exists(local_dir):
        raise RuntimeError(f"Local output already exists: {local_dir}")
    os.makedirs(local_dir)

    config = ARM_CONFIGS[arm]
    connector_path = pathlib.Path(
        "/opt/efficientagent/efficientagent/kvtier/connector.py"
    )
    initial = {
        "status": "running",
        "campaign": campaign,
        "run_id": run_id,
        "repetition": repetition,
        "arm": arm,
        "arm_config": config,
        "started_at": time.time(),
        "source": {
            "knlp_harness_commit": knlp_commit,
            "harness_sha256": _sha256(__file__),
            "replay_validation_sha256": _sha256(REPLAY_VALIDATION_REMOTE),
            "counter_patch_sha256": _sha256(PATCH_REMOTE),
            "patched_connector_sha256": _sha256(connector_path),
            "efficientagent_revision": EFFICIENTAGENT_REVISION,
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
        "measurement_boundary": (
            "Serving metrics and LMCache transfer logs cover replay only; "
            "post-replay greedy probes are disabled."
        ),
    }
    _write_json(os.path.join(local_dir, "result.json"), initial)
    _commit(local_dir, remote_dir)

    trace_dir = pathlib.Path(local_dir, "trace")
    replay_dir = pathlib.Path(local_dir, "run")
    try:
        manifest = _trace(trace_dir, seed=10_000 + repetition)
        model_path = snapshot_download(
            repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True
        )
        arguments = [
            "--admission",
            str(config["admission"]),
            "--host-gib",
            str(HOST_GIB),
            "--model",
            model_path,
            "--served-model-name",
            MODEL,
            "--trace",
            str(trace_dir),
            "--out",
            str(replay_dir),
            "--workers",
            "1",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.25",
            "--kv-bytes-per-token",
            str(KV_BYTES_PER_TOKEN),
            "--write-threshold",
            str(config["write_threshold"]),
            "--occupancy-threshold",
            "0.95",
            "--chunk-tokens",
            str(CHUNK_TOKENS),
            "--window-s",
            "60",
            "--telemetry-max-age-s",
            "5",
            "--max-seconds",
            "180",
            "--tmpdir",
            f"/tmp/e{repetition}{arm[0]}",
            "--server-arg=--enforce-eager",
            f"--server-arg=--num-gpu-blocks-override={GPU_BLOCKS}",
            "--execute",
        ]
        return_code = launch.main(arguments)
        run = json.loads(pathlib.Path(replay_dir, "run.json").read_text())
        replay_validation = validate_replay_artifacts(
            replay_dir, manifest, return_code=return_code
        )
        metrics = json.loads(pathlib.Path(replay_dir, "run_metrics.json").read_text())
        counters = run.get("kvtier_counters") or {}
        target = _target_receipt(replay_dir)
        counter_files = sorted(
            path.name
            for path in pathlib.Path(replay_dir, "kvtier_stats").glob("*.json")
            if "telemetry" not in path.name
        )
        checks = {
            **replay_validation["checks"],
            "aggregate_decisions_survived": counters.get("scheduler.decisions", 0)
            == manifest["steps"],
            "role_counter_files_preserved": len(counter_files) >= 2,
            "target_lookup_joined": target["lookup"]["total_tokens"] == PROMPT_TOKENS,
            "no_post_replay_probe": "greedy_check_requests" not in run,
        }
        if arm == "unfiltered":
            checks["unfiltered_saved_every_request"] = counters.get(
                "scheduler.saved_requests", 0
            ) == manifest["steps"] and not counters.get("scheduler.skipped_requests", 0)
        elif arm == "fixed":
            checks["fixed_filter_skipped_requests"] = (
                counters.get("scheduler.skipped_requests", 0) > 0
            )
        else:
            checks.update(
                {
                    "conditioned_pressure_observed": counters.get(
                        "scheduler.pressure_requests", 0
                    )
                    > 0,
                    "conditioned_skip_observed": counters.get(
                        "scheduler.skipped_requests", 0
                    )
                    > 0,
                    "host_evictions_observed": counters.get(
                        "worker.cpu_evicted_chunks", 0
                    )
                    > 0,
                    "later_physical_restoration": target["physical_retrieved_tokens"]
                    > 0,
                }
            )
        result = {
            **initial,
            "status": "passed" if all(checks.values()) else "failed",
            "finished_at": time.time(),
            "checks": checks,
            "replay_validation": replay_validation,
            "trace_manifest_sha256": _sha256(trace_dir / "trace_manifest.json"),
            "launcher_arguments": arguments,
            "counter_files": counter_files,
            "kvtier_counters": counters,
            "telemetry_last": run.get("telemetry_last"),
            "target_revisit": target,
            "metrics": {
                key: metrics.get(key)
                for key in (
                    "makespan_s",
                    "prompt_tokens",
                    "prefill_computed_tokens",
                    "retrieved_per_rank",
                    "stored_per_rank",
                    "gpu_prefix_hit_tokens",
                    "request_latency",
                    "task_jct",
                    "gpu_kv_tokens",
                )
            },
        }
        _write_json(os.path.join(local_dir, "result.json"), result)
        _commit(local_dir, remote_dir)
        if result["status"] != "passed":
            raise RuntimeError(f"Policy/restore checks failed: {checks}")
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
        _commit(local_dir, remote_dir)
        raise


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=1800,
    volumes={"/runs": runs},
)
def summarize(
    campaign: str, results: list[dict[str, Any]], knlp_commit: str
) -> dict[str, Any]:
    _safe_component(campaign, "campaign")
    _validate_commit(knlp_commit)
    grouped: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_CONFIGS}
    for result in results:
        grouped[result["arm"]].append(result)
    medians = {}
    for arm, rows in grouped.items():
        medians[arm] = {}
        for metric in (
            "prefill_computed_tokens",
            "retrieved_per_rank",
            "stored_per_rank",
            "makespan_s",
        ):
            values = [row["metrics"][metric] for row in rows]
            medians[arm][metric] = statistics.median(values) if values else None
        values = [row["target_revisit"]["client"]["latency_s"] for row in rows]
        medians[arm]["target_revisit_latency_s"] = (
            statistics.median(values) if values else None
        )
    conditioned = grouped["conditioned"]
    unfiltered = grouped["unfiltered"]
    fixed = grouped["fixed"]
    gates = {
        "all_arms_passed": all(row.get("status") == "passed" for row in results),
        "balanced_repetitions": bool(conditioned)
        and len(conditioned) == len(unfiltered) == len(fixed),
        "dedup_implementation_identical": len(
            {row["source"]["patched_connector_sha256"] for row in results}
        )
        == 1,
        "trace_identical_within_repetition": all(
            len(
                {
                    row["trace_manifest_sha256"]
                    for row in results
                    if row["repetition"] == repetition
                }
            )
            == 1
            for repetition in {row["repetition"] for row in results}
        ),
        "conditioned_restored_every_repeat": bool(conditioned)
        and all(
            row["target_revisit"]["physical_retrieved_tokens"] > 0
            for row in conditioned
        ),
        "conditioned_recomputed_less_than_unfiltered_each_repeat": bool(conditioned)
        and all(
            next(row for row in conditioned if row["repetition"] == repetition)[
                "metrics"
            ]["prefill_computed_tokens"]
            < next(row for row in unfiltered if row["repetition"] == repetition)[
                "metrics"
            ]["prefill_computed_tokens"]
            for repetition in {row["repetition"] for row in conditioned}
        ),
    }
    summary = {
        "status": "passed" if all(gates.values()) else "completed_with_issues",
        "campaign": campaign,
        "knlp_harness_commit": knlp_commit,
        "finished_at": time.time(),
        "arm_orders": [list(order) for order in ARM_ORDERS],
        "gates": gates,
        "medians": medians,
        "results": [
            {
                "run_id": row["run_id"],
                "repetition": row["repetition"],
                "arm": row["arm"],
                "metrics": row["metrics"],
                "target_revisit": row["target_revisit"],
                "kvtier_counters": row["kvtier_counters"],
            }
            for row in results
        ],
        "claim_boundary": (
            "Synthetic mechanism comparison only. No live-agent quality or "
            "general PrefixPlan benefit is established."
        ),
    }
    local_dir = f"/tmp/{campaign}-summary"
    os.makedirs(local_dir, exist_ok=True)
    _write_json(os.path.join(local_dir, "SUMMARY.json"), summary)
    _commit(local_dir, os.path.join(REMOTE_ROOT, campaign))
    return summary


@app.local_entrypoint()
def main(
    campaign: str,
    knlp_commit: str,
    repetitions: int = 1,
):
    _safe_component(campaign, "campaign")
    _validate_commit(knlp_commit)
    if not 1 <= repetitions <= len(ARM_ORDERS):
        raise ValueError(f"repetitions must be between 1 and {len(ARM_ORDERS)}")
    fetched = fetch_model.remote()
    print("MODEL_FETCH " + json.dumps(fetched, sort_keys=True), flush=True)
    results = []
    for repetition in range(1, repetitions + 1):
        for arm in ARM_ORDERS[repetition - 1]:
            result = run_arm.remote(campaign, repetition, arm, knlp_commit)
            results.append(result)
            print("ARM_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    summary = summarize.remote(campaign, results, knlp_commit)
    print("CAMPAIGN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
