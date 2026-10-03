#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a bounded layout plan with uniform and static controls on Modal.

The logical workload has two request families.  One family reuses repository
context and the other reuses tool instructions.  A uniform layout can expose
only one reuse pattern at the prompt prefix.  The planner arm chooses one of
the same two approved layouts per family.  A static per-family arm renders the
same assignment without using the planner and is the attribution control.

All arms use the same pressure-conditioned EfficientAgent connector and
copy-time deduplication.  This is a synthetic systems mechanism check, not a
live-agent quality result or a general compiler-performance claim.
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

APP_NAME = "knlp-prefixplan-joint-layout"
PREFIX_PLAN_SCHEMA = "knlp.agent_baselines.prefix_plan.v2"
EFFICIENTAGENT_REVISION = "b21b12a07a3ccf473c82acf00820b1aa2073044b"
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.13.0@"
    "sha256:1ae45c63e0578cf8ea690299c20b6a32a5c3ddf659e92ea15c0ae6b8040ba741"
)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REMOTE_ROOT = "/runs/agent-cache-baselines-20261002/prefixplan-joint-layout"
PATCH_NAME = "efficientagent_stats_instances.patch"
PATCH_REMOTE = f"/opt/knlp/{PATCH_NAME}"
REPLAY_VALIDATION_NAME = "replay_validation.py"
REPLAY_VALIDATION_REMOTE = f"/opt/knlp/{REPLAY_VALIDATION_NAME}"
HOST_GIB = 0.26
CHUNK_TOKENS = 256
KV_BYTES_PER_TOKEN = 28672
SEGMENT_TOKENS = 1024
PROMPT_TOKENS = 3 * SEGMENT_TOKENS
GPU_BLOCKS = 256
NEW_CHUNK_THRESHOLD = 2
FAMILY_REQUESTS = 5
STATIC_PER_FAMILY_ASSIGNMENT = {
    "repo-cluster": "repo-first",
    "tool-cluster": "tool-first",
}
ARMS = ("fixed-repo", "fixed-tool", "fixed-per-family", "joint")
ARM_ORDERS = (
    ("fixed-repo", "fixed-tool", "fixed-per-family", "joint"),
    ("fixed-tool", "fixed-per-family", "joint", "fixed-repo"),
    ("fixed-per-family", "joint", "fixed-repo", "fixed-tool"),
    ("joint", "fixed-repo", "fixed-tool", "fixed-per-family"),
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


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_component(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be one safe path component")
    return value


def _validate_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("knlp_commit must be a full 40-character Git object ID")
    return value


def workload_families() -> list[dict]:
    return [
        {
            "family_id": "repo-cluster",
            "candidates": {
                "repo-first": {
                    "prefix_tokens": SEGMENT_TOKENS,
                    "prefixes": ["repo-a"] * FAMILY_REQUESTS,
                },
                "tool-first": {
                    "prefix_tokens": SEGMENT_TOKENS,
                    "prefixes": [f"tool-a{index}" for index in range(FAMILY_REQUESTS)],
                },
            },
        },
        {
            "family_id": "tool-cluster",
            "candidates": {
                "repo-first": {
                    "prefix_tokens": SEGMENT_TOKENS,
                    "prefixes": [f"repo-z{index}" for index in range(FAMILY_REQUESTS)],
                },
                "tool-first": {
                    "prefix_tokens": SEGMENT_TOKENS,
                    "prefixes": ["tool-z"] * FAMILY_REQUESTS,
                },
            },
        },
    ]


def compile_experiment_plan() -> dict:
    from research.agent_baselines.prefix_plan import compile_prefix_plan

    return compile_prefix_plan(
        workload_families(),
        host_capacity_bytes=int(HOST_GIB * 2**30),
        kv_bytes_per_token=KV_BYTES_PER_TOKEN,
    )


def _arm_assignment(arm: str, plan: dict) -> dict[str, str]:
    if arm == "fixed-repo":
        return {family["family_id"]: "repo-first" for family in workload_families()}
    if arm == "fixed-tool":
        return {family["family_id"]: "tool-first" for family in workload_families()}
    if arm == "fixed-per-family":
        return dict(STATIC_PER_FAMILY_ASSIGNMENT)
    if arm == "joint":
        return dict(plan["selected"]["assignment"])
    raise ValueError(f"Unknown arm: {arm}")


def _trace(out: pathlib.Path, seed: int, arm: str, plan: dict) -> dict[str, Any]:
    """Render the same logical requests with the arm's approved segment order."""
    assignment = _arm_assignment(arm, plan)
    rng = random.Random(seed)

    def tokens() -> list[int]:
        return [rng.randrange(1000, 20000) for _ in range(SEGMENT_TOKENS)]

    repositories = {"repo-a": tokens()}
    repositories.update({f"repo-z{i}": tokens() for i in range(FAMILY_REQUESTS)})
    tools = {"tool-z": tokens()}
    tools.update({f"tool-a{i}": tokens() for i in range(FAMILY_REQUESTS)})
    unique = {
        f"{family}-{index}": tokens()
        for family in ("repo-cluster", "tool-cluster")
        for index in range(FAMILY_REQUESTS)
    }

    logical = []
    for index in range(FAMILY_REQUESTS):
        logical.append(
            {
                "request_id": f"repo-{index}",
                "family_id": "repo-cluster",
                "repo": "repo-a",
                "tool": f"tool-a{index}",
                "unique": f"repo-cluster-{index}",
            }
        )
        logical.append(
            {
                "request_id": f"tool-{index}",
                "family_id": "tool-cluster",
                "repo": f"repo-z{index}",
                "tool": "tool-z",
                "unique": f"tool-cluster-{index}",
            }
        )

    by_id = {request["request_id"]: request for request in logical}

    def render(request: dict) -> list[int]:
        first = assignment[request["family_id"]]
        ordered = (
            (repositories[request["repo"]], tools[request["tool"]])
            if first == "repo-first"
            else (tools[request["tool"]], repositories[request["repo"]])
        )
        return [*ordered[0], *ordered[1], *unique[request["unique"]]]

    sequence = [
        ("sacrificial-filler", [*tokens(), *tokens(), *tokens()], None),
        ("seed-repo", render(by_id["repo-0"]), "repo-cluster"),
        ("seed-tool", render(by_id["tool-0"]), "tool-cluster"),
        ("pressure-trigger", [*tokens(), *tokens(), *tokens()], None),
    ]
    target_ids = []
    for index in range(1, FAMILY_REQUESTS):
        for family in ("repo", "tool"):
            sequence.append(
                (
                    f"evict-{family}-{index}",
                    [*tokens(), *tokens(), *tokens()],
                    None,
                )
            )
            request_id = f"{family}-{index}"
            request = by_id[request_id]
            target_id = f"target-{request_id}"
            sequence.append((target_id, render(request), request["family_id"]))
            target_ids.append(target_id)

    out.mkdir(parents=True)
    files = []
    replay_workload = []
    for order, (instance_id, prompt, family_id) in enumerate(sequence):
        output = [rng.randrange(1000, 20000) for _ in range(8)]
        row = {
            "seq": 1,
            "gap_before_s": 0.0,
            "prompt_shared": 0,
            "prompt_suffix": prompt,
            "prompt_len": len(prompt),
            "output": output,
            "output_len": len(output),
            "recorded_client_elapsed_s": 0.0,
            "prompt_sha256": _json_sha256(prompt),
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
            {
                "order": order,
                "instance_id": instance_id,
                "family_id": family_id,
                "file": filename,
                "steps": 1,
            }
        )
        replay_workload.append(
            {
                "order": order,
                "instance_id": instance_id,
                "prompt_sha256": row["prompt_sha256"],
                "output": output,
            }
        )
    logical_receipt = {
        "families": workload_families(),
        "logical_requests": logical,
        "sequence_roles": [
            "target" if name in target_ids else "pressure" for name, _, _ in sequence
        ],
    }
    manifest = {
        "schema": "knlp.agent_baselines.joint_layout_trace.v1",
        "synthetic": True,
        "seed": seed,
        "arm": arm,
        "assignment": assignment,
        "plan_schema": PREFIX_PLAN_SCHEMA,
        "plan_sha256": _json_sha256(plan),
        "logical_workload_sha256": _json_sha256(logical_receipt),
        "replay_workload_sha256": _json_sha256(replay_workload),
        "target_ids": target_ids,
        "tasks": len(files),
        "steps": len(files),
        "files": files,
    }
    _write_json(out / "trace_manifest.json", manifest)
    return manifest


def _jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _target_receipts(run_dir: pathlib.Path, target_ids: list[str]) -> dict[str, Any]:
    from efficientagent.analysis import common

    requests = _jsonl(run_dir / "replay" / "requests.jsonl")
    by_id = {row["instance_id"]: row for row in requests}
    lookups, _storing, _stored, retrieved = common.parse_lmcache_log(
        run_dir / "server.log"
    )
    receipts = []
    for target_id in target_ids:
        target = by_id[target_id]
        candidates = [
            row
            for row in lookups
            if row[2] == target["prompt_len"]
            and target["start"] - 0.5 <= row[0] <= target["end"] + 0.5
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one lookup for {target_id}, got {len(candidates)}"
            )
        lookup = candidates[0]
        physical = [
            row
            for row in retrieved
            if target["start"] - 0.2 <= row[0] <= target["end"] + 0.5
        ]
        receipts.append(
            {
                "instance_id": target_id,
                "client": target,
                "host_hit_tokens": lookup[3],
                "physical_retrieved_tokens": sum(row[1] for row in physical),
            }
        )
    return {
        "requests": receipts,
        "host_hit_tokens": sum(row["host_hit_tokens"] for row in receipts),
        "physical_retrieved_tokens": sum(
            row["physical_retrieved_tokens"] for row in receipts
        ),
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
    campaign: str,
    repetition: int,
    arm: str,
    knlp_commit: str,
    plan: dict,
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
    if arm not in ARMS or repetition < 1:
        raise ValueError("Invalid arm or repetition")
    if plan.get("schema") != PREFIX_PLAN_SCHEMA:
        raise ValueError("Unexpected planner receipt")
    source_revision = subprocess.check_output(
        ["git", "-C", "/opt/efficientagent", "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != EFFICIENTAGENT_REVISION:
        raise RuntimeError("EfficientAgent source revision mismatch")

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
    initial = {
        "status": "running",
        "campaign": campaign,
        "run_id": run_id,
        "repetition": repetition,
        "arm": arm,
        "assignment": _arm_assignment(arm, plan),
        "plan": plan,
        "started_at": time.time(),
        "source": {
            "knlp_harness_commit": knlp_commit,
            "harness_sha256": _sha256(__file__),
            "replay_validation_sha256": _sha256(REPLAY_VALIDATION_REMOTE),
            "counter_patch_sha256": _sha256(PATCH_REMOTE),
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
        "measurement_boundary": "Replay only; no post-replay greedy probes.",
    }
    _write_json(os.path.join(local_dir, "result.json"), initial)
    _commit(local_dir, remote_dir)

    trace_dir = pathlib.Path(local_dir, "trace")
    replay_dir = pathlib.Path(local_dir, "run")
    try:
        manifest = _trace(trace_dir, 20_000 + repetition, arm, plan)
        model_path = snapshot_download(
            repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True
        )
        arguments = [
            "--admission",
            "conditioned",
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
            str(NEW_CHUNK_THRESHOLD),
            "--occupancy-threshold",
            "0.95",
            "--chunk-tokens",
            str(CHUNK_TOKENS),
            "--window-s",
            "120",
            "--telemetry-max-age-s",
            "5",
            "--max-seconds",
            "300",
            "--tmpdir",
            f"/tmp/j{repetition}{arm[-1]}",
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
        targets = _target_receipts(replay_dir, manifest["target_ids"])
        expected = next(
            score["estimated_reuse_tokens"]
            for score in plan["joint_candidates"]
            if score["assignment"] == _arm_assignment(arm, plan)
        )
        checks = {
            **replay_validation["checks"],
            "aggregate_decisions_survived": counters.get("scheduler.decisions", 0)
            == manifest["steps"],
            "conditioned_pressure_observed": counters.get(
                "scheduler.pressure_requests", 0
            )
            > 0,
            "conditioned_skip_observed": counters.get("scheduler.skipped_requests", 0)
            > 0,
            "host_evictions_observed": counters.get("worker.cpu_evicted_chunks", 0) > 0,
            "physical_reuse_meets_structural_estimate": targets[
                "physical_retrieved_tokens"
            ]
            >= expected,
            "no_post_replay_probe": "greedy_check_requests" not in run,
        }
        result = {
            **initial,
            "status": "passed" if all(checks.values()) else "failed",
            "finished_at": time.time(),
            "checks": checks,
            "replay_validation": replay_validation,
            "trace_manifest_sha256": _sha256(trace_dir / "trace_manifest.json"),
            "logical_workload_sha256": manifest["logical_workload_sha256"],
            "replay_workload_sha256": manifest["replay_workload_sha256"],
            "estimated_reuse_tokens": expected,
            "launcher_arguments": arguments,
            "kvtier_counters": counters,
            "targets": targets,
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
            raise RuntimeError(f"Joint-layout checks failed: {checks}")
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


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800, volumes={"/runs": runs})
def summarize(
    campaign: str, results: list[dict[str, Any]], knlp_commit: str, plan: dict
) -> dict[str, Any]:
    grouped = {arm: [row for row in results if row["arm"] == arm] for arm in ARMS}
    medians = {}
    for arm, rows in grouped.items():
        medians[arm] = {
            metric: statistics.median(row["metrics"][metric] for row in rows)
            for metric in (
                "prefill_computed_tokens",
                "retrieved_per_rank",
                "stored_per_rank",
                "makespan_s",
            )
        }
        medians[arm]["target_retrieved_tokens"] = statistics.median(
            row["targets"]["physical_retrieved_tokens"] for row in rows
        )
    repetitions = sorted({row["repetition"] for row in results})
    gates = {
        "all_arms_passed": all(row.get("status") == "passed" for row in results),
        "balanced_repetitions": all(
            len(grouped[arm]) == len(repetitions) for arm in ARMS
        ),
        "logical_workload_identical_within_repetition": all(
            len(
                {
                    row["logical_workload_sha256"]
                    for row in results
                    if row["repetition"] == repetition
                }
            )
            == 1
            for repetition in repetitions
        ),
        "static_control_matches_selected_assignment": (
            plan["selected"]["assignment"] == STATIC_PER_FAMILY_ASSIGNMENT
        ),
        "static_control_replay_identical_to_planner_each_repeat": all(
            next(
                row
                for row in grouped["fixed-per-family"]
                if row["repetition"] == repetition
            )["replay_workload_sha256"]
            == next(row for row in grouped["joint"] if row["repetition"] == repetition)[
                "replay_workload_sha256"
            ]
            for repetition in repetitions
        ),
        "static_control_prefill_equals_planner_each_repeat": all(
            next(
                row
                for row in grouped["fixed-per-family"]
                if row["repetition"] == repetition
            )["metrics"]["prefill_computed_tokens"]
            == next(row for row in grouped["joint"] if row["repetition"] == repetition)[
                "metrics"
            ]["prefill_computed_tokens"]
            for repetition in repetitions
        ),
        "static_control_restoration_equals_planner_each_repeat": all(
            next(
                row
                for row in grouped["fixed-per-family"]
                if row["repetition"] == repetition
            )["targets"]["physical_retrieved_tokens"]
            == next(row for row in grouped["joint"] if row["repetition"] == repetition)[
                "targets"
            ]["physical_retrieved_tokens"]
            for repetition in repetitions
        ),
        "joint_recomputed_less_than_both_fixed_each_repeat": all(
            next(row for row in grouped["joint"] if row["repetition"] == repetition)[
                "metrics"
            ]["prefill_computed_tokens"]
            < min(
                next(row for row in grouped[arm] if row["repetition"] == repetition)[
                    "metrics"
                ]["prefill_computed_tokens"]
                for arm in ("fixed-repo", "fixed-tool")
            )
            for repetition in repetitions
        ),
        "joint_restored_more_than_both_fixed_each_repeat": all(
            next(row for row in grouped["joint"] if row["repetition"] == repetition)[
                "targets"
            ]["physical_retrieved_tokens"]
            > max(
                next(row for row in grouped[arm] if row["repetition"] == repetition)[
                    "targets"
                ]["physical_retrieved_tokens"]
                for arm in ("fixed-repo", "fixed-tool")
            )
            for repetition in repetitions
        ),
    }
    summary = {
        "status": "passed" if all(gates.values()) else "completed_with_issues",
        "campaign": campaign,
        "knlp_harness_commit": knlp_commit,
        "finished_at": time.time(),
        "arm_orders": [list(order) for order in ARM_ORDERS],
        "plan": plan,
        "gates": gates,
        "medians": medians,
        "results": [
            {
                "run_id": row["run_id"],
                "repetition": row["repetition"],
                "arm": row["arm"],
                "assignment": row["assignment"],
                "estimated_reuse_tokens": row["estimated_reuse_tokens"],
                "replay_workload_sha256": row["replay_workload_sha256"],
                "metrics": row["metrics"],
                "targets": row["targets"],
                "kvtier_counters": row["kvtier_counters"],
            }
            for row in results
        ],
        "claim_boundary": (
            "Synthetic fixed-output mechanism comparison only. The static "
            "per-family control intentionally emits the planner-selected workload, "
            "so equality cannot establish a compiler benefit. No live-agent quality, "
            "latency, or general PrefixPlan benefit is established."
        ),
    }
    local_dir = f"/tmp/{campaign}-summary"
    os.makedirs(local_dir, exist_ok=True)
    _write_json(os.path.join(local_dir, "SUMMARY.json"), summary)
    _commit(local_dir, os.path.join(REMOTE_ROOT, campaign))
    return summary


@app.local_entrypoint()
def main(campaign: str, knlp_commit: str, repetitions: int = 1):
    from research.agent_baselines.prefix_plan import validate_prefix_plan

    _safe_component(campaign, "campaign")
    _validate_commit(knlp_commit)
    if not 1 <= repetitions <= len(ARM_ORDERS):
        raise ValueError(f"repetitions must be between 1 and {len(ARM_ORDERS)}")
    plan = validate_prefix_plan(compile_experiment_plan())
    print("PREFIX_PLAN " + json.dumps(plan, sort_keys=True), flush=True)
    fetched = fetch_model.remote()
    print("MODEL_FETCH " + json.dumps(fetched, sort_keys=True), flush=True)
    results = []
    for repetition in range(1, repetitions + 1):
        for arm in ARM_ORDERS[repetition - 1]:
            result = run_arm.remote(campaign, repetition, arm, knlp_commit, plan)
            results.append(result)
            print("ARM_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    summary = summarize.remote(campaign, results, knlp_commit, plan)
    print("CAMPAIGN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
