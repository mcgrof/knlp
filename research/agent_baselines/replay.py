# SPDX-License-Identifier: MIT
"""Pinned EfficientAgent trace, CPU profile, and replay adapter.

Every command prints its plan unless --execute is supplied. Outputs must be
new directories. The GPU replay uses an externally installed reference stack;
it does not provision hardware or install dependencies.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

EFFICIENTAGENT_REVISION = "b21b12a07a3ccf473c82acf00820b1aa2073044b"
EFFICIENTAGENT_URL = "https://github.com/KunmingSHAO/efficientagent_release"
TRACE_SCHEMA = "efficientagent.replay_trace.v1"
STACK_VERSIONS = {"vllm": "0.13.0", "lmcache": "0.3.12"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def token_ids(value, label: str, *, allow_empty: bool = False) -> list[int]:
    require(isinstance(value, list), f"{label}: expected a token-ID list")
    require(allow_empty or bool(value), f"{label}: empty token sequence")
    require(
        all(type(x) is int and x >= 0 for x in value),
        f"{label}: token IDs must be nonnegative integers",
    )
    return value


def finite_number(value, label: str, minimum: float = 0) -> float:
    require(
        type(value) in (float, int) and math.isfinite(value) and value >= minimum,
        f"{label}: expected a finite number >= {minimum}",
    )
    return value


def verify_checkout(path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    revision = git("rev-parse", "HEAD")
    require(
        revision == EFFICIENTAGENT_REVISION, "EfficientAgent checkout revision mismatch"
    )
    require(
        not git(
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
        "EfficientAgent source has modified or untracked files",
    )
    require(
        (path / "efficientagent/replay/launch.py").is_file(),
        "missing upstream launcher",
    )
    return {"path": str(path), "revision": revision, "url": EFFICIENTAGENT_URL}


def inspect_source(run: Path, limit: int | None = None) -> dict:
    """Reject unverified IDs instead of accepting upstream's missing=true default."""
    tasks = []
    for path in (run / "tasks").glob("*.json"):
        record = read_json(path)
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.stem) is not None,
            "unsafe task ID",
        )
        require(
            record.get("recording_complete", True) is True, "incomplete task recording"
        )
        require(
            record.get("capture_errors", 0) == 0, "task has exact token capture errors"
        )
        timing = record["task_timing"]
        assigned = finite_number(timing["assigned_ns"], "assigned_ns")
        finished = finite_number(timing["finished_ns"], "finished_ns", assigned)
        tasks.append((assigned, path.stem, path, record, finished))
    tasks.sort(key=lambda item: item[:2])
    if limit is not None:
        tasks = tasks[:limit]
    require(bool(tasks), "source run contains no task records")
    files, good, failed = {}, 0, 0
    for assigned, task_id, path, record, finished in tasks:
        attempt = record.get("attempt_index", 0)
        require(type(attempt) is int and attempt >= 0, "invalid attempt_index")
        trial = record.get("trial_id")
        require(isinstance(trial, str) and bool(trial), "missing trial_id")
        log = (
            run
            / "attempts"
            / task_id
            / f"attempt_{attempt:02d}"
            / "telemetry/llm_requests.jsonl"
        )
        rows = [
            json.loads(line) for line in log.read_text().splitlines() if line.strip()
        ]
        require(bool(rows), f"{task_id}: no recorded calls")
        sequences = [row["request_sequence"] for row in rows]
        require(
            all(type(x) is int and x > 0 for x in sequences)
            and len(set(sequences)) == len(sequences),
            f"{task_id}: invalid/duplicate request sequence",
        )
        previous_end, task_good = assigned, 0
        for row in sorted(rows, key=lambda item: item["request_sequence"]):
            require(row.get("trial_id") == trial, f"{task_id}: trial mismatch")
            start = finite_number(row["start_wall_ns"], "start_wall_ns", previous_end)
            end = finite_number(row["end_wall_ns"], "end_wall_ns", start)
            require(end <= finished, f"{task_id}: call ends after task completion")
            finite_number(row["client_elapsed_ns"], "client_elapsed_ns")
            previous_end = end
            if row.get("error") is not None:
                require(
                    "TokenCaptureError" not in str(row["error"]),
                    f"{task_id}: exact token capture failed; re-record the run",
                )
                failed += 1
                continue
            require(
                row.get("token_identity_verified") is True,
                f"{task_id}: unverified token identity",
            )
            token_ids(row.get("prompt_token_ids"), "prompt_token_ids")
            outputs = row.get("generated_token_ids")
            require(
                isinstance(outputs, list) and len(outputs) == 1,
                "exactly one generated sequence required",
            )
            token_ids(outputs[0], "generated_token_ids[0]")
            task_good += 1
        require(task_good > 0, f"{task_id}: no replayable calls")
        good += task_good
        for source in (path, log):
            files[str(source.relative_to(run))] = digest(source)
    return {
        "tasks": len(tasks),
        "steps": good,
        "failed_calls_omitted": failed,
        "files_sha256": files,
    }


def inspect_trace(trace: Path, limit: int | None = None) -> dict:
    """Validate exact replay inputs, including delta reconstruction and checksums."""
    manifest = read_json(trace / "trace_manifest.json")
    require(manifest.get("schema") == TRACE_SCHEMA, "unsupported trace schema")
    entries = manifest.get("files", [])
    require(bool(entries), "trace contains no tasks")
    require(manifest.get("tasks") == len(entries), "manifest task count mismatch")
    require(
        manifest.get("steps") == sum(entry["steps"] for entry in entries),
        "manifest step count mismatch",
    )
    names = [entry["file"] for entry in entries]
    require(len(names) == len(set(names)), "duplicate trace files")
    require(
        names == sorted(names),
        "manifest order differs from upstream sorted-file dispatch",
    )
    require(
        set(names) == {p.name for p in trace.glob("*.jsonl.gz")},
        "trace files differ from manifest",
    )
    selected = entries if limit is None else entries[:limit]
    checksums, total, max_context, ids = {}, 0, 0, set()
    for order, entry in enumerate(selected):
        name = entry["file"]
        require(Path(name).name == name, "unsafe trace filename")
        path = trace / name
        require(
            path.resolve().parent == trace.resolve(),
            "trace symlinks outside trace directory are unsupported",
        )
        checksums[name] = digest(path)
        if "sha256" in entry:
            require(checksums[name] == entry["sha256"], f"{name}: checksum mismatch")
        with gzip.open(path, "rt") as stream:
            head = json.loads(stream.readline())
            require(head["order"] == order == entry["order"], "trace order mismatch")
            require(head["instance_id"] == entry["instance_id"], "task ID mismatch")
            require(head["instance_id"] not in ids, "duplicate trace task ID")
            ids.add(head["instance_id"])
            finite_number(head.get("tail_gap_s", 0), "tail_gap_s")
            prompt, count, last_sequence = [], 0, 0
            for line in stream:
                row = json.loads(line)
                shared = row["prompt_shared"]
                require(
                    type(shared) is int and 0 <= shared <= len(prompt),
                    "invalid shared prefix length",
                )
                suffix = token_ids(
                    row["prompt_suffix"], "prompt_suffix", allow_empty=True
                )
                prompt = prompt[:shared] + suffix
                token_ids(prompt, "reconstructed prompt")
                output = token_ids(row["output"], "forced output")
                require(len(prompt) == row["prompt_len"], "prompt length mismatch")
                require(len(output) == row["output_len"], "output length mismatch")
                expected = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                require(
                    row["prompt_sha256"] == expected, "prompt token digest mismatch"
                )
                finite_number(row["gap_before_s"], "gap_before_s")
                seq = row["seq"]
                require(
                    type(seq) is int and seq > last_sequence,
                    "non-increasing call sequence",
                )
                last_sequence = seq
                count += 1
                max_context = max(max_context, len(prompt) + len(output))
            require(count > 0 and count == entry["steps"], "trace step count mismatch")
            require(count == head["calls_replayed"], "trace header step count mismatch")
            total += count
    origin_file = trace / "adapter_trace.json"
    origin = read_json(origin_file) if origin_file.exists() else {}
    kind = (
        "synthetic"
        if manifest.get("synthetic")
        else origin.get("data_kind", "external-unverified-origin")
    )
    return {
        "path": str(trace),
        "tasks": len(selected),
        "steps": total,
        "max_context_tokens": max_context,
        "data_kind": kind,
        "manifest_sha256": digest(trace / "trace_manifest.json"),
        "files_sha256": checksums,
        "server_provenance": origin.get("server_provenance"),
    }


def validate_replay_result(run: Path, expected: dict) -> dict:
    """Upstream 'passed' alone does not establish complete, token-correct replay."""
    result = read_json(run / "run.json")
    replay = read_json(run / "replay/replay_summary.json")
    require(result.get("status") == "passed", "upstream replay did not pass")
    require(
        replay.get("interrupted") is False, "replay was interrupted or dispatch-capped"
    )
    require(
        replay.get("requests") == expected["steps"],
        "not all expected calls were replayed",
    )
    require(replay.get("tasks") == expected["tasks"], "replay task count mismatch")
    for field in ("errors", "forced_mismatch", "prompt_len_mismatch"):
        require(replay.get(field) == 0, f"replay has {field}")
    require(
        replay.get("forced_match") == expected["steps"],
        "not all output tokens were verified",
    )
    task_file = run / "replay/tasks.jsonl"
    tasks = [
        json.loads(line) for line in task_file.read_text().splitlines() if line.strip()
    ]
    require(
        len(tasks) == expected["tasks"] and all(t.get("ok") is True for t in tasks),
        "incomplete task completions",
    )
    require(
        len({t["instance_id"] for t in tasks}) == len(tasks),
        "duplicate completed tasks",
    )
    return {
        "status": "complete-token-replay",
        "task_quality_validated": False,
        "replay": replay,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build-trace", "profile", "replay", "grid"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkout", type=Path, required=True)
        sub.add_argument("--python", default=sys.executable)
        sub.add_argument(
            "--out", type=Path, required=True, help="new adapter output directory"
        )
        sub.add_argument(
            "--execute", action="store_true", help="run command (grid: save plan only)"
        )
        sub.add_argument(
            "--wall-timeout",
            type=float,
            default=3600,
            help="adapter wall-clock limit in seconds",
        )
        if name == "build-trace":
            sub.add_argument("--source-run", type=Path, required=True)
            sub.add_argument(
                "--data-kind", choices=("recorded-agent", "synthetic"), required=True
            )
            sub.add_argument("--limit", type=int)
        else:
            sub.add_argument("--trace", type=Path, required=True)
        if name in ("replay", "grid"):
            sub.add_argument(
                "--model",
                type=Path,
                required=True,
                help="local model directory with config.json",
            )
            sub.add_argument(
                "--model-revision",
                help="model revision receipt; user supplied, not independently verified",
            )
            sub.add_argument(
                "--tokenizer-revision",
                help="tokenizer revision receipt matching recorded-agent traces",
            )
            sub.add_argument(
                "--dtype",
                choices=("bfloat16", "float16"),
                default="bfloat16",
                help="explicit model and resolved KV dtype for matched reference runs",
            )
            sub.add_argument("--tp", type=int, default=1)
            sub.add_argument("--workers", type=int, default=1)
            sub.add_argument("--max-num-seqs", type=int, default=1)
            sub.add_argument("--max-model-len", type=int, required=True)
            sub.add_argument("--gpu-memory-utilization", type=float, default=0.8)
            sub.add_argument(
                "--kv-bytes-per-token",
                type=int,
                required=True,
                help="KV bytes per token per TP rank; derive for the actual model/dtype/TP",
            )
            sub.add_argument("--chunk-tokens", type=int, default=1024)
            sub.add_argument("--limit", type=int)
            sub.add_argument(
                "--max-seconds",
                type=float,
                default=600,
                help="upstream dispatch cap; partial runs fail adapter validation",
            )
            sub.add_argument("--startup-timeout", type=float, default=600)
            sub.add_argument("--port", type=int, default=8000)
            sub.add_argument("--write-threshold", type=int, default=8)
            sub.add_argument("--occupancy-threshold", type=float, default=0.95)
            sub.add_argument("--greedy-check", type=int, default=1)
        if name == "replay":
            sub.add_argument(
                "--host-gib",
                type=float,
                required=True,
                help="CPU tier GiB per TP rank; 0 = recompute with GPU APC",
            )
            sub.add_argument(
                "--admission", choices=("none", "fixed", "conditioned"), default="none"
            )
        if name == "grid":
            sub.add_argument(
                "--host-capacities",
                type=float,
                nargs="+",
                required=True,
                help="positive GiB per rank; includes one host0 reference automatically",
            )
            sub.add_argument("--worker-counts", type=int, nargs="+", required=True)
    return parser


def replay_args(
    a, out: Path, workers: int, capacity: float, admission: str
) -> list[str]:
    argv = [
        "--admission",
        admission,
        "--host-gib",
        str(capacity),
        "--model",
        str(a.model.resolve()),
        "--tp",
        str(a.tp),
        "--trace",
        str(a.trace.resolve()),
        "--out",
        str(out),
        "--workers",
        str(workers),
        "--max-num-seqs",
        str(a.max_num_seqs),
        "--max-model-len",
        str(a.max_model_len),
        "--gpu-memory-utilization",
        str(a.gpu_memory_utilization),
        "--kv-bytes-per-token",
        str(a.kv_bytes_per_token),
        "--chunk-tokens",
        str(a.chunk_tokens),
        "--max-seconds",
        str(a.max_seconds),
        "--startup-timeout",
        str(a.startup_timeout),
        "--port",
        str(a.port),
        "--write-threshold",
        str(a.write_threshold),
        "--occupancy-threshold",
        str(a.occupancy_threshold),
        "--greedy-check",
        str(a.greedy_check),
    ]
    if a.limit is not None:
        argv += ["--limit", str(a.limit)]
    return argv


def prepare(a) -> dict:
    a.checkout, a.out = a.checkout.resolve(), a.out.resolve()
    require(not a.out.exists(), f"output already exists: {a.out}")
    python = shutil.which(a.python)
    require(python is not None, f"Python interpreter not found: {a.python}")
    a.python = os.path.abspath(python)
    finite_number(a.wall_timeout, "wall-timeout", 1)
    if getattr(a, "limit", None) is not None:
        require(a.limit > 0, "limit must be positive")
    plan = {
        "schema": "knlp.agent_baselines.efficientagent.v1",
        "command": a.command,
        "upstream": verify_checkout(a.checkout),
        "python": a.python,
        "output": str(a.out),
        "adapter_sha256": digest(Path(__file__)),
        "scope": "reference tooling; not a reproduction result or SWE-bench quality score",
    }
    base = [a.python, "-m"]
    if a.command == "build-trace":
        a.source_run = a.source_run.resolve()
        plan["source"] = inspect_source(a.source_run, a.limit)
        plan["data_kind"] = a.data_kind
        if a.data_kind == "recorded-agent":
            source_manifest = read_json(a.source_run / "manifest.json")
            require(
                source_manifest.get("capture_errors", 0) == 0,
                "source run has exact token capture errors; re-record the run",
            )
            provenance = source_manifest.get("server_provenance", {})
            for field in (
                "model_id",
                "model_revision",
                "tokenizer_revision",
                "kv_cache_dtype",
                "prefix_caching",
            ):
                require(
                    field in provenance,
                    f"recorded-agent source missing server_provenance.{field}",
                )
            for field in (
                "model_id",
                "model_revision",
                "tokenizer_revision",
                "kv_cache_dtype",
            ):
                require(
                    isinstance(provenance[field], str) and bool(provenance[field]),
                    f"invalid server_provenance.{field}",
                )
            require(
                type(provenance["prefix_caching"]) is bool,
                "prefix_caching must be a boolean",
            )
            plan["server_provenance"] = provenance
            plan["source_manifest_sha256"] = digest(a.source_run / "manifest.json")
        argv = base + [
            "efficientagent.replay.build_trace",
            "--run-dir",
            str(a.source_run),
            "--out",
            str(a.out / "trace"),
        ]
        if a.limit is not None:
            argv += ["--limit", str(a.limit)]
    else:
        a.trace = a.trace.resolve()
        plan["trace"] = inspect_trace(a.trace, getattr(a, "limit", None))
        argv = base + [
            "efficientagent.analysis.trace_profile",
            "--trace",
            str(a.trace),
            "--out",
            str(a.out / "profile.json"),
        ]
    if a.command in ("replay", "grid"):
        for field in (
            "tp",
            "workers",
            "max_num_seqs",
            "max_model_len",
            "kv_bytes_per_token",
            "chunk_tokens",
        ):
            require(getattr(a, field) > 0, f"{field} must be positive")
        require(
            a.greedy_check >= 0 and a.write_threshold >= 0,
            "negative greedy-check or write threshold",
        )
        require(1 <= a.port <= 65535, "invalid server port")
        for field in ("gpu_memory_utilization", "occupancy_threshold"):
            require(0 < getattr(a, field) <= 1, f"{field} must be in (0, 1]")
        finite_number(a.max_seconds, "max-seconds", 1)
        finite_number(a.startup_timeout, "startup-timeout", 1)
        require(
            a.max_model_len >= plan["trace"]["max_context_tokens"],
            "max-model-len is shorter than a recorded prompt plus output",
        )
        a.model = a.model.resolve()
        require(
            (a.model / "config.json").is_file(),
            "model must be a local directory with config.json",
        )
        plan["model"] = {
            "path": str(a.model),
            "config_sha256": digest(a.model / "config.json"),
            "revision_label": a.model_revision,
            "tokenizer_revision_label": a.tokenizer_revision,
            "dtype": a.dtype,
            "resolved_kv_cache_dtype": a.dtype,
        }
        provenance = plan["trace"].get("server_provenance")
        if plan["trace"]["data_kind"] == "recorded-agent":
            require(bool(provenance), "recorded-agent trace lacks server provenance")
            require(
                a.model_revision == provenance["model_revision"],
                "--model-revision must match the recorded server model_revision",
            )
            require(
                a.tokenizer_revision == provenance["tokenizer_revision"],
                "--tokenizer-revision must match the recorded server tokenizer_revision",
            )
            require(
                provenance.get("kv_cache_dtype") == a.dtype,
                "recorded kv_cache_dtype must equal --dtype; record the resolved "
                "bfloat16/float16 dtype, not 'auto' (FP8 is outside this reference adapter)",
            )
        plan["required_stack"] = STACK_VERSIONS
        plan["hardware_support"] = (
            "upstream documents NVIDIA; ROCm serving is unvalidated"
        )
    if a.command == "replay":
        finite_number(a.host_gib, "host-gib")
        require(
            a.host_gib > 0 or a.admission == "none", "host0 requires admission=none"
        )
        require(
            round(a.host_gib, 1) == a.host_gib,
            "host-gib must use 0.1 GiB increments (upstream serialization precision)",
        )
        tmpdir = Path("/tmp") / (
            "knlp-ea-" + hashlib.sha256(str(a.out).encode()).hexdigest()[:12]
        )
        require(
            not tmpdir.exists(), f"temporary socket directory already exists: {tmpdir}"
        )
        argv = base + [
            "efficientagent.replay.launch",
            *replay_args(a, a.out / "run", a.workers, a.host_gib, a.admission),
            "--python",
            a.python,
            "--tmpdir",
            str(tmpdir),
            f"--server-arg=--dtype={a.dtype}",
            "--server-arg=--kv-cache-dtype=auto",
            "--execute",
        ]
        plan["host_allocation_gib"] = a.host_gib * a.tp
    if a.command == "grid":
        capacities = sorted(set(a.host_capacities))
        workers = sorted(set(a.worker_counts))
        require(all(x > 0 for x in workers), "worker-counts must be positive")
        for value in capacities:
            finite_number(value, "host-capacity", 0.1)
            require(
                round(value, 1) == value, "host capacities must use 0.1 GiB increments"
            )
        plan["arms"] = []
        for worker_count in workers:
            arms = [(0.0, "none")] + [
                (capacity, admission)
                for capacity in capacities
                for admission in ("none", "fixed", "conditioned")
            ]
            for capacity, admission in arms:
                name = f"workers{worker_count}_host{capacity:g}_{admission}"
                arm_out = a.out / "runs" / name
                command = [
                    sys.executable,
                    "-m",
                    "research.agent_baselines.replay",
                    "replay",
                    "--checkout",
                    str(a.checkout),
                    "--python",
                    a.python,
                    *replay_args(a, arm_out, worker_count, capacity, admission),
                    "--wall-timeout",
                    str(a.wall_timeout),
                    "--dtype",
                    a.dtype,
                ]
                if a.model_revision:
                    command += ["--model-revision", a.model_revision]
                if a.tokenizer_revision:
                    command += ["--tokenizer-revision", a.tokenizer_revision]
                plan["arms"].append(
                    {
                        "name": name,
                        "argv": command,
                        "host_allocation_gib": capacity * a.tp,
                    }
                )
        plan["execute_meaning"] = "save this plan only; each arm still needs --execute"
    else:
        plan["argv"] = argv
    return plan


def check_stack(python: str, env: dict, cwd: Path) -> dict:
    code = "import importlib.metadata,json; print(json.dumps({name:importlib.metadata.version(name) for name in ['vllm','lmcache']}))"
    result = subprocess.run(
        [python, "-c", code],
        env=env,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    versions = json.loads(result.stdout)
    require(versions == STACK_VERSIONS, f"reference serving stack mismatch: {versions}")
    return versions


def execute(a, plan: dict) -> int:
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out / "adapter_manifest.json", plan)
    if a.command == "grid":
        write_json(a.out / "grid.json", plan)
        return 0
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(a.checkout), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1"
    )
    status = {"started_unix_s": time.time(), "status": "failed"}
    process = None
    try:
        if a.command == "replay":
            status["stack_versions"] = check_stack(a.python, env, a.checkout)
        with (a.out / "adapter.log").open("x") as log:
            process = subprocess.Popen(
                plan["argv"],
                cwd=a.checkout,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            status["returncode"] = process.wait(timeout=a.wall_timeout)
        require(
            status["returncode"] == 0, "upstream command failed; inspect adapter.log"
        )
        if a.command == "replay":
            status["validation"] = validate_replay_result(a.out / "run", plan["trace"])
        elif a.command == "build-trace":
            write_json(
                a.out / "trace/adapter_trace.json",
                {
                    "data_kind": a.data_kind,
                    "source": plan["source"],
                    "upstream_revision": EFFICIENTAGENT_REVISION,
                    "server_provenance": plan.get("server_provenance"),
                },
            )
            status["validation"] = inspect_trace(a.out / "trace")
            require(
                status["validation"]["steps"] == plan["source"]["steps"],
                "trace dropped successful source calls",
            )
        else:
            result = read_json(a.out / "profile.json")
            require(
                result["profile"]["calls"] == plan["trace"]["steps"],
                "profile call count differs from trace",
            )
        status["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        status["error"] = str(error) or type(error).__name__
    finally:
        if process is not None and process.poll() is None:
            # The upstream launcher handles TERM by stopping its separate server group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    status["cleanup_warning"] = (
                        "upstream launcher did not exit gracefully; inspect GPU processes"
                    )
            except ProcessLookupError:
                process.wait()
        status["finished_unix_s"] = time.time()
        write_json(a.out / "adapter_result.json", status)
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "passed" else 1


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = prepare(args)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    if not args.execute:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
