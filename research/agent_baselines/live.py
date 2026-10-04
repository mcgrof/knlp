# SPDX-License-Identifier: MIT
"""Run paired mini-SWE-agent workloads against an existing local server."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

MINI_REVISION = "04d809ceab9df28f9adaed044884180159172930"
SWEBENCH_REVISION = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"

FATAL_ENVIRONMENT_ERRORS = (
    (
        "container_runtime_no_space",
        re.compile(
            r"OCI runtime exec failed:.*(?:no space left on device|runc-process)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

TERMINAL_CAPTURE_SCHEMA = "knlp.agent_baselines.terminal_capture.v2"
TERMINAL_CAPTURE_COMMANDS = {
    "status": "git status --short --untracked-files=all",
    "changed_paths": "git diff --name-only -- .",
    "untracked_paths": "git ls-files --others --exclude-standard",
    "source_diff": "git diff --no-ext-diff --binary -- .",
    "staged_paths": "git diff --cached --name-only -- .",
    "staged_diff": "git diff --cached --no-ext-diff --binary -- .",
}
VERIFICATION_COMMAND = re.compile(
    r"(?:^|[;&|\n]\s*|\b)(?:pytest|tox|nox|make\s+(?:test|check)|"
    r"python\S*\s+-m\s+(?:pytest|unittest|compileall)|"
    r"(?:npm|pnpm|yarn)\s+test|cargo\s+test|go\s+test|assert\b|"
    r"git\s+diff\s+--check)",
    re.IGNORECASE,
)


class TerminalCaptureError(RuntimeError):
    """A terminal-state receipt is present but incomplete or invalid."""


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_checkout(path: Path, revision: str) -> Path:
    path = path.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if head != revision or dirty:
        raise ValueError(f"Expected clean checkout at {revision}: {path}")
    return path


def code_provenance() -> dict:
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    harness = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "research/agent_baselines",
        ],
        text=True,
    )
    return {"knlp_revision": revision, "knlp_dirty": bool(tracked or harness)}


def stable_prefix(template: str) -> str:
    """Move intact stable instructions ahead of the varying task block."""
    from .quality_profiles import render_instance_template

    return render_instance_template(template, "stable-prefix")


def endpoint(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Endpoint must be HTTP(S), without credentials/query/fragment")
    return value.rstrip("/")


def read_instances(path: Path) -> tuple[dict, list[dict]]:
    return _validate_selection(json.loads(path.read_text()))


def fatal_environment_error(messages: list[dict]) -> str | None:
    """Return a stable reason for a fatal tool-environment observation."""
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        for reason, pattern in FATAL_ENVIRONMENT_ERRORS:
            if pattern.search(content):
                return reason
    return None


def command_observations(messages: list[dict]) -> list[dict]:
    """Pair recorded tool observations with their originating commands."""
    commands = {}
    observations = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for action in message.get("extra", {}).get("actions", []):
                tool_call_id = action.get("tool_call_id")
                command = action.get("command")
                if isinstance(tool_call_id, str) and isinstance(command, str):
                    commands[tool_call_id] = command
        elif message.get("role") == "tool":
            content = message.get("content", "")
            if not isinstance(content, str):
                content = ""
            tool_call_id = message.get("tool_call_id")
            observations.append(
                {
                    "message_index": index,
                    "tool_call_id": tool_call_id,
                    "command": commands.get(tool_call_id),
                    "content": content,
                }
            )
    return observations


def capture_terminal_state(env, out: Path, iid: str, result: dict, messages) -> None:
    """Capture final worktree evidence without returning it to the model."""
    observations = command_observations(messages)
    verification = next(
        (
            item
            for item in reversed(observations)
            if isinstance(item["command"], str)
            and VERIFICATION_COMMAND.search(item["command"])
        ),
        None,
    )
    capture = {
        "schema": TERMINAL_CAPTURE_SCHEMA,
        "instance_id": iid,
        "exit_status": result["exit_status"],
        "submitted": result["exit_status"] == "Submitted",
        "submission_characters": len(result["model_patch"]),
        "submission_sha256": (
            hashlib.sha256(result["model_patch"].encode()).hexdigest()
            if result["model_patch"]
            else None
        ),
        "last_command_result": observations[-1] if observations else None,
        "last_verification_result": verification,
        "verification_selector": VERIFICATION_COMMAND.pattern,
        "command_scopes": {
            "changed_paths": "unstaged tracked paths",
            "source_diff": "unstaged tracked changes",
            "staged_paths": "staged tracked paths",
            "staged_diff": "staged tracked changes",
            "untracked_paths": "untracked path names only; contents are not captured",
        },
        "commands": {},
        "failures": [],
    }
    for name, command in TERMINAL_CAPTURE_COMMANDS.items():
        try:
            output = env.execute({"command": command}, timeout=60)
        except Exception as exc:
            capture["commands"][name] = {
                "returncode": None,
                "output": None,
                "exception_info": None,
                "error_type": type(exc).__name__,
            }
            capture["failures"].append(
                {"command": name, "reasons": ["execute_exception"]}
            )
            continue
        if not isinstance(output, dict):
            capture["commands"][name] = {
                "returncode": None,
                "output": None,
                "exception_info": None,
            }
            capture["failures"].append({"command": name, "reasons": ["invalid_result"]})
            continue
        returncode = output.get("returncode")
        command_output = output.get("output")
        exception_info = output.get("exception_info")
        receipt = {
            "returncode": returncode if type(returncode) is int else None,
            "output": command_output if isinstance(command_output, str) else None,
            "exception_info": (
                exception_info if isinstance(exception_info, str) else None
            ),
        }
        capture["commands"][name] = receipt
        reasons = []
        for key in receipt:
            if key not in output:
                reasons.append(f"missing_{key}")
        if type(returncode) is not int:
            reasons.append("invalid_returncode")
        elif returncode != 0:
            reasons.append("nonzero_returncode")
        if not isinstance(command_output, str):
            reasons.append("invalid_output")
        if not isinstance(exception_info, str):
            reasons.append("invalid_exception_info")
        elif exception_info:
            reasons.append("exception_info_present")
        if reasons:
            capture["failures"].append({"command": name, "reasons": reasons})
    capture["capture_status"] = "failed" if capture["failures"] else "complete"
    destination = out / "terminal_captures" / f"{iid}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, capture)
    if capture["failures"]:
        raise TerminalCaptureError("Terminal capture contains failed commands")


def _validate_selection(data: dict) -> tuple[dict, list[dict]]:
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("instances"), list)
        or not data["instances"]
    ):
        raise ValueError("Expected a nonempty frozen selection made by select")
    ids = []
    for row in data["instances"]:
        if not isinstance(row, dict):
            raise ValueError("Each frozen instance must be an object")
        iid = row.get("instance_id", "")
        if not isinstance(iid, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", iid
        ):
            raise ValueError("Invalid instance ID")
        for name in (
            "repo",
            "version",
            "base_commit",
            "problem_statement",
            "test_patch",
        ):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"Each instance requires a nonempty {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError("Instance base_commit must be an immutable commit SHA")
        for name in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            tests = row.get(name)
            if isinstance(tests, str):
                try:
                    tests = json.loads(tests)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{name} must be a JSON array") from error
            if not isinstance(tests, list) or any(
                not isinstance(test, str) for test in tests
            ):
                raise ValueError(f"{name} must contain test names")
        if any(row.get(name) for name in ("image", "image_name", "docker_image")):
            raise ValueError(
                "Custom images are unsupported by the canonical SWE-bench baseline"
            )
        ids.append(iid)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate instance IDs")
    revision = data.get("dataset_revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be an immutable 40-character SHA")
    return {k: v for k, v in data.items() if k != "instances"}, data["instances"]


def select(args) -> int:
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be an immutable dataset commit SHA")
    if args.out.exists():
        raise FileExistsError(args.out)
    plan = {
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "split": args.split,
        "ids": args.ids,
        "limit": None if args.ids else args.limit,
        "requested_task_count": len(args.ids) if args.ids else args.limit,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    from datasets import load_dataset

    rows = list(load_dataset(args.dataset, split=args.split, revision=args.revision))
    by_id = {row["instance_id"]: row for row in rows}
    ids = args.ids or sorted(by_id)[: args.limit]
    if len(set(ids)) != len(ids) or set(ids) - by_id.keys():
        raise ValueError("IDs must be unique and present in the pinned dataset")
    plan["instances"] = [by_id[iid] for iid in ids]
    plan["effective_task_count"] = len(ids)
    _validate_selection(plan)
    # Retain grading fields in the snapshot; only problem_statement reaches the agent.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    return 0


def make_config(args, checkout: Path, server: dict | None = None) -> dict:
    import yaml

    from .quality_profiles import apply_observation_protocol, render_instance_template

    config = yaml.safe_load(
        (checkout / "src/minisweagent/config/benchmarks/swebench.yaml").read_text()
    )
    config["agent"]["instance_template"] = render_instance_template(
        config["agent"]["instance_template"], args.arm
    )
    config["model"]["observation_template"] = apply_observation_protocol(
        config["model"]["observation_template"], args.arm
    )
    config["agent"].update(
        step_limit=args.steps, wall_time_limit_seconds=args.task_seconds, cost_limit=0
    )
    config["model"].update(
        model_name=args.model,
        model_class="litellm",
        cost_tracking="ignore_errors",
        set_cache_control=None,
    )
    backend = "vllm" if server is None else server["backend"]
    extra_body = (
        {"return_token_ids": True}
        if backend == "vllm"
        else {
            "return_input_ids_in_sglext": True,
            "return_output_ids_in_sglext": True,
            "return_cached_tokens_details": bool(
                server["capability_receipt"]["features"]["cache_source_report"][
                    "enabled"
                ]
            ),
        }
    )
    config["model"]["model_kwargs"].update(
        api_base=endpoint(args.endpoint),
        temperature=0,
        seed=args.seed,
        max_tokens=args.max_tokens,
        # Let mini's recorded retry loop own each HTTP attempt.
        max_retries=0,
        timeout=args.request_seconds,
        extra_body=extra_body,
        stream=False,
        n=1,
    )
    return config


def server_provenance(path: Path) -> dict:
    data = json.loads(path.read_text())
    required = {
        "backend",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "kv_cache_dtype",
        "prefix_caching",
        "server_version",
        "hardware",
        "gpu_memory_utilization",
        "max_model_len",
        "tensor_parallel_size",
        "cache_start_state",
        "chat_template_sha256",
        "tool_call_parser",
    }
    if not isinstance(data, dict) or required - data.keys():
        raise ValueError(f"Server provenance requires {sorted(required)}")
    if any(data[k] is None or data[k] == "" for k in required):
        raise ValueError("Server provenance fields cannot be empty")
    if "REPLACE" in json.dumps({key: data[key] for key in required}).upper():
        raise ValueError("Replace all server provenance placeholders before running")
    if not isinstance(data["chat_template_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", data["chat_template_sha256"]
    ):
        raise ValueError("chat_template_sha256 must identify the serving template")
    if type(data["prefix_caching"]) is not bool:
        raise ValueError("prefix_caching must be a JSON boolean")
    if data["cache_start_state"] not in {"cold", "warm", "unknown"}:
        raise ValueError("cache_start_state must be cold, warm, or unknown")
    if data["backend"] not in {"vllm", "sglang"}:
        raise ValueError("backend must be vllm or sglang")
    if data["backend"] == "sglang":
        from .sglang_adapter import (
            SGLANG_REVISION,
            capability_receipt,
            validate_cache_scope,
            validate_capability_receipt,
        )

        if data.get("server_revision") != SGLANG_REVISION:
            raise ValueError("SGLang server_revision must match the audited source pin")
        enabled = data.get("enabled_capabilities")
        if not isinstance(enabled, list) or any(
            not isinstance(item, str) for item in enabled
        ):
            raise ValueError("SGLang enabled_capabilities must be a list of names")
        data["capability_receipt"] = validate_capability_receipt(
            capability_receipt(enabled), require_live=True
        )
        data["cache_scope"] = validate_cache_scope(data.get("cache_scope"))
    elif "cache_scope" in data:
        raise ValueError("cache_scope is only valid for the sglang backend")
    return data


def run_task(
    row,
    config,
    out,
    model_factory,
    environment_factory,
    agent_class,
    *,
    token_protocol="vllm",
    sglang_receipt=None,
    cache_scope=None,
):
    from .recording_model import attach_recorder

    iid = row["instance_id"]
    result = {
        "instance_id": iid,
        "exit_status": "InfrastructureError",
        "model_patch": "",
        "error_type": None,
    }
    agent = env = recorder = None
    try:
        model = model_factory(config=copy.deepcopy(config["model"]))
        recorder = attach_recorder(
            model,
            out,
            iid,
            token_protocol=token_protocol,
            sglang_receipt=sglang_receipt,
            cache_scope=cache_scope,
        )
        env = environment_factory(copy.deepcopy(config), row)
        agent_config = copy.deepcopy(config["agent"])
        agent_config["output_path"] = out / "trajectories" / f"{iid}.json"
        agent = agent_class(model, env, **agent_config)
        info = agent.run(row["problem_statement"])
        result.update(
            exit_status=info.get("exit_status", "Unknown"),
            model_patch=info.get("submission") or "",
        )
        fatal_reason = fatal_environment_error(getattr(agent, "messages", []))
        if fatal_reason is not None:
            result.update(
                exit_status="InfrastructureError",
                model_patch="",
                error_type="FatalEnvironmentError",
                infrastructure_reason=fatal_reason,
            )
    except Exception as exc:
        # Provider exceptions can carry credentials; retain only the type here.
        result["error_type"] = type(exc).__name__
    finally:
        if env is not None:
            try:
                capture_terminal_state(
                    env,
                    out,
                    iid,
                    result,
                    getattr(agent, "messages", []) if agent is not None else [],
                )
            except Exception as exc:
                result["terminal_capture_error_type"] = type(exc).__name__
            try:
                env.cleanup()
            except Exception as exc:
                result["cleanup_error_type"] = type(exc).__name__
        if recorder is not None:
            try:
                recorder.finish(exit_status=result["exit_status"])
            except Exception as exc:
                result["recording_error_type"] = type(exc).__name__
    return result


def run(args) -> int:
    from .metrics import compare_snapshots, snapshot
    from .quality_profiles import quality_profile_receipt
    from .recording_model import summarize_calls

    checkout = check_checkout(args.checkout, MINI_REVISION)
    metadata, rows = read_instances(args.instances)
    provenance = server_provenance(args.server_info)
    if not args.model.startswith("openai/"):
        raise ValueError(
            "Use openai/<served-model-name> for this local-server baseline"
        )
    config = make_config(args, checkout, provenance)
    quality_profile = quality_profile_receipt(
        config["agent"]["instance_template"],
        args.arm,
        config["model"]["observation_template"],
    )
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.metrics_url:
        endpoint(args.metrics_url)
    manifest = {
        "schema_version": 1,
        "baseline": "mini-swe-agent",
        "mini_revision": MINI_REVISION,
        "model": args.model,
        "endpoint": endpoint(args.endpoint),
        "arm": args.arm,
        "quality_experiment": {
            "primary_objective": "official tasks resolved within fixed budgets",
            "instruction_profile": quality_profile["instruction_profile"],
            "layout": quality_profile["layout"],
            "evidence_state": quality_profile["evidence_state"],
            "finish_window": quality_profile["finish_window"],
            "execution_protocol": quality_profile["execution_protocol"],
            "profile_receipt": quality_profile,
            "budgets": {
                "steps_per_task": args.steps,
                "wall_seconds_per_task": args.task_seconds,
                "request_seconds": args.request_seconds,
                "max_output_tokens_per_call": args.max_tokens,
            },
        },
        "server_provenance": provenance,
        "dataset": metadata,
        "instances_sha256": digest(args.instances),
        "task_order": [r["instance_id"] for r in rows],
        "workers": args.workers,
        "seed": args.seed,
        "config": config,
        "metrics_url": args.metrics_url,
        "quality": "ungraded",
        "cache_metrics_scope": "whole server; attribution requires exclusive use",
        **code_provenance(),
    }
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return 0
    sys.path.insert(0, str(checkout / "src"))
    import minisweagent
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    if not Path(minisweagent.__file__).resolve().is_relative_to(checkout):
        raise ValueError("Imported mini-SWE-agent is not from the pinned checkout")
    args.out.mkdir(parents=True, mode=0o700)
    # Apply to all worker writes, including trajectories and captured workload tokens.
    os.umask(0o077)
    write_json(args.out / "instances.json", rows)
    manifest.update(
        start_ns=time.time_ns(),
        platform=platform.platform(),
        python=sys.version,
        packages={
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        },
    )
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "config.json", config)
    before = snapshot(args.metrics_url) if args.metrics_url else None
    if before is not None:
        write_json(args.out / "metrics_before.json", before)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_task,
                row,
                config,
                args.out,
                get_model,
                get_sb_environment,
                DefaultAgent,
                token_protocol=provenance["backend"],
                sglang_receipt=provenance.get("capability_receipt"),
                cache_scope=provenance.get("cache_scope"),
            )
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result["instance_id"]] = result
            write_json(args.out / "task_results.json", results)
    after = snapshot(args.metrics_url) if args.metrics_url else None
    if after is not None:
        write_json(args.out / "metrics_after.json", after)
    ordered = [results[row["instance_id"]] for row in rows]
    with (args.out / "predictions.jsonl").open("w") as stream:
        for result in ordered:
            stream.write(
                json.dumps(
                    {
                        "instance_id": result["instance_id"],
                        "model_name_or_path": args.model,
                        "model_patch": result["model_patch"],
                    }
                )
                + "\n"
            )
    failed = sum(
        bool(
            r["error_type"]
            or r.get("cleanup_error_type")
            or r.get("recording_error_type")
            or r.get("terminal_capture_error_type")
        )
        for r in ordered
    )
    write_json(
        args.out / "summary.json",
        {
            "finished_ns": time.time_ns(),
            "tasks": len(rows),
            "infrastructure_failures": failed,
            "submitted": sum(r["exit_status"] == "Submitted" for r in ordered),
            "quality": "ungraded",
            "resolved": None,
            "metrics_coverage": "interval_only; final stats flush not verified",
            "call_metrics": summarize_calls(args.out),
            "cache_metrics": compare_snapshots(before, after) if before else None,
        },
    )
    return 1 if failed else 0


def evaluate(args) -> int:
    checkout = check_checkout(args.checkout, SWEBENCH_REVISION)
    run_dir = args.run.resolve()
    for name in [
        "manifest.json",
        "instances.json",
        "predictions.jsonl",
        "summary.json",
    ]:
        if not (run_dir / name).is_file():
            raise ValueError(f"Missing completed-run artifact: {name}")
    if args.out.exists():
        raise FileExistsError(args.out)
    interpreter = shutil.which(args.python)
    if interpreter is None:
        raise ValueError("Evaluation Python executable was not found")
    command = [
        os.path.abspath(interpreter),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(run_dir / "instances.json"),
        "--predictions_path",
        str(run_dir / "predictions.jsonl"),
        "--max_workers",
        str(args.workers),
        "--run_id",
        "evaluation",
    ]
    plan = {
        "argv": command,
        "harness_revision": SWEBENCH_REVISION,
        "predictions_sha256": digest(run_dir / "predictions.jsonl"),
        "instances_sha256": digest(run_dir / "instances.json"),
        "cwd": str(args.out.resolve()),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    args.out.mkdir(parents=True, mode=0o700)
    write_json(args.out / "evaluation_manifest.json", plan)
    env = dict(os.environ, PYTHONPATH=str(checkout))
    with (args.out / "evaluation.log").open("w") as stream:
        result = subprocess.run(
            command, cwd=args.out, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    write_json(
        args.out / "exit.json",
        {
            "returncode": result.returncode,
            "quality": "inspect harness reports; exit code is not resolved rate",
        },
    )
    return result.returncode


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    select_p = commands.add_parser(
        "select", help="Freeze dataset rows in deterministic order"
    )
    select_p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    select_p.add_argument("--revision", required=True)
    select_p.add_argument("--split", default="test")
    selection = select_p.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--limit", type=positive, default=2)
    select_p.set_defaults(func=select)
    run_p = commands.add_parser("run", help="Run a live, freely generating agent arm")
    run_p.add_argument("--checkout", type=Path, required=True)
    run_p.add_argument("--instances", type=Path, required=True)
    run_p.add_argument("--server-info", type=Path, required=True)
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--endpoint", required=True)
    run_p.add_argument("--metrics-url")
    run_p.add_argument(
        "--arm",
        choices=[
            "stock",
            "stable-prefix",
            "clear",
            "clear-stable-prefix",
            "evidence-state",
            "finish-window",
            "finish-protocol",
        ],
        default="stock",
    )
    run_p.add_argument("--workers", type=positive, default=1)
    run_p.add_argument("--steps", type=positive, default=30)
    run_p.add_argument("--task-seconds", type=positive, default=1800)
    run_p.add_argument("--request-seconds", type=positive, default=120)
    run_p.add_argument("--max-tokens", type=positive, default=2048)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.set_defaults(func=run)
    eval_p = commands.add_parser(
        "evaluate", help="Grade patches using frozen dataset rows"
    )
    eval_p.add_argument("--checkout", type=Path, required=True)
    eval_p.add_argument("--run", type=Path, required=True)
    eval_p.add_argument("--workers", type=positive, default=1)
    eval_p.add_argument("--python", default=sys.executable)
    eval_p.set_defaults(func=evaluate)
    for sub in [select_p, run_p, eval_p]:
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument(
            "--execute", action="store_true", help="Execute; default only prints plan"
        )
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
