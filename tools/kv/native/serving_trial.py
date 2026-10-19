# SPDX-License-Identifier: MIT
"""Replay a frozen serving trial after independently audited qualification.

Each invocation starts a fresh server for one arm and one block. The caller
owns counterbalancing and the prospective statistical family. This module
neither selects loads from observed outcomes nor adds confirmation blocks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .current import save, sha
from .serving_pilot import URL, batch, rpc, run_trace

CHECKS = (
    "worker_identity",
    "calibration_before_capture",
    "physical_storage",
    "weight_dispatch",
    "changing_graphs",
    "inprocess_parity",
    "mixed_and_idle_routes",
    "nll",
    "retrieval",
    "gsm8k",
)
IMPLEMENTATION = (
    "serving_server.py",
    "serving_worker.py",
    "serving_client.py",
    "serving_events.py",
    "serving_verify.py",
    "serving_pilot.py",
    "instrument.py",
    "memory_worker.py",
    "october.py",
    "graph.py",
    "storage_receipts.py",
)


def read_frozen(root, item):
    path = (root / item["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("frozen input escapes the plan directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("frozen trial input changed: " + item["file"])
    return json.loads(raw)


def validate(plan, qualification, implementation):
    if plan["stage"] not in ("descriptive", "confirmation"):
        raise ValueError("trial needs a prospective stage label")
    if qualification.get("passed") is not True or any(
        qualification.get("checks", {}).get(name) is not True for name in CHECKS
    ):
        raise ValueError("server and quality qualification is incomplete")
    if qualification["implementation_sha256"] != implementation:
        raise ValueError("qualified serving implementation changed")
    for name in ("format", "arm", "length", "cache_bytes", "backend_sources"):
        if plan[name] != qualification[name]:
            raise ValueError("trial differs from qualified scope: " + name)
    if plan["length"] not in (1024, 8192) or plan["output_tokens"] != 128:
        raise ValueError("trial requires frozen input and output lengths")
    if plan["retry_count"] != 0:
        raise ValueError("failed confirmation requests cannot be retried")
    if plan["warmup_repetitions"] < 1 or plan["settle_seconds"] < 0:
        raise ValueError("invalid warmup policy")
    labels = [cell["label"] for cell in plan["cells"]]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("trial needs distinct nonempty cells")
    if any(not label or Path(label).name != label for label in labels):
        raise ValueError("cell labels must be single path components")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--install", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("trial plan differs from its committed identity")
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    qualification = read_frozen(root, plan["qualification"])
    implementation = {
        name: sha(Path(__file__).with_name(name)) for name in IMPLEMENTATION
    }
    validate(plan, qualification, implementation)
    source = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != plan["source_sha"]:
        raise ValueError("trial harness differs from its frozen source")
    traces = []
    for cell in plan["cells"]:
        trace = read_frozen(root, cell["trace"])
        if any(
            row["max_tokens"] != plan["output_tokens"]
            or len(row["prompt_token_ids"]) != plan["length"]
            for row in trace["requests"]
        ):
            raise ValueError("cell token lengths differ from the frozen policy")
        traces.append((cell, trace))
    warmup = read_frozen(root, plan["warmup_trace"])
    prompts = [row["prompt_token_ids"] for row in warmup["requests"]]
    if not prompts or any(len(prompt) != plan["length"] for prompt in prompts):
        raise ValueError("warmup prompts have the wrong length")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "plan.json", plan)
    save(args.output / "qualification.json", qualification)
    argv = [
        sys.executable,
        "-m",
        "tools.kv.native.serving_server",
        "--install",
        str(args.install.resolve()),
        "--output",
        str(args.output / "server"),
        "--format",
        plan["format"],
        "--arm",
        plan["arm"],
        "--cache-bytes",
        str(plan["cache_bytes"]),
    ]
    save(args.output / "command.json", {"argv": argv, "plan_sha256": args.plan_sha256})
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + plan["startup_timeout_seconds"]
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(URL + "/health", timeout=2) as health:
                        if health.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("server startup timeout")
            for index in range(plan["warmup_repetitions"]):
                save(
                    args.output / f"warmup-{index}.json",
                    batch(prompts, plan["output_tokens"]),
                )
            save(args.output / "before-timing-storage.json", rpc("study_snapshot"))
            for cell, trace in traces:
                time.sleep(plan["settle_seconds"])
                run_trace(trace, args.output, cell["label"])
            save(args.output / "after-timing-storage.json", rpc("study_snapshot"))
            save(
                args.output / "DONE.json",
                {
                    "complete": True,
                    "stage": plan["stage"],
                    "block": plan["block"],
                    "arm": plan["arm"],
                },
            )
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            save(args.output / "server-exit.json", {"returncode": server.returncode})


if __name__ == "__main__":
    main()
