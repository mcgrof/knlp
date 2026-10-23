# SPDX-License-Identifier: MIT
"""Excluded HTTP qualification/pilot; never treat these trials as confirmation."""

import argparse
import asyncio
import gzip
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .current import fixture_root, save
from .serving_client import replay
from .serving_trace import trace
from .serving_verify import verify
from .serving_telemetry import MetricSampler

URL = "http://127.0.0.1:8000"


def post(endpoint, value, timeout=600):
    req = urllib.request.Request(
        URL + endpoint,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def rpc(method, *args):
    return post(
        "/collective_rpc", {"method": method, "args": list(args), "timeout": 600}
    )


def run_trace(plan, output, label, mode="timing"):
    save(output / (label + "-trace.json"), plan)
    rpc("study_begin", label, mode)
    try:
        with MetricSampler(URL) as metrics:
            result = asyncio.run(replay(plan, URL))
        save(output / (label + "-server-metrics.json"), metrics.rows)
        result["plan"] = plan
        save(output / (label + "-client.json"), result)
        audit = verify(result)
        save(output / (label + "-audit.json"), audit)
    finally:
        save(output / (label + "-worker.json"), rpc("study_end"))
    return result


def batch(prompts, output_tokens=128, *, timeout=600):
    return post(
        "/v1/completions",
        {
            "model": "study",
            "prompt": prompts,
            "max_tokens": output_tokens,
            "ignore_eos": True,
            "temperature": 0,
            "seed": 20261019,
            "return_token_ids": True,
        },
        timeout=timeout,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--install", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--format", choices=("awq", "fp16"), required=True)
    p.add_argument(
        "--arm", choices=("candidate", "native_fi", "native_fa3"), required=True
    )
    p.add_argument("--length", type=int, choices=(1024, 8192), default=8192)
    p.add_argument("--cache-bytes", type=int)
    p.add_argument("--skip-heavy", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(
        gzip.decompress((fixture_root() / args.format / "qwen.json.gz").read_bytes())
    )
    base = trace(manifest, length=args.length, count=32, concurrency=32)
    prompts = [row["prompt_token_ids"] for row in base["requests"]]
    argv = [
        sys.executable,
        "-m",
        "tools.kv.native.serving_server",
        "--install",
        str(args.install),
        "--output",
        str(args.output / "server"),
        "--format",
        args.format,
        "--arm",
        args.arm,
    ]
    if args.cache_bytes is not None:
        argv.extend(["--cache-bytes", str(args.cache_bytes)])
    save(
        args.output / "command.json", {"argv": argv, "excluded_from_confirmation": True}
    )
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 1800
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
            save(args.output / "warmup.json", batch(prompts))
            if not args.skip_heavy:
                for order in range(2):
                    rpc("study_begin", f"qualification-{order}", "qualification")
                    try:
                        selected = prompts if order == 0 else prompts[1:] + prompts[:1]
                        save(
                            args.output / f"batch-{order}.json",
                            batch(selected, timeout=3600),
                        )
                    finally:
                        save(
                            args.output / f"qualification-{order}.json",
                            rpc("study_end"),
                        )
                rpc("study_begin", "profile", "profile")
                try:
                    save(args.output / "profile-output.json", batch(prompts))
                finally:
                    save(args.output / "profile.json", rpc("study_end"))
            # Separate idle, staggered arrivals, and closed-loop concurrency.
            idle = trace(manifest, length=args.length, count=2, concurrency=1)
            idle["requests"][1]["offset_ns"] = 5_000_000_000
            idle["arrival_duration_ns"] = 5_000_000_000
            run_trace(idle, args.output, "idle")
            for concurrency in (8, 32):
                plan = trace(
                    manifest, length=args.length, count=96, concurrency=concurrency
                )
                run_trace(plan, args.output, f"closed-{concurrency}")
            plan = trace(manifest, length=args.length, count=64, rate=4)
            run_trace(plan, args.output, "open-probe")
            # A short matched pair describes buffered-counter overhead only.
            for mode in ("timing", "overhead"):
                plan = trace(manifest, length=args.length, count=64, concurrency=32)
                run_trace(plan, args.output, f"observer-{mode}", mode)
            save(
                args.output / "DONE.json",
                {
                    "complete": True,
                    "excluded_from_confirmation": True,
                    "note": "Raw qualification requires independent audit before any confirmation unlock.",
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
