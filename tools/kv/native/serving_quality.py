# SPDX-License-Identifier: MIT
"""Score frozen natural completions and teacher targets through the HTTP worker.

This is untimed qualification. It neither reuses a failed parity check nor
changes preservation thresholds after observing a new execution path.
"""

import argparse
import asyncio
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .current import fixture_root, save
from .serving_pilot import URL, post, rpc


def register_teacher_requests(pending, state, seen, new_requests):
    for request in new_requests:
        matches = [
            item
            for api_id, item in pending.items()
            if request.req_id.startswith(f"cmpl-{api_id}-0-")
        ]
        if len(matches) != 1:
            raise RuntimeError("HTTP teacher request identity is ambiguous")
        item = matches[0]
        if list(request.prompt_token_ids) != item["prompt_token_ids"]:
            raise RuntimeError("HTTP teacher prompt differs from frozen IDs")
        if item["sample_id"] in seen:
            raise RuntimeError("HTTP teacher sample was admitted twice")
        seen.add(item["sample_id"])
        state["requests"][request.req_id] = item
        state["counts"][request.req_id] = 0


def completion_body(sample, stage, task=None):
    body = {
        "model": "study",
        "prompt": sample["token_ids"],
        "request_id": stage + "-" + sample["id"],
        "temperature": 0,
        "seed": 20261019,
        "return_token_ids": True,
        "stream": False,
        "ignore_eos": stage in ("nll", "parity"),
        "max_tokens": 129 if stage == "nll" else 128,
    }
    if stage in ("nll", "parity"):
        body["prompt"] = sample["token_ids"][: sample["length"]]
    if stage == "gsm8k":
        body["max_tokens"] = task["max_output_tokens"]
        body["stop"] = task["stop"]
    return body


async def cohort(samples, stage, output, task=None, concurrency=32):
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    rows = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:

        async def one(sample):
            row = {
                "id": sample["id"],
                "prompt_hash": sample["prompt_hash"],
                "gold": sample.get("gold", sample.get("answer")),
                "stage": stage,
            }
            body = completion_body(sample, stage, task)
            async with semaphore:
                row["started_ns"] = time.monotonic_ns()
                try:
                    async with session.post(
                        URL + "/v1/completions", json=body
                    ) as response:
                        row["http_status"] = response.status
                        row["raw_response"] = await response.text()
                        if response.status != 200:
                            raise RuntimeError(f"HTTP {response.status}")
                        value = json.loads(row["raw_response"])
                        if len(value["choices"]) != 1:
                            raise RuntimeError(
                                "quality request returned multiple choices"
                            )
                        choice = value["choices"][0]
                        tokens = choice["token_ids"]
                        if not isinstance(tokens, list):
                            raise RuntimeError(
                                "quality response lacks generated token IDs"
                            )
                        row.update(
                            text=choice["text"],
                            generated_token_ids=tokens,
                            finish_reason=choice["finish_reason"],
                            stop_reason=choice.get("stop_reason"),
                            error=None,
                        )
                        if (
                            stage == "nll"
                            and tokens
                            != sample["token_ids"][
                                sample["length"] : sample["length"] + 129
                            ]
                        ):
                            raise RuntimeError("HTTP teacher generated targets differ")
                        if stage == "parity" and len(tokens) != 128:
                            raise RuntimeError("parity output length differs")
                except Exception as exc:
                    row["error"] = repr(exc)
                row["ended_ns"] = time.monotonic_ns()
                # Persist each completed or failed request, including partial runs.
                with output.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)

        await asyncio.gather(*(one(sample) for sample in samples))
    if len(rows) != len(samples) or any(row["error"] for row in rows):
        raise RuntimeError("HTTP quality cohort failed; raw failures retained")
    return rows


def score(rows, stage, manifest):
    from .gsm8k import extract
    from .runner import parse_native

    result = []
    for row in rows:
        truncated = row["finish_reason"] == "length"
        unexpected = row["finish_reason"] not in ("stop", "length")
        answer = (
            extract(row["text"])
            if stage == "gsm8k"
            else parse_native(row["text"], manifest.get("retrieval_final_marker"))
        )
        result.append(
            {
                **row,
                "strict_answer": answer,
                "correct": answer == row["gold"] and not truncated and not unexpected,
                "truncated": truncated,
                "parser_failure": answer is None,
            }
        )
    return result


def run(args):
    from .gsm8k import read

    manifest = json.loads(
        gzip.decompress((fixture_root() / args.format / "qwen.json.gz").read_bytes())
    )
    task = read(args.output / "server/inputs/gsm8k-frozen.json")
    samples = [
        s
        for s in manifest["samples"]
        if s["kind"] == "nll" and s["length"] == args.length and not s.get("reserve")
    ]
    if len(samples) != 32:
        raise ValueError("HTTP quality requires 32 frozen NLL documents")
    stages = (
        [args.stage] if args.stage != "all" else ["parity", "nll", "retrieval", "gsm8k"]
    )
    for stage in stages:
        rpc("study_begin", "quality-" + stage, "timing")
        try:
            if stage == "nll":
                requests = {
                    "nll-"
                    + s["id"]: {
                        "sample_id": s["id"],
                        "replica": i,
                        "prompt_token_ids": s["token_ids"][: s["length"]],
                        "teacher": s["token_ids"][s["length"] : s["length"] + 129],
                    }
                    for i, s in enumerate(samples)
                }
                rpc("study_teacher_begin", "nll", requests)
                try:
                    asyncio.run(cohort(samples, stage, args.output / "nll.jsonl"))
                finally:
                    receipt = rpc("study_teacher_end")
                    save(args.output / "nll-teacher-end.json", receipt)
                if not all(r["complete"] for r in receipt["results"]):
                    raise RuntimeError("HTTP teacher coverage incomplete")
            elif stage == "parity":
                asyncio.run(
                    cohort(samples, stage, args.output / "parity.jsonl", concurrency=1)
                )
            else:
                selected = (
                    task["samples"]
                    if stage == "gsm8k"
                    else [
                        s
                        for s in manifest["samples"]
                        if s["kind"] == "retrieval"
                        and s["length"] == args.length
                        and not s.get("reserve")
                    ]
                )
                assert len(selected) == (1319 if stage == "gsm8k" else 100)
                rows = asyncio.run(
                    cohort(
                        selected, stage, args.output / (stage + "-raw.jsonl"), task=task
                    )
                )
                save(
                    args.output / (stage + "-scored.json"), score(rows, stage, manifest)
                )
        finally:
            save(args.output / (stage + "-worker.json"), rpc("study_end"))
    save(
        args.output / "DONE.json",
        {
            "capture_complete": True,
            "preservation_gate_passed": False,
            "requires_independent_paired_analysis": True,
        },
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
    p.add_argument(
        "--stage", choices=("all", "parity", "nll", "retrieval", "gsm8k"), default="all"
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
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
        args.output / "command.json",
        {
            "argv": argv,
            "purpose": "actual HTTP-worker quality; not timing",
            "length": args.length,
            "concurrency": 32,
            "parity_concurrency": 1,
        },
    )
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"quality server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(URL + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("quality server startup timeout")
            run(args)
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
