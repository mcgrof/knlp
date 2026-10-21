# SPDX-License-Identifier: MIT
"""Collect fixed worker quality cohorts, preserving every HTTP attempt.

The natural-response parser deliberately retains the reference's truncation
semantics. A correct labelled answer at the length cap is still scored;
truncation remains a separate diagnostic. No failed request is retried.
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import re
import time

from tools.kv.native.gsm8k import number
from tools.kv.native.serving_pilot import rpc
from .production import read_inputs, sha


def body(sample, eos):
    kind = sample["kind"]
    if kind not in ("nll", "retrieval", "gsm8k"):
        raise ValueError("Unknown fixed quality task")
    return {
        "model": "study",
        "prompt": (
            sample["token_ids"][: sample["length"]]
            if kind == "nll"
            else sample["token_ids"]
        ),
        "request_id": "decision-" + sample["id"],
        "temperature": 0,
        "seed": 20261020,
        "return_token_ids": True,
        "stream": False,
        "ignore_eos": kind == "nll",
        "stop_token_ids": [] if kind == "nll" else eos,
        "skip_special_tokens": True,
        "max_tokens": 129 if kind == "nll" else 2048 if kind == "gsm8k" else 128,
    }


def score(sample, response, data, tokenizer, eos):
    if len(response["choices"]) != 1:
        raise ValueError("Expected one completion")
    choice = response["choices"][0]
    raw_tokens = choice["token_ids"]
    if not isinstance(raw_tokens, list) or any(
        type(t) is not int or t < 0 for t in raw_tokens
    ):
        raise ValueError("Missing or invalid actual output token IDs")
    text = tokenizer.decode(raw_tokens, skip_special_tokens=True)
    if text != choice["text"]:
        raise ValueError("API text differs from independently decoded token IDs")
    cap = 2048 if sample["kind"] == "gsm8k" else 128
    reason = choice["finish_reason"]
    stop = choice.get("stop_reason")
    if reason not in ("stop", "length") or len(raw_tokens) > cap:
        raise ValueError("Unexpected completion termination")
    if any(t in eos for t in raw_tokens[:-1]):
        raise ValueError("Generation continued after an EOS token")
    tokens = raw_tokens[:-1] if raw_tokens and raw_tokens[-1] in eos else raw_tokens
    if reason == "length" and (len(raw_tokens) != cap or tokens != raw_tokens):
        raise ValueError("Length termination disagrees with retained tokens")
    if reason == "stop" and stop is not None and stop not in eos:
        raise ValueError("Unfrozen stop rule")
    if sample["kind"] == "gsm8k":
        matches = re.findall(data["gsm_regex"], text)
        answer = number(matches[-1]) if matches else None
        gold = sample["gold"]
        correct = answer == gold
    elif sample["kind"] == "retrieval":
        answer = text.strip().strip(" \n\t.\"'`")
        gold = sample["answer"]
        correct = answer.casefold() == gold.casefold()
    else:
        raise ValueError("Teacher rows must be scored from raw logits")
    return {
        "id": sample["id"],
        "length": sample.get("length"),
        "text": text,
        "tokens": tokens,
        "output_tokens": len(tokens),
        "raw_api_token_ids": raw_tokens,
        "finish_reason": reason,
        "stop_reason": stop,
        "eos": reason == "stop",
        "truncated": reason == "length",
        "answer": answer,
        "gold": gold,
        "correct": correct,
        "parser_failure": answer is None,
        "error": None,
    }


def teacher_rows(samples, path):
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {s["id"]: [] for s in samples}
    for row in raw:
        if row["sample_id"] not in by_id:
            raise ValueError("Unexpected teacher sample")
        by_id[row["sample_id"]].append(row)
    result = []
    for sample in samples:
        rows = sorted(by_id[sample["id"]], key=lambda r: r["step"])
        targets = sample["token_ids"][sample["length"] : sample["length"] + 129]
        if (
            len({r["request_id"] for r in rows}) != 1
            or [r["step"] for r in rows] != list(range(129))
            or [r["target"] for r in rows] != targets
        ):
            raise ValueError("Teacher request/step/target identity differs")
        if any(not math.isfinite(r["nll"]) for r in rows):
            raise ValueError("Nonfinite teacher likelihood")
        # The reference consumes teacher token zero, then scores 128 decode
        # continuations. Do not silently substitute the prefill next-token loss.
        losses = [r["nll"] for r in rows[1:]]
        result.append(
            {
                "id": sample["id"],
                "length": sample["length"],
                "article_id": sample["article_id"],
                "nll": losses,
                "mean_nll": sum(losses) / 128,
                "teacher_file_sha256": sha(path),
            }
        )
    return result


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


async def cohort(samples, eos, output, *, url):
    import aiohttp

    rows = []
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=300)
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=1800)
    ) as session:

        async def one(sample):
            request = body(sample, eos)
            row = {
                "id": sample["id"],
                "request": request,
                "started_monotonic_ns": time.monotonic_ns(),
            }
            try:
                async with session.post(
                    url + "/v1/completions",
                    json=request,
                    headers={"X-KNLP-Request-ID": request["request_id"]},
                ) as response:
                    row["status"] = response.status
                    row["raw_response"] = await response.text()
                    if response.status != 200:
                        raise RuntimeError("HTTP " + str(response.status))
                    row["response"] = json.loads(row["raw_response"])
                    row["error"] = None
            except Exception as error:
                row["error"] = repr(error)
            row["ended_monotonic_ns"] = time.monotonic_ns()
            append(output, row)
            rows.append(row)

        await asyncio.gather(*(one(s) for s in samples))
    if len(rows) != len(samples) or any(r["error"] is not None for r in rows):
        raise RuntimeError("Fixed quality cohort failed; all attempts retained")
    return {r["id"]: r for r in rows}


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    protocol, data = read_inputs(args.inputs, args.protocol_sha256)
    qualified = json.loads(args.qualification.read_text())
    worker = json.loads((args.worker_output / "worker-identity.json").read_text())
    if worker["allocation"] != "capacity":
        raise ValueError("Quality requires the full-budget production worker")
    if (
        not qualified["passed"]
        or qualified["arm"] != worker["arm"]
        or qualified["protocol_sha256"] != args.protocol_sha256
        or worker["protocol_sha256"] != args.protocol_sha256
        or qualified["worker_identity_sha256"]
        != sha(args.worker_output / "worker-identity.json")
    ):
        raise ValueError("This worker requires its own numerical qualification")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.inputs / "tokenizer", local_files_only=True
    )
    eos = protocol["eos_token_ids"]
    start = {
        "arm": worker["arm"],
        "protocol_sha256": args.protocol_sha256,
        "numerical_qualification_sha256": sha(args.qualification),
        "source_sha": protocol["source_sha"],
        "at": time.time(),
        "worker_identity_sha256": sha(args.worker_output / "worker-identity.json"),
        "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    (args.output / "START.json").write_text(json.dumps(start, indent=2) + "\n")
    for kind in ("nll", "retrieval", "gsm8k"):
        selected = [s for s in data["samples"]["confirmation"] if s["kind"] == kind]
        for first in range(0, len(selected), protocol["quality_concurrency"]):
            samples = selected[first : first + protocol["quality_concurrency"]]
            label = f"quality-{kind}-{first:04d}"
            rpc("study_begin", label, "timing")
            try:
                if kind == "nll":
                    requests = {
                        "decision-"
                        + s["id"]: {
                            "sample_id": s["id"],
                            "replica": i,
                            "prompt_token_ids": s["token_ids"][: s["length"]],
                            "teacher": s["token_ids"][s["length"] : s["length"] + 129],
                        }
                        for i, s in enumerate(samples)
                    }
                    rpc("study_teacher_begin", label, requests)
                try:
                    rows = asyncio.run(
                        cohort(
                            samples,
                            eos,
                            args.output / "http-attempts.jsonl",
                            url="http://127.0.0.1:8000",
                        )
                    )
                finally:
                    if kind == "nll":
                        receipt = rpc("study_teacher_end")
                        (args.output / (label + "-teacher-end.json")).write_text(
                            json.dumps(receipt, indent=2) + "\n"
                        )
                if kind == "nll":
                    if not all(r["complete"] for r in receipt["results"]):
                        raise RuntimeError("Teacher coverage incomplete")
                    for s in samples:
                        choice = rows[s["id"]]["response"]["choices"]
                        targets = requests["decision-" + s["id"]]["teacher"]
                        if (
                            len(choice) != 1
                            or choice[0]["token_ids"] != targets
                            or choice[0]["finish_reason"] != "length"
                        ):
                            raise RuntimeError(
                                "HTTP teacher output differs from forced targets"
                            )
                    scored = teacher_rows(
                        samples, args.worker_output / (label + "-teacher.jsonl")
                    )
                else:
                    scored = [
                        score(s, rows[s["id"]]["response"], data, tokenizer, eos)
                        for s in samples
                    ]
                for row in scored:
                    append(args.output / (kind + ".jsonl"), row)
            finally:
                receipt = rpc("study_end")
                (args.output / (label + "-worker.json")).write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
            if any(r["preemption_events"] for r in receipt["results"]):
                raise RuntimeError("Teacher/quality cohort was preempted")
    (args.output / "DONE.json").write_text(
        json.dumps(
            {
                **start,
                "completed_at": time.time(),
                "build_identity": rpc("study_build_identity", "quality"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
