# SPDX-License-Identifier: MIT
"""Independently audit host clocks, output coverage and actual cohort activity."""

from collections import Counter


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(row, prompt_token_ids):
    start, end = row["start_monotonic_ns"], row["end_monotonic_ns"]
    require(type(start) is int and type(end) is int and end > start, "invalid clocks")
    outputs = row["outputs"]
    require(len(outputs) == len(prompt_token_ids) > 0, "incomplete outputs")
    require(
        [r["prompt_token_ids"] for r in outputs] == prompt_token_ids,
        "prompt identity differs",
    )
    ids = {r["internal_request_id"] for r in outputs}
    require(len(ids) == len(outputs), "duplicate request identity")
    require(
        all(
            len(r["output_token_ids"]) == 128
            and r["num_cached_tokens"] == 0
            and r["finish_reason"] == "length"
            for r in outputs
        ),
        "output length, stop or prefix-cache contract failed",
    )
    counts, seen_batches, full_steps = Counter(), set(), 0
    previous = start / 1e9
    for event in row["completions"]:
        stamp = event["completed_monotonic"]
        require(previous <= stamp <= end / 1e9, "completion outside measured interval")
        previous = stamp
        require(event["batch_id"] not in seen_batches, "duplicate batch identity")
        seen_batches.add(event["batch_id"])
        require(not event["preempted"], "preempted cohort")
        require(set(event["completed_tokens"]) <= ids, "unexpected completion")
        require(
            all(
                type(n) is int and n in (0, 1)
                for n in event["completed_tokens"].values()
            ),
            "invalid completion count",
        )
        counts.update(event["completed_tokens"])
        if (
            set(event["scheduled_tokens"]) == ids
            and set(event["decode_request_ids"]) == ids
            and all(event["scheduled_tokens"][rid] == 1 for rid in ids)
            and all(event["completed_tokens"].get(rid) == 1 for rid in ids)
        ):
            full_steps += 1
    require(
        set(counts) == ids and set(counts.values()) == {128}, "token coverage differs"
    )
    require(full_steps >= 32, "actual full cohort activity not established")
    seconds = (end - start) / 1e9
    return {
        "requests": len(outputs),
        "output_tokens": len(outputs) * 128,
        "seconds": seconds,
        "complete_request_tokens_per_second": len(outputs) * 128 / seconds,
        "full_active_completed_decode_steps": full_steps,
        "preemptions": 0,
        "scope": "synchronized host complete-request throughput; no GPU-step latency claim",
    }
