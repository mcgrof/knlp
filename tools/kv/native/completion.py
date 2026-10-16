# SPDX-License-Identifier: MIT
"""Read completion-linked intervals without mistaking scheduling for execution."""

import re


def request_metrics(metrics):
    """Read the v1 request clocks, including metrics objects without __dict__."""
    fields = ("arrival_time", "first_token_ts", "last_token_ts", "first_token_latency")
    if metrics is None or any(not hasattr(metrics, name) for name in fields):
        raise RuntimeError(
            "request timing requires vLLM statistics; freeze disable_log_stats=False"
        )
    return {name: getattr(metrics, name) for name in fields}


def final_text(text, final_marker=None):
    """Apply a frozen final-channel delimiter, then exact text normalization."""
    if final_marker is not None:
        if final_marker not in text:
            return ""
        text = text.rsplit(final_marker, 1)[1]
    text = re.split(r"<\|(?:im_end|end)\|>", text)[0]
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    return text.strip(" \n\t.\"'`")


def decode_intervals(rows, request_ids):
    """Exclude prefill, new admissions, preemption and a draining batch tail.

    Each completed row carries the scheduler's batch identity and the actual
    returned token count for each request. The first eligible completion is
    an endpoint only: its duration may include prefill still in flight.
    """
    wanted = set(request_ids)
    seen = set()
    previous = None
    intervals = []
    for row in rows:
        counts = row["completed_tokens"]
        eligible = (
            wanted <= seen
            and set(row["scheduled_tokens"]) == wanted
            and set(row["decode_request_ids"]) == wanted
            and all(counts.get(rid) == 1 for rid in wanted)
            and not row["preempted"]
        )
        stamp = row["completed_monotonic"]
        if eligible and previous is not None:
            seconds = stamp - previous
            if seconds <= 0:
                raise ValueError("completion clock must increase")
            intervals.append(
                {"batch_id": row["batch_id"], "seconds": seconds, "tokens": len(wanted)}
            )
        previous = stamp if eligible else None
        seen.update(rid for rid, count in counts.items() if count > 0)
    seconds = sum(row["seconds"] for row in intervals)
    return {
        "intervals": intervals,
        "steps": len(intervals),
        "seconds": seconds,
        "tokens_per_second": (
            sum(row["tokens"] for row in intervals) / seconds if seconds else None
        ),
        "endpoint": "host completion-to-completion; not isolated GPU time",
    }
