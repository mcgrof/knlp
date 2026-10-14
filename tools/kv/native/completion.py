# SPDX-License-Identifier: MIT
"""Read completion-linked intervals without mistaking scheduling for execution."""


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
