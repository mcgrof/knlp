# SPDX-License-Identifier: MIT
"""Audit raw completion and worker records without using producer summaries."""

import argparse
import json
import math
from pathlib import Path


def verify(path):
    """Recompute one timing cell and require one worker event per decode end."""
    path = Path(path)
    row = json.loads(path.read_text())
    batch = row["batch"]
    if row["output_tokens"] != batch * row["output_tokens_per_request"]:
        raise ValueError("output count differs from the requested batch")
    rate = row["output_tokens"] / row["seconds"]
    if not math.isclose(
        rate, row["end_to_end_output_tokens_per_second"], rel_tol=1e-12
    ):
        raise ValueError("complete-request throughput does not match elapsed time")
    metrics = row["request_metrics"]
    ids = {m["internal_request_id"] for m in metrics}
    if len(metrics) != batch or len(ids) != batch:
        raise ValueError("request coverage differs from the requested batch")
    completions = json.loads(path.with_suffix(".completions.json").read_text())
    seen, previous, intervals = set(), None, []
    for event in completions:
        counts = event["completed_tokens"]
        full = (
            ids <= seen
            and set(event["scheduled_tokens"]) == ids
            and set(event["decode_request_ids"]) == ids
            and not event["preempted"]
            and all(counts.get(rid) == 1 for rid in ids)
        )
        stamp = event["completed_monotonic"]
        if full and previous is not None:
            seconds = stamp - previous
            if seconds <= 0:
                raise ValueError("completion clock does not increase")
            intervals.append((event["batch_id"], seconds))
        previous = stamp if full else None
        seen.update(rid for rid, count in counts.items() if count > 0)
    seconds = sum(duration for _, duration in intervals)
    linked = row["completion_linked_decode"]
    if linked["steps"] != len(intervals) or not math.isclose(
        linked["seconds"], seconds, abs_tol=1e-12
    ):
        raise ValueError("completion summary differs from raw endpoints")
    if intervals and not math.isclose(
        linked["tokens_per_second"], batch * len(intervals) / seconds, rel_tol=1e-12
    ):
        raise ValueError("completion throughput differs from raw token counts")
    if row["active_concurrency_qualified"] != (len(intervals) >= 32):
        raise ValueError("active concurrency qualification differs from completions")
    device = json.loads(path.with_suffix(".device.json").read_text())
    wanted = {bid for bid, _ in intervals}
    events = [e for group in device for e in group["events"] if e["batch_id"] in wanted]
    if (
        len(wanted) != len(intervals)
        or len(events) != len(wanted)
        or {e["batch_id"] for e in events} != wanted
    ):
        raise ValueError("decode endpoints do not join one-to-one to worker events")
    if any(
        e["model_execute_cuda_ms"] < 0
        or set(e["scheduled_tokens"]) != ids
        or any(n != 1 for n in e["scheduled_tokens"].values())
        for e in events
    ):
        raise ValueError("worker decode shape or CUDA duration is invalid")
    return {
        "path": str(path),
        "batch": batch,
        "completed_decode_intervals": len(intervals),
        "completed_host_seconds": seconds,
        "worker_execute_cuda_ms": sum(e["model_execute_cuda_ms"] for e in events),
        "complete_request_tokens_per_second": rate,
        "device_endpoint": "current-stream execute_model; excludes later sampling",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.records:
        print(json.dumps(verify(path)))


if __name__ == "__main__":
    main()
