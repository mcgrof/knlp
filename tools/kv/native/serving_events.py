# SPDX-License-Identifier: MIT
"""Lossless SSE framing and client-clock measurements (no GPU dependency)."""

import json
import math


class SSEDecoder:
    """Frame arbitrary byte reads; timestamp events when their delimiter arrives."""

    def __init__(self):
        self.buffer = b""
        self.lines = []

    def feed(self, data, timestamp_ns):
        self.buffer += data
        events = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            if line:
                self.lines.append(line)
                continue
            fields = [
                x[5:].removeprefix(b" ") for x in self.lines if x.startswith(b"data:")
            ]
            self.lines = []
            if fields:
                events.append(
                    {"at_ns": timestamp_ns, "data": b"\n".join(fields).decode("utf-8")}
                )
        return events

    def finish(self):
        if self.buffer or self.lines:
            raise ValueError("truncated SSE frame")


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def request_metrics(row):
    """TPOT is a per-request mean, never a percentile of all token gaps.

    Token-bearing chunks can contain several tokens or no visible text. Preserve
    that uncertainty: token timestamps inside one chunk cannot be recovered.
    """
    token_events, visible, tokens, done, finishes = [], [], [], [], []
    for event in row["events"]:
        if event["data"] == "[DONE]":
            done.append(event["at_ns"])
            continue
        value = json.loads(event["data"])
        if "error" in value:
            raise ValueError("server reported an SSE error")
        for choice in value.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("recorder expects exactly one completion")
            ids = choice.get("token_ids")
            if choice.get("text") and ids is None:
                raise ValueError("server omitted requested streamed token IDs")
            if ids:
                if not all(isinstance(token, int) for token in ids):
                    raise ValueError("invalid token IDs")
                tokens.extend(ids)
                token_events.append((event["at_ns"], len(ids)))
            if choice.get("text"):
                visible.append(event["at_ns"])
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    times = [t for t, _ in token_events]
    first, last = (times[0], times[-1]) if times else (None, None)
    send, intended = row.get("send_ns"), row["intended_ns"]
    eos = done[-1] if done else None
    valid = (
        row["status"] == "completed"
        and len(done) == 1
        and bool(finishes)
        and bool(times)
    )
    ms = lambda end, start: (
        None if end is None or start is None else (end - start) / 1e6
    )
    return {
        "valid_completion": valid,
        "output_tokens": len(tokens),
        "output_token_ids": tokens,
        "token_events": len(times),
        "tokens_per_event": [n for _, n in token_events],
        "inter_token_event_gaps_ms": [(b - a) / 1e6 for a, b in zip(times, times[1:])],
        "first_token_ns": first,
        "last_token_ns": last,
        "eos_ns": eos,
        "ttft_send_ms": ms(first, send),
        "ttft_arrival_ms": ms(first, intended),
        "first_visible_send_ms": ms(visible[0] if visible else None, send),
        "tpot_ms": (
            (last - first) / 1e6 / (len(tokens) - 1) if len(tokens) > 1 else None
        ),
        "latency_send_ms": ms(eos, send),
        "latency_arrival_ms": ms(eos, intended),
        "dispatch_lag_ms": ms(row.get("dispatch_ns"), intended),
        "admission_wait_ms": ms(row.get("admitted_ns"), row.get("dispatch_ns")),
        "finish_reasons": finishes,
    }


def summarize(rows, start_ns, end_ns, arrival_end_ns):
    measured = []
    counts = {
        k: 0
        for k in ("completed", "error", "timeout", "cancelled", "unfinished", "invalid")
    }
    for row in rows:
        status = row["status"]
        counts[status] += 1
        try:
            value = request_metrics(row)
        except (ValueError, TypeError, KeyError):
            value = {"valid_completion": False}
        if value["valid_completion"]:
            measured.append(value)
        elif status == "completed":
            counts["completed"] -= 1
            counts["invalid"] += 1
    elapsed = (end_ns - start_ns) / 1e9
    if elapsed <= 0:
        raise ValueError("nonpositive observation interval")
    metrics = {}
    for field in (
        "ttft_send_ms",
        "ttft_arrival_ms",
        "tpot_ms",
        "latency_send_ms",
        "latency_arrival_ms",
        "dispatch_lag_ms",
        "admission_wait_ms",
    ):
        values = [x[field] for x in measured if x[field] is not None]
        metrics[field] = {
            "n": len(values),
            "p50": quantile(values, 0.5),
            "p95": quantile(values, 0.95),
        }
    completed = len(measured)
    return {
        "offered": len(rows),
        "counts": counts,
        "valid_completed": completed,
        "observation_seconds_including_drain": elapsed,
        "arrival_seconds": (arrival_end_ns - start_ns) / 1e9,
        "drain_seconds": max(0, end_ns - arrival_end_ns) / 1e9,
        "request_throughput_per_s": completed / elapsed,
        "output_token_throughput_per_s": sum(x["output_tokens"] for x in measured)
        / elapsed,
        "metrics_conditional_on_valid_completion": metrics,
        "event_token_counts": [n for x in measured for n in x["tokens_per_event"]],
        "scope": "Client monotonic clock; finite offered trace including drain. No SLO goodput claim.",
    }
