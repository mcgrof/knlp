# SPDX-License-Identifier: MIT
"""Independent standard-library reader of saved HTTP stream evidence.

Does not import the recorder or trust its summaries. Verifies raw ordering,
reliability counts, elapsed endpoints, and each reported latency percentile.
"""

import argparse
import json
import math
from pathlib import Path


def verify(record):
    start, end = record["start_ns"], record["end_ns"]
    elapsed = (end - start) / 1e9
    assert elapsed > 0
    plan = record["plan"]
    assert record["arrival_end_ns"] == start + plan["arrival_duration_ns"]
    assert len(record["requests"]) == len(plan["requests"])
    fields = {
        key: []
        for key in (
            "ttft_send_ms",
            "ttft_arrival_ms",
            "tpot_ms",
            "latency_send_ms",
            "latency_arrival_ms",
            "dispatch_lag_ms",
            "admission_wait_ms",
        )
    }
    completed, total_tokens, chunks = 0, 0, []
    statuses = {
        key: 0
        for key in (
            "completed",
            "error",
            "timeout",
            "cancelled",
            "unfinished",
            "invalid",
        )
    }
    for row, intended in zip(record["requests"], plan["requests"]):
        assert (
            row["id"] == intended["id"]
            and row["prompt_token_ids"] == intended["prompt_token_ids"]
        )
        assert row["intended_ns"] == start + intended["offset_ns"]
        stamps = [
            row[k]
            for k in (
                "intended_ns",
                "dispatch_ns",
                "admitted_ns",
                "send_ns",
                "headers_ns",
                "end_ns",
            )
            if k in row
        ]
        assert stamps == sorted(stamps) and stamps[-1] <= end
        events = row["events"]
        event_stamps = [x["at_ns"] for x in events]
        assert event_stamps == sorted(event_stamps)
        if events:
            assert (
                row["headers_ns"]
                <= event_stamps[0]
                <= event_stamps[-1]
                <= row["end_ns"]
            )
        if row["status"] != "completed":
            statuses[row["status"]] += 1
            continue
        tokens, token_stamps, sizes, eos, finish = [], [], [], [], []
        for event in events:
            data = event["data"]
            if data == "[DONE]":
                eos.append(event["at_ns"])
                continue
            value = json.loads(data)
            assert "error" not in value
            for choice in value.get("choices", []):
                assert choice.get("index", 0) == 0
                ids = choice.get("token_ids", [])
                if choice.get("text"):
                    assert "token_ids" in choice
                if ids:
                    tokens.extend(ids)
                    token_stamps.append(event["at_ns"])
                    sizes.append(len(ids))
                if choice.get("finish_reason") is not None:
                    finish.append(choice["finish_reason"])
        if len(eos) != 1 or not finish or not tokens:
            statuses["invalid"] += 1
            continue
        assert token_stamps[-1] <= eos[0]
        # These controlled traces force a fixed output length.
        assert len(tokens) == intended["max_tokens"]
        statuses["completed"] += 1
        completed += 1
        total_tokens += len(tokens)
        chunks.extend(sizes)
        fields["ttft_send_ms"].append((token_stamps[0] - row["send_ns"]) / 1e6)
        fields["ttft_arrival_ms"].append((token_stamps[0] - row["intended_ns"]) / 1e6)
        if len(tokens) > 1:
            fields["tpot_ms"].append(
                (token_stamps[-1] - token_stamps[0]) / (len(tokens) - 1) / 1e6
            )
        fields["latency_send_ms"].append((eos[0] - row["send_ns"]) / 1e6)
        fields["latency_arrival_ms"].append((eos[0] - row["intended_ns"]) / 1e6)
        fields["dispatch_lag_ms"].append(
            (row["dispatch_ns"] - row["intended_ns"]) / 1e6
        )
        fields["admission_wait_ms"].append(
            (row["admitted_ns"] - row["dispatch_ns"]) / 1e6
        )
    summary = record["summary"]
    assert summary["offered"] == len(record["requests"])
    assert summary["counts"] == statuses
    assert summary["valid_completed"] == completed
    assert summary["event_token_counts"] == chunks
    assert math.isclose(summary["request_throughput_per_s"], completed / elapsed)
    assert math.isclose(
        summary["output_token_throughput_per_s"], total_tokens / elapsed
    )
    for key, values in fields.items():
        stated = summary["metrics_conditional_on_valid_completion"][key]
        assert stated["n"] == len(values)
        for label, q in (("p50", 0.5), ("p95", 0.95)):
            if not values:
                assert stated[label] is None
                continue
            ordered = sorted(values)
            index = (len(values) - 1) * q
            lower, upper = math.floor(index), math.ceil(index)
            expected = (
                ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
                if upper != lower
                else ordered[lower]
            )
            assert math.isclose(
                stated[label], expected, rel_tol=1e-10, abs_tol=1e-10
            ), key
    return {
        "passed": True,
        "offered": len(record["requests"]),
        "completed": completed,
        "output_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.record.read_text())), indent=2))


if __name__ == "__main__":
    main()
