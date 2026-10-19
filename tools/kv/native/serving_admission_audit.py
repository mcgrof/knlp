# SPDX-License-Identifier: MIT
"""Describe scheduler admission from the exact independently audited HTTP steps.

This CPU supplement distinguishes mixed prefill/decode scheduling from client
concurrency. It does not add experimental replicates or infer kernel dispatch
from a schedule. Production qualification establishes the supported routes.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .serving_scheduler import admissions

if not __debug__:
    raise RuntimeError("Evidence audits require Python assertions enabled")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path, expected_sha256, input_tokens):
    if input_tokens not in (1024, 8192):
        raise ValueError("this audit covers the frozen Qwen3 HTTP contexts")
    if sha(path) != expected_sha256:
        raise ValueError("worker steps differ from the independent final audit")
    rows = json.loads(path.read_text())
    kinds = Counter()
    prefill_sizes, admission_sizes, decode_sizes = Counter(), Counter(), Counter()
    seen = set()
    resumed = preempted = new_steps = new_mixed_steps = recompute_steps = 0
    for row, admission in zip(rows, admissions(rows, input_tokens), strict=True):
        assert row["batch_id"] not in seen
        seen.add(row["batch_id"])
        tokens = row["scheduled_tokens"]
        assert all(
            type(n) is int and (n == 1 or input_tokens <= n < input_tokens + 128)
            for n in tokens.values()
        )
        prefill = sum(n > 1 for n in tokens.values())
        decode = sum(n == 1 for n in tokens.values())
        assert len(row["new_request_ids"]) == len(set(row["new_request_ids"]))
        kind = (
            "mixed_prefill_or_recompute_decode"
            if prefill and decode
            else (
                "prefill_or_recompute_only"
                if prefill
                else "decode_only" if decode else "empty"
            )
        )
        kinds[kind] += 1
        if prefill:
            prefill_sizes[prefill] += 1
        admitted = len(admission["fresh"])
        if admitted:
            admission_sizes[admitted] += 1
            new_steps += 1
            new_mixed_steps += bool(decode)
        if decode:
            decode_sizes[decode] += 1
        recompute_steps += any(n > input_tokens for n in tokens.values())
        resumed += len(admission["resumed_via_new"]) + len(
            admission["resumed_via_cached"]
        )
        preempted += len(row["preempted_request_ids"])
    assert sum(kinds.values()) == len(rows)
    return {
        "worker_steps_sha256": expected_sha256,
        "steps": len(rows),
        "steps_by_kind": {
            k: kinds[k]
            for k in [
                "prefill_or_recompute_only",
                "mixed_prefill_or_recompute_decode",
                "decode_only",
                "empty",
            ]
        },
        "new_request_admission_steps": new_steps,
        "new_request_admission_mixed_with_decode_steps": new_mixed_steps,
        "new_request_admission_count_histogram": dict(sorted(admission_sizes.items())),
        "scheduled_prefill_or_recompute_count_histogram": dict(
            sorted(prefill_sizes.items())
        ),
        "scheduled_decode_request_count_histogram": dict(sorted(decode_sizes.items())),
        "steps_with_longer_recomputed_context": recompute_steps,
        "resumed_request_events": resumed,
        "preemption_events": preempted,
    }


def audit(root, result_path, prefix):
    result = json.loads(result_path.read_text())
    assert result["complete"] and not result["pending"]
    records = []
    captures = [
        (b["block"], arm, row)
        for b in result["blocks"]
        for arm, row in b["arms"].items()
    ]
    captures += [
        ("descriptive", arm, row) for arm, row in result["descriptive"].items()
    ]
    for block, arm, capture in captures:
        base = root / f"{prefix}-{block}-{arm}"
        assert sha(base / "server/inputs/protocol.json") == capture["protocol_sha256"]
        plan = json.loads((base / "plan.json").read_text())
        command = json.loads((base / "command.json").read_text())
        assert sha(base / "plan.json") == command["plan_sha256"]
        assert plan["frozen_design_sha256"] == result["design_sha256"]
        assert plan["source_sha"] == result["source_sha"]
        for label, cell in capture["cells"].items():
            path = base / "server" / label / "steps.json"
            value = summarize(path, cell["worker_steps_sha256"], plan["length"])
            assert value["steps"] == cell["actual_worker_steps"]
            assert value["preemption_events"] == cell["preemption_events"]
            records.append({"block": block, "arm": arm, "cell": label, **value})
    return {
        "confirmation_sha256": sha(result_path),
        "source_sha": result["source_sha"],
        "records": records,
        "failed_launches_without_schedules": len(result.get("failed_launches", [])),
        "scope": "Descriptive scheduler counts at the exact step hashes retained by the independent final reader. Each request has a whole fresh prompt and 128 forced outputs, without chunked prefill or prefix caching. Mixed prefill can include recomputation after preemption; newly admitted requests and their mixing with decode are counted separately. This does not itself prove a kernel route or graph replay; it adds no performance blocks or statistical hypotheses.",
        "reader_sha256": sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.confirmation, args.prefix)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Read", len(result["records"]), "captured scheduler cells")


if __name__ == "__main__":
    main()
