# SPDX-License-Identifier: MIT
"""Recompute reasoning-demand evidence and export aggregate public results."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .cpu import VLLM_COMMIT
from .headroom import analyze
from .publish import require
from .tot_workload import (
    grade,
    MODEL,
    MODEL_REVISION,
    TASK_INDICES,
    TOT_COMMIT,
    WORKLOAD_SECONDS,
)

PUZZLES = ([1, 1, 4, 6], [1, 1, 11, 11], [1, 1, 3, 8])


def export(root):
    """Validate the complete capture without copying private trace fields."""
    root = Path(root)
    manifest = json.loads((root / "workload.json").read_text())
    rows = [
        json.loads(line)
        for line in (root / "invocations.jsonl").read_text().splitlines()
    ]
    require(
        all(row.get("finish_reason") in {"stop", "length"} for row in rows),
        "unexpected model finish reason",
    )
    search = json.loads((root / "search.json").read_text())
    events = json.loads((root / "events.json").read_text())
    bound = analyze(root)
    require(manifest["schema"] == "knlp.kmesh.workload-capture.v1", "unknown capture")
    require(manifest["successful_capture"] is True, "incomplete capture")
    require(
        manifest["workload_source_commit"] == TOT_COMMIT, "unexpected search source"
    )
    workload_files = json.loads(
        Path(__file__).with_name("tot-workload-files.json").read_text()
    )
    require(
        manifest["workload_files_sha256"] == workload_files, "unexpected workload files"
    )
    identity = manifest["identity"]
    expected_files = json.loads(
        Path(__file__).with_name("tot-model-files.json").read_text()
    )
    require(manifest["model_files_sha256"] == expected_files, "unexpected model files")
    require(
        identity
        == {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": expected_files["config.json"],
            "vllm_commit": VLLM_COMMIT,
            "dtype": "bfloat16",
            "block_tokens": 16,
            "adapter": None,
        },
        "unexpected model or cache identity",
    )
    configuration = {
        "cache_bytes": 512 << 20,
        "max_model_len": 2048,
        "active_requests": 1,
        "generate": "propose",
        "evaluate": "value",
        "select": "greedy",
        "selected_states": 2,
        "evaluation_samples": 1,
        "max_output_tokens": 192,
        "temperature": 0.7,
        "seed_base": 123,
        "request_limit": 384,
        "workload_deadline_seconds": WORKLOAD_SECONDS,
    }
    require(manifest["configuration"] == configuration, "unexpected search budget")
    require(manifest["task_indices"] == list(TASK_INDICES), "unexpected task selection")
    require(
        [item["task_index"] for item in search] == list(TASK_INDICES), "missing task"
    )
    tasks = []
    later_selections = whitespace_extensions = 0
    for item, puzzle in zip(search, PUZZLES):
        require(item["puzzle"] == puzzle, "unexpected puzzle")
        grades = [grade(item["puzzle"], answer) for answer in item["answers"]]
        require(item["correct"] == grades, "arithmetic grades disagree")
        steps = item["search"]["steps"]
        require(len(steps) == 4, "incomplete search steps")
        previous = [""]
        for index, step in enumerate(steps):
            require(
                step["step"] == index
                and step["x"] == " ".join(map(str, puzzle))
                and step["ys"] == previous,
                "search continuity disagrees",
            )
            require(
                len(step["new_ys"]) == len(step["values"])
                and all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in step["values"]
                ),
                "invalid candidate values",
            )
            selected = sorted(
                range(len(step["new_ys"])),
                key=lambda i: step["values"][i],
                reverse=True,
            )[:2]
            selected_states = [step["new_ys"][i] for i in selected]
            if index:
                later_selections += len(selected_states)
                whitespace_extensions += sum(
                    any(candidate.rstrip() == old.rstrip() for old in previous)
                    for candidate in selected_states
                )
            previous = selected_states
            require(step["select_new_ys"] == previous, "greedy selection disagrees")
        require(item["answers"] == previous, "selected answers disagree")
        family = f"game24-{item['task_index']:04}"
        require(
            sum(
                row["purpose"] == "propose" and row["family_id"] == family
                for row in rows
            )
            == sum(len(step["ys"]) for step in steps),
            "proposal invocation count disagrees",
        )
        require(
            sum(
                row["purpose"] == "value" and row["family_id"] == family for row in rows
            )
            + sum(
                event["family_id"] == family for event in manifest["application_events"]
            )
            == sum(len(set(step["new_ys"])) for step in steps),
            "value invocation count disagrees",
        )
        tasks.append(
            {
                "task_index": item["task_index"],
                "solved": any(grades),
                "selected_answers": len(item["answers"]),
                "proposed_states": sum(len(step["new_ys"]) for step in steps),
                "selected_states": sum(len(step["select_new_ys"]) for step in steps),
            }
        )
    require(tasks == manifest["task_results"], "task summary disagrees")
    require(
        all(task["proposed_states"] >= task["selected_states"] for task in tasks)
        and sum(task["proposed_states"] - task["selected_states"] for task in tasks)
        > 0,
        "expected observed candidate pruning",
    )
    require(
        bound["purpose_counts"].get("propose", 0) >= len(TASK_INDICES)
        and bound["purpose_counts"].get("value", 0) > 0
        and set(bound["purpose_counts"]) <= {"propose", "value"},
        "expected proposal and evaluation invocations",
    )
    require(
        len(rows) <= 384
        and all(
            len(row["input_token_ids"]) <= 2048 - 192
            and len(row["output_token_ids"]) <= 192
            for row in rows
        ),
        "invocation budget exceeded",
    )
    require(
        {row["family_id"] for row in rows}
        == {f"game24-{index:04}" for index in TASK_INDICES}
        and all(
            any(
                row["family_id"] == f"game24-{index:04}" and row["purpose"] == "propose"
                for row in rows
            )
            for index in TASK_INDICES
        ),
        "missing task invocations",
    )
    edges = bound["lineage"]
    require(
        edges["observed_edges"] > 0
        and edges["unknown_parents"]
        == edges["cross_clock_edges"]
        == edges["cross_engine_edges"]
        == 0,
        "incomplete invocation lineage",
    )
    require(
        events == [event for row in rows for event in row["cache_observations"]],
        "allocator events disagree",
    )
    cache = manifest["engine_cache"]
    cache_fields = (
        "physical_pool_blocks",
        "block_tokens",
        "initial_free_blocks",
        "final_free_blocks",
        "final_nonnull_references",
        "completed_requests",
        "evictions",
    )
    require(
        all(type(cache[key]) is int and cache[key] >= 0 for key in cache_fields),
        "invalid allocator counters",
    )
    require(
        cache["final_free_blocks"] == cache["physical_pool_blocks"] - 1
        and cache["evictions"] == sum(event["action"] == "evicted" for event in events)
        and all(
            event["prior_ref_count"] == 0
            for event in events
            if event["action"] == "evicted"
        ),
        "allocator summary disagrees",
    )
    null = cache["reserved_null_block_id"]
    require(
        type(null) is int and 0 <= null < cache["physical_pool_blocks"],
        "invalid reserved block",
    )
    for row in rows:
        observations = row["cache_observations"]
        before = [event for event in observations if event["action"] == "before-free"]
        after = [event for event in observations if event["action"] == "after-free"]
        require(
            len(before) == len(after) == 1 and observations[-1] == after[0],
            "missing terminal allocator release",
        )
        times = [event["time_ns"] for event in observations]
        require(
            all(
                type(stamp) is int and row["start_ns"] <= stamp <= row["end_ns"]
                for stamp in times
            )
            and times == sorted(times),
            "invalid allocator event order",
        )
        require(
            before[0]["time_ns"] <= after[0]["time_ns"]
            and after[0]["free_blocks"] == cache["physical_pool_blocks"] - 1
            and all(
                event["request_id"] == row["request_id"]
                for event in (before[0], after[0])
            ),
            "allocator release disagrees",
        )
        blocks = before[0]["blocks"]
        require(
            len({block["block_id"] for block in blocks}) == len(blocks)
            and all(
                type(block["block_id"]) is int
                and 0 <= block["block_id"] < cache["physical_pool_blocks"]
                and block["block_id"] != null
                and type(block["ref_count"]) is int
                and block["ref_count"] > 0
                for block in blocks
            ),
            "invalid referenced block table",
        )
    application = manifest["application_events"]
    require(
        all(
            event["event"] == "application-value-cache-hit"
            and event["family_id"] in {f"game24-{index:04}" for index in TASK_INDICES}
            and isinstance(event["state_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", event["state_sha256"])
            for event in application
        ),
        "unknown application event",
    )
    runtime = {}
    for key in (
        "python",
        "torch",
        "vllm",
        "numpy",
        "pandas",
        "sympy",
        "transformers",
        "cuda",
    ):
        value = manifest["runtime"][key]
        require(
            isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._+()-]+", value),
            "invalid runtime version",
        )
        runtime[key] = value
    require(
        type(manifest["runtime"]["cpu_threads"]) is int
        and manifest["runtime"]["cpu_threads"] == 2,
        "unexpected thread count",
    )
    runtime["cpu_threads"] = 2
    return {
        "schema": "knlp.kmesh.workload-public.v1",
        "aggregate_evidence_validated": True,
        "production_workload_qualified": False,
        "paper_accuracy_reproduced": False,
        "storage_tested": False,
        "latency_benchmark": False,
        "workload_source_commit": TOT_COMMIT,
        "workload_files_sha256": workload_files,
        "identity": identity,
        "runtime": runtime,
        "configuration": configuration,
        "tasks": tasks,
        "solved_tasks": sum(task["solved"] for task in tasks),
        "search_progress": {
            "selected_extensions_after_first_step": later_selections,
            "selected_whitespace_only_extensions": whitespace_extensions,
        },
        "application_value_cache_hits": len(application),
        "model_finish_counts": {
            purpose: dict(
                Counter(
                    row["finish_reason"] for row in rows if row["purpose"] == purpose
                )
            )
            for purpose in ("propose", "value")
        },
        "cache": {key: cache[key] for key in cache_fields},
        "lineage": {
            key: edges[key]
            for key in (
                "observed_edges",
                "unknown_parents",
                "cross_clock_edges",
                "cross_engine_edges",
                "non_exact_parent_prefix_edges",
                "live_forks",
            )
        },
        "headroom": {
            key: bound[key]
            for key in (
                "requests",
                "purpose_counts",
                "capacity_blocks",
                "eligible_prompt_blocks",
                "observed_hit_blocks",
                "optimistic_hit_blocks",
                "computed_write_blocks",
                "observed_prefill_tokens",
                "optimistic_minimum_prefill_tokens",
                "maximum_prefill_tokens_saved",
                "maximum_prefill_fraction_saved",
            )
        },
        "capture_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "workload.json",
                "invocations.jsonl",
                "search.json",
                "events.json",
            )
        },
        "limits": [
            "Export checks internal capture consistency, not independent tokenized-search replay or authentication of device execution.",
            "Three unfiltered Game24 tasks with a smaller local-model search budget; not paper accuracy or production demand.",
            "Logical producer relationships do not imply exact ancestor-prefix reuse; candidate pruning is not engine cancellation.",
            "The offline bound knows the future and relaxes prefix dependencies and active pins; it is not an implementable policy.",
            "Capture instrumentation is active; token savings are not latency, throughput or DMA-BUF results.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    result = export(args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
