# SPDX-License-Identifier: MIT
"""Bound cache-hit headroom from exact observed requests without a GPU."""

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path

from .audit import lineage


def maximum_intervals(intervals, capacity):
    """Select the most unit-value reuse intervals within a residency budget.

    Process starts in order. When too many selected intervals overlap, discard
    one with the latest end; it leaves the least room for later reuse. This is an
    offline independent-block bound. It relaxes prefix-contiguity and active-request
    pinning and does not estimate runtime or implement an admissible engine policy.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive block count")
    active, chosen = set(), set()
    ending, latest = [], []
    for index, (start, end) in sorted(
        enumerate(intervals), key=lambda x: (x[1][0], x[1][1])
    ):
        if not 0 <= start < end:
            raise ValueError("invalid reuse interval")
        while ending and ending[0][0] <= start:
            _, done = heapq.heappop(ending)
            active.discard(done)
        active.add(index)
        chosen.add(index)
        heapq.heappush(ending, (end, index))
        heapq.heappush(latest, (-end, index))
        if len(active) > capacity:
            while latest and latest[0][1] not in active:
                heapq.heappop(latest)
            _, dropped = heapq.heappop(latest)
            active.remove(dropped)
            chosen.remove(dropped)
    return len(chosen)


def analyze(root):
    root = Path(root)
    path = root / "invocations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    manifest = json.loads((root / "workload.json").read_text())
    if (
        not manifest.get("successful_capture")
        or not rows
        or any(r["status"] != "complete" for r in rows)
    ):
        raise ValueError("headroom requires a complete invocation capture")
    edges = lineage(rows)
    cache = manifest["engine_cache"]
    if (
        cache["initial_free_blocks"] != cache["physical_pool_blocks"] - 1
        or cache["final_nonnull_references"] != 0
    ):
        raise ValueError("capture must begin empty and finish drained")
    if {(r["engine_id"], r["clock_id"]) for r in rows} != {
        (cache["engine_id"], cache["clock_id"])
    } or cache["completed_requests"] != len(rows):
        raise ValueError("expected one complete engine and clock domain")
    rows.sort(key=lambda r: r["start_ns"])
    if any(a["end_ns"] > b["start_ns"] for a, b in zip(rows, rows[1:])):
        raise ValueError("the first bound supports sequential observed requests")
    block_size = cache["block_tokens"]
    if block_size != 16:
        raise ValueError("unexpected block geometry")
    intervals, observed_intervals, previous = [], [], {}
    position = reads = observed_hits = remainder = writes = 0
    per_purpose = Counter()
    namespace = json.dumps(manifest["identity"], sort_keys=True).encode()
    for row in rows:
        prompt, ready = row["input_token_ids"], row["ready"]
        if not prompt:
            raise ValueError("empty model prompt")
        tokens = prompt + row["output_token_ids"]
        if ready["all_token_ids"] != tokens or not len(prompt) <= ready[
            "computed_tokens"
        ] <= len(tokens):
            raise ValueError("completed scheduler token evidence disagrees")
        hits = ready["cached_tokens"]
        eligible = (len(prompt) - 1) // block_size
        if (
            type(hits) is not int
            or hits < 0
            or hits % block_size
            or hits > eligible * block_size
        ):
            raise ValueError("unsupported partial or invalid observed prefix hit")
        before_free = [
            e for e in row["cache_observations"] if e["action"] == "before-free"
        ]
        lookups = [e for e in row["cache_observations"] if e["action"] == "lookup"]
        if len(lookups) != 1 or lookups[0]["hit_tokens"] != hits:
            raise ValueError("observed prefix lookup disagrees")
        if (
            len(before_free) != 1
            or before_free[0]["computed_tokens"] != ready["computed_tokens"]
            or before_free[0]["blocks"] != ready["blocks"]
        ):
            raise ValueError("missing completed allocator observation")
        observed_hits += hits // block_size
        reads += eligible
        remainder += len(prompt) - eligible * block_size
        per_purpose[row["purpose"]] += 1
        count = ready["computed_tokens"] // block_size
        if len(ready["blocks"]) < count:
            raise ValueError("not enough observed computed blocks")
        digest = hashlib.sha256(namespace).digest()
        for index in range(count):
            block = ready["blocks"][index]
            if (
                block["hash"] is None
                or block["hash_tokens"] != (index + 1) * block_size
            ):
                raise ValueError("missing exact full-block evidence")
            encoded = json.dumps(
                tokens[index * block_size : (index + 1) * block_size],
                separators=(",", ":"),
            ).encode()
            digest = hashlib.sha256(digest + encoded).digest()
            if index < hits // block_size:
                if digest not in previous:
                    raise ValueError("observed hit lacks a prior computed prefix")
                observed_intervals.append((previous[digest], position))
            if index < eligible and digest in previous:
                intervals.append((previous[digest], position))
            if index >= eligible:
                writes += 1
            previous[digest] = position
            position += 1
    capacity = cache["physical_pool_blocks"] - 1
    if maximum_intervals(observed_intervals, capacity) != observed_hits:
        raise ValueError("observed hit intervals exceed the cache capacity")
    bound_hits = maximum_intervals(intervals, capacity)
    if bound_hits < observed_hits:
        raise ValueError(
            "optimistic bound is below observed hits; replay model is invalid"
        )
    observed = remainder + (reads - observed_hits) * block_size
    bound = remainder + (reads - bound_hits) * block_size
    return {
        "schema": "knlp.kmesh.workload-headroom.v1",
        "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "requests": len(rows),
        "purpose_counts": dict(per_purpose),
        "lineage": edges,
        "capacity_blocks": capacity,
        "eligible_prompt_blocks": reads,
        "observed_hit_blocks": observed_hits,
        "optimistic_hit_blocks": bound_hits,
        "computed_write_blocks": writes,
        "observed_prefill_tokens": observed,
        "optimistic_minimum_prefill_tokens": bound,
        "maximum_prefill_tokens_saved": observed - bound,
        "maximum_prefill_fraction_saved": (
            (observed - bound) / observed if observed else 0
        ),
        "production_workload_qualified": False,
        "limits": [
            "Observed demand comes from one bounded local-model benchmark adaptation, not production traffic.",
            "The bound uses future knowledge and independent full blocks; real prefix-contiguity and active-request pinning can only reduce opportunity.",
            "Every block has equal token value; this does not weight computation, storage, metadata or scheduling cost.",
            "Invocation observation is active; no latency or throughput verdict follows.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be fresh")
    result = analyze(args.root)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "requests",
                    "observed_prefill_tokens",
                    "maximum_prefill_tokens_saved",
                    "maximum_prefill_fraction_saved",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
