# SPDX-License-Identifier: MIT
"""Verify restored outputs, backend reads and simultaneous ancestor destinations."""

import itertools
import json
import math

from restore_common import ANCESTOR, CHUNKS, CHUNK_BYTES
from verify import check_core, read_delta


def live_stats(evidence):
    events = evidence["events"]
    if not events or events[-1]["action"] != "free" or events[-1]["tables"]:
        raise ValueError("live request ownership was not drained")
    peak_blocks, peak_requests, shared_pairs, ready_blocks, ready_requests = (
        0,
        0,
        0,
        0,
        0,
    )
    for event in events:
        tables = event["tables"]
        if set(tables) != set(event["statuses"]) or set(tables) != set(
            event["computed_tokens"]
        ):
            raise ValueError("missing request status")
        for blocks in tables.values():
            if len(blocks) != ANCESTOR // 16 or len(set(blocks)) != len(blocks):
                raise ValueError("incomplete or repeated ancestor blocks")
            if any(
                type(block) is not int
                or not 0 <= block < evidence["total_blocks"]
                or block == evidence["null_block_id"]
                for block in blocks
            ):
                raise ValueError("invalid ancestor block")
        sets = [set(blocks) for blocks in tables.values()]
        ready = [
            set(blocks)
            for name, blocks in tables.items()
            if event["computed_tokens"][name] >= ANCESTOR
            and (
                event["statuses"][name] == "RUNNING"
                or event["statuses"][name].startswith("FINISHED_")
            )
        ]
        ready_blocks = max(ready_blocks, len(set().union(*ready)))
        ready_requests = max(ready_requests, len(ready))
        peak_requests = max(peak_requests, len(sets))
        peak_blocks = max(peak_blocks, len(set().union(*sets)))
        shared_pairs = max(
            shared_pairs,
            sum(bool(a & b) for a, b in itertools.combinations(sets, 2)),
        )
    resolved = {
        row["request_id"]: row
        for row in evidence.get("lookups", [])
        if row["external_new_tokens"] is not None
    }
    return {
        "max_live_requests": peak_requests,
        "max_live_distinct_ancestor_blocks": peak_blocks,
        "max_live_pairs_sharing_ancestor_blocks": shared_pairs,
        "max_live_ready_ancestor_blocks": ready_blocks,
        "max_live_ready_requests": ready_requests,
        "resolved_local_hit_requests": sum(
            row["local_tokens"] >= ANCESTOR for row in resolved.values()
        ),
        "resolved_external_load_requests": sum(
            row["external_new_tokens"] > 0 for row in resolved.values()
        ),
    }


def verify_coldburst(root):
    plan = json.loads((root / "plan.json").read_text())
    samples = plan["serving_samples"]
    oracle = json.loads((root / "oracle/oracle.json").read_text())
    baseline = {row["fanout"]: row["rows"] for row in oracle}
    summary = {}
    observations = 0
    for mode in ("staged", "direct"):
        corruption = json.loads((root / f"corruption-{mode}-rejected.json").read_text())
        if (
            corruption["returncode"] == 0
            or not corruption["expected_payload_rejection"]
        ):
            raise ValueError("missing deliberate corruption rejection")
        for arrival, fanouts in (("burst", [2, 4, 8]), ("stagger", [4])):
            cell = root / (mode + "-" + arrival)
            rows = json.loads((cell / "consumer.json").read_text())
            if len(rows) != len(fanouts) * (samples + 3):
                raise ValueError("incomplete cold-burst samples")
            terminal = json.loads((cell / "terminal.json").read_text())[
                "storage_manager"
            ]
            check_core(terminal["l2_adapters"][0]["core"], mode, terminal=True)
            if terminal["l1_manager"]["external_io_lease_count"]:
                raise ValueError("external buffer ownership remains")
            for fanout in fanouts:
                selected = [row for row in rows if row["fanout"] == fanout]
                if sorted(row["sample"] for row in selected) != list(
                    range(-3, samples)
                ):
                    raise ValueError("missing or repeated family")
                counts, destinations = [], []
                for row in selected:
                    if len(row["rows"]) != fanout:
                        raise ValueError("missing child")
                    for child, expected in zip(row["rows"], baseline[fanout]):
                        if child["token_ids"] != expected["token_ids"]:
                            raise ValueError(
                                "generated tokens differ from recomputation"
                            )
                        actual_probs = child["chosen_logprobs"]
                        expected_probs = expected["chosen_logprobs"]
                        if (
                            len(child["token_ids"]) != 8
                            or len(actual_probs) != 8
                            or len(expected_probs) != 8
                        ):
                            raise ValueError("incomplete eight-token numerical control")
                        if any(
                            not math.isfinite(abs(a - b)) or abs(a - b) > 0.02
                            for a, b in zip(actual_probs, expected_probs, strict=True)
                        ):
                            raise ValueError(
                                "chosen log probability differs from recomputation"
                            )
                        if child["num_cached_tokens"] != ANCESTOR:
                            raise ValueError("child recomputed its ancestor")
                        observations += 1
                    before = row["before"]["l2_adapters"][0]["core"]
                    after = row["after"]["l2_adapters"][0]["core"]
                    delta = read_delta(before, after, mode)
                    if not CHUNKS <= delta["logical_requests"] <= fanout * CHUNKS:
                        raise ValueError(
                            "backend read count outside the bounded family"
                        )
                    live = live_stats(row["live_prefix"])
                    if not row["warmup"]:
                        counts.append(delta["logical_requests"])
                        destinations.append(live)
                producer = json.loads((cell / "producer-prefix.json").read_text())
                expected = producer["members"][0]["checksums"]["chunk_checksums"]
                if not expected:
                    raise ValueError("missing producer KV digests")
                consumers = json.loads(
                    (cell / f"consumer-prefix-n{fanout}.json").read_text()
                )
                if len(consumers["members"]) != fanout or any(
                    member["checksums"]["chunk_checksums"] != expected
                    for member in consumers["members"]
                ):
                    raise ValueError("restored prefix digests differ from producer")
                summary[f"{mode}-{arrival}-n{fanout}"] = {
                    "backend_chunk_reads_per_family": counts,
                    "backend_payload_bytes_per_family": [
                        count * CHUNK_BYTES for count in counts
                    ],
                    "ancestor_chunks": CHUNKS,
                    "live_destinations": destinations,
                }
    return {
        "cases": summary,
        "restored_continuation_observations_including_warmups": observations,
        "limits": [
            "Backend payload reads are not physical NVMe command traces.",
            "Allocation/free observations show simultaneous block destinations, not HBM bandwidth.",
            "Instrumented eight-token synthetic continuations do not qualify serving latency or quality.",
            "Both paths use NVMe; the historical host-RAM tier is not retested here.",
        ],
    }
