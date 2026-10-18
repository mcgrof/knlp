# SPDX-License-Identifier: MIT
"""Verify retained-L1 read suppression independently of GPU ancestor sharing."""

import json
import math
from pathlib import Path

from coldburst_verify import live_stats
from family_verify import check_family_receipt, check_outputs
from restore_common import ANCESTOR, CHUNK_BYTES, CHUNKS, distribution
from retained_worker import GAP_MS
from verify import check_core, read_delta


def check_l1(l1, objects=None):
    """Require healthy, unlocked residency with no outstanding external I/O."""
    if not l1["is_healthy"] or l1["backing_resources_retained"]:
        raise ValueError("L1 is unhealthy")
    if any(
        l1[key]
        for key in (
            "external_io_lease_count",
            "write_locked_count",
            "read_locked_count",
            "temporary_count",
            "staging_object_count",
            "staging_bytes",
        )
    ):
        raise ValueError("temporary objects or ownership remain in L1")
    if objects is not None and l1["total_object_count"] != objects:
        raise ValueError("retained ancestor object count differs")


def check_retention(row, mode, unused=False):
    """Reject free prefetch work, recomputation and unproved storage savings."""
    if row["policy"] != "family" or row["arm"] not in ("engine", "retained"):
        raise ValueError("unexpected retention arm or declaration policy")
    if unused and row["arm"] != "retained":
        raise ValueError("unused control must perform retained prefetch")
    check_family_receipt(row)
    check_l1(row["before"]["l1_manager"], 0)
    check_l1(row["after"]["l1_manager"], CHUNKS if row["arm"] == "retained" else None)
    if row["requested_gap_ms"] != GAP_MS or not row["warm_completed_before_release"]:
        raise ValueError("control did not finish warming before release")
    preparation = row["preparation"]
    if row["arm"] == "retained":
        if (
            not preparation
            or preparation["found_keys"] != CHUNKS
            or preparation["total_keys"] != CHUNKS
        ):
            raise ValueError("missing complete retained prefetch")
        elapsed = preparation["completion_ms"]
        if not math.isfinite(elapsed) or not 0 < elapsed < row["warm_complete_ms"]:
            raise ValueError("external prefetch is outside the warm clock")
        check_l1(preparation["l1"], CHUNKS)
    elif preparation is not None:
        raise ValueError("engine-only control contains retained prefetch work")
    if unused != (not row["rows"]):
        raise ValueError("unused control child count differs")
    requests = row["warm_rows"] + row["rows"]
    loads = [
        child["kv_transfer_params"]["cached_token_stats"][
            "num_lmcache_extra_cached_tokens"
        ]
        for child in requests
    ]
    if loads != [ANCESTOR] + [0] * len(row["rows"]):
        raise ValueError("family lacks one complete external engine ancestor load")
    if any(
        child["kv_transfer_params"]["cached_token_stats"]["num_vllm_cached_tokens"]
        != ANCESTOR
        for child in row["rows"]
    ):
        raise ValueError("child did not reuse its local ancestor")
    native = read_delta(
        row["before"]["l2_adapters"][0]["core"],
        row["after"]["l2_adapters"][0]["core"],
        mode,
    )
    reads = native["logical_requests"]
    if row["arm"] == "retained" or unused:
        if reads != CHUNKS:
            raise ValueError(
                "retained family did not read exactly one backend ancestor"
            )
    elif not CHUNKS <= reads <= CHUNKS * (row["fanout"] + 1):
        raise ValueError("engine-only backend reads outside bounded family")
    return reads


def verify_retained(root: Path):
    """Recheck all warmups, paired families, discarded preloads and terminal state."""
    plan = json.loads((root / "plan.json").read_text())
    if plan["section"] != "retained":
        raise ValueError("not a retained-L1 replay")
    samples = plan["serving_samples"]
    oracle = {
        row["fanout"]: row["rows"]
        for row in json.loads((root / "oracle/oracle.json").read_text())
    }
    cases, wasted = {}, {}
    child_checks, warm_checks, deltas = 0, 0, []
    for mode in ("staged", "direct"):
        corruption = json.loads((root / f"corruption-{mode}-rejected.json").read_text())
        if (
            corruption["returncode"] == 0
            or not corruption["expected_payload_rejection"]
        ):
            raise ValueError("missing deliberate corruption rejection")
        for fanout in (4, 8):
            cell = root / f"{mode}-n{fanout}"
            rows = json.loads((cell / "consumer.json").read_text())
            if len(rows) != 2 * (samples + 3):
                raise ValueError("incomplete paired retention cases")
            producer = json.loads((cell / "producer.json").read_text())["rows"]
            expected = json.loads((cell / "producer-prefix.json").read_text())[
                "members"
            ][0]["checksums"]["chunk_checksums"]
            for sample in range(-3, samples):
                order = [r["arm"] for r in rows if r["sample"] == sample]
                if order != (
                    ["engine", "retained"]
                    if (sample + 3) % 2 == 0
                    else ["retained", "engine"]
                ):
                    raise ValueError(
                        "paired arm order differs from the balanced schedule"
                    )
            for arm in ("engine", "retained"):
                retained = []
                selected = [r for r in rows if r["arm"] == arm]
                if sorted(r["sample"] for r in selected) != list(range(-3, samples)):
                    raise ValueError("missing or repeated retention sample")
                for row in selected:
                    if row["fanout"] != fanout or row["warmup"] != (row["sample"] < 0):
                        raise ValueError("retention family label differs")
                    reads = check_retention(row, mode)
                    deltas.extend(check_outputs(row["rows"], oracle[fanout], 8))
                    deltas.extend(check_outputs(row["warm_rows"], producer, 1))
                    child_checks += fanout
                    warm_checks += 1
                    if row["sample"] == -3:
                        if (
                            live_stats(row["live_prefix"])[
                                "max_live_ready_ancestor_blocks"
                            ]
                            != ANCESTOR // 16
                        ):
                            raise ValueError("duplicate ready GPU ancestors")
                        digests = json.loads(
                            (cell / f"n{fanout}-{arm}-s0-prefix.json").read_text()
                        )["members"]
                        if len(digests) != fanout + 1 or any(
                            m["checksums"]["chunk_checksums"] != expected
                            for m in digests
                        ):
                            raise ValueError("restored model KV digests differ")
                    if not row["warmup"]:
                        retained.append(
                            {
                                "sample": row["sample"],
                                "backend_chunk_reads": reads,
                                "backend_payload_bytes": reads * CHUNK_BYTES,
                                "first_child_ttft_ms": row["rows"][0]["ttft_ms"],
                                "actual_fork_gap_ms": row["actual_gap_ms"],
                                "warm_complete_ms": row["warm_complete_ms"],
                                "external_warm_completion_ms": (
                                    None
                                    if row["preparation"] is None
                                    else row["preparation"]["completion_ms"]
                                ),
                                "children_wall_ms": row["children_wall_ms"],
                                "preparation_to_completion_ms": row[
                                    "declaration_to_completion_ms"
                                ],
                                "resident_objects_after_family": row["after"][
                                    "l1_manager"
                                ]["total_object_count"],
                                "l1_memory_used_bytes_after_family": row["after"][
                                    "l1_manager"
                                ]["memory_used_bytes"],
                            }
                        )
                cases[f"{mode}-n{fanout}-{arm}"] = {
                    "retained_samples": retained,
                    "first_child_ttft": distribution(
                        [r["first_child_ttft_ms"] for r in retained]
                    ),
                    "children_wall": distribution(
                        [r["children_wall_ms"] for r in retained]
                    ),
                }
            unused = json.loads((cell / "wasted.json").read_text())
            reads = check_retention(unused, mode, unused=True)
            deltas.extend(check_outputs(unused["warm_rows"], producer, 1))
            warm_checks += 1
            wasted[f"{mode}-n{fanout}"] = {
                "backend_chunk_reads": reads,
                "unused_declarations_charged": unused["family_after"][
                    "family_prefetch_wasted_total"
                ]
                - unused["family_before"]["family_prefetch_wasted_total"],
                "resident_objects": unused["after"]["l1_manager"]["total_object_count"],
            }
            terminal = json.loads((cell / "terminal.json").read_text())[
                "storage_manager"
            ]
            check_core(terminal["l2_adapters"][0]["core"], mode, terminal=True)
            check_l1(terminal["l1_manager"], 0)
    return {
        "cases": cases,
        "unused_preload_controls": wasted,
        "child_continuations_checked": child_checks,
        "warm_continuations_checked": warm_checks,
        "chosen_logprob_comparisons": len(deltas),
        "maximum_logprob_delta": max(deltas),
        "requested_fork_gap_ms": GAP_MS,
        "limits": [
            "Synthetic 250 ms foreknowledge control, not the minimum useful lead time.",
            "External warm API polling and engine preload both consume the measured lead-time interval.",
            "One engine per path/fanout, paired alternating arms and five retained samples; not independent randomized process pairs.",
            "Retained external L1 chunks consume extra residency inside the same arena; no eviction-pressure or lease qualification.",
            "Native payload read counts are not physical NVMe commands or general serving throughput.",
        ],
    }
