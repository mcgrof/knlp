# SPDX-License-Identifier: MIT
"""Verify actual fork-gap preloading without modeled child wait times."""

import json
import math
from pathlib import Path

from coldburst_verify import live_stats
from restore_common import ANCESTOR, CHUNKS, distribution
from verify import check_core, read_delta


def check_outputs(rows, expected, length):
    """Require complete chosen tokens and fixed-bound likelihoods for every row."""
    if len(rows) != len(expected):
        raise ValueError("missing continuation")
    deltas = []
    for got, want in zip(rows, expected, strict=True):
        if got["num_cached_tokens"] != ANCESTOR:
            raise ValueError("ancestor was recomputed")
        if (
            len(got["token_ids"]) != length
            or got["token_ids"] != want["token_ids"][:length]
        ):
            raise ValueError("tokens differ from recomputation")
        if len(got["chosen_logprobs"]) != length:
            raise ValueError("missing chosen-token log probabilities")
        errors = [
            abs(x - y)
            for x, y in zip(
                got["chosen_logprobs"], want["chosen_logprobs"][:length], strict=True
            )
        ]
        if any(not math.isfinite(e) or e > 0.02 for e in errors):
            raise ValueError("chosen log probability differs from recomputation")
        deltas.extend(errors)
    return deltas


def check_family_receipt(row):
    """Cross-check attained release timing and actual declared child attribution."""
    if row["singleflight"]["inflight"] or row["singleflight"]["joined"]:
        raise ValueError("singleflight state remains")
    before, after = row["family_before"], row["family_after"]
    if after["pending"]:
        raise ValueError("family declaration remains")
    delta = {key: after[key] - before[key] for key in before if key != "pending"}
    if any(v < 0 for v in delta.values()):
        raise ValueError("family counters moved backwards")
    if not math.isfinite(row["actual_gap_ms"]) or row["actual_gap_ms"] < 0:
        raise ValueError("invalid attained fork gap")
    policy = row["policy"]
    if policy == "reactive":
        if row["warm_rows"] or row["warm_complete_ms"] is not None:
            raise ValueError("reactive control contains a preload")
    else:
        warm = row["warm_complete_ms"]
        if not isinstance(warm, (float, int)) or not math.isfinite(warm) or warm <= 0:
            raise ValueError("missing observed warm completion")
        if row["warm_completed_before_release"] != (warm <= row["actual_gap_ms"]):
            raise ValueError("warm completion and release disagree")
        if policy == "oracle" and not row["warm_completed_before_release"]:
            raise ValueError("oracle released children before warming completed")
    if policy == "family":
        if row["actual_gap_ms"] < row["requested_gap_ms"]:
            raise ValueError("children released before requested fork gap")
        local_hits = sum(
            child["kv_transfer_params"]["cached_token_stats"]["num_vllm_cached_tokens"]
            >= ANCESTOR
            for child in row["rows"]
        )
        if (
            delta["family_ensure_total"] != 1
            or delta["family_prefetch_hit_total"] != local_hits
        ):
            raise ValueError("family attribution differs from actual local child hits")
        if delta["family_prefetch_wasted_total"] != int(local_hits == 0):
            raise ValueError("unused family declaration was not charged")
    elif any(delta.values()):
        raise ValueError("reference arm contains a family declaration")


def verify_family(root: Path) -> dict:
    """Check every warmup and sample, then summarize quiet attained-gap trials."""
    plan = json.loads((root / "plan.json").read_text())
    if plan["section"] != "family":
        raise ValueError("not a family replay")
    samples = plan["serving_samples"]
    oracle = {
        row["fanout"]: row["rows"]
        for row in json.loads((root / "oracle/oracle.json").read_text())
    }
    cases = [("reactive", 0), ("oracle", 0)] + [
        ("family", gap) for gap in (0, 10, 25, 100)
    ]
    summary = {}
    child_checks, warm_checks, deltas = 0, 0, []
    deferred_requests = 0
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
            if len(rows) != 6 * (samples + 3):
                raise ValueError("incomplete family cases")
            producer = json.loads((cell / "producer.json").read_text())["rows"]
            expected_digests = json.loads((cell / "producer-prefix.json").read_text())[
                "members"
            ][0]["checksums"]["chunk_checksums"]
            for policy, gap in cases:
                selected = [
                    row
                    for row in rows
                    if (row["fanout"], row["policy"], row["requested_gap_ms"])
                    == (fanout, policy, gap)
                ]
                if sorted(row["sample"] for row in selected) != list(
                    range(-3, samples)
                ):
                    raise ValueError("missing or repeated family sample")
                reads, first, actual, warmed, warm_times, walls = [], [], [], [], [], []
                deferred = []
                for row in selected:
                    deltas.extend(check_outputs(row["rows"], oracle[fanout], 8))
                    child_checks += fanout
                    if policy != "reactive":
                        deltas.extend(check_outputs(row["warm_rows"], producer, 1))
                        warm_checks += 1
                    check_family_receipt(row)
                    deferred_requests += (
                        row["family_after"]["family_prefetch_deferred_total"]
                        - row["family_before"]["family_prefetch_deferred_total"]
                    )
                    native = read_delta(
                        row["before"]["l2_adapters"][0]["core"],
                        row["after"]["l2_adapters"][0]["core"],
                        mode,
                    )
                    if (
                        not CHUNKS
                        <= native["logical_requests"]
                        <= CHUNKS * (fanout + int(policy != "reactive"))
                    ):
                        raise ValueError("backend reads outside bounded family")
                    if row["after"]["l1_manager"]["external_io_lease_count"]:
                        raise ValueError("external ownership remains")
                    if row["sample"] == -3:
                        live = live_stats(row["live_prefix"])
                        if live["max_live_ready_ancestor_blocks"] != ANCESTOR // 16:
                            raise ValueError(
                                "integrity family has duplicate ready ancestors"
                            )
                        receipt = json.loads(
                            (
                                cell / f"n{fanout}-{policy}-g{gap}-s0-prefix.json"
                            ).read_text()
                        )
                        if len(receipt["members"]) != fanout + int(
                            policy != "reactive"
                        ) or any(
                            m["checksums"]["chunk_checksums"] != expected_digests
                            for m in receipt["members"]
                        ):
                            raise ValueError("model KV digests differ")
                    if not row["warmup"]:
                        reads.append(native["logical_requests"])
                        deferred.append(
                            row["family_after"]["family_prefetch_deferred_total"]
                            - row["family_before"]["family_prefetch_deferred_total"]
                        )
                        first.append(row["rows"][0]["ttft_ms"])
                        actual.append(row["actual_gap_ms"])
                        warmed.append(row["warm_completed_before_release"])
                        walls.append(row["children_wall_ms"])
                        if row["warm_complete_ms"] is not None:
                            warm_times.append(row["warm_complete_ms"])
                summary[f"{mode}-n{fanout}-{policy}-g{gap}"] = {
                    "backend_chunk_reads_per_family": reads,
                    "declared_children_deferred_before_lookup": deferred,
                    "first_child_ttft": distribution(first),
                    "actual_fork_gap": distribution(actual),
                    "warm_completed_before_release": warmed,
                    "observed_warm_completion": (
                        None if not warm_times else distribution(warm_times)
                    ),
                    "children_wall": distribution(walls),
                }
            wasted = json.loads((cell / "wasted.json").read_text())
            before, after = wasted["family_before"], wasted["family_after"]
            if (
                wasted["rows"]
                or after["pending"]
                or after["family_prefetch_hit_total"]
                != before["family_prefetch_hit_total"]
                or after["family_prefetch_wasted_total"]
                - before["family_prefetch_wasted_total"]
                != 1
            ):
                raise ValueError("unused preload was not charged without a child hit")
            deltas.extend(check_outputs(wasted["warm_rows"], producer, 1))
            warm_checks += 1
            if (
                read_delta(
                    wasted["before"]["l2_adapters"][0]["core"],
                    wasted["after"]["l2_adapters"][0]["core"],
                    mode,
                )["logical_requests"]
                != CHUNKS
            ):
                raise ValueError("unused preload work missing from backend accounting")
            terminal = json.loads((cell / "terminal.json").read_text())[
                "storage_manager"
            ]
            check_core(terminal["l2_adapters"][0]["core"], mode, terminal=True)
            if terminal["l1_manager"]["external_io_lease_count"]:
                raise ValueError("terminal external ownership remains")
    if not deferred_requests:
        raise ValueError("declared replay never deferred a child before lookup")
    return {
        "cases": summary,
        "declared_children_deferred_including_warmups": deferred_requests,
        "child_continuations_checked": child_checks,
        "warm_continuations_checked": warm_checks,
        "chosen_logprob_comparisons": len(deltas),
        "maximum_logprob_delta": max(deltas),
        "limits": [
            "Public engine loop measures attained gaps; a synchronous step can delay release beyond the requested gap.",
            "One engine per path/fanout and fixed case order, not independent randomized process pairs.",
            "Only the first warmup is instrumented for ownership; retained timings use the same driver without allocation hooks.",
            "Backend payload reads are not physical NVMe commands; declaration bytes are not storage savings.",
            "Synthetic exact-prefix requests do not establish task quality or a general production latency benefit.",
        ],
    }
