# SPDX-License-Identifier: MIT
"""Check fixed-arrival, read-once restore evidence without a device runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re

from coldburst_verify import live_stats
from prefetch_analysis import LEADS, verify_clock
from restore_common import ANCESTOR, CHUNK_BYTES, CHUNKS
from retained_verify import check_l1
from verify import check_core, read_delta

ARENA = 512 << 20
PAYLOAD = CHUNKS * CHUNK_BYTES
CASES = (
    [("demand", 0)]
    + [("early", lead) for lead in LEADS]
    + [("declaration", 0), ("external_resident", 0), ("engine_resident", 0)]
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, message, minimum=0):
    require(type(value) is int and value >= minimum, message)
    return value


def finite(value, message):
    require(type(value) in (int, float) and math.isfinite(value), message)
    return value


def outputs(rows, expected, length, *, restored=True):
    require(
        isinstance(rows, list) and len(rows) == len(expected), "missing continuation"
    )
    for got, want in zip(rows, expected, strict=True):
        if restored:
            require(got["num_cached_tokens"] == ANCESTOR, "ancestor was recomputed")
        tokens = got["token_ids"]
        require(
            isinstance(tokens, list)
            and len(tokens) == length
            and all(type(t) is int and t >= 0 for t in tokens)
            and tokens == want["token_ids"][:length],
            "tokens differ from recomputation",
        )
        probs, baseline = got["chosen_logprobs"], want["chosen_logprobs"]
        require(
            isinstance(probs, list)
            and isinstance(baseline, list)
            and len(probs) == length
            and len(baseline) >= length,
            "missing chosen-token log probabilities",
        )
        for actual, expected in zip(probs, baseline[:length], strict=True):
            finite(actual, "invalid chosen-token log probability")
            finite(expected, "invalid oracle log probability")
            require(
                actual == expected, "chosen log probability differs from recomputation"
            )
    return len(rows) * length


def core(status):
    adapters = status["l2_adapters"]
    require(
        isinstance(adapters, list) and len(adapters) == 1, "unexpected backend count"
    )
    return adapters[0]["core"]


def residency(status, count):
    check_l1(status["l1_manager"], count)
    require(
        status["l1_manager"]["memory_used_bytes"] == count * CHUNK_BYTES,
        "retained payload bytes differ",
    )


def reads(before, after, mode, count):
    result = read_delta(before, after, mode, allow_empty=count == 0)
    require(
        result["logical_requests"] == count,
        (
            "resident trial performed SSD reads"
            if count == 0
            else "expected exactly one backend ancestor"
        ),
    )
    require(
        result["logical_bytes"] == count * CHUNK_BYTES, "backend payload bytes differ"
    )
    return result


def engine_loads(rows, resident=False):
    stats = [r["kv_transfer_params"]["cached_token_stats"] for r in rows]
    for stat in stats:
        for key in ("num_lmcache_extra_cached_tokens", "num_vllm_cached_tokens"):
            integer(stat[key], "invalid engine cache receipt")
            require(stat[key] in (0, ANCESTOR), "partial engine ancestor load")
    loaded = [s["num_lmcache_extra_cached_tokens"] for s in stats]
    require(
        sum(loaded) == (0 if resident else ANCESTOR)
        and loaded.count(ANCESTOR) == (0 if resident else 1),
        "engine ancestor was not loaded exactly once",
    )
    require(
        all(
            s["num_vllm_cached_tokens"] == (0 if load else ANCESTOR)
            for s, load in zip(stats, loaded, strict=True)
        ),
        "child did not reuse its local ancestor",
    )


def preparation(receipt):
    require(
        receipt["found_keys"] == receipt["total_keys"] == CHUNKS,
        "external preparation missed ancestor chunks",
    )
    integer(receipt["polls"], "missing external completion poll", 1)
    require(
        finite(receipt["completion_ms"], "invalid preparation duration") > 0,
        "invalid preparation duration",
    )
    # A child may acquire a read reference after the external completion but
    # before this asynchronous diagnostic snapshot. Final snapshots must drain.
    l1 = receipt["l1"]
    require(
        l1["is_healthy"]
        and not l1["backing_resources_retained"]
        and l1["total_object_count"] == CHUNKS
        and l1["memory_used_bytes"] == PAYLOAD,
        "preparation residency differs",
    )


def controller(status):
    receipt = status["prefetch_controller"]
    require(
        receipt["is_healthy"] is True
        and receipt["thread_alive"] is True
        and receipt["join_overlaps"] is True,
        "prefetch controller is unhealthy or joining is disabled",
    )
    for key in (
        "submission_queue_size",
        "pending_queue_size",
        "in_flight_request_count",
        "lookup_phase_count",
        "load_phase_count",
        "completed_results_count",
        "accepted_handle_count",
        "dependency_waiter_count",
    ):
        require(
            type(receipt[key]) is int and receipt[key] == 0,
            "prefetch controller ownership remains",
        )
    integer(receipt["overlap_joined_total"], "invalid overlap counter")


def declarations(row, unused):
    require(
        row["singleflight"]["inflight"] == row["singleflight"]["joined"] == 0,
        "singleflight state remains",
    )
    before, after = row["family_before"], row["family_after"]
    require(before["pending"] == after["pending"] == 0, "family declaration remains")
    require(set(before) == set(after), "family counter schema changed")
    for key in before.keys() - {"pending"}:
        integer(before[key], "invalid family counter")
        integer(after[key], "invalid family counter")
        require(after[key] >= before[key], "family counters moved backwards")
    if row["arm"] in ("early", "declaration"):
        require(
            isinstance(row["declaration_identity"], str)
            and bool(row["declaration_identity"]),
            "missing declaration",
        )
    else:
        require(
            row["declaration_identity"] is None and before == after,
            "reference arm contains a family declaration",
        )
    if unused:
        require(
            after["family_prefetch_hit_total"] == before["family_prefetch_hit_total"]
            and after["family_prefetch_wasted_total"]
            - before["family_prefetch_wasted_total"]
            == 1,
            "unused hint received child credit or was not charged",
        )


def digests(receipt):
    require(
        receipt["status"] == "success"
        and receipt["layerwise"] is True
        and receipt["chunk_size"] == 16
        and receipt["num_chunks"] == CHUNKS,
        "incomplete model KV checksum geometry",
    )
    result = receipt["chunk_checksums"]
    require(
        isinstance(result, dict) and set(result) == {f"layer_{n}" for n in range(24)},
        "missing model KV layers",
    )
    require(
        all(
            isinstance(values, list)
            and len(values) == CHUNKS
            and all(
                isinstance(v, str) and re.fullmatch(r"[0-9a-f]{32}", v) for v in values
            )
            for values in result.values()
        ),
        "missing actual model KV digests",
    )
    return result


def integrity(row, producer):
    if row["sample"] != -3:
        require(
            row["live_prefix"] is None and row["integrity"] is None,
            "timed sample contains integrity instrumentation",
        )
        return
    require(
        live_stats(row["live_prefix"])["max_live_ready_ancestor_blocks"]
        == ANCESTOR // 16,
        "duplicate or missing ready GPU ancestor",
    )
    receipt = row["integrity"]
    require(
        receipt["instance_id"] != producer["instance_id"],
        "producer KV was used as consumer proof",
    )
    expected = digests(producer["members"][0]["checksums"])
    aliases = [{r["request_id"], r["engine_request_id"]} for r in row["rows"]]
    if row["warm_rows"]:
        names = {row["warm_rows"][0]["request_id"]}
        names.update(
            e["warm"]
            for e in row["events"]
            if e["event"] == "declaration" and e["warm"] is not None
        )
        aliases.append(names)
    members, used = receipt["members"], set()
    require(
        isinstance(members, list) and len(members) == len(aliases),
        "missing model KV member",
    )
    for member in members:
        matched = [
            i for i, names in enumerate(aliases) if member["request_id"] in names
        ]
        require(
            len(matched) == 1 and matched[0] not in used,
            "model KV receipt has unknown or repeated request",
        )
        used.add(matched[0])
        blocks = member["block_ids"]
        require(
            isinstance(blocks, list)
            and len(blocks) == ANCESTOR // 16
            and len(set(blocks)) == len(blocks)
            and all(
                type(b) is int
                and 0 <= b < row["live_prefix"]["total_blocks"]
                and b != row["live_prefix"]["null_block_id"]
                for b in blocks
            ),
            "invalid model KV block evidence",
        )
        require(
            digests(member["checksums"]) == expected, "restored model KV digests differ"
        )


def unused_clock(row):
    require(
        type(row["sample"]) is int and row["sample"] == 0 and row["warmup"] is False,
        "mislabeled unused-hint qualification",
    )
    require(
        row["arm"] == "early"
        and row["lead_ms"] == 250
        and not row["rows"]
        and len(row["warm_rows"]) == 1
        and row["warm_skipped_due_to_arrival"] is False,
        "unused hint has child use or lacks warming",
    )
    stamps = [
        integer(row[k], "invalid unused-hint clock")
        for k in (
            "trace_origin_ns",
            "prepare_start_ns",
            "external_ready_ns",
            "warm_submit_ns",
            "warm_done_ns",
        )
    ]
    require(stamps == sorted(stamps), "invalid unused-hint preparation order")
    arrival = integer(row["scheduled_arrival_ns"], "invalid unused-hint arrival")
    require(arrival - stamps[0] == 250_000_000, "unused hint moved fixed arrival")
    require(
        arrival <= row["group_submit_ns"] <= row["drain_end_ns"]
        and stamps[-1] <= row["drain_end_ns"],
        "unused hint did not drain",
    )
    require(
        type(row["ready_by_scheduled_arrival"]) is bool
        and row["ready_by_scheduled_arrival"] == (stamps[-1] <= arrival),
        "unused hint readiness disagrees",
    )
    require(
        all(
            row[k] is None
            for k in (
                "group_end_ns",
                "children_wall_ms",
                "deadline_group_completion_ms",
                "workflow_completion_ms",
            )
        ),
        "unused hint has child completion timing",
    )


def trial(row, config, producer_rows, unused=False):
    mode, arm = config["mode"], row["arm"]
    require(
        row["mode"] == mode
        and row["block_id"] == config["block_id"]
        and row["fanout"] == config["fanouts"][0],
        "trial identity differs from process block",
    )
    declarations(row, unused)
    resident = arm in ("external_resident", "engine_resident")
    for name, count in (
        ("reset_before", 0),
        ("before", CHUNKS if resident else 0),
        ("after", CHUNKS),
        ("cleared", 0),
    ):
        residency(row[name], count)
        check_core(core(row[name]), mode, terminal=True)
        controller(row[name])
    overlap_counts = [
        row[name]["prefetch_controller"]["overlap_joined_total"]
        for name in ("reset_before", "before", "after", "cleared")
    ]
    require(overlap_counts == sorted(overlap_counts), "overlap counter moved backwards")
    native = reads(
        core(row["before"]), core(row["after"]), mode, 0 if resident else CHUNKS
    )
    require(
        core(row["cleared"])["payload_reads"] == core(row["after"])["payload_reads"],
        "cleanup performed uncharged payload reads",
    )
    placed, comparisons = row["placement"], 0
    if resident:
        require(isinstance(placed, dict), "missing residency placement")
        integer(placed["start_ns"], "invalid placement clock")
        require(
            placed["start_ns"] <= placed["ready_ns"] <= row["trace_origin_ns"],
            "placement was not established before trace",
        )
        reads(placed["native"]["before"], placed["native"]["after"], mode, CHUNKS)
        require(
            placed["native"]["before"]["payload_reads"]
            == core(row["reset_before"])["payload_reads"]
            and placed["native"]["after"]["payload_reads"]
            == core(row["before"])["payload_reads"],
            "placement accounting does not meet trial boundary",
        )
        if arm == "external_resident":
            preparation(placed["external"])
            require(not placed.get("warm_rows"), "external placement warmed the engine")
        else:
            require(
                not placed.get("external"), "engine placement contains an external hint"
            )
            comparisons += outputs(placed["warm_rows"], producer_rows, 1)
            engine_loads(placed["warm_rows"])
    else:
        require(placed is None, "cold trial contains resident placement")
        require(
            core(row["reset_before"])["payload_reads"]
            == core(row["before"])["payload_reads"],
            "cold trial hid preparation reads",
        )
    if arm == "early":
        preparation(row["preparation"])
    else:
        require(row["preparation"] is None, "demand/reference contains early loading")
    comparisons += outputs(
        row["warm_rows"], producer_rows if row["warm_rows"] else [], 1
    )
    engine_loads(row["warm_rows"] + row["rows"], arm == "engine_resident")
    if unused:
        unused_clock(row)
        require(
            row["live_prefix"] is None and row["integrity"] is None,
            "unused hint contains child integrity evidence",
        )
    else:
        verify_clock(row)
    memory = row["memory"]
    require(
        memory["engine_configured_bytes"]
        == memory["external_configured_bytes"]
        == ARENA,
        "configured memory budgets differ",
    )
    require(
        memory["external_used_bytes"] == PAYLOAD,
        "memory ledger omits retained ancestor",
    )
    require(
        memory["external_backed_bytes"] == ARENA,
        "external backing differs from fixed arena",
    )
    require(
        memory["registered_host_arena_bytes"] == (ARENA if mode == "staged" else 0)
        and memory["external_gpu_arena_bytes"] == (ARENA if mode == "direct" else 0),
        "memory placement differs from transfer arm",
    )
    occupied = integer(
        memory["engine_valid_full_block_bytes"], "invalid occupied engine memory"
    )
    require(PAYLOAD <= occupied <= ARENA, "occupied engine memory violates budget")
    pool = integer(memory["engine_block_pool_bytes"], "invalid engine block pool")
    require(occupied <= pool <= ARENA, "engine block pool violates budget")
    require(
        type(memory["engine_live_read_blocks"]) is int
        and memory["engine_live_read_blocks"] == 0,
        "engine read ownership remains",
    )
    low = finite(
        memory["external_payload_byte_seconds_lower_bound"], "invalid residency bound"
    )
    high = finite(
        memory["external_payload_byte_seconds_upper_bound"], "invalid residency bound"
    )
    require(0 <= low <= high, "residency bounds disagree")
    cleanup = integer(row["cleanup_completed_ns"], "missing completed cleanup clock")
    native_drained = integer(row["native_drained_ns"], "missing native drain clock")
    require(
        row["drain_end_ns"]
        <= cleanup
        <= native_drained
        <= row["residency_clear_start_ns"]
        <= row["residency_clear_end_ns"],
        "residency cleared before ownership drained",
    )
    events = memory["diagnostic_engine_occupancy_events"]
    peak = memory["diagnostic_engine_peak_occupied_bytes"]
    if row["sample"] == -3 and not unused:
        require(
            isinstance(events, list) and events, "missing diagnostic engine occupancy"
        )
        stamps = [integer(e["time_ns"], "invalid occupancy clock") for e in events]
        occupied = [
            integer(e["occupied_bytes"], "invalid engine occupancy") for e in events
        ]
        require(
            stamps == sorted(stamps)
            and peak == max(occupied)
            and PAYLOAD <= peak <= ARENA,
            "diagnostic engine peak differs",
        )
        external = memory["diagnostic_external"]
        require(
            external["errors"] == []
            and isinstance(external["samples"], list)
            and external["samples"],
            "missing or failed external diagnostics",
        )
        require(
            finite(external["interval_seconds"], "invalid sampling interval") > 0,
            "invalid sampling interval",
        )
        sampled_bytes, ends = [], []
        for snapshot in external["samples"]:
            start = integer(snapshot["start_ns"], "invalid external diagnostic clock")
            end = integer(snapshot["end_ns"], "invalid external diagnostic clock")
            require(
                start <= end and (not ends or ends[-1] <= start),
                "external diagnostic clocks overlap or moved backwards",
            )
            ends.append(end)
            sampled_bytes.append(
                integer(
                    snapshot["l1"]["memory_used_bytes"],
                    "invalid sampled external residency",
                )
            )
            require(
                snapshot["prefetch"]["is_healthy"] is True,
                "sampled prefetch controller is unhealthy",
            )
        require(
            memory["diagnostic_external_observed_peak_bytes"] == max(sampled_bytes)
            and max(sampled_bytes) <= ARENA,
            "sampled external peak differs",
        )
    else:
        require(
            events is None
            and peak is None
            and memory["diagnostic_external"] is None
            and memory["diagnostic_external_observed_peak_bytes"] is None,
            "timed sample contains occupancy instrumentation",
        )
    return native, comparisons


def _verify_block(cell):
    config = json.loads((cell / "config.json").read_text())
    require(config["mode"] in ("staged", "direct"), "unknown transfer mode")
    require(
        config["phase"] in ("qualification", "pilot", "confirmation"), "unknown phase"
    )
    require(
        isinstance(config["fanouts"], list)
        and len(config["fanouts"]) == 1
        and type(config["fanouts"][0]) is int
        and config["fanouts"][0] in (4, 8),
        "unexpected fanout",
    )
    samples = integer(config["serving_samples"], "invalid sample count", 1)
    index = integer(config["block_index"], "invalid process block index")
    require(
        isinstance(config["block_id"], str) and bool(config["block_id"]),
        "missing block identity",
    )
    process = json.loads((cell / "consumer-process.json").read_text())
    integer(process["pid"], "invalid consumer PID", 1)
    require(
        isinstance(process["boot_id"], str)
        and bool(process["boot_id"])
        and isinstance(process["start_ticks"], str)
        and process["start_ticks"].isdecimal()
        and int(process["start_ticks"]) > 0
        and process["block_id"] == config["block_id"],
        "missing consumer process incarnation",
    )
    cases = CASES[:]
    if config["phase"] == "confirmation":
        lead = config["selected_lead_ms"]
        require(type(lead) is int and lead in LEADS, "invalid frozen lead")
        cases = [("demand", 0), ("early", lead)]
    random.Random(20261020 + index * 17 + int(config["mode"] == "direct")).shuffle(
        cases
    )
    require(
        json.loads((cell / "control-order.json").read_text())
        == [list(c) for c in cases],
        "control order differs from frozen randomized schedule",
    )
    oracle_path = Path(config["oracle"])
    oracle = json.loads(
        (oracle_path if oracle_path.is_absolute() else cell / oracle_path).read_text()
    )
    oracle_rows = [r for r in oracle if r["fanout"] == config["fanouts"][0]]
    require(
        sorted(r["sample"] for r in oracle_rows) == [-3, -2, -1, 0]
        and all(
            type(r["warmup"]) is bool and r["warmup"] == (r["sample"] < 0)
            for r in oracle_rows
        ),
        "missing or mislabeled recomputation controls",
    )
    selected = [r["rows"] for r in oracle_rows if r["sample"] == 0 and not r["warmup"]]
    require(
        len(selected) == 1 and len(selected[0]) == config["fanouts"][0],
        "missing or duplicated oracle fanout",
    )
    for baseline in oracle_rows:
        outputs(baseline["rows"], selected[0], 8, restored=False)
    producer_rows = json.loads((cell / "producer.json").read_text())["rows"]
    producer = json.loads((cell / "producer-prefix.json").read_text())
    require(
        len(producer_rows) == len(producer["members"]) == 1, "missing producer evidence"
    )
    digests(producer["members"][0]["checksums"])
    rows = json.loads((cell / "consumer.json").read_text())
    require(
        isinstance(rows, list) and len(rows) == len(cases) * (samples + 3),
        "incomplete process-block cases",
    )
    comparisons = child_checks = warm_checks = 0
    by_case = {}
    for offset, sample in enumerate(range(-3, samples)):
        batch = rows[offset * len(cases) : (offset + 1) * len(cases)]
        rotation = offset % len(cases)
        require(
            [(r["arm"], r["lead_ms"]) for r in batch]
            == cases[rotation:] + cases[:rotation],
            "missing, duplicated or unbalanced control order",
        )
        for row in batch:
            require(
                type(row["sample"]) is int
                and row["sample"] == sample
                and type(row["warmup"]) is bool
                and row["warmup"] == (sample < 0),
                "missing, repeated or mislabeled sample",
            )
            native, extra = trial(row, config, producer_rows)
            comparisons += extra + outputs(row["rows"], selected[0], 8)
            child_checks += len(row["rows"])
            warm_checks += len(row["warm_rows"]) + len(
                (row["placement"] or {}).get("warm_rows", [])
            )
            integrity(row, producer)
            by_case.setdefault(f"{row['arm']}-g{row['lead_ms']}", []).append(
                native["logical_requests"]
            )
    unused = None
    if config["phase"] == "qualification":
        native, extra = trial(
            json.loads((cell / "unused.json").read_text()), config, producer_rows, True
        )
        comparisons += extra
        warm_checks += 1
        unused = {
            "backend_chunk_reads": native["logical_requests"],
            "backend_payload_bytes": native["logical_bytes"],
            "unused_hints_charged": 1,
        }
    else:
        require(
            not (cell / "unused.json").exists(),
            "qualification artifact mixed into another phase",
        )
    terminal = json.loads((cell / "terminal.json").read_text())["storage_manager"]
    check_core(core(terminal), config["mode"], terminal=True)
    residency(terminal, 0)
    controller(terminal)
    return {
        "mode": config["mode"],
        "fanout": config["fanouts"][0],
        "phase": config["phase"],
        "block_id": config["block_id"],
        "process": process,
        "rows": rows,
        "backend_chunk_reads_by_case_including_warmups": by_case,
        "child_continuations_checked": child_checks,
        "warm_continuations_checked": warm_checks,
        "chosen_logprob_comparisons": comparisons,
        "maximum_logprob_delta": 0,
        "unused_hint_control": unused,
        "limits": [
            "Offline consistency does not authenticate fresh processes or hardware execution.",
            "Backend payload operations are not physical NVMe command traces.",
        ],
    }


def verify_block(cell: Path) -> dict:
    """Reject incomplete numerical, memory, route, ownership and clock evidence."""
    try:
        return _verify_block(Path(cell))
    except (KeyError, TypeError, IndexError, OSError, OverflowError) as error:
        raise ValueError("incomplete prefetch block evidence") from error
