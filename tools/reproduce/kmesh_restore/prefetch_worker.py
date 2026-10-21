# SPDX-License-Identifier: MIT
"""Compare read-once demand and early loading at frozen request deadlines."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import random
import sys
from threading import Event, Thread
import time

from prefetch_clock import replay_fixed
from restore_common import (
    ANCESTOR,
    CHUNK_BYTES,
    cold_l1,
    compact_core_pair,
    http,
    prompts,
    write_json,
)
from retained_worker import warm_external
from worker import (
    create_llm,
    prefix_checksums,
    record_live_prefix_blocks,
    record_prefix_blocks,
    result_rows,
)

PILOT_CASES = (
    [("demand", 0)]
    + [("early", lead) for lead in (0, 25, 100, 250)]
    + [("declaration", 0), ("external_resident", 0), ("engine_resident", 0)]
)


@contextmanager
def observe_engine_occupancy(manager):
    """Measure pool occupancy at allocation/free boundaries in diagnostics only."""
    allocate, free = manager.allocate_slots, manager.free
    events = []

    def snapshot(action):
        occupied = sum(
            not block.is_null and (block.ref_cnt > 0 or block.block_hash is not None)
            for block in manager.block_pool.blocks
        )
        events.append(
            {
                "time_ns": time.perf_counter_ns(),
                "action": action,
                "occupied_bytes": occupied * (CHUNK_BYTES // 16),
            }
        )

    def on_allocate(*args, **kwargs):
        result = allocate(*args, **kwargs)
        snapshot("allocate")
        return result

    def on_free(*args, **kwargs):
        snapshot("before_free")
        result = free(*args, **kwargs)
        snapshot("free")
        return result

    manager.allocate_slots, manager.free = on_allocate, on_free
    snapshot("start")
    try:
        yield events
    finally:
        snapshot("end")
        manager.allocate_slots, manager.free = allocate, free


@contextmanager
def observe_external(port):
    """Sample external residency and control stages outside retained timings."""
    stopped = Event()
    evidence = {"samples": [], "errors": [], "interval_seconds": 0.001}

    def collect():
        while not stopped.is_set():
            start = time.perf_counter_ns()
            try:
                status = http(port, "status", timeout=2)["storage_manager"]
                evidence["samples"].append(
                    {
                        "start_ns": start,
                        "end_ns": time.perf_counter_ns(),
                        "l1": status["l1_manager"],
                        "prefetch": status["prefetch_controller"],
                    }
                )
            except Exception as error:
                evidence["errors"].append(str(error))
                break
            stopped.wait(0.001)

    thread = Thread(target=collect)
    thread.start()
    try:
        yield evidence
    finally:
        stopped.set()
        thread.join(timeout=3)
        if thread.is_alive():
            raise RuntimeError("diagnostic HTTP observer did not drain")


def settled_status(port):
    """Observe asynchronous completion without clearing the retained objects."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = http(port, "status", timeout=2)["storage_manager"]
        l1, prefetch = status["l1_manager"], status["prefetch_controller"]
        core = status["l2_adapters"][0]["core"]
        if (
            not l1["is_healthy"]
            or not prefetch["is_healthy"]
            or not prefetch["thread_alive"]
            or not core["is_healthy"]
            or core["poisoned"]
            or core["worker_error"]
            or core["quarantined_slot_count"]
        ):
            raise RuntimeError("storage owner is unhealthy during drainage")
        pending = (
            [
                l1[key]
                for key in (
                    "external_io_lease_count",
                    "write_locked_count",
                    "read_locked_count",
                    "temporary_count",
                    "staging_object_count",
                    "staging_bytes",
                )
            ]
            + [
                prefetch[key]
                for key in (
                    "submission_queue_size",
                    "pending_queue_size",
                    "in_flight_request_count",
                    "lookup_phase_count",
                    "load_phase_count",
                    "completed_results_count",
                    "accepted_handle_count",
                    "dependency_waiter_count",
                )
            ]
            + [
                core[key]
                for key in (
                    "inflight_io_count",
                    "quarantined_slot_count",
                    "locked_key_count",
                    "inflight_key_count",
                )
            ]
        )
        if not any(pending):
            return status
        time.sleep(0.001)
    raise TimeoutError("retained storage ownership did not drain")


def generate_preparation(llm, prompts, params):
    """Keep offline generation's parameter mutation out of streaming trials."""
    return llm.generate(prompts, deepcopy(params), use_tqdm=False)


def main(config):
    """Run one independent engine block with balanced randomized control order."""
    llm = create_llm(config, "consumer")
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    connector = scheduler.connector
    shared = {
        "temperature": 0,
        "ignore_eos": True,
        "seed": 0,
        "logprobs": 1,
        "output_kind": RequestOutputKind.CUMULATIVE,
        "extra_args": {"kv_transfer_params": {"cached_token_stats": {}}},
    }
    params = SamplingParams(max_tokens=8, **shared)
    warm_params = SamplingParams(max_tokens=1, **shared)
    ancestor, children = prompts(8)
    fanout = config["fanouts"][0]
    output = Path(config["output"])
    producer = json.loads((output / "producer-prefix.json").read_text())
    expected = producer["members"][0]["checksums"]["chunk_checksums"]
    records = []
    write_json(
        output / "consumer-process.json",
        {
            "pid": os.getpid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start_ticks": Path("/proc/self/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
            "block_id": config["block_id"],
        },
    )
    request = partial(http, timeout=2)
    cases = (
        [("demand", 0), ("early", config["selected_lead_ms"])]
        if config["phase"] == "confirmation"
        else PILOT_CASES[:]
    )
    rng = random.Random(
        20261020 + config["block_index"] * 17 + int(config["mode"] == "direct")
    )
    rng.shuffle(cases)
    write_json(output / "control-order.json", cases)

    def run_trial(arm, lead, sample, executor, ledger, *, unused=False):
        if not llm.reset_prefix_cache():
            raise RuntimeError("engine prefix reset refused")
        reset_before = cold_l1(config["http_port"])
        ledger["reset_before"] = reset_before
        placement = None
        if arm in ("external_resident", "engine_resident"):
            start = time.perf_counter_ns()
            placement = {"start_ns": start}
            if arm == "external_resident":
                placement["external"] = warm_external(config, ancestor, request=request)
            else:
                placement["warm_rows"] = result_rows(
                    generate_preparation(
                        llm,
                        [{"prompt_token_ids": ancestor + [7]}],
                        warm_params,
                    ),
                    1,
                )
            placement["ready_ns"] = time.perf_counter_ns()
        ledger["placement"] = placement
        before = settled_status(config["http_port"])
        ledger["before"] = before
        family_before = connector.get_family_prefetch_stats()
        label = f"{config['block_id']}-{arm}-g{lead}-s{sample}"
        if unused:
            label += "-unused"
        observe = sample == -3 and not unused
        with (
            observe_external(config["http_port"]) if observe else nullcontext(None)
        ) as external_observation, (
            observe_engine_occupancy(scheduler.kv_cache_manager)
            if observe
            else nullcontext(None)
        ) as occupancy, (
            record_live_prefix_blocks(llm) if observe else nullcontext(None)
        ) as live, (
            record_prefix_blocks(llm) if observe else nullcontext(None)
        ) as tables:
            row = replay_fixed(
                engine,
                ancestor,
                [] if unused else children[:fanout],
                params,
                warm_params,
                arm,
                lead,
                label,
                start_prefetch=lambda: executor.submit(
                    warm_external, config, ancestor, request=request
                ),
                declare=lambda warm_id: connector.declare_family_prefetch(
                    ancestor,
                    ANCESTOR,
                    fanout,
                    bytes_per_token=CHUNK_BYTES // 256,
                    prefetch_request_id=warm_id,
                ),
                release_declaration=connector.release_family_prefetch,
                on_partial=lambda partial_row: write_json(
                    output / (label + "-failed.json"), partial_row
                ),
            )
            ledger.update(row)
            write_json(output / (label + "-raw.json"), ledger)
        integrity = None
        if tables is not None:
            integrity = prefix_checksums(config, tables, [producer["instance_id"]])
            ledger["integrity"] = integrity
            write_json(output / (label + "-prefix.json"), integrity)
            if len(integrity["members"]) != fanout + len(row["warm_rows"]) or any(
                member["checksums"]["chunk_checksums"] != expected
                for member in integrity["members"]
            ):
                raise ValueError("prepared or demand-loaded model KV differs")
        after = settled_status(config["http_port"])
        row["native_drained_ns"] = time.perf_counter_ns()
        ledger["after"] = after
        pool = scheduler.kv_cache_manager.block_pool
        memory = {
            "engine_configured_bytes": 512 << 20,
            "engine_block_pool_bytes": len(pool.blocks) * (CHUNK_BYTES // 16),
            "external_configured_bytes": 512 << 20,
            "engine_valid_full_block_bytes": sum(
                not b.is_null and b.block_hash is not None for b in pool.blocks
            )
            * (CHUNK_BYTES // 16),
            "engine_live_read_blocks": sum(
                not b.is_null and b.ref_cnt > 0 for b in pool.blocks
            ),
            "external_used_bytes": after["l1_manager"]["memory_used_bytes"],
            "external_backed_bytes": after["l1_manager"]["memory_total_bytes"],
            "registered_host_arena_bytes": (
                (512 << 20) if config["mode"] == "staged" else 0
            ),
            "external_gpu_arena_bytes": (
                (512 << 20) if config["mode"] == "direct" else 0
            ),
            "diagnostic_engine_occupancy_events": occupancy,
            "diagnostic_external": external_observation,
            "diagnostic_external_observed_peak_bytes": (
                max(
                    sample["l1"]["memory_used_bytes"]
                    for sample in external_observation["samples"]
                )
                if external_observation is not None and external_observation["samples"]
                else None
            ),
            "diagnostic_engine_peak_occupied_bytes": (
                max(event["occupied_bytes"] for event in occupancy)
                if occupancy is not None
                else None
            ),
        }
        clear_start = time.perf_counter_ns()
        cleared = cold_l1(config["http_port"])
        ledger["cleared"] = cleared
        clear_end = time.perf_counter_ns()
        known_ready = placement["ready_ns"] if placement else row["external_ready_ns"]
        if known_ready is None and row["rows"]:
            known_ready = min(child["first_token_observed_ns"] for child in row["rows"])
        possible_start = (
            placement["start_ns"]
            if placement
            else (
                row["prepare_start_ns"]
                if row["prepare_start_ns"] is not None
                else row["group_submit_ns"]
            )
        )
        resident_bytes = memory["external_used_bytes"]
        memory["external_payload_byte_seconds_lower_bound"] = (
            resident_bytes * max(0, clear_start - known_ready) / 1e9
        )
        memory["external_payload_byte_seconds_upper_bound"] = (
            resident_bytes * (clear_end - possible_start) / 1e9
        )
        reset_core = reset_before["l2_adapters"][0]["core"]
        before_core = before["l2_adapters"][0]["core"]
        after_core = after["l2_adapters"][0]["core"]
        placement_before, placement_after = compact_core_pair(reset_core, before_core)
        if placement is not None:
            placement["native"] = {"before": placement_before, "after": placement_after}
        reset_before["l2_adapters"][0]["core"] = placement_before
        before["l2_adapters"][0]["core"], after["l2_adapters"][0]["core"] = (
            compact_core_pair(before_core, after_core)
        )
        _, cleared["l2_adapters"][0]["core"] = compact_core_pair(
            after_core, cleared["l2_adapters"][0]["core"]
        )
        complete = row | {
            "mode": config["mode"],
            "block_id": config["block_id"],
            "fanout": fanout,
            "sample": sample,
            "warmup": sample < 0,
            "before": before,
            "after": after,
            "reset_before": reset_before,
            "cleared": cleared,
            "placement": placement,
            "live_prefix": live,
            "integrity": integrity,
            "singleflight": connector.get_singleflight_stats(),
            "family_before": family_before,
            "family_after": connector.get_family_prefetch_stats(),
            "memory": memory,
            "residency_clear_start_ns": clear_start,
            "residency_clear_end_ns": clear_end,
        }
        write_json(output / (label + "-raw.json"), complete)
        return complete

    def trial(arm, lead, sample, executor, *, unused=False):
        ledger = {
            "arm": arm,
            "lead_ms": lead,
            "sample": sample,
            "mode": config["mode"],
            "block_id": config["block_id"],
            "fanout": fanout,
            "unused": unused,
        }
        try:
            return run_trial(arm, lead, sample, executor, ledger, unused=unused)
        except BaseException as error:
            ledger.update(error_type=type(error).__name__, error=str(error))
            write_json(
                output / f"failed-s{sample}-{arm}-g{lead}-u{int(unused)}.json", ledger
            )
            raise

    try:
        generate_preparation(
            llm, [{"prompt_token_ids": [44, 55, 66, 77, 88] * 20}], params
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            for index, sample in enumerate(range(-3, config["serving_samples"])):
                offset = index % len(cases)
                for arm, lead in cases[offset:] + cases[:offset]:
                    records.append(trial(arm, lead, sample, executor))
                    write_json(output / "consumer.json", records)
            if config["phase"] == "qualification":
                write_json(
                    output / "unused.json",
                    trial("early", 250, 0, executor, unused=True),
                )
    finally:
        write_json(output / "consumer.json", records)
        engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
