# SPDX-License-Identifier: MIT
"""Reject duplicate reads, hidden placement work and incomplete model receipts."""

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
from prefetch_verify import ARENA, CASES, PAYLOAD, verify_block
from restore_common import ANCESTOR, CHUNK_BYTES, CHUNKS
from test_kmesh_prefetch_analysis import trial as clock_row
from test_kmesh_restore import core
from test_kmesh_retained import residency


def status(mode, reads=0, objects=0):
    return {
        "l1_manager": residency(objects) | {"memory_used_bytes": objects * CHUNK_BYTES},
        "l2_adapters": [{"core": core(mode, reads)}],
        "prefetch_controller": {
            "is_healthy": True,
            "thread_alive": True,
            "join_overlaps": True,
            "submission_queue_size": 0,
            "pending_queue_size": 0,
            "in_flight_request_count": 0,
            "lookup_phase_count": 0,
            "load_phase_count": 0,
            "completed_results_count": 0,
            "accepted_handle_count": 0,
            "dependency_waiter_count": 0,
            "overlap_joined_total": 1,
        },
    }


def output(name, length=8, external=False):
    return {
        "request_id": name,
        "token_ids": [7] * length,
        "chosen_logprobs": [-1.0] * length,
        "num_cached_tokens": ANCESTOR,
        "kv_transfer_params": {
            "cached_token_stats": {
                "num_vllm_cached_tokens": 0 if external else ANCESTOR,
                "num_lmcache_extra_cached_tokens": ANCESTOR if external else 0,
            }
        },
    }


def checksum():
    return {
        "status": "success",
        "layerwise": True,
        "chunk_size": 16,
        "num_chunks": CHUNKS,
        "chunk_checksums": {
            f"layer_{n}": [f"{n:02x}{j:02x}" * 8 for j in range(CHUNKS)]
            for n in range(24)
        },
    }


def prepared():
    return {
        "found_keys": CHUNKS,
        "total_keys": CHUNKS,
        "polls": 1,
        "completion_ms": 1,
        "l1": status("direct", objects=CHUNKS)["l1_manager"],
    }


def fixture_row(config, sample, arm, lead):
    row = clock_row(
        mode=config["mode"],
        fanout=config["fanouts"][0],
        sample=sample,
        arm=arm,
        lead=lead,
    )
    row["block_id"] = config["block_id"]
    for child in row["rows"]:
        child.update(output(child["request_id"]))
    for warm in row["warm_rows"]:
        warm.update(output(warm["request_id"], 1, external=True))
    if not row["warm_rows"] and arm != "engine_resident":
        row["rows"][0].update(output(row["rows"][0]["request_id"], external=True))
    resident = arm in ("external_resident", "engine_resident")
    row.update(
        {
            "reset_before": status(config["mode"]),
            "before": status(
                config["mode"], CHUNKS if resident else 0, CHUNKS if resident else 0
            ),
            "after": status(config["mode"], CHUNKS, CHUNKS),
            "cleared": status(config["mode"], CHUNKS),
            "preparation": prepared() if arm == "early" else None,
            "placement": None,
            "events": [],
            "live_prefix": None,
            "integrity": None,
            "cleanup_completed_ns": row["drain_end_ns"] + 1,
            "native_drained_ns": row["drain_end_ns"] + 2,
            "residency_clear_start_ns": row["drain_end_ns"] + 3,
            "residency_clear_end_ns": row["drain_end_ns"] + 4,
        }
    )
    if resident:
        row["placement"] = {
            "start_ns": row["trace_origin_ns"] - 20_000_000,
            "ready_ns": row["trace_origin_ns"] - 10_000_000,
            "native": {
                "before": core(config["mode"]),
                "after": core(config["mode"], CHUNKS),
            },
        }
        # Compact trial journal excludes the earlier placement operations.
        for name in ("before", "after", "cleared"):
            row[name]["l2_adapters"][0]["core"]["io_by_request"]["operations"] = []
        if arm == "external_resident":
            row["placement"]["external"] = prepared()
        else:
            row["placement"]["warm_rows"] = [output("placement-warm", 1, True)]
    elif row["warm_rows"]:
        row["events"] = [
            {"event": "declaration", "warm": row["warm_rows"][0]["request_id"]}
        ]
    counters = {
        "pending": 0,
        "family_prefetch_hit_total": 0,
        "family_prefetch_wasted_total": 0,
        "family_ensure_total": 0,
    }
    row["family_before"] = counters.copy()
    row["family_after"] = counters | (
        {"family_ensure_total": 1} if arm in ("early", "declaration") else {}
    )
    row["declaration_identity"] = "family" if arm in ("early", "declaration") else None
    row["singleflight"] = {"inflight": 0, "joined": 0}
    row["memory"] = {
        "engine_configured_bytes": ARENA,
        "external_configured_bytes": ARENA,
        "external_used_bytes": PAYLOAD,
        "external_backed_bytes": ARENA,
        "engine_valid_full_block_bytes": PAYLOAD,
        "engine_block_pool_bytes": ARENA,
        "engine_live_read_blocks": 0,
        "registered_host_arena_bytes": ARENA if config["mode"] == "staged" else 0,
        "external_gpu_arena_bytes": ARENA if config["mode"] == "direct" else 0,
        "external_payload_byte_seconds_lower_bound": 0,
        "external_payload_byte_seconds_upper_bound": PAYLOAD,
        "diagnostic_engine_occupancy_events": None,
        "diagnostic_engine_peak_occupied_bytes": None,
        "diagnostic_external": None,
        "diagnostic_external_observed_peak_bytes": None,
    }
    if sample == -3:
        names = [c["engine_request_id"] for c in row["rows"]]
        names += [w["request_id"] for w in row["warm_rows"]]
        blocks = list(range(1, 257))
        row["live_prefix"] = {
            "total_blocks": 2048,
            "null_block_id": 0,
            "events": [
                {
                    "action": "allocate",
                    "tables": {n: blocks for n in names},
                    "statuses": {n: "RUNNING" for n in names},
                    "computed_tokens": {n: ANCESTOR for n in names},
                },
                {"action": "free", "tables": {}, "statuses": {}, "computed_tokens": {}},
            ],
        }
        row["integrity"] = {
            "instance_id": 2,
            "members": [
                {"request_id": n, "block_ids": blocks, "checksums": checksum()}
                for n in names
            ],
        }
        row["memory"]["diagnostic_engine_occupancy_events"] = [
            {"time_ns": 1, "action": "start", "occupied_bytes": 0},
            {"time_ns": 2, "action": "end", "occupied_bytes": PAYLOAD},
        ]
        row["memory"]["diagnostic_engine_peak_occupied_bytes"] = PAYLOAD
        row["memory"]["diagnostic_external"] = {
            "samples": [
                {
                    "start_ns": 1,
                    "end_ns": 2,
                    "l1": status(config["mode"], objects=16)["l1_manager"],
                    "prefetch": status(config["mode"])["prefetch_controller"],
                }
            ],
            "errors": [],
            "interval_seconds": 0.001,
        }
        row["memory"]["diagnostic_external_observed_peak_bytes"] = PAYLOAD
    return row


class PrefetchVerificationTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.cell = Path(self.folder.name)
        self.make()

    def write(self, name, value):
        (self.cell / name).write_text(json.dumps(value))

    def read(self, name):
        return json.loads((self.cell / name).read_text())

    def make(self, mode="direct", phase="pilot", fanout=4):
        config = {
            "mode": mode,
            "phase": phase,
            "fanouts": [fanout],
            "serving_samples": 1,
            "block_index": 0,
            "block_id": "block-0",
            "selected_lead_ms": 100,
            "oracle": str(self.cell / "oracle.json"),
        }
        self.write("config.json", config)
        self.write(
            "consumer-process.json",
            {
                "pid": 123,
                "boot_id": "boot",
                "start_ticks": "12345",
                "block_id": "block-0",
            },
        )
        cases = [("demand", 0), ("early", 100)] if phase == "confirmation" else CASES[:]
        random.Random(20261020 + int(mode == "direct")).shuffle(cases)
        self.write("control-order.json", cases)
        rows = []
        for index, sample in enumerate(range(-3, 1)):
            offset = index % len(cases)
            rows.extend(
                fixture_row(config, sample, arm, lead)
                for arm, lead in cases[offset:] + cases[:offset]
            )
        self.write("consumer.json", rows)
        self.write(
            "oracle.json",
            [
                {
                    "fanout": fanout,
                    "sample": sample,
                    "warmup": sample < 0,
                    "rows": [
                        output(str(n))
                        | {"num_cached_tokens": 0, "kv_transfer_params": None}
                        for n in range(fanout)
                    ],
                }
                for sample in range(-3, 1)
            ],
        )
        self.write("producer.json", {"rows": [output("producer", 1)]})
        self.write(
            "producer-prefix.json",
            {"instance_id": 1, "members": [{"checksums": checksum()}]},
        )
        self.write("terminal.json", {"storage_manager": status(mode, CHUNKS)})
        if phase == "qualification":
            unused = fixture_row(config, 0, "early", 250)
            unused["rows"] = []
            unused["warm_rows"] = [output("unused-warm", 1, True)]
            unused["family_after"]["family_prefetch_wasted_total"] = 1
            unused["group_end_ns"] = None
            for key in (
                "children_wall_ms",
                "deadline_group_completion_ms",
                "workflow_completion_ms",
            ):
                unused[key] = None
            self.write("unused.json", unused)

    def mutate(self, action, arm="demand", sample=-3):
        rows = self.read("consumer.json")
        action(next(r for r in rows if r["arm"] == arm and r["sample"] == sample))
        self.write("consumer.json", rows)

    def reject(self, action, message, **selection):
        self.mutate(action, **selection)
        with self.assertRaisesRegex(ValueError, message):
            verify_block(self.cell)

    def test_full_pilot_and_raw_rows_preserved(self):
        original = self.read("consumer.json")
        result = verify_block(self.cell)
        self.assertEqual(result["rows"], original)
        self.assertEqual(result["maximum_logprob_delta"], 0)
        self.assertEqual(result["child_continuations_checked"], 128)
        self.assertEqual(
            result["backend_chunk_reads_by_case_including_warmups"][
                "engine_resident-g0"
            ],
            [0] * 4,
        )
        self.make(mode="staged")
        verify_block(self.cell)

    def test_confirmation_and_unused_qualification(self):
        self.make(phase="confirmation", fanout=8)
        self.assertEqual(verify_block(self.cell)["child_continuations_checked"], 64)
        self.make(phase="qualification")
        result = verify_block(self.cell)
        self.assertEqual(result["unused_hint_control"]["backend_chunk_reads"], 16)
        unused = self.read("unused.json")
        unused["family_after"]["family_prefetch_hit_total"] = 1
        self.write("unused.json", unused)
        with self.assertRaisesRegex(ValueError, "child credit"):
            verify_block(self.cell)

    def test_any_nonzero_logprob_delta_and_nonfinite_rejected(self):
        for value in (-1.00000001, float("nan")):
            with self.subTest(value=value):
                self.make()
                self.reject(
                    lambda r: r["rows"][0]["chosen_logprobs"].__setitem__(0, value),
                    "log probability",
                )

    def test_oracle_warmups_are_preserved_and_must_agree(self):
        oracle = self.read("oracle.json")
        self.assertEqual(len(oracle), 4)
        oracle[0]["rows"][0]["chosen_logprobs"][0] = -2.0
        self.write("oracle.json", oracle)
        with self.assertRaisesRegex(ValueError, "log probability"):
            verify_block(self.cell)

    def test_duplicate_reads_rejected_despite_one_engine_load(self):
        self.reject(
            lambda r: r["after"]["l2_adapters"][0].update(core=core("direct", 32)),
            "exactly one backend ancestor",
        )

    def test_resident_reads_must_be_charged_before_trial(self):
        self.reject(
            lambda r: r["placement"]["native"].update(after=core("direct")),
            "exactly one backend ancestor|payload route",
            arm="external_resident",
        )
        self.make()

        def extra_trial_read(row):
            extra = core("direct", 32)
            extra["io_by_request"]["operations"] = extra["io_by_request"]["operations"][
                16:
            ]
            row["after"]["l2_adapters"][0]["core"] = extra

        self.reject(
            extra_trial_read,
            "resident trial performed SSD reads",
            arm="engine_resident",
        )

    def test_partial_or_multiple_engine_loads_rejected(self):
        self.reject(
            lambda r: r["rows"][1]["kv_transfer_params"]["cached_token_stats"].update(
                num_lmcache_extra_cached_tokens=ANCESTOR
            ),
            "loaded exactly once",
        )
        self.make()
        self.reject(
            lambda r: r["rows"][0]["kv_transfer_params"]["cached_token_stats"].update(
                num_lmcache_extra_cached_tokens=256
            ),
            "partial engine",
        )

    def test_local_reference_and_no_demand_warming(self):
        self.reject(
            lambda r: r["rows"][0]["kv_transfer_params"]["cached_token_stats"].update(
                num_vllm_cached_tokens=0
            ),
            "reuse its local",
            arm="engine_resident",
        )
        self.make()
        self.reject(
            lambda r: r.update(warm_rows=[output("forbidden", 1, True)]),
            "loaded exactly once",
        )

    def test_temporary_locked_and_retained_memory_rejected(self):
        for key in ("read_locked_count", "external_io_lease_count", "temporary_count"):
            with self.subTest(key=key):
                self.make()
                self.reject(
                    lambda r: r["after"]["l1_manager"].update({key: 1}),
                    "ownership remain",
                )
        self.make()
        self.reject(
            lambda r: r["after"]["l1_manager"].update(memory_used_bytes=PAYLOAD - 1),
            "payload bytes",
        )

    def test_async_preparation_read_refs_are_allowed_but_final_refs_are_not(self):
        self.mutate(
            lambda r: r["preparation"]["l1"].update(read_locked_count=1), arm="early"
        )
        verify_block(self.cell)
        self.reject(
            lambda r: r["after"]["l1_manager"].update(read_locked_count=1),
            "ownership remain",
            arm="early",
        )

    def test_native_route_errors_and_controller_owners_rejected(self):
        changes = [
            lambda r: r["after"]["l2_adapters"][0]["core"]["io_by_request"].update(
                dropped_rows=1
            ),
            lambda r: r["after"]["l2_adapters"][0]["core"].update(poisoned=True),
            lambda r: r["after"]["l2_adapters"][0]["core"]["io_by_request"][
                "operations"
            ][0].update(path="host_fixed"),
            lambda r: r["after"]["prefetch_controller"].update(accepted_handle_count=1),
            lambda r: r["after"]["prefetch_controller"].update(
                dependency_waiter_count=1
            ),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.make()
                self.reject(change, "native|unhealthy|route|ownership")

    def test_model_integrity_cannot_use_empty_or_foreign_receipts(self):
        self.reject(
            lambda r: r["integrity"]["members"][0]["checksums"]["chunk_checksums"][
                "layer_0"
            ].__setitem__(0, "f" * 32),
            "digests differ",
        )
        self.make()
        self.reject(lambda r: r["integrity"].update(instance_id=1), "producer KV")
        self.make()
        self.reject(
            lambda r: r["integrity"]["members"][0].update(request_id="unrelated"),
            "unknown or repeated",
        )
        self.make()
        self.reject(
            lambda r: r["integrity"]["members"][0]["checksums"].update(
                chunk_checksums={}
            ),
            "missing model KV layers",
        )

    def test_duplicate_ready_ancestors_and_retained_instrumentation_rejected(self):
        def duplicate(row):
            event = row["live_prefix"]["events"][0]
            name = row["rows"][1]["engine_request_id"]
            event["tables"][name] = list(range(257, 513))

        self.reject(duplicate, "ready GPU ancestor")
        self.make()
        self.reject(
            lambda r: r["memory"].update(diagnostic_engine_occupancy_events=[]),
            "occupancy instrumentation",
            sample=0,
        )

    def test_sampled_external_evidence_is_not_a_continuous_peak(self):
        self.mutate(
            lambda r: r["memory"].update(
                diagnostic_external_observed_peak_bytes=0,
                diagnostic_external={
                    "samples": [
                        {
                            "start_ns": 1,
                            "end_ns": 2,
                            "l1": {"memory_used_bytes": 0},
                            "prefetch": {"is_healthy": True},
                        }
                    ],
                    "errors": [],
                    "interval_seconds": 0.001,
                },
            )
        )
        verify_block(self.cell)
        self.reject(
            lambda r: r["memory"]["diagnostic_external"]["errors"].append(
                "status failed"
            ),
            "failed external diagnostics",
        )

    def test_fixed_deadline_and_cleanup_are_verified(self):
        self.reject(
            lambda r: r.update(scheduled_arrival_ns=r["scheduled_arrival_ns"] + 1),
            "fixed origin",
        )
        self.make()
        self.reject(
            lambda r: r.update(cleanup_completed_ns=r["residency_clear_end_ns"] + 1),
            "cleared before",
        )

    def test_native_drain_precedes_clear_and_follows_cleanup(self):
        for field, offset in (
            ("cleanup_completed_ns", -1),
            ("residency_clear_start_ns", 1),
        ):
            with self.subTest(boundary=field):
                self.make()
                self.reject(
                    lambda r: r.update(native_drained_ns=r[field] + offset),
                    "cleared before",
                )
        self.make()
        self.reject(lambda r: r.pop("native_drained_ns"), "incomplete")

    def test_engine_read_ownership_and_physical_memory_budgets(self):
        for field, value, message in (
            ("engine_live_read_blocks", 1, "engine read ownership"),
            ("engine_live_read_blocks", False, "engine read ownership"),
            ("engine_block_pool_bytes", PAYLOAD - 1, "block pool violates"),
            ("engine_block_pool_bytes", ARENA + 1, "block pool violates"),
            ("external_backed_bytes", ARENA - 1, "external backing"),
        ):
            with self.subTest(field=field, value=value):
                self.make()
                self.reject(lambda r: r["memory"].update({field: value}), message)

    def test_native_overlap_counter_cannot_move_backwards(self):
        self.reject(
            lambda r: r["after"]["prefetch_controller"].update(overlap_joined_total=0),
            "overlap counter moved backwards",
        )

    def test_missing_duplicate_and_mislabeled_samples_rejected(self):
        rows = self.read("consumer.json")
        self.write("consumer.json", rows[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete process"):
            verify_block(self.cell)
        self.make()
        self.reject(lambda r: r.update(warmup=False), "mislabeled sample")
        self.make()
        order = self.read("control-order.json")
        self.write("control-order.json", order[::-1])
        with self.assertRaisesRegex(ValueError, "frozen randomized"):
            verify_block(self.cell)

    def test_terminal_and_process_receipts_are_required(self):
        terminal = self.read("terminal.json")
        terminal["storage_manager"]["l2_adapters"][0]["core"]["locked_key_count"] = 1
        self.write("terminal.json", terminal)
        with self.assertRaisesRegex(ValueError, "terminal key ownership"):
            verify_block(self.cell)
        self.make()
        process = self.read("consumer-process.json")
        process["start_ticks"] = ""
        self.write("consumer-process.json", process)
        with self.assertRaisesRegex(ValueError, "process incarnation"):
            verify_block(self.cell)


if __name__ == "__main__":
    unittest.main()
