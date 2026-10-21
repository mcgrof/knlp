# SPDX-License-Identifier: MIT
"""Wait for asynchronous owners without discarding retained cache evidence."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
import prefetch_worker as worker


def status():
    return {
        "l1_manager": {
            "is_healthy": True,
            "total_object_count": 16,
            "memory_used_bytes": 48 << 20,
            "memory_total_bytes": 512 << 20,
            "external_io_lease_count": 0,
            "write_locked_count": 0,
            "read_locked_count": 0,
            "temporary_count": 0,
            "staging_object_count": 0,
            "staging_bytes": 0,
            "backing_resources_retained": False,
        },
        "prefetch_controller": {
            "is_healthy": True,
            "thread_alive": True,
            "submission_queue_size": 0,
            "pending_queue_size": 0,
            "in_flight_request_count": 0,
            "lookup_phase_count": 0,
            "load_phase_count": 0,
            "completed_results_count": 0,
            "accepted_handle_count": 0,
            "dependency_waiter_count": 0,
        },
        "l2_adapters": [
            {
                "core": {
                    "is_healthy": True,
                    "poisoned": False,
                    "worker_error": None,
                    "inflight_io_count": 0,
                    "quarantined_slot_count": 0,
                    "locked_key_count": 0,
                    "inflight_key_count": 0,
                }
            }
        ],
    }


class Clock:
    def __init__(self, step=0.001):
        self.now = 0
        self.step = step
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += max(self.step, seconds)


@contextmanager
def drainage(responses, *, step=0.001):
    clock = Clock(step)
    with (
        patch.object(worker.time, "monotonic", side_effect=clock.monotonic),
        patch.object(worker.time, "sleep", side_effect=clock.sleep),
        patch.object(
            worker.time,
            "perf_counter_ns",
            side_effect=AssertionError("drainage touched the request stopwatch"),
        ),
        patch.object(worker, "http", side_effect=responses) as request,
        patch.object(
            worker,
            "cold_l1",
            side_effect=AssertionError("drainage cleared retained KV"),
        ),
        patch.object(
            worker,
            "replay_fixed",
            side_effect=AssertionError("drainage replayed requests"),
        ),
    ):
        yield clock, request


class PreparationTest(unittest.TestCase):
    def test_offline_generation_cannot_change_trial_output_delivery(self):
        params = SimpleNamespace(output_kind="cumulative", extra_args={"nested": []})

        def generate(prompts, supplied, *, use_tqdm):
            self.assertFalse(use_tqdm)
            supplied.output_kind = "final-only"
            supplied.extra_args["nested"].append("mutated")
            return ["prepared"]

        llm = SimpleNamespace(generate=generate)
        self.assertEqual(worker.generate_preparation(llm, [[1]], params), ["prepared"])
        self.assertEqual(params.output_kind, "cumulative")
        self.assertEqual(params.extra_args, {"nested": []})


class SettledStatusTest(unittest.TestCase):
    def test_read_references_native_io_and_handles_settle_without_cache_clear(self):
        leased, native, handle, settled = [status() for _ in range(4)]
        leased["l1_manager"].update(read_locked_count=16, external_io_lease_count=1)
        native["l2_adapters"][0]["core"]["inflight_io_count"] = 1
        handle["prefetch_controller"]["accepted_handle_count"] = 1
        snapshots = [leased, native, handle, settled]
        original = deepcopy(snapshots)
        with drainage([{"storage_manager": row} for row in snapshots]) as (
            clock,
            request,
        ):
            result = worker.settled_status(1234)
            self.assertIs(result, settled)
            self.assertEqual(result["l1_manager"]["total_object_count"], 16)
            self.assertEqual(result["l1_manager"]["memory_used_bytes"], 48 << 20)
            self.assertEqual(request.call_count, 4)
            self.assertEqual(len(clock.sleeps), 3)
            self.assertTrue(
                all(args == (1234, "status") for args, _ in request.call_args_list)
            )
            self.assertTrue(
                all(kwargs == {"timeout": 2} for _, kwargs in request.call_args_list)
            )
        self.assertEqual(snapshots, original)

    def test_each_asynchronous_owner_prevents_premature_settlement(self):
        owners = {
            "l1_manager": (
                "external_io_lease_count",
                "write_locked_count",
                "read_locked_count",
                "temporary_count",
                "staging_object_count",
                "staging_bytes",
            ),
            "prefetch_controller": (
                "submission_queue_size",
                "pending_queue_size",
                "in_flight_request_count",
                "lookup_phase_count",
                "load_phase_count",
                "completed_results_count",
                "accepted_handle_count",
                "dependency_waiter_count",
            ),
            "core": ("inflight_io_count", "locked_key_count", "inflight_key_count"),
        }
        for owner, fields in owners.items():
            for field in fields:
                with self.subTest(owner=owner, field=field):
                    pending, done = status(), status()
                    group = (
                        pending["l2_adapters"][0]["core"]
                        if owner == "core"
                        else pending[owner]
                    )
                    group[field] = 1
                    with drainage(
                        [{"storage_manager": pending}, {"storage_manager": done}]
                    ) as (clock, _):
                        self.assertIs(worker.settled_status(1234), done)
                        self.assertTrue(clock.sleeps)

    def test_unhealthy_owner_refuses_without_waiting_for_a_quiet_snapshot(self):
        for owner in ("l1_manager", "prefetch_controller", "core"):
            with self.subTest(owner=owner):
                broken = status()
                group = (
                    broken["l2_adapters"][0]["core"]
                    if owner == "core"
                    else broken[owner]
                )
                group["is_healthy"] = False
                with drainage([{"storage_manager": broken}]) as (clock, request):
                    with self.assertRaisesRegex(RuntimeError, "unhealthy"):
                        worker.settled_status(1234)
                    request.assert_called_once()
                    self.assertEqual(clock.sleeps, [])

    def test_health_failure_during_read_reference_drain_is_not_accepted(self):
        pending, broken = status(), status()
        pending["l1_manager"]["read_locked_count"] = 16
        broken["l2_adapters"][0]["core"]["is_healthy"] = False
        with drainage([{"storage_manager": pending}, {"storage_manager": broken}]):
            with self.assertRaisesRegex(RuntimeError, "unhealthy"):
                worker.settled_status(1234)

    def test_persistent_quarantine_cannot_be_reported_as_settled(self):
        quarantined = status()
        quarantined["l2_adapters"][0]["core"]["quarantined_slot_count"] = 1
        with drainage(
            lambda *args, **kwargs: {"storage_manager": quarantined}, step=20
        ) as (clock, request):
            with self.assertRaises(RuntimeError):
                worker.settled_status(1234)
            request.assert_called_once()
            self.assertEqual(clock.sleeps, [])

    def test_dead_controller_or_native_fault_refuses_immediately(self):
        for owner, field, value in (
            ("prefetch_controller", "thread_alive", False),
            ("core", "poisoned", True),
            ("core", "worker_error", "native worker failed"),
        ):
            with self.subTest(field=field):
                broken = status()
                group = (
                    broken["l2_adapters"][0]["core"]
                    if owner == "core"
                    else broken[owner]
                )
                group[field] = value
                with drainage([{"storage_manager": broken}]) as (clock, request):
                    with self.assertRaises(RuntimeError):
                        worker.settled_status(1234)
                    request.assert_called_once()
                    self.assertEqual(clock.sleeps, [])

    def test_changing_pending_owners_does_not_restart_the_drain_deadline(self):
        waiting = []
        for owner, field in (
            ("l1_manager", "read_locked_count"),
            ("prefetch_controller", "accepted_handle_count"),
            ("l1_manager", "external_io_lease_count"),
        ):
            item = status()
            item[owner][field] = 1
            waiting.append({"storage_manager": item})
        with drainage(waiting, step=20) as (clock, request):
            with self.assertRaisesRegex(TimeoutError, "did not drain"):
                worker.settled_status(1234)
            self.assertEqual(clock.now, 60)
            self.assertEqual(request.call_count, 3)

    def test_missing_status_evidence_propagates_failure(self):
        with drainage(ConnectionError("status unavailable")):
            with self.assertRaisesRegex(ConnectionError, "status unavailable"):
                worker.settled_status(1234)


class OccupancyHookTest(unittest.TestCase):
    def test_hooks_restore_after_failed_diagnostic_and_ignore_null_blocks(self):
        blocks = [
            SimpleNamespace(is_null=True, ref_cnt=1, block_hash="null"),
            SimpleNamespace(is_null=False, ref_cnt=1, block_hash=None),
            SimpleNamespace(is_null=False, ref_cnt=0, block_hash="retained"),
            SimpleNamespace(is_null=False, ref_cnt=0, block_hash=None),
        ]
        allocate, free = Mock(return_value="allocation"), Mock()
        manager = SimpleNamespace(
            allocate_slots=allocate,
            free=free,
            block_pool=SimpleNamespace(blocks=blocks),
        )
        with self.assertRaisesRegex(RuntimeError, "trial failed"):
            with worker.observe_engine_occupancy(manager) as events:
                self.assertEqual(manager.allocate_slots("request"), "allocation")
                manager.free("request")
                raise RuntimeError("trial failed")
        self.assertIs(manager.allocate_slots, allocate)
        self.assertIs(manager.free, free)
        allocate.assert_called_once_with("request")
        free.assert_called_once_with("request")
        self.assertEqual(events[0]["occupied_bytes"], 2 * (worker.CHUNK_BYTES // 16))
        self.assertEqual(events[-1]["action"], "end")


if __name__ == "__main__":
    unittest.main()
