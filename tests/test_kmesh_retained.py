# SPDX-License-Identifier: MIT
"""Reject uncharged retention, locked residency and storage-read duplication."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
from family_worker import replay
from retained_verify import check_retention
from retained_worker import warm_external
from test_kmesh_family import Clock, Engine
import run as restore_run


def residency(objects=0):
    return {
        "is_healthy": True,
        "backing_resources_retained": False,
        "total_object_count": objects,
        "external_io_lease_count": 0,
        "write_locked_count": 0,
        "read_locked_count": 0,
        "temporary_count": 0,
        "staging_object_count": 0,
        "staging_bytes": 0,
    }


class RetainedPrefetchTest(unittest.TestCase):
    def setUp(self):
        stats = {
            "pending": 0,
            "family_ensure_total": 0,
            "family_prefetch_hit_total": 0,
            "family_prefetch_wasted_total": 0,
        }
        self.row = {
            "policy": "family",
            "arm": "retained",
            "fanout": 1,
            "requested_gap_ms": 250,
            "actual_gap_ms": 250,
            "warm_complete_ms": 50,
            "warm_completed_before_release": True,
            "singleflight": {"inflight": 0, "joined": 0},
            "family_before": stats,
            "family_after": stats
            | {"family_ensure_total": 1, "family_prefetch_hit_total": 1},
            "preparation": {
                "found_keys": 16,
                "total_keys": 16,
                "completion_ms": 20,
                "l1": residency(16),
            },
            "rows": [
                {
                    "kv_transfer_params": {
                        "cached_token_stats": {
                            "num_vllm_cached_tokens": 4096,
                            "num_lmcache_extra_cached_tokens": 0,
                        }
                    }
                }
            ],
            "warm_rows": [
                {
                    "kv_transfer_params": {
                        "cached_token_stats": {
                            "num_vllm_cached_tokens": 0,
                            "num_lmcache_extra_cached_tokens": 4096,
                        }
                    }
                }
            ],
            "before": {"l1_manager": residency(), "l2_adapters": [{"core": {}}]},
            "after": {"l1_manager": residency(16), "l2_adapters": [{"core": {}}]},
        }

    def verify(self, reads=16, unused=False):
        with patch(
            "retained_verify.read_delta", return_value={"logical_requests": reads}
        ):
            return check_retention(self.row, "direct", unused=unused)

    def test_complete_retained_family(self):
        self.assertEqual(self.verify(), 16)

    def test_duplicate_backend_reads_rejected_despite_gpu_sharing(self):
        with self.assertRaisesRegex(ValueError, "exactly one backend ancestor"):
            self.verify(reads=32)

    def test_prefetch_time_cannot_be_excluded(self):
        self.row["preparation"]["completion_ms"] = 60
        with self.assertRaisesRegex(ValueError, "outside the warm clock"):
            self.verify()

    def test_partial_warm_and_temporary_or_locked_residency_rejected(self):
        original = deepcopy(self.row)
        for field in (
            "temporary_count",
            "read_locked_count",
            "write_locked_count",
            "external_io_lease_count",
        ):
            with self.subTest(field=field):
                self.row = deepcopy(original)
                self.row["after"]["l1_manager"][field] = 1
                with self.assertRaisesRegex(ValueError, "ownership remain"):
                    self.verify()
        self.row = deepcopy(original)
        self.row["preparation"]["found_keys"] = 15
        with self.assertRaisesRegex(ValueError, "complete retained prefetch"):
            self.verify()

    def test_unused_work_is_charged_without_a_child_credit(self):
        self.row["rows"] = []
        self.row["family_after"] = self.row["family_before"] | {
            "family_ensure_total": 1,
            "family_prefetch_wasted_total": 1,
        }
        self.assertEqual(self.verify(unused=True), 16)
        self.row["family_after"]["family_prefetch_wasted_total"] = 0
        with self.assertRaisesRegex(ValueError, "unused family declaration"):
            self.verify(unused=True)

    def test_engine_control_cannot_hide_external_warm(self):
        self.row["arm"] = "engine"
        with self.assertRaisesRegex(ValueError, "engine-only control contains"):
            self.verify()

    def test_prepare_is_inside_release_clock_and_precedes_engine_warm(self):
        clock = Clock()
        engine = Engine(clock)

        def prepare():
            self.assertEqual(engine.events, [])
            clock.now += 30_000_000
            return {"completion_ms": 30}

        with (
            patch("family_worker.time.perf_counter_ns", side_effect=lambda: clock.now),
            patch("family_worker.time.sleep", side_effect=clock.sleep),
        ):
            result = replay(
                engine,
                [1],
                [[1, 2]],
                SimpleNamespace(max_tokens=8),
                SimpleNamespace(max_tokens=1),
                "family",
                250,
                "retained",
                prepare=prepare,
            )
        self.assertEqual(result["actual_gap_ms"], 250)
        self.assertEqual(engine.events[0][2], 30_000_000)
        self.assertGreater(result["warm_complete_ms"], 30)
        self.assertEqual(result["preparation"], {"completion_ms": 30})

    def test_stock_warm_submission_and_completed_poll(self):
        answers = [
            {"status": "submitted", "chunks": 16, "request_id": "warm"},
            {"status": "pending"},
            {"status": "completed", "found_keys": 16, "total_keys": 16},
            {"storage_manager": {"l1_manager": residency(16)}},
        ]
        with (
            patch("retained_worker.http", side_effect=answers) as api,
            patch("retained_worker.time.sleep"),
        ):
            result = warm_external({"http_port": 1234, "model": "model"}, [1] * 4096)
        self.assertEqual(result["polls"], 2)
        self.assertEqual(result["found_keys"], 16)
        self.assertEqual(api.call_args_list[0].args[1], "cache/prefetches")
        self.assertEqual(api.call_args_list[1].args[1], "cache/prefetches/warm")

    def test_stock_warm_miss_rejected(self):
        with patch(
            "retained_worker.http",
            side_effect=[
                {"status": "submitted", "chunks": 16, "request_id": "warm"},
                {"status": "completed", "found_keys": 15, "total_keys": 16},
            ],
        ):
            with self.assertRaisesRegex(ValueError, "missed ancestor"):
                warm_external({"http_port": 1234, "model": "model"}, [1] * 4096)


class ConsumerRoutingTest(unittest.TestCase):
    def test_each_consumer_uses_its_supported_entry_arguments(self):
        for worker in ("worker.py", "family_worker.py", "retained_worker.py"):
            with self.subTest(worker=worker), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "cell"
                process = SimpleNamespace(poll=lambda: None, returncode=0)
                with (
                    patch.object(restore_run, "create_media"),
                    patch.object(restore_run, "launch", return_value=process),
                    patch.object(restore_run, "execute") as execute,
                    patch.object(restore_run, "stop"),
                    patch.object(restore_run, "http", return_value={}),
                    patch.object(restore_run, "cold_l1"),
                ):
                    restore_run.serving({}, output, 5, worker)
                arguments = [str(output / "config.json")]
                if worker == "worker.py":
                    arguments.append("consumer")
                self.assertEqual(
                    execute.call_args_list[1].args[:2], (worker, arguments)
                )


if __name__ == "__main__":
    unittest.main()
