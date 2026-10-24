# SPDX-License-Identifier: MIT
"""Reject cold-read claims without exact admission, route and completion proof."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.reproduce.kmesh_registered import storage


class BackingFile(unittest.TestCase):
    def test_short_writes_still_initialize_the_entire_fresh_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backing.raw"
            write = storage.os.write
            written = []

            def short_write(fd, data):
                count = write(fd, data[:10003])
                written.append(count)
                return count

            with patch.object(storage.os, "write", side_effect=short_write):
                storage.initialize_backing_file(path, (2 << 20) + 17)
            self.assertEqual(sum(written), (2 << 20) + 17)
            self.assertEqual(path.read_bytes(), bytes((2 << 20) + 17))

    def test_existing_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backing.raw"
            path.write_bytes(b"existing data")
            with self.assertRaises(FileExistsError):
                storage.initialize_backing_file(path, 4096)
            self.assertEqual(path.read_bytes(), b"existing data")

    def test_zero_progress_refuses_incomplete_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(storage.os, "write", return_value=0):
                with self.assertRaisesRegex(OSError, "no progress"):
                    storage.initialize_backing_file(
                        Path(directory) / "backing.raw", 4096
                    )


class ModelGeometry(unittest.TestCase):
    def test_low_memlock_refuses_before_model_or_storage_work(self):
        with patch.object(storage.resource, "getrlimit", return_value=(8 << 20,) * 2):
            with self.assertRaisesRegex(ValueError, "RLIMIT_MEMLOCK.*host arena"):
                storage.require_host_memlock({"arena_bytes": 64 << 20})

    def test_larger_and_unlimited_memlock_allow_the_host_registration(self):
        for limit in (1 << 30, storage.resource.RLIM_INFINITY):
            with self.subTest(limit=limit), patch.object(
                storage.resource, "getrlimit", return_value=(limit, limit)
            ):
                storage.require_host_memlock({"arena_bytes": 64 << 20})

    def test_packed_content_width_is_not_a_doubled_rotary_head(self):
        settings = {
            "chunk_tokens": 256,
            "model_layers": 24,
            "logical_kv_planes": 2,
            "kv_heads": 2,
            "head_size": 64,
            "dtype_bytes": 2,
            "source_objects": 4,
            "children": 4,
            "chunk_payload_bytes": 3 << 20,
            "arena_bytes": 64 << 20,
            "metadata_bytes": 1 << 20,
            "max_transfer_bytes": 128 << 10,
            "prefetch_policy": "retain",
        }
        storage.validate_settings(settings)
        for change in (
            {"head_size": 128, "chunk_payload_bytes": 6 << 20},
            {"chunk_payload_bytes": (3 << 20) - 4096},
            {"arena_bytes": 4 << 20},
            {"chunk_tokens": True},
            {"prefetch_policy": "default"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                storage.validate_settings(settings | change)


class LookupBoundary(unittest.TestCase):
    def test_eviction_retries_only_keys_with_actual_remaining_leases(self):
        calls = []

        class Manager:
            def delete_l1_keys(self, keys):
                calls.append(tuple(keys))
                return (0, 1) if calls == [("a",)] else (1, 0)

        self.assertEqual(
            storage.evict_settled_l1_keys(Manager(), ["a", "b"], float("inf")), 2
        )
        self.assertEqual(calls, [("a",), ("b",), ("a",)])

    def test_missing_retained_l1_key_is_not_counted_as_our_eviction(self):
        class Manager:
            def delete_l1_keys(self, keys):
                return 0, 0

        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            storage.evict_settled_l1_keys(Manager(), ["a"], float("inf"))

    def test_partial_lookup_unlocks_only_hits_before_retry(self):
        calls = []

        class Adapter:
            attempt = 0

            def submit_lookup_and_lock_task(self, keys, layouts):
                self.attempt += 1
                calls.append(("lookup", self.attempt))
                return self.attempt

            def query_lookup_and_lock_result(self, task):
                self.assert_task(task)
                return type("Bitmap", (), {"test": lambda _, i: task == 2 or i == 1})()

            def assert_task(self, task):
                if task != self.attempt:
                    raise AssertionError("query consumed a foreign task receipt")

            def submit_unlock(self, keys):
                calls.append(("unlock", keys))

        storage.await_exact_l2_keys(Adapter(), ["a", "b"], {}, float("inf"))
        self.assertEqual(
            calls,
            [("lookup", 1), ("unlock", ["b"]), ("lookup", 2), ("unlock", ["a", "b"])],
        )

    def test_unknown_lookup_never_invents_an_unlock(self):
        class Adapter:
            def submit_lookup_and_lock_task(self, *args):
                return 7

            def query_lookup_and_lock_result(self, task):
                if task != 7:
                    raise AssertionError("foreign receipt")
                return None

            def submit_unlock(self, keys):
                raise AssertionError("unknown native lock was treated as complete")

        with patch.object(
            storage.time, "monotonic", side_effect=[0, 0, 2]
        ), patch.object(storage.time, "sleep"), self.assertRaisesRegex(
            RuntimeError, "completion is unknown"
        ):
            storage.await_exact_l2_keys(Adapter(), ["a"], {}, 1)
