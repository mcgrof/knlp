# SPDX-License-Identifier: MIT
"""Reject cold-read claims without exact admission, route and completion proof."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.reproduce.kmesh_registered import storage, storage_verify


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


def synthetic_evidence(route):
    """Build a small protocol fixture, never a captured storage measurement."""
    settings = {
        "children": 2,
        "source_objects": 2,
        "chunk_payload_bytes": 4096,
        "arena_bytes": 1 << 20,
    }
    mode = "classic" if route == "staged" else "dmabuf"
    native = "host_fixed" if route == "staged" else "dmabuf_fixed"
    rows, operations = [], []

    def adapter(count):
        return {
            "is_healthy": True,
            "core": {
                "buffer_registration_mode": mode,
                "io_engine": "io_uring",
                "use_uring_cmd": False,
                "poisoned": False,
                "inflight_io_count": 0,
                "quarantined_slot_count": 0,
                "payload_reads": {
                    "logical_requests": count,
                    "logical_bytes": count * 4096,
                    "padded_bytes": count * 4096,
                    "submitted_operations": count,
                    "submitted_padded_bytes": count * 4096,
                    "completed_operations": count,
                    "completed_padded_bytes": count * 4096,
                },
                "io_by_request": {
                    "unanswered": {},
                    "evidence_failures": [],
                    "operations": copy.deepcopy(operations),
                },
            },
        }

    for child in range(2):
        request = f"child-{child}"
        rows.append(
            {
                "kind": "cold",
                "request_id": request,
                "source_key_hashes": ["a", "b"],
                "confirmed_l2_keys": 2,
                "deleted_l1_keys": 2,
                "elapsed_ns": 1,
                "before": adapter(child * 2),
            }
        )
        for index in range(2):
            operations.append(
                {
                    "device_instance_id": "synthetic",
                    "operation_id": child * 2 + index,
                    "direction": "read",
                    "path": native,
                    "complete": True,
                    "requested_bytes": 4096,
                    "completed_bytes": 4096,
                    "context": {"request_id": f"load-{child}-{index}"},
                }
            )
            rows.extend(
                [
                    {
                        "kind": "storage_read",
                        "phase": request,
                        "elapsed_ns": 1,
                        "io_request_id": f"load-{child}-{index}",
                        "source_key_hashes": [["a", "b"][index]],
                        "payload_bytes": 4096,
                        "success": True,
                    },
                    {
                        "kind": "restore",
                        "phase": request,
                        "state": "loaded-needs-repair",
                        "elapsed_ns": 2,
                    },
                    {
                        "kind": "checksum",
                        "phase": request,
                        "payload_bytes": 4096,
                        "host_payload_copy_bytes": 4096 if route == "dma-buf" else 0,
                        "elapsed_ns": 1,
                    },
                ]
            )
        rows.append(
            {
                "kind": "after",
                "request_id": request,
                "adapter": adapter((child + 1) * 2),
            }
        )
    return settings, {
        "close_buffers_reusable": True,
        "rows": rows,
        "storage_manager": {"is_healthy": True},
        "location": {"synthetic": True},
        "poisoned_l1_arena": False,
    }


class StorageReceipt(unittest.TestCase):
    def test_poison_receipt_requires_every_byte_and_a_checked_drain(self):
        for route in ("staged", "dma-buf"):
            settings, evidence = synthetic_evidence(route)
            evidence["poisoned_l1_arena"] = True
            for cold in evidence["rows"]:
                if cold["kind"] == "cold":
                    cold["poison"] = {
                        "bytes": settings["arena_bytes"],
                        "device": "cpu" if route == "staged" else "cuda:0",
                        "drained": True,
                    }
            self.assertTrue(
                storage_verify.verify(route, evidence, settings, require_poison=True)[
                    "passed"
                ]
            )
            for change in ({"bytes": 1}, {"device": "cuda:1"}, {"drained": False}):
                invalid = copy.deepcopy(evidence)
                invalid["rows"][0]["poison"].update(change)
                with self.subTest(route=route, change=change), self.assertRaises(
                    ValueError
                ):
                    storage_verify.verify(route, invalid, settings, require_poison=True)

    def test_timing_receipt_cannot_replace_poisoned_data_arrival_qualification(self):
        settings, evidence = synthetic_evidence("dma-buf")
        with self.assertRaisesRegex(ValueError, "conditioning differs"):
            storage_verify.verify("dma-buf", evidence, settings, require_poison=True)

    def test_complete_receipt_preserves_integrity_copy_accounting(self):
        for route in ("staged", "dma-buf"):
            with self.subTest(route=route):
                settings, evidence = synthetic_evidence(route)
                result = storage_verify.verify(route, evidence, settings)
                self.assertTrue(result["passed"])
                self.assertEqual(
                    sum(r["integrity_host_copy_bytes"] for r in result["children"]),
                    16384 if route == "dma-buf" else 0,
                )

    def test_gpu_checksum_requires_digests_without_a_payload_download(self):
        settings, evidence = synthetic_evidence("dma-buf")
        evidence["checksum_backend"] = "cuda-sha256"
        for row in evidence["rows"]:
            if row["kind"] == "checksum":
                row.update(
                    checksum_backend="cuda-sha256",
                    host_payload_copy_bytes=0,
                    host_digest_copy_bytes=32,
                )
        result = storage_verify.verify(
            "dma-buf", evidence, settings, checksum_backend="cuda-sha256"
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            [child["integrity_host_copy_bytes"] for child in result["children"]],
            [0, 0],
        )
        self.assertEqual(
            [child["integrity_host_digest_bytes"] for child in result["children"]],
            [64, 64],
        )
        for change in (
            {"host_payload_copy_bytes": 4096},
            {"host_digest_copy_bytes": 31},
            {"checksum_backend": "cpu"},
        ):
            bad = copy.deepcopy(evidence)
            next(row for row in bad["rows"] if row["kind"] == "checksum").update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                storage_verify.verify(
                    "dma-buf", bad, settings, checksum_backend="cuda-sha256"
                )
        with self.assertRaises(ValueError):
            storage_verify.verify("dma-buf", evidence, settings)
        with self.assertRaises(ValueError):
            storage_verify.verify(
                "staged", evidence, settings, checksum_backend="cuda-sha256"
            )

    def test_incomplete_receipts_never_qualify_a_storage_route(self):
        for route in ("staged", "dma-buf"):
            for case in (
                "admission",
                "retirement",
                "short",
                "route",
                "native",
                "unanswered",
                "hash",
                "source-identity",
                "model-request",
                "storage-task",
                "storage-source",
            ):
                with self.subTest(route=route, case=case):
                    settings, evidence = synthetic_evidence(route)
                    cold = [r for r in evidence["rows"] if r["kind"] == "cold"]
                    after = [r for r in evidence["rows"] if r["kind"] == "after"]
                    core = after[0]["adapter"]["core"]
                    if case == "admission":
                        cold[0]["confirmed_l2_keys"] = 1
                    elif case == "retirement":
                        evidence["close_buffers_reusable"] = False
                    elif case == "short":
                        core["payload_reads"]["completed_padded_bytes"] -= 1
                    elif case == "route":
                        core["io_by_request"]["operations"][0]["path"] = "regular"
                    elif case == "native":
                        core["io_by_request"]["operations"][0]["complete"] = False
                    elif case == "unanswered":
                        core["io_by_request"]["unanswered"] = {"pending": 1}
                    elif case == "hash":
                        evidence["rows"] = [
                            r for r in evidence["rows"] if r["kind"] != "checksum"
                        ]
                    elif case == "source-identity":
                        cold[1]["source_key_hashes"] = ["c", "d"]
                    elif case == "storage-task":
                        core["io_by_request"]["operations"][0]["context"][
                            "request_id"
                        ] = "foreign"
                    elif case == "storage-source":
                        next(
                            row
                            for row in evidence["rows"]
                            if row["kind"] == "storage_read"
                        )["source_key_hashes"] = ["foreign"]
                    else:
                        after[1]["request_id"] = "unmatched"
                    with self.assertRaises(ValueError):
                        storage_verify.verify(route, evidence, settings)


if __name__ == "__main__":
    unittest.main()
