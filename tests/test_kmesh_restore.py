# SPDX-License-Identifier: MIT
"""Reject false restore-latency claims before any GPU run is accepted."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("restore_verify", HERE / "verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
from restore_common import (
    CHUNK_BYTES,
    CHUNKS,
    compact_core_pair,
    create_media,
    distribution,
    prompts,
)


def core(mode="direct", count=0):
    tally = {
        "logical_requests": count,
        "logical_bytes": count * CHUNK_BYTES,
        "padded_bytes": count * CHUNK_BYTES,
        "submitted_operations": count,
        "submitted_padded_bytes": count * CHUNK_BYTES,
        "completed_operations": count,
        "completed_padded_bytes": count * CHUNK_BYTES,
    }
    return {
        "buffer_registration_mode": "dmabuf" if mode == "direct" else "classic",
        "is_healthy": True,
        "poisoned": False,
        "worker_error": None,
        "use_odirect": True,
        "use_uring_cmd": False,
        "iouring_queue_depth": 8,
        "inflight_io_count": 0,
        "quarantined_slot_count": 0,
        "locked_key_count": 0,
        "inflight_key_count": 0,
        "payload_reads": tally,
        "io_by_kind_and_path": {"read/payload/iouring_batched": tally.copy()},
        "io_by_request": {
            "unanswered": {},
            "evidence_failures": [],
            "untagged_operations": 0,
            "dropped_rows": 0,
            "evicted_requests": 0,
            "malformed_rows": 0,
            "duplicate_rows": 0,
            "context_conflicts": 0,
            "collection_failures": 0,
            "operations": [
                {
                    "device_instance_id": "fixture",
                    "operation_id": i,
                    "batch_id": 1,
                    "direction": "read",
                    "path": "dmabuf_fixed" if mode == "direct" else "host_fixed",
                    "complete": True,
                    "requested_bytes": CHUNK_BYTES,
                    "completed_bytes": CHUNK_BYTES,
                    "attempts": [
                        {
                            "attempt": 0,
                            "submitted_bytes": CHUNK_BYTES,
                            "outcome": "completed",
                            "completed_bytes": CHUNK_BYTES,
                        }
                    ],
                }
                for i in range(count)
            ],
        },
    }


class RestoreVerificationTests(unittest.TestCase):
    def make_capture(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        golden, checksum = verify.expected_payload()
        write(
            root / "plan.json",
            {"rounds": 1, "adapter_samples": 2, "serving_samples": 1},
        )
        rows = {
            n: [
                {
                    "token_ids": [7],
                    "logprob": -0.5,
                    "ttft_ms": 10,
                    "num_cached_tokens": 4096,
                }
                for _ in range(n)
            ]
            for n in (1, 5)
        }
        write(
            root / "oracle/oracle.json",
            [{"fanout": n, "rows": rows[n]} for n in (1, 5)],
        )
        for mode in ("direct", "staged"):
            write(
                root / f"corruption-{mode}-rejected.json",
                {"returncode": 1, "expected_payload_rejection": True},
            )
            (root / f"corruption-{mode}.log").write_text(
                "AssertionError: restored GPU consumer bytes differ\n"
            )
            adapter = root / f"smoke-0-{mode}-adapter"
            write(
                adapter / "result.json",
                {"terminal": {"core": core(mode)}, "samples": 2},
            )
            write(
                adapter / "samples.json",
                [
                    {
                        "before": {"core": core(mode)},
                        "after": {"core": core(mode, CHUNKS)},
                        "payload_sha256": golden,
                        "gpu_checksum": checksum,
                        "sample": i,
                        "warmup": False,
                        "load_ms": 2,
                        "ready_ms": 3,
                    }
                    for i in range(2)
                ],
            )
            serving = root / f"smoke-0-{mode}-serving"
            prefix = {
                "block_ids": list(range(1, 257)),
                "checksums": {
                    "chunk_checksums": {
                        f"layer_{i}": ["digest"] * CHUNKS for i in range(24)
                    }
                },
            }
            write(serving / "producer-prefix.json", {"members": [prefix]})
            for n in (1, 5):
                write(
                    serving / f"consumer-prefix-n{n}.json",
                    {"members": [prefix for _ in range(n)]},
                )
            write(
                serving / "consumer.json",
                [
                    {
                        "fanout": n,
                        "before": {"l2_adapters": [{"core": core(mode)}]},
                        "after": {"l2_adapters": [{"core": core(mode, CHUNKS)}]},
                        "sample": 0,
                        "warmup": False,
                        "rows": rows[n],
                        "batch_wall_ms": 11,
                    }
                    for n in (1, 5)
                ],
            )
            write(
                serving / "terminal.json",
                {
                    "storage_manager": {
                        "l2_adapters": [{"core": core(mode)}],
                        "l1_manager": {"external_io_lease_count": 0},
                    }
                },
            )

    def test_complete_capture_and_missing_serving_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_capture(root)
            accepted = verify.verify(root, "smoke")
            self.assertEqual(accepted["serving"]["direct-n5"]["ttft"]["count"], 5)
            path = root / "smoke-0-direct-serving/consumer.json"
            rows = json.loads(path.read_text())
            rows[1]["rows"].pop()
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "lost a member"):
                verify.verify(root, "smoke")

    def test_transport_qualification_does_not_require_model_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_capture(root)
            (root / "oracle/oracle.json").unlink()
            result = verify.verify(root, "smoke", "adapter")
            self.assertEqual(result["adapter"]["direct"]["ready"]["count"], 2)
            self.assertNotIn("output_tokens_exact", result["checks"])
            with self.assertRaises(FileNotFoundError):
                verify.verify(root, "smoke")

    def test_output_token_drift_rejects_the_entire_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_capture(root)
            path = root / "smoke-0-staged-serving/consumer.json"
            rows = json.loads(path.read_text())
            rows[0]["rows"][0]["token_ids"] = [8]
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "output tokens differ"):
                verify.verify(root, "smoke")

    def test_wrong_model_kv_rejects_an_otherwise_valid_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_capture(root)
            path = root / "smoke-0-direct-serving/consumer-prefix-n5.json"
            control = json.loads(path.read_text())
            control["members"][0]["checksums"]["chunk_checksums"]["layer_0"][
                0
            ] = "wrong"
            path.write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError, "model KV differs"):
                verify.verify(root, "smoke")

    def test_likelihood_gate_rejects_drift_and_nonfinite_values(self):
        for likelihood, message in (
            (-0.55, "fixed 0.02 bound"),
            (float("nan"), "not finite"),
            (float("inf"), "not finite"),
            (float("-inf"), "not finite"),
        ):
            with self.subTest(likelihood=likelihood):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self.make_capture(root)
                    path = root / "smoke-0-direct-serving/consumer.json"
                    rows = json.loads(path.read_text())
                    rows[0]["rows"][0]["logprob"] = likelihood
                    path.write_text(json.dumps(rows))
                    with self.assertRaisesRegex(ValueError, message):
                        verify.verify(root, "smoke")

    def test_all_model_layers_must_be_identified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_capture(root)
            path = root / "smoke-0-direct-serving/producer-prefix.json"
            control = json.loads(path.read_text())
            hashes = control["members"][0]["checksums"]["chunk_checksums"]
            hashes["layer_24"] = hashes.pop("layer_0")
            path.write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError, "digest geometry"):
                verify.verify(root, "smoke")

    def test_geometry_and_exact_causal_prefix(self):
        ancestor, children = prompts()
        self.assertEqual(len(ancestor), 4096)
        self.assertEqual(CHUNK_BYTES * CHUNKS, 48 << 20)
        self.assertTrue(all(p[:4096] == ancestor and len(p) == 4608 for p in children))
        self.assertEqual(len({tuple(p[4096:]) for p in children}), 5)

    def test_percentiles_reject_empty_nan_and_negative_samples(self):
        self.assertEqual(distribution([1, 2, 3])["p50_ms"], 2)
        for values in ([], [float("nan")], [-1]):
            with self.assertRaises(ValueError):
                distribution(values)

    def test_media_creation_refuses_existing_files_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing"
            path.write_bytes(b"do not overwrite")
            with self.assertRaises(FileExistsError):
                create_media(path)
            self.assertEqual(path.read_bytes(), b"do not overwrite")

    def test_both_registered_paths_account_for_all_bytes(self):
        for mode in ("direct", "staged"):
            delta = verify.read_delta(core(mode), core(mode, CHUNKS), mode)
            self.assertEqual(delta["completed_padded_bytes"], 48 << 20)

    def test_resident_control_requires_explicit_zero_read_permission(self):
        for mode in ("direct", "staged"):
            with self.assertRaisesRegex(ValueError, "route counts"):
                verify.read_delta(core(mode), core(mode), mode)
            delta = verify.read_delta(core(mode), core(mode), mode, allow_empty=True)
            self.assertEqual(delta["completed_padded_bytes"], 0)
            after = core(mode)
            after["payload_reads"]["logical_bytes"] = CHUNK_BYTES
            with self.assertRaisesRegex(ValueError, "model chunk geometry"):
                verify.read_delta(core(mode), after, mode, allow_empty=True)
            with self.assertRaisesRegex(ValueError, "registered route"):
                verify.read_delta(
                    core("staged"), core("staged"), "direct", allow_empty=True
                )

    def test_compaction_retains_every_new_native_operation(self):
        before, after = core(count=16), core(count=32)
        lean_before, lean_after = compact_core_pair(before, after)
        self.assertEqual(len(before["io_by_request"]["operations"]), 16)
        self.assertEqual(len(after["io_by_request"]["operations"]), 32)
        self.assertEqual(lean_before["io_by_request"]["operations"], [])
        self.assertEqual(len(lean_after["io_by_request"]["operations"]), 16)
        self.assertEqual(
            verify.read_delta(lean_before, lean_after, "direct")["logical_requests"], 16
        )

    def test_wrong_registration_cannot_pass_as_direct(self):
        with self.assertRaisesRegex(ValueError, "registered route"):
            verify.read_delta(core("staged"), core("staged", CHUNKS), "direct")

    def test_fallback_payload_route_is_rejected(self):
        after = core(count=CHUNKS)
        after["io_by_request"]["operations"][0]["path"] = "host_fixed"
        with self.assertRaisesRegex(ValueError, "unexpected payload route"):
            verify.read_delta(core(), after, "direct")

    def test_lost_native_evidence_cannot_certify_a_route(self):
        after = core(count=CHUNKS)
        after["io_by_request"]["dropped_rows"] = 1
        with self.assertRaisesRegex(ValueError, "native operation evidence"):
            verify.read_delta(core(), after, "direct")

    def test_unanswered_read_is_rejected(self):
        after = core(count=CHUNKS)
        after["payload_reads"]["completed_padded_bytes"] -= 4096
        with self.assertRaisesRegex(ValueError, "fully complete"):
            verify.read_delta(core(), after, "direct")

    def test_wrong_geometry_and_missing_route_counts_are_rejected(self):
        for field in ("logical_bytes", "logical_requests"):
            after = core(count=CHUNKS)
            after["payload_reads"][field] -= 1
            with self.assertRaises(ValueError):
                verify.read_delta(core(), after, "direct")

    def test_unresolved_ownership_cannot_be_published(self):
        for field in (
            "inflight_io_count",
            "quarantined_slot_count",
            "locked_key_count",
            "inflight_key_count",
        ):
            state = core()
            state[field] = 1
            with self.assertRaises(ValueError):
                verify.check_core(state, "direct", terminal=True)


if __name__ == "__main__":
    unittest.main()
