# SPDX-License-Identifier: MIT
"""Reject incomplete or misleading copies of a real allocator capture."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "fork_verify", ROOT / "tools/reproduce/kmesh_forks/verify.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = json.loads(
            (ROOT / "docs/data/kmesh-fork-accounting.json").read_text()
        )

    def setUp(self):
        self.result = copy.deepcopy(self.capture)

    def test_accept_real_complete_capture(self):
        report = verifier.verify(self.result)
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["gpu_inference_measured"])

    def test_missing_uncached_control_is_not_a_pass(self):
        del self.result["scenarios"]["a4096-b5-t512"]["pool_pressure_uncached"]
        with self.assertRaisesRegex(ValueError, "incomplete scenarios"):
            verifier.verify(self.result)

    def test_missing_eviction_point_is_not_a_pass(self):
        self.result["loss_window_sweep"]["points"].pop()
        with self.assertRaisesRegex(ValueError, "eviction sweep"):
            verifier.verify(self.result)

    def test_optimistic_uncached_counts_are_rejected(self):
        value = self.result["scenarios"]["a4096-b5-t512"]["concurrent_uncached"]
        value["summary"]["written_blocks"] = 416
        with self.assertRaisesRegex(ValueError, "changed accounting"):
            verifier.verify(self.result)

    def test_shared_writable_boundary_is_rejected(self):
        rows = self.result["scenarios"]["a4101-b5-t509"]["concurrent_cached"]["rows"]
        rows[2]["block_ids"][256] = rows[1]["block_ids"][256]
        with self.assertRaisesRegex(ValueError, "writable tail"):
            verifier.verify(self.result)

    def test_correct_counts_without_ancestor_aliasing_are_rejected(self):
        rows = self.result["scenarios"]["a4096-b5-t512"]["concurrent_cached"]["rows"]
        rows[1]["block_ids"][0] = 8000
        with self.assertRaisesRegex(ValueError, "not aliased"):
            verifier.verify(self.result)

    def test_inconsistent_boundary_recompute_is_rejected(self):
        rows = self.result["scenarios"]["a4101-b5-t509"]["concurrent_cached"]["rows"]
        rows[1]["duplicated_boundary_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "boundary tokens"):
            verifier.verify(self.result)

    def test_different_runtime_is_not_the_frozen_experiment(self):
        self.result["vllm_version"] = "another-build"
        with self.assertRaisesRegex(ValueError, "wrong vllm"):
            verifier.verify(self.result)

    def test_current_capture_requires_its_explicit_profile(self):
        current = json.loads(
            (ROOT / "docs/data/kmesh-fork-accounting-current.json").read_text()
        )
        report = verifier.verify(current, "current")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["profile"], "current")
        with self.assertRaisesRegex(ValueError, "wrong python"):
            verifier.verify(current)

    def test_historical_runtime_cannot_pass_as_current(self):
        with self.assertRaisesRegex(ValueError, "wrong python"):
            verifier.verify(self.result, "current")


if __name__ == "__main__":
    unittest.main()
