# SPDX-License-Identifier: MIT
"""Check trace evidence boundaries without a model, network or GPU."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.reproduce.kmesh_residency.audit import (
    TRACE_REVISION,
    audit,
    check_invocation,
    lineage,
    timestamp_seconds,
)


def invocation(name="parent", parent=None, start=1000000, end=2000000):
    return {
        "schema": "knlp.kmesh.invocation.v1",
        "request_id": name,
        "family_id": "family",
        "parent_request_id": parent,
        "engine_id": "engine",
        "clock_id": "clock",
        "start_ns": start,
        "end_ns": end,
        "input_token_ids": [1, 2],
        "output_token_ids": [3],
        "status": "complete",
        "cache_observations": [],
    }


class TraceEvidenceTest(unittest.TestCase):
    def test_epoch_units_are_only_plausible_finite_values(self):
        for factor in (1, 1000, 1000000, 1000000000):
            self.assertEqual(timestamp_seconds(1700000000 * factor), 1700000000)
        for value in (None, True, "1700000000", -1, float("nan"), float("inf")):
            self.assertIsNone(timestamp_seconds(value))

    def test_signed_gap_and_exact_parent_prefix(self):
        parent = invocation()
        child = invocation("child", "parent", start=1500000, end=3000000)
        child["input_token_ids"] = [1, 2, 3, 4]
        result = lineage([parent, child])
        self.assertEqual(result["signed_fork_gap_ms"], [-0.5])
        self.assertEqual(result["live_forks"], 1)
        self.assertEqual(result["non_exact_parent_prefix_edges"], 0)
        child["input_token_ids"][1] = 7
        self.assertEqual(lineage([parent, child])["non_exact_parent_prefix_edges"], 1)

    def test_missing_parent_or_clock_is_not_an_observed_gap(self):
        parent, child = invocation(), invocation("child", "parent")
        self.assertEqual(lineage([child])["unknown_parents"], 1)
        child["clock_id"] = "other-clock"
        result = lineage([parent, child])
        self.assertEqual(result["cross_clock_edges"], 1)
        self.assertEqual(result["observed_edges"], 0)

    def test_duplicate_cycle_and_family_conflicts_fail(self):
        parent, child = invocation(), invocation("child", "parent")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            lineage([parent, parent])
        parent["parent_request_id"] = "child"
        with self.assertRaisesRegex(ValueError, "cyclic"):
            lineage([parent, child])
        parent["parent_request_id"] = None
        child["family_id"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "different families"):
            lineage([parent, child])

    def test_child_cannot_precede_its_parent_start(self):
        with self.assertRaisesRegex(ValueError, "before its parent"):
            lineage([invocation(), invocation("child", "parent", start=0)])

    def test_invalid_interval_and_token_types_fail(self):
        for key, value in (
            ("start_ns", True),
            ("end_ns", -1),
            ("input_token_ids", [True]),
            ("output_token_ids", [-1]),
        ):
            row = invocation()
            row[key] = value
            with self.assertRaises(ValueError):
                check_invocation(row)

    def test_complete_invocation_metadata_does_not_claim_cache_pressure(self):
        parent, child = invocation(), invocation(
            "child", "parent", start=3000000, end=4000000
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.jsonl").write_text(
                "\n".join(json.dumps(r) for r in (parent, child))
            )
            result = audit(root, "invocations")
        self.assertTrue(result["timed_lineage_complete"])
        self.assertFalse(result["pressure_replay_qualified"])
        self.assertEqual(result["decision"], "physical-cache-capture-required")

    def test_raw_text_spacing_never_becomes_a_fork_gap(self):
        row = {
            "session_id": "s",
            "timestamp": 1700000000,
            "input": "abc",
            "output": "d",
        }
        other = {**row, "timestamp": 1700000002, "input": "abcd"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "framework").mkdir()
            (root / "framework/a.jsonl").write_text(
                "\n".join(json.dumps(r) for r in (row, row, other)) + "\nnot-json\n"
            )
            with patch(
                "tools.reproduce.kmesh_residency.audit.subprocess.check_output",
                side_effect=[TRACE_REVISION + "\n", b"", "framework/a.jsonl\0"],
            ):
                result = audit(root)
        stats = result["frameworks"]["framework"]
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["parse_errors"], 1)
        self.assertEqual(stats["fields"]["timestamp"], 2)
        self.assertEqual(stats["record_spacing_seconds_median"], 2)
        self.assertEqual(stats["input_character_prefix_fraction_median"], 0.75)
        self.assertFalse(result["timed_lineage_complete"])
        self.assertEqual(result["lineage"]["observed_edges"], 0)

    def test_wrong_source_revision_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "tools.reproduce.kmesh_residency.audit.subprocess.check_output",
                return_value="wrong\n",
            ):
                with self.assertRaisesRegex(ValueError, "pinned revision"):
                    audit(Path(directory))

    def test_cli_fresh_output_and_plan_are_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trace.jsonl").write_text(json.dumps(invocation()) + "\n")
            args = [
                sys.executable,
                "-m",
                "tools.reproduce.kmesh_residency.audit",
                "--config",
                str(root / "absent"),
                "--root",
                str(root),
                "--mode",
                "invocations",
                "--output",
                str(root / "output"),
            ]
            subprocess.run(args, check=True, capture_output=True)
            second = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("fresh directory", second.stderr)
            plan = subprocess.run(
                args + ["--plan"], check=True, capture_output=True, text=True
            )
            self.assertFalse(json.loads(plan.stdout)["gpu_execution"])


if __name__ == "__main__":
    unittest.main()
