# SPDX-License-Identifier: MIT
"""Receipt acceptance and deployment guards; no GPU or storage is accessed."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "workflow", Path(__file__).resolve().parents[1] / "workflow.py"
)
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cell = {"io_bytes": 65536, "span_bytes": 1048576, "batch_depth": 8}
        self.rows = [
            dict(
                schema_version=1,
                backend="tutti",
                workload="synthetic-transfer",
                scheduling="submit-and-drain",
                access="sequential",
                seed=42,
                repeat=0,
                operation=op,
                completed_bytes=1048576,
                operations=16,
                wall_seconds=0.01,
                process_cpu_seconds=0.002,
                batch_count=2,
                batch_p50_us=100,
                batch_p99_us=200,
                verified_bytes=1048576,
                correctness="full-readback",
                **self.cell,
            )
            for op in ("write", "read")
        ]

    def validate(self, rows):
        w.validate_rows(rows, self.cell, 1, 42, "sequential")

    def receipt(self):
        w.write_json(
            self.root / "manifest.json",
            {
                "status": "PASS",
                "kind": "synthetic-transfer",
                "cells": [self.cell],
                "repeats": 1,
                "seed": 42,
                "access": "sequential",
                "files": {},
            },
        )
        measurements = self.root / "measurements.jsonl"
        measurements.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["files"][measurements.name] = w.sha(measurements)
        w.write_json(self.root / "manifest.json", manifest)

    def test_rows_accept_full_result_and_log_noise(self):
        raw = "runtime initialized\n" + "".join(
            "KNLP_TUTTI_JSON " + json.dumps(r) + "\n" for r in self.rows
        )
        self.validate(w.measurement_rows(raw))
        with self.assertRaises(ValueError):
            w.measurement_rows("KNLP_TUTTI_JSON broken")

    def test_rows_reject_incomplete_or_duplicate_pairs(self):
        for rows in ([], self.rows[:1], [self.rows[0], self.rows[0]], self.rows * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.validate(rows)

    def test_rows_reject_wrong_scope_geometry_bytes_and_timing(self):
        bad = {
            "correctness": "sampled",
            "completed_bytes": 65536,
            "verified_bytes": 65536,
            "io_bytes": 4096,
            "repeat": True,
            "operation": "flush",
            "operations": 15,
            "seed": 0,
            "scheduling": "refill",
            "batch_count": 3,
            "wall_seconds": float("nan"),
            "process_cpu_seconds": -1,
            "batch_p99_us": 50,
            "batch_p50_us": float("inf"),
        }
        for key, value in bad.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                self.validate(rows)

    def test_report_rejects_tampering_and_removes_stale_summary(self):
        self.receipt()
        w.report(self.root)
        self.assertTrue((self.root / "summary.md").exists())
        with (self.root / "measurements.jsonl").open("a") as f:
            f.write("{}\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            w.report(self.root)
        self.assertFalse((self.root / "summary.md").exists())

    def test_report_rejects_failure_and_unhashed_measurements(self):
        for field, value in (
            ("status", "FAIL"),
            ("files", {}),
            ("cells", [self.cell, self.cell]),
            ("kind", "upstream-synthetic-overlap"),
        ):
            self.receipt()
            manifest = json.loads((self.root / "manifest.json").read_text())
            manifest[field] = value
            w.write_json(self.root / "manifest.json", manifest)
            with self.subTest(field=field), self.assertRaises(ValueError):
                w.report(self.root)

    def test_config_is_literal_and_pin_is_exact(self):
        config = self.root / ".config"
        config.write_text('CONFIG_TUTTI=y\nCONFIG_TUTTI_DIRECTORY="$(touch NEVER)"\n')
        self.assertEqual(w.config(config)["TUTTI_DIRECTORY"], "$(touch NEVER)")
        config.write_text('CONFIG_TUTTI=y\nCONFIG_TUTTI_REV="main"\n')
        with self.assertRaisesRegex(ValueError, "SHA"):
            w.config(config)

    def test_matrix_rejects_duplicates_misalignment_and_excess_depth(self):
        for key, value in (
            ("TUTTI_IO_KIB", "64 64"),
            ("TUTTI_IO_KIB", "3"),
            ("TUTTI_BATCH_DEPTHS", "0"),
            ("TUTTI_BATCH_DEPTHS", "4097"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                w.cells({**w.DEFAULTS, key: value})
        self.assertEqual(w.cells(w.DEFAULTS), w.cells(w.DEFAULTS))

    def test_runtime_requires_opt_in_and_one_explicit_device(self):
        spec = {
            "accelerator": {"profile": "CUDA"},
            "runtime": {"accel_id": 0},
            "storage": {
                "datapaths": [{"type": "local-nvme"}],
                "resources": [
                    {
                        "type": "nvme",
                        "allocation": {"selection": "explicit", "device_ids": [0]},
                    }
                ],
            },
        }
        runtime = self.root / "runtime.json"
        w.write_json(runtime, spec)
        c = {"TUTTI_DIRECTORY": str(self.root), "TUTTI_RUNTIME_CONFIG": str(runtime)}
        with self.assertRaisesRegex(ValueError, "ALLOW_FILE_WRITES"):
            w.validate_runtime(c)
        c["TUTTI_ALLOW_FILE_WRITES"] = "y"
        w.validate_runtime(c)
        for directory in ("/", "/dev", "/proc", "/sys"):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                w.validate_runtime({**c, "TUTTI_DIRECTORY": directory})
        spec["storage"]["resources"][0]["allocation"]["device_ids"] = [0, 1]
        w.write_json(runtime, spec)
        with self.assertRaisesRegex(ValueError, "one explicitly"):
            w.validate_runtime(c)

    def test_pipeline_orders_stages_and_does_not_run_build_defconfig(self):
        config = self.root / ".config"
        config.write_text(f'CONFIG_TUTTI=y\nCONFIG_TUTTI_WORK_DIR="{self.root}"\n')
        calls = []
        with patch.object(
            w.sys, "argv", ["workflow", "pipeline", "--config", str(config)]
        ):
            with patch.object(
                w, "doctor", side_effect=lambda c: calls.append("doctor")
            ), patch.object(
                w, "fetch", side_effect=lambda c: calls.append("fetch")
            ), patch.object(
                w, "build", side_effect=lambda c: calls.append("build")
            ), patch.object(
                w, "benchmark"
            ) as benchmark:
                w.main()
                benchmark.assert_not_called()
        self.assertEqual(calls, ["doctor", "fetch", "build"])


if __name__ == "__main__":
    unittest.main()
