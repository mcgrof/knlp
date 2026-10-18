# SPDX-License-Identifier: MIT
"""Reject false cascade comparisons without importing a GPU runtime."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_cascade"
SPEC = importlib.util.spec_from_file_location("cascade_verify", HERE / "verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class CascadeEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((HERE / "profile.json").read_text())
        w = self.profile["workload"]
        cell = {
            "workload": w,
            "runner": "vllm.v1.worker.gpu_model_runner.GPUModelRunner",
            "backends": ["FLASH_ATTN"],
            "source_commit": self.profile["vllm_commit"],
            "prompts_sha256": "fixture",
            "native_sha256": {"library": "fixture"},
            "runtime": {"fixture": True},
            "outputs": [
                [[1] * w["decode"] for _ in range(w["fanout"])]
                for _ in range(w["repeats"])
            ],
            "cached_tokens": [[w["ancestor"]] * w["fanout"]] * w["repeats"],
            "generation_wall_ms": [100] * w["repeats"],
            "engagement": {"kernel_calls": 0, "runner_calls": 1},
        }
        self.cells = {name: copy.deepcopy(cell) for name in VERIFY.MODES}
        for name, value in self.cells.items():
            value["mode"] = name
            value["engagement"]["kernel_calls"] = 0 if name == "disabled" else 1

    def check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, cell in self.cells.items():
                (root / (name + ".json")).write_text(json.dumps(cell))
            return VERIFY.verify(root)

    def test_enabled_without_engagement_has_no_speed_comparison(self):
        self.cells["enabled"]["engagement"]["kernel_calls"] = 0
        result = self.check()
        self.assertFalse(result["cells"]["enabled"]["speed_comparison_has_engagement"])

    def test_forced_without_kernel_is_rejected(self):
        self.cells["forced"]["engagement"]["kernel_calls"] = 0
        with self.assertRaisesRegex(ValueError, "did not reach"):
            self.check()

    def test_disabled_kernel_is_rejected(self):
        self.cells["disabled"]["engagement"]["kernel_calls"] = 1
        with self.assertRaisesRegex(ValueError, "disabled control"):
            self.check()

    def test_recomputed_ancestor_is_rejected(self):
        self.cells["enabled"]["cached_tokens"][0][0] = 0
        with self.assertRaisesRegex(ValueError, "did not reuse"):
            self.check()

    def test_output_changes_are_reported_separately(self):
        self.cells["forced"]["outputs"][0][0][0] = 2
        result = self.check()
        self.assertEqual(
            result["cells"]["forced"]["changed_continuations_vs_disabled"], 1
        )


if __name__ == "__main__":
    unittest.main()
