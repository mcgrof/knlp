# SPDX-License-Identifier: MIT
"""CPU checks for the joint-layout experiment's attribution control."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
import types
import unittest
from unittest import mock


class _Image:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: self


class _App:
    def function(self, *args, **kwargs):
        return lambda function: function

    def local_entrypoint(self, *args, **kwargs):
        return lambda function: function


def _load_harness():
    modal = types.ModuleType("modal")
    modal.App = lambda *args, **kwargs: _App()
    modal.Image = types.SimpleNamespace(from_registry=lambda *args, **kwargs: _Image())
    modal.Volume = types.SimpleNamespace(from_name=lambda *args, **kwargs: object())
    modal.is_local = lambda: False
    path = Path(__file__).parents[1] / "modal_joint_layout_compare.py"
    spec = importlib.util.spec_from_file_location("_joint_layout_harness", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"modal": modal}):
        spec.loader.exec_module(module)
    return module


class JointLayoutControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = _load_harness()

    def test_static_control_matches_planner_assignment_and_replay(self):
        plan = self.harness.compile_experiment_plan()
        self.assertEqual(
            plan["selected"]["assignment"],
            self.harness.STATIC_PER_FAMILY_ASSIGNMENT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = self.harness._trace(
                root / "control", 20_001, "fixed-per-family", plan
            )
            selected = self.harness._trace(root / "selected", 20_001, "joint", plan)
        self.assertEqual(control["assignment"], selected["assignment"])
        self.assertEqual(
            control["replay_workload_sha256"],
            selected["replay_workload_sha256"],
        )

    def test_explicit_uniform_controls_remain_distinct(self):
        plan = self.harness.compile_experiment_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self.harness._trace(root / "repo", 20_001, "fixed-repo", plan)
            selected = self.harness._trace(root / "selected", 20_001, "joint", plan)
        self.assertNotEqual(
            repo["replay_workload_sha256"], selected["replay_workload_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
