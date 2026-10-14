# SPDX-License-Identifier: MIT
"""Ensure independent experiments cannot silently share or overwrite a run."""

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run.py"
SPEC = importlib.util.spec_from_file_location("google_muvera_run", SCRIPT)
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class RunIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.args = argparse.Namespace(
            config=base / "selected.config", build_dir=base / "build", run_dir=None
        )
        self.cfg = {"CONFIG_MUVERA_OUTPUT_BASE": str(base / "results")}

    def test_new_runs_are_unique_and_latest_is_scoped(self):
        first = run.select_run(self.args, self.cfg, fresh=True)
        second = run.select_run(self.args, self.cfg, fresh=True)
        self.assertNotEqual(first, second)
        self.assertEqual(run.select_run(self.args, self.cfg), second)
        changed = dict(self.cfg, CONFIG_GOOGLE_MUVERA_SEEDS="3")
        with self.assertRaises(ValueError):
            run.select_run(self.args, changed)

    def test_explicit_path_rejects_overwrite_and_config_drift(self):
        self.args.run_dir = Path(self.tmp.name) / "fixed-run"
        run.select_run(self.args, self.cfg, fresh=True)
        with self.assertRaises(ValueError):
            run.select_run(self.args, self.cfg, fresh=True)
        with self.assertRaises(ValueError):
            run.select_run(self.args, dict(self.cfg, changed="y"))

    def test_legacy_scripts_receive_explicit_directory(self):
        path = run.select_run(self.args, self.cfg, fresh=True)
        config_path = run.effective_config(path, self.cfg)
        parsed = run.cf.parse_kconfig(config_path)
        self.assertEqual(run.cf.get_run_dir(parsed), path)
        self.assertEqual(
            parsed["CONFIG_MUVERA_OUTPUT_BASE"], self.cfg["CONFIG_MUVERA_OUTPUT_BASE"]
        )

    def test_existing_results_cannot_be_overwritten(self):
        path = run.select_run(self.args, self.cfg, fresh=True)
        (path / "systems_byte_floor.csv").write_text("previous data\n")
        with self.assertRaises(ValueError):
            run.systems(path, self.cfg)

    def test_unicode_and_literal_paths_round_trip(self):
        self.args.run_dir = Path(self.tmp.name) / 'café "quoted" \\ cache'
        path = run.select_run(self.args, self.cfg, fresh=True)
        parsed = run.cf.parse_kconfig(run.effective_config(path, self.cfg))
        self.assertEqual(run.cf.get_run_dir(parsed), path)

    def test_multiline_values_are_rejected(self):
        path = run.select_run(self.args, self.cfg, fresh=True)
        with self.assertRaises(ValueError):
            run.effective_config(path, dict(self.cfg, CONFIG_BAD="one\ntwo"))


if __name__ == "__main__":
    unittest.main()
