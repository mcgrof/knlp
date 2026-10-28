# SPDX-License-Identifier: MIT
"""The weight-io orchestrator reads its configuration and plans without a
device, a network or root."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.reproduce.weight_io import run as wio

CONTROLS = """\
CONFIG_KNLP_WEIGHT_IO=y
CONFIG_KNLP_WEIGHT_IO_ROOT="{root}"
CONFIG_KNLP_WEIGHT_IO_REPOS="a/one b/two@abc"
CONFIG_KNLP_WEIGHT_IO_CONTROL_REPOS="a/one"
CONFIG_KNLP_WEIGHT_IO_FETCH_COMPACT=n
CONFIG_KNLP_WEIGHT_IO_GROUPS="layers rows"
CONFIG_KNLP_WEIGHT_IO_TRIALS=3
CONFIG_KNLP_WEIGHT_IO_INTENT=y
CONFIG_KNLP_WEIGHT_IO_GPU=n
"""


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        self.cfg_path = Path(self.tmp.name) / ".config"
        self.cfg_path.write_text(CONTROLS.format(root=self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_parses_lists_and_flags(self):
        cfg = wio.Config.from_file(self.cfg_path)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.repos, ["a/one", "b/two@abc"])
        self.assertEqual(cfg.control_repos, ["a/one"])
        self.assertEqual(cfg.groups, ["layers", "rows"])
        self.assertEqual(cfg.trials, 3)
        self.assertTrue(cfg.intent)
        self.assertFalse(cfg.gpu)
        self.assertEqual(
            cfg.full_dir("a/one"), self.root.resolve() / "models" / "a__one" / "full"
        )

    def test_disabled_config_refuses(self):
        self.cfg_path.write_text("CONFIG_KNLP_WEIGHT_IO=n\n")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(wio.main(["plan", "--config", str(self.cfg_path)]), 1)

    def test_plan_names_every_stage(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(wio.main(["plan", "--config", str(self.cfg_path)]), 0)
        text = out.getvalue()
        for word in (
            "inventory: 2 repositories",
            "fetch: 1 controls",
            "groups ['layers', 'rows']",
            "gpu: off",
        ):
            self.assertIn(word, text)

    def test_geometry_is_geometry_only(self):
        geo = wio.device_geometry(self.root)
        for key in (
            "disk",
            "partition",
            "part_start_sectors",
            "logical_block_bytes",
            "max_sectors_kb",
            "max_segments",
        ):
            self.assertIn(key, geo)
        self.assertNotIn("serial", json_keys(geo))
        self.assertNotIn("model", json_keys(geo))

    def test_tracer_dir_default_under_root(self):
        cfg = wio.Config.from_file(self.cfg_path)
        self.assertEqual(
            wio.tracer_dir(cfg), self.root.resolve() / "tools" / "ebpf-syscall"
        )
        self.assertFalse(wio.have_tracers(cfg))


def json_keys(d):
    return {k.lower() for k in d}


if __name__ == "__main__":
    unittest.main()
