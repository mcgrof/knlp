# SPDX-License-Identifier: MIT
"""Planner checks for the CS312 tutorial lane; no torch, no network."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/reproduce/cs312"))

import run as cs312  # noqa: E402


def defaults(**overrides):
    cfg = dict(cs312.DEFAULTS)
    cfg["RUN_SUFFIX"] = "test"
    cfg.update(overrides)
    return cfg


class LogGridTest(unittest.TestCase):
    def test_base_three_is_centred_on_the_default(self):
        grid = cs312.log_grid(0.003, 3.0, 5)
        self.assertEqual([k for k, _ in grid], [-2, -1, 0, 1, 2])
        self.assertEqual(
            [v for _, v in grid], [0.000333333, 0.001, 0.003, 0.009, 0.027]
        )

    def test_integer_axis_rounds_and_stays_positive(self):
        grid = cs312.log_grid(64, 2.0, 5, integer=True)
        self.assertEqual([v for _, v in grid], [16, 32, 64, 128, 256])
        grid = cs312.log_grid(64, 3.0, 5, integer=True)
        self.assertEqual([v for _, v in grid], [7, 21, 64, 192, 576])


class PlanTest(unittest.TestCase):
    def test_default_plan_has_seventeen_unique_runs(self):
        plan = cs312.build_plan(defaults())
        self.assertEqual(len(plan["runs"]), 17)
        self.assertEqual(plan["runs"][0]["axis"], "baseline")
        self.assertEqual(plan["runs"][0]["overrides"], {})
        self.assertEqual(
            sorted(plan["runs"][0]["axes"]),
            ["batch_size", "learning_rate", "warmup_percent", "weight_decay"],
        )
        # Inner ring before outer ring so a sequential sweep covers all axes early.
        self.assertEqual([r["exponent"] for r in plan["runs"][1:9]].count(1), 4)
        self.assertEqual([abs(r["exponent"]) for r in plan["runs"][9:]], [2] * 8)

    def test_train_config_carries_plan_tags_and_suffix(self):
        plan = cs312.build_plan(defaults())
        config = plan["runs"][3]["train_config"]
        self.assertEqual(config["model_name"], "d8")
        self.assertEqual(config["num_train_sequences"], 600000)
        self.assertEqual(config["run_name_suffix"], "test")
        self.assertIn(f"cs312-plan-{plan['digest']}", config["wandb_tags"])
        self.assertIn("cs312-test", config["wandb_tags"])

    def test_digest_ignores_backend_and_suffix(self):
        modal = cs312.build_plan(defaults(BACKEND="modal", RUN_SUFFIX="modal"))
        local = cs312.build_plan(defaults(BACKEND="local", RUN_SUFFIX="local"))
        self.assertEqual(modal["digest"], local["digest"])
        smaller = cs312.build_plan(defaults(NUM_TRAIN_SEQUENCES="1000"))
        self.assertNotEqual(modal["digest"], smaller["digest"])

    def test_settings_reads_kconfig_choices(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".config", delete=False) as f:
            f.write(
                "CONFIG_KNLP_CS312=y\nCONFIG_KNLP_CS312_BACKEND_LOCAL=y\n"
                'CONFIG_KNLP_CS312_P1A_BASE="2"\nCONFIG_KNLP_CS312_DEPTH=6\n'
            )
        cfg = cs312.settings(f.name)
        self.assertEqual(cfg["BACKEND"], "local")
        self.assertEqual(cfg["RUN_SUFFIX"], "local")
        self.assertEqual(cfg["P1A_BASE"], "2")
        self.assertEqual(cfg["DEPTH"], "6")


if __name__ == "__main__":
    unittest.main()
