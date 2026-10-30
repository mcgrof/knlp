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


class PlanP1bTest(unittest.TestCase):
    def test_trimmed_design_adds_23_new_runs_off_the_1a_grid(self):
        p1a = cs312.build_plan(defaults())
        p1b = cs312.build_plan(defaults(PROBLEM="p1b"))
        self.assertEqual(len(p1b["runs"]), 23)
        self.assertNotEqual(p1a["digest"], p1b["digest"])
        old = {cs312.json.dumps(r["overrides"], sort_keys=True) for r in p1a["runs"]}
        new = [cs312.json.dumps(r["overrides"], sort_keys=True) for r in p1b["runs"]]
        self.assertFalse(old & set(new))
        self.assertEqual(len(set(new)), len(new))
        # pair cells use exactly the 1(a) grid values
        grid = {axis: set(v) for axis, v in p1a["axis_values"].items()}
        for run in p1b["runs"]:
            if run["axis"] == "seed":
                continue
            for axis, value in run["overrides"].items():
                self.assertIn(value, grid[axis])
        seeds = [r for r in p1b["runs"] if r["axis"] == "seed"]
        self.assertEqual(
            [r["overrides"] for r in seeds],
            [{"model_seed": 43, "data_seed": 43}, {"model_seed": 44, "data_seed": 44}],
        )

    def test_full_corners_gives_four_cells_per_pair(self):
        p1b = cs312.build_plan(
            defaults(PROBLEM="p1b", P1B_FULL_CORNERS="y", P1B_SEED_REPEATS="0")
        )
        self.assertEqual(len(p1b["runs"]), 24)
        self.assertEqual(set(p1b["runs"][0]["exponents"].values()), {1})


if __name__ == "__main__":
    unittest.main()


class KvqPlanTest(unittest.TestCase):
    def test_frontier_crosses_levels_grans_qknorm(self):
        plan = cs312.build_plan(defaults(PROBLEM="kvq"))
        runs = plan["runs"]
        # 3 levels x 2 grans x 2 qknorm
        self.assertEqual(len(runs), 12)
        self.assertTrue(all(r["axis"] == "frontier" for r in runs))
        for r in runs:
            ov = r["overrides"]
            self.assertEqual(ov["model_builder"], "cs312_quant:build_quant_model")
            self.assertIn("granularity", ov["model_builder_kwargs"])
            self.assertIn(ov["qk_norm"], (True, False))
        # K4V16 -> k_bits 4, v_bits 16
        k4 = next(r for r in runs if r["level"] == "K4V16")
        self.assertEqual(k4["overrides"]["model_builder_kwargs"]["k_bits"], 4)
        self.assertEqual(k4["overrides"]["model_builder_kwargs"]["v_bits"], 16)

    def test_suffix_is_unique_per_cell_under_build_plan(self):
        plan = cs312.build_plan(defaults(PROBLEM="kvq", RUN_SUFFIX="kvq"))
        names = []
        for r in plan["runs"]:
            tc = r["train_config"]
            # run_name_suffix encodes level+gran; qk_norm off adds -noqknorm
            # in the course run_name, so suffix need not encode qk
            key = (
                tc["run_name_suffix"],
                tc["qk_norm"],
                tuple(sorted(tc["model_builder_kwargs"].items())),
            )
            names.append(key)
        self.assertEqual(len(set(names)), len(names))

    def test_sweep_requires_stress_and_builds_hp_cells(self):
        with self.assertRaises(SystemExit):
            cs312.build_plan(defaults(PROBLEM="kvq", KVQ_PHASE="sweep"))
        plan = cs312.build_plan(
            defaults(PROBLEM="kvq", KVQ_PHASE="sweep", KVQ_STRESS="K4V4:per_tensor:off")
        )
        axes = sorted({r["axis"] for r in plan["runs"]})
        self.assertEqual(
            axes, ["learning_rate", "seed", "stress-center", "warmup", "weight_decay"]
        )
        center = next(r for r in plan["runs"] if r["axis"] == "stress-center")
        self.assertFalse(center["qk_norm"])
        self.assertEqual(center["granularity"], "per_tensor")
        self.assertEqual(center["overrides"]["model_builder_kwargs"]["k_bits"], 4)
