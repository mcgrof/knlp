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


class PlanP1cTest(unittest.TestCase):
    def test_cells_become_labelled_runs_at_the_raised_lr_and_wd(self):
        plan = cs312.build_plan(defaults(PROBLEM="p1c"))
        self.assertEqual([r["label"] for r in plan["runs"]], ["A", "B", "C"])
        self.assertEqual(
            [(r["lr_schedule"], r["warmup_percent"]) for r in plan["runs"]],
            [("wsd0.2", 0.0), ("wsd0.2", 0.2), ("linear", 0.1)],
        )
        for run in plan["runs"]:
            self.assertEqual(run["overrides"]["learning_rate"], 0.009)
            self.assertEqual(run["overrides"]["weight_decay"], 1.0)
            # the token budget stays standard, so the cells are compute matched
            self.assertNotIn("num_train_sequences", run["overrides"])

    def test_every_cell_gets_its_own_run_name(self):
        plan = cs312.build_plan(defaults(PROBLEM="p1c"))
        names = {cs312.json.dumps(r["train_config"], sort_keys=True) for r in plan["runs"]}
        self.assertEqual(len(names), 3)

    def test_a_malformed_cell_is_rejected(self):
        with self.assertRaises(SystemExit):
            cs312.build_plan(defaults(PROBLEM="p1c", P1C_CELLS="wsd0.2"))


class PlanP1cMultiRateTest(unittest.TestCase):
    def test_a_list_of_rates_repeats_the_cells_at_each_rate(self):
        plan = cs312.build_plan(defaults(PROBLEM="p1c", P1C_LR="0.009,0.003,0.001"))
        self.assertEqual(len(plan["runs"]), 9)
        self.assertEqual(plan["learning_rates"], [0.009, 0.003, 0.001])
        self.assertEqual(
            [r["label"] for r in plan["runs"]],
            ["A", "B", "C"] * 3,
        )
        for rate in (0.009, 0.003, 0.001):
            group = [r for r in plan["runs"] if r["learning_rate"] == rate]
            self.assertEqual(len(group), 3)
            for run in group:
                self.assertEqual(run["overrides"]["learning_rate"], rate)
                self.assertEqual(run["overrides"]["weight_decay"], 1.0)

    def test_indices_stay_unique_across_rates(self):
        plan = cs312.build_plan(defaults(PROBLEM="p1c", P1C_LR="0.009,0.003"))
        self.assertEqual([r["index"] for r in plan["runs"]], list(range(6)))


class PlanP2Test(unittest.TestCase):
    def test_every_variant_is_trained_at_every_depth(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2", P2_DEPTHS="4,5"))
        self.assertEqual(len(plan["runs"]), 8)
        self.assertEqual(plan["depths"], [4, 5])
        for run in plan["runs"]:
            self.assertEqual(run["overrides"]["model_name"], f"d{run['depth']}")

    def test_variants_carry_their_recipe(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2", P2_DEPTHS="8"))
        by = {r["variant"]: r["overrides"] for r in plan["runs"]}
        self.assertEqual(by["baseline"]["lr_schedule"], "linear")
        self.assertEqual(by["constant"]["lr_schedule"], "constant")
        self.assertEqual(by["dropout0.2"]["dropout"], 0.2)
        self.assertEqual(by["lr0.03"]["learning_rate"], 0.03)

    def test_param_counts_rise_with_depth_and_exclude_embeddings(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2", P2_DEPTHS="4,8"))
        d4 = next(r for r in plan["runs"] if r["depth"] == 4)
        d8 = next(r for r in plan["runs"] if r["depth"] == 8)
        self.assertLess(d4["params"], d8["params"])
        self.assertLess(d8["params_embedding_free"], d8["params"])

    def test_a_malformed_variant_is_rejected(self):
        with self.assertRaises(SystemExit):
            cs312.build_plan(defaults(PROBLEM="p2", P2_VARIANTS="baseline=0.003:linear"))


class PlanP2ArchTest(unittest.TestCase):
    ARCH = ("hc=0.003:linear:0.0:hc,mhc=0.003:linear:0.0:mhc,"
            "layerrope=0.003:linear:0.0:layerrope:-0.5,"
            "mhc_layerrope=0.003:linear:0.0:mhc_layerrope:0.0")

    def test_arch_variants_carry_the_builder_and_kwargs(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2", P2_DEPTHS="4",
                                         P2_VARIANTS=self.ARCH))
        by = {r["variant"]: r for r in plan["runs"]}
        self.assertEqual(by["hc"]["overrides"]["model_builder"],
                         "cs312_arch:build_arch_model")
        self.assertEqual(by["mhc"]["overrides"]["model_builder_kwargs"]["variant"], "mhc")
        self.assertEqual(
            by["layerrope"]["overrides"]["model_builder_kwargs"]["layerrope_alpha"], -0.5)
        self.assertEqual(
            by["mhc_layerrope"]["overrides"]["model_builder_kwargs"]["layerrope_alpha"], 0.0)

    def test_architectures_do_not_collide_on_one_run_name(self):
        # the course run name ignores model_builder, so the suffix must separate them
        plan = cs312.build_plan(defaults(PROBLEM="p2", P2_DEPTHS="4",
                                         P2_VARIANTS=self.ARCH))
        names = {cs312.json.dumps(r["train_config"], sort_keys=True) for r in plan["runs"]}
        self.assertEqual(len(names), 4)
        suffixes = {r["suffix"] for r in plan["runs"]}
        self.assertEqual(len(suffixes), 4)

    def test_plain_recipe_variants_keep_their_original_suffix(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2", P2_DEPTHS="4"))
        for run in plan["runs"]:
            self.assertEqual(run["suffix"], "test")
            self.assertNotIn("model_builder", run["overrides"])


class PlanP2bTest(unittest.TestCase):
    def test_mup_scales_the_rate_as_one_over_width(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2b", P2_DEPTHS="4,8,16",
                                         P2B_VARIANTS="mup_lr"))
        rates = {r["depth"]: r["overrides"]["learning_rate"] for r in plan["runs"]}
        # width is 64*depth, so halving depth doubles the rate
        self.assertAlmostEqual(rates[8], 0.003)
        self.assertAlmostEqual(rates[4], 0.006)
        self.assertAlmostEqual(rates[16], 0.0015)

    def test_each_intervention_touches_its_own_axis(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2b", P2_DEPTHS="8"))
        by = {r["variant"]: r["overrides"] for r in plan["runs"]}
        self.assertEqual(by["tokens2x"]["num_train_sequences"], 1_200_000)
        self.assertEqual(by["batch128"]["batch_size"], 128)
        self.assertEqual(by["kv16v8"]["model_builder_kwargs"]["v_bits"], 8)
        self.assertEqual(by["kv16v8"]["model_builder_kwargs"]["k_bits"], 16)
        self.assertNotIn("learning_rate", by["tokens2x"])

    def test_interventions_get_distinct_run_names(self):
        plan = cs312.build_plan(defaults(PROBLEM="p2b", P2_DEPTHS="8"))
        self.assertEqual(len({r["suffix"] for r in plan["runs"]}), 4)

    def test_an_unknown_intervention_is_rejected(self):
        with self.assertRaises(SystemExit):
            cs312.build_plan(defaults(PROBLEM="p2b", P2B_VARIANTS="not_a_thing"))


class PlanP3P4Test(unittest.TestCase):
    def test_p3_has_the_assigned_run_count_and_parts(self):
        plan = cs312.build_plan(defaults(PROBLEM="p3"))
        self.assertEqual(len(plan["runs"]), 25)
        parts = {}
        for r in plan["runs"]:
            parts[r["part"]] = parts.get(r["part"], 0) + 1
        self.assertEqual(parts, {"ref": 2, "a": 3, "b": 9, "c": 11})

    def test_p3_isolates_one_source_per_arm(self):
        plan = cs312.build_plan(defaults(PROBLEM="p3"))
        by = {}
        for r in plan["runs"]:
            by.setdefault(r["label"], []).append(r["overrides"])
        # initialisation arm: deterministic, seeds differ, data seed untouched
        init = by["initialisation only"]
        self.assertTrue(all(o["deterministic"] for o in init))
        self.assertEqual(len({o["model_seed"] for o in init}), 3)
        self.assertTrue(all("data_seed" not in o for o in init))
        # data arm: deterministic, one model seed, data seeds differ
        data = by["data order only"]
        self.assertEqual(len({o["model_seed"] for o in data}), 1)
        self.assertEqual(len({o["data_seed"] for o in data}), 3)
        # hardware arm: every setting identical, only the GPU is free
        hw = by["hardware only"]
        self.assertEqual(len({tuple(sorted(o.items())) for o in hw}), 1)
        self.assertTrue(all(not o["deterministic"] for o in hw))

    def test_identical_runs_still_get_distinct_names(self):
        for problem, n in (("p3", 25), ("p4", 15)):
            plan = cs312.build_plan(defaults(PROBLEM=problem))
            self.assertEqual(len({r["suffix"] for r in plan["runs"]}), n)

    def test_p4_pairs_differ_only_by_the_perturbation(self):
        plan = cs312.build_plan(defaults(PROBLEM="p4"))
        self.assertEqual(len(plan["runs"]), 15)
        ctl = [r for r in plan["runs"] if r["part"] == "control"]
        pert = [r for r in plan["runs"] if r["part"] == "perturbed"]
        self.assertEqual((len(ctl), len(pert)), (7, 7))
        for c_run, p_run in zip(ctl, pert):
            a, b = dict(c_run["overrides"]), dict(p_run["overrides"])
            self.assertTrue(b.pop("perturb_one_token"))
            self.assertEqual(a, b)
        self.assertTrue(all(r["overrides"]["deterministic"] for r in plan["runs"]))


class PlanP5Test(unittest.TestCase):
    def test_beta1_sweep_skips_the_value_already_measured(self):
        plan = cs312.build_plan(defaults(PROBLEM="p5", P5_BETA1="0.0,0.5,0.9,0.98"))
        betas = [r["overrides"]["beta1"] for r in plan["runs"]]
        self.assertNotIn(0.9, betas)           # the corpus already has it
        self.assertEqual(sorted(betas), [0.0, 0.5, 0.98])

    def test_beta1_is_the_only_thing_that_changes(self):
        plan = cs312.build_plan(defaults(PROBLEM="p5"))
        for r in plan["runs"]:
            self.assertEqual(list(r["overrides"]), ["beta1"])
