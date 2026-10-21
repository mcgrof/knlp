# SPDX-License-Identifier: MIT
"""Expose fixed loading phases without requiring a device for planning."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tools/reproduce/kmesh_restore"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("prefetch_entry", HERE / "entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
PRESET = ROOT / "defconfigs/kmesh-prefetch-cuda"


class PrefetchEntryTest(unittest.TestCase):
    def invoke(self, action, environment, *, config=PRESET, failure=False):
        stdout, stderr = StringIO(), StringIO()
        with (
            patch.dict(entry.os.environ, environment, clear=True),
            patch.object(sys, "argv", ["entry.py", action, "--config", str(config)]),
            patch.object(entry.subprocess, "run") as child,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            if failure:
                with self.assertRaises(SystemExit) as error:
                    entry.main()
                self.assertEqual(error.exception.code, 2)
                child.assert_not_called()
            else:
                entry.main()
        return stdout.getvalue(), stderr.getvalue(), child

    def inputs(self, root, *, phase="qualification", section="prefetch"):
        return {
            "KMESH_RESTORE_SECTION": section,
            "KMESH_RESTORE_PHASE": phase,
            "KMESH_RESTORE_OUTPUT": str(root / "fresh output"),
            "KMESH_RESTORE_MODEL_DIR": str(root / "model with spaces"),
            "KMESH_RESTORE_VLLM_SOURCE": str(root / "vllm source"),
            "KMESH_RESTORE_LMCACHE_SOURCE": str(root / "lmcache source"),
            "KMESH_RESTORE_GPU_PROVIDER": "vulkan_rm",
            "KMESH_RESTORE_QUALIFIED": str(root / "qualification capture"),
            "KMESH_RESTORE_PILOT": str(root / "pilot capture"),
        }

    def test_public_preset_and_environment_precedence(self):
        preset = entry.settings(PRESET, {})
        self.assertEqual(preset["SECTION"], "prefetch")
        self.assertEqual(preset["PHASE"], "qualification")
        self.assertEqual(preset["OUTPUT"], "results/kmesh-prefetch/qualification")
        self.assertEqual(preset["GPU_PROVIDER"], "vulkan_rm")
        self.assertEqual(preset["MODEL_DIR"], "")
        changed = entry.settings(
            PRESET,
            {
                "KMESH_RESTORE_PHASE": "confirmation",
                "KMESH_RESTORE_QUALIFIED": "q with spaces",
                "KMESH_RESTORE_PILOT": "p with spaces",
                "KMESH_RESTORE_OUTPUT": "new confirmation",
                "KMESH_RESTORE_GPU_PROVIDER": "native",
            },
        )
        self.assertEqual(changed["PHASE"], "confirmation")
        self.assertEqual(changed["QUALIFIED"], "q with spaces")
        self.assertEqual(changed["PILOT"], "p with spaces")
        self.assertEqual(changed["OUTPUT"], "new confirmation")
        self.assertEqual(changed["GPU_PROVIDER"], "native")

    def test_all_cpu_phase_plans_keep_source_pins_geometry_and_complete_grid(self):
        for phase, blocks, samples, fanouts in (
            ("qualification", 1, 1, [4]),
            ("pilot", 2, 5, [4]),
            ("confirmation", 10, 20, [4, 8]),
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                text, _, child = self.invoke(
                    "plan",
                    {
                        "KMESH_RESTORE_PHASE": phase,
                        "KMESH_RESTORE_OUTPUT": str(root / "not-created"),
                    },
                )
                plan = json.loads(text)
                self.assertEqual(
                    plan["vllm_commit"], "1b76bf9dd9d52efbd75fb89bf2619898d8f01233"
                )
                self.assertEqual(
                    plan["lmcache_commit"], "fb71fb40abe8ca3fdbe2b0a6b4823332d1a76f37"
                )
                self.assertEqual(plan["ancestor_bytes"], 48 << 20)
                self.assertEqual(plan["phase"], phase)
                self.assertEqual(plan["blocks_per_path_fanout"], blocks)
                self.assertEqual(plan["warmups_per_arm_block"], 3)
                self.assertEqual(plan["measured_groups_per_arm_block"], samples)
                self.assertEqual(plan["fanouts"], fanouts)
                self.assertEqual(
                    set(plan["missing_inputs"]),
                    {"MODEL_DIR", "VLLM_SOURCE", "LMCACHE_SOURCE"},
                )
                child.assert_not_called()
                self.assertFalse((root / "not-created").exists())

    def test_phase_dispatch_keeps_paths_literal_and_admits_only_required_receipts(self):
        for phase in ("qualification", "pilot", "confirmation"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                environment = self.inputs(Path(temporary), phase=phase)
                # Passing shell-looking text in an argv array must retain it as
                # one literal path; the test never invokes a child process.
                environment["KMESH_RESTORE_MODEL_DIR"] += " $(never-execute)"
                _, _, child = self.invoke("run", environment)
                argv = child.call_args.args[0]
                self.assertEqual(
                    argv[:2], [sys.executable, str(HERE / "prefetch_run.py")]
                )
                flags = dict(zip(argv[2::2], argv[3::2], strict=True))
                self.assertEqual(
                    flags,
                    {
                        "--output": environment["KMESH_RESTORE_OUTPUT"],
                        "--model": environment["KMESH_RESTORE_MODEL_DIR"],
                        "--vllm-source": environment["KMESH_RESTORE_VLLM_SOURCE"],
                        "--lmcache-source": environment["KMESH_RESTORE_LMCACHE_SOURCE"],
                        "--gpu-provider": "vulkan_rm",
                        "--phase": phase,
                    }
                    | (
                        {"--qualified": environment["KMESH_RESTORE_QUALIFIED"]}
                        if phase != "qualification"
                        else {}
                    )
                    | (
                        {"--pilot": environment["KMESH_RESTORE_PILOT"]}
                        if phase == "confirmation"
                        else {}
                    ),
                )
                self.assertEqual(child.call_args.kwargs, {"check": True})

    def test_offline_verify_needs_only_capture_output(self):
        text, _, child = self.invoke(
            "verify",
            {
                "KMESH_RESTORE_PHASE": "confirmation",
                "KMESH_RESTORE_OUTPUT": "saved capture with spaces",
                "KMESH_RESTORE_MODEL_DIR": "",
                "KMESH_RESTORE_VLLM_SOURCE": "",
                "KMESH_RESTORE_LMCACHE_SOURCE": "",
                "KMESH_RESTORE_GPU_PROVIDER": "",
                "KMESH_RESTORE_QUALIFIED": "",
                "KMESH_RESTORE_PILOT": "",
            },
        )
        self.assertEqual(text, "")
        child.assert_called_once_with(
            [
                sys.executable,
                str(HERE / "prefetch_campaign.py"),
                "saved capture with spaces",
            ],
            check=True,
        )

    def test_invalid_phase_refuses_before_dispatch(self):
        for action in ("plan", "run", "verify"):
            with self.subTest(action=action):
                _, error, _ = self.invoke(
                    action, {"KMESH_RESTORE_PHASE": "not-a-phase"}, failure=True
                )
                self.assertIn("invalid KMESH_RESTORE_PHASE", error)

    def test_run_refuses_missing_paths_or_phase_admissions(self):
        for phase, missing in (
            ("qualification", "MODEL_DIR"),
            ("qualification", "VLLM_SOURCE"),
            ("qualification", "LMCACHE_SOURCE"),
            ("qualification", "OUTPUT"),
            ("qualification", "GPU_PROVIDER"),
            ("pilot", "QUALIFIED"),
            ("confirmation", "QUALIFIED"),
            ("confirmation", "PILOT"),
        ):
            with (
                self.subTest(phase=phase, missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                environment = self.inputs(Path(temporary), phase=phase)
                environment["KMESH_RESTORE_" + missing] = ""
                _, error, _ = self.invoke("run", environment, failure=True)
                self.assertIn("KMESH_RESTORE_" + missing, error)
                self.assertFalse((Path(temporary) / "fresh output").exists())

    def test_missing_admissions_are_visible_in_cpu_plan(self):
        text, _, child = self.invoke(
            "plan",
            {
                "KMESH_RESTORE_PHASE": "confirmation",
                "KMESH_RESTORE_QUALIFIED": "",
                "KMESH_RESTORE_PILOT": "",
            },
        )
        self.assertTrue(
            {"QUALIFIED", "PILOT"}.issubset(json.loads(text)["missing_inputs"])
        )
        child.assert_not_called()

    def test_legacy_profile_commands_and_pins_are_unchanged(self):
        for section, script, pin, samples in (
            ("adapter", "run.py", "007a3388331209737ee19c095e430907fc467952", None),
            ("all", "run.py", "007a3388331209737ee19c095e430907fc467952", None),
            ("coldburst", "run.py", "007a3388331209737ee19c095e430907fc467952", "3"),
            (
                "singleflight",
                "singleflight.py",
                "100442de5a499af165fa4b797d9be75df2834943",
                None,
            ),
            ("family", "run.py", "44a52bda3b494e3d427e65b08d7196c70f631c47", "5"),
            ("retained", "run.py", "44a52bda3b494e3d427e65b08d7196c70f631c47", "5"),
        ):
            with (
                self.subTest(section=section),
                tempfile.TemporaryDirectory() as temporary,
            ):
                environment = self.inputs(
                    Path(temporary), phase="irrelevant", section=section
                )
                text, _, child = self.invoke("plan", environment)
                plan = json.loads(text)
                self.assertEqual(plan["lmcache_commit"], pin)
                argv = plan["command"]
                self.assertEqual(argv[:2], [sys.executable, str(HERE / script)])
                flags = dict(zip(argv[2::2], argv[3::2], strict=True))
                self.assertNotIn("--phase", flags)
                self.assertNotIn("--qualified", flags)
                self.assertNotIn("--pilot", flags)
                if section == "singleflight":
                    self.assertNotIn("--section", flags)
                else:
                    self.assertEqual(flags["--section"], section)
                if samples:
                    self.assertEqual(flags["--rounds"], "1")
                    self.assertEqual(flags["--serving-samples"], samples)
                    self.assertEqual(flags["--adapter-samples"], "1")
                else:
                    self.assertNotIn("--rounds", flags)
                child.assert_not_called()

    def test_legacy_verifiers_still_route_to_their_original_cpu_commands(self):
        for section in (
            "adapter",
            "all",
            "coldburst",
            "singleflight",
            "family",
            "retained",
        ):
            with self.subTest(section=section):
                _, _, child = self.invoke(
                    "verify",
                    {
                        "KMESH_RESTORE_SECTION": section,
                        "KMESH_RESTORE_OUTPUT": "old capture",
                    },
                )
                expected = (
                    [
                        sys.executable,
                        str(HERE / "singleflight.py"),
                        "--verify",
                        "--output",
                        "old capture",
                    ]
                    if section == "singleflight"
                    else [
                        sys.executable,
                        str(HERE / "verify.py"),
                        "old capture",
                        "--section",
                        section,
                    ]
                )
                child.assert_called_once_with(expected, check=True)


if __name__ == "__main__":
    unittest.main()
