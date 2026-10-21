# SPDX-License-Identifier: MIT
"""Exercise public qualification dispatch without importing a GPU runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ResidencyDispatchTest(unittest.TestCase):
    def run_cli(self, config, *args, **env):
        environment = {
            k: v for k, v in os.environ.items() if not k.startswith("KMESH_RESIDENCY_")
        }
        environment.update(env)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.reproduce.kmesh_residency.run",
                "--config",
                str(config),
                *args,
            ],
            env=environment,
            text=True,
            capture_output=True,
        )

    def test_plan_exposes_cpu_and_gpu_modes_without_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            for mode in ("cpu", "serving", "tot"):
                config.write_text(
                    "CONFIG_KNLP_KMESH_RESIDENCY_MODE=" + json.dumps(mode) + "\n"
                )
                result = self.run_cli(config, "--plan")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    json.loads(result.stdout)["gpu_execution"], mode != "cpu"
                )

    def test_environment_overrides_config_and_unknown_mode_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config"
            config.write_text('CONFIG_KNLP_KMESH_RESIDENCY_SOURCE="configured"\n')
            result = self.run_cli(config, "--plan", KMESH_RESIDENCY_SOURCE="override")
            self.assertEqual(json.loads(result.stdout)["source"], "override")
            result = self.run_cli(config, "--plan", KMESH_RESIDENCY_MODE="guess")
            self.assertNotEqual(result.returncode, 0)

    def test_missing_inputs_fail_before_runtime_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "absent"
            output = Path(directory) / "output"
            for mode in ("cpu", "serving", "tot"):
                result = self.run_cli(
                    config,
                    KMESH_RESIDENCY_MODE=mode,
                    KMESH_RESIDENCY_OUTPUT=str(output),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("set KMESH_RESIDENCY_SOURCE", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
