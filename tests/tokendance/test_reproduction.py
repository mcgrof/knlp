# SPDX-License-Identifier: MIT
"""Exercise source/workload identity and the portable execution boundary."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tools/reproduce/tokendance"
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "harness"))
from contract import (
    digest,
    profile,
    source_identity,
    validate_assets,
    validate_sources,
    validate_trace,
)

spec = importlib.util.spec_from_file_location("tokendance_runner", PACKAGE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReproductionTest(unittest.TestCase):
    def test_both_explicit_source_pairs_and_wrong_commit_rejection(self):
        for pair in ("dated", "reference"):
            source = copy.deepcopy(profile()["source_pairs"][pair])
            validate_sources(source, pair)
            source["vllm"]["commit"] = "0" * 40
            with self.assertRaisesRegex(ValueError, "source pins"):
                validate_sources(source, pair)

    def test_dirty_source_is_rejected(self):
        source = copy.deepcopy(profile()["source_pairs"]["dated"])
        source["lmcache"]["status_porcelain"] = " M cache_engine.py"
        with self.assertRaisesRegex(ValueError, "dirty"):
            validate_sources(source)

    def test_archived_pair_is_not_a_live_run_substitute(self):
        with self.assertRaisesRegex(ValueError, "source pins"):
            validate_sources(profile()["source_pairs"]["reference"])

    def test_assets_are_bound(self):
        self.assertIn(
            "harness/verify_online_collective_lockstep_gate.py", validate_assets()
        )
        import collective_gate_common as common

        self.assertEqual(common.SERIAL_GATE_SHA256, digest(common._serial_gate_path()))

    def test_cpu_profile_generates_exact_trace_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            command = [
                sys.executable,
                str(PACKAGE / "run.py"),
                "cpu",
                "--output",
                str(output),
            ]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(
                validate_trace(output / "trace")["trace_sha256"],
                profile()["trace_sha256"],
            )
            result = json.loads((output / "result.json").read_text())
            self.assertFalse(result["executed_kv_reuse"])
            self.assertFalse(result["gpu_execution"])
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(json.loads((output / "result.json").read_text()), result)
            trace = output / "trace/trace.jsonl"
            trace.write_text(trace.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "pinned synthetic"):
                validate_trace(output / "trace")

    def test_gpu_requires_a_reservation_before_any_runtime_work(self):
        env = dict(os.environ)
        env.pop("GPUQ_RUN_ID", None)
        result = subprocess.run(
            [sys.executable, str(PACKAGE / "run.py"), "gpu"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires gpuq", result.stderr)

    def test_unknown_config_mode_cannot_fall_through_to_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / ".config"
            config.write_text('CONFIG_KNLP_TOKENDANCE_MODE="typo"\n')
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKAGE / "run.py"),
                    "--config",
                    str(config),
                    "run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be cpu or gpu", result.stderr)

    def test_container_cpu_has_no_gpu_and_sources_are_read_only(self):
        cfg = {"DOCKER": "docker", "RUNTIME_VOLUME": "test-runtime"}
        identity = {
            name: {
                "path": "/example/" + name,
                "git_common_dir": "/example/" + name + "/.git",
            }
            for name in ("vllm", "lmcache")
        }
        _, command = runner.container_command(
            cfg,
            identity,
            Path("/example/output"),
            "verify_numerical_warmup_contract.py",
            [],
            False,
        )
        self.assertNotIn("--device", command)
        self.assertIn("HIP_VISIBLE_DEVICES=", command)
        self.assertIn("/example/vllm:/sources/vllm:ro", command)
        self.assertIn("none", command)

    def test_config_and_environment_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".config"
            path.write_text(
                'CONFIG_KNLP_TOKENDANCE_MODE="gpu"\nCONFIG_KNLP_TOKENDANCE_OUTPUT="one"\n'
            )
            with patch.dict(os.environ, {"TOKENDANCE_OUTPUT": "two"}):
                settings = runner.settings(path)
            self.assertEqual(settings["MODE"], "gpu")
            self.assertEqual(settings["OUTPUT"], "two")

    def test_model_snapshot_external_symlink_is_mounted_as_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            blob = root / "model-blob"
            blob.write_bytes(b"weights")
            (snapshot / "model.safetensors").symlink_to(blob)
            cfg = {
                "DOCKER": "docker",
                "RUNTIME_VOLUME": "runtime",
                "MODEL_DIR": str(snapshot),
            }
            _, command = runner.container_command(
                cfg, {}, root / "out", "preflight_resolved_profile.py", [], True
            )
            self.assertIn(
                f"{blob}:{runner.MODEL_CONTAINER}/model.safetensors:ro", command
            )
            self.assertNotIn(f"{snapshot}:{runner.MODEL_CONTAINER}:ro", command)

    def test_shared_git_clone_keeps_its_object_dependency_in_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, clone = root / "source", root / "clone"
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "input").write_text("frozen input\n")
            subprocess.run(["git", "-C", str(source), "add", "input"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.org",
                    "commit",
                    "-qm",
                    "input",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "clone", "-q", "--shared", str(source), str(clone)],
                check=True,
            )
            identity = source_identity(clone)
            objects = str(source / ".git/objects")
            self.assertEqual(identity["git_alternates"], [objects])
            _, command = runner.container_command(
                {"DOCKER": "docker", "RUNTIME_VOLUME": "runtime"},
                {"vllm": identity},
                root / "output",
                "measure_family.py",
                [],
                False,
            )
            self.assertIn(f"{objects}:{objects}:ro", command)

    def test_report_rejects_self_reported_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "verified.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "claims": {"full_collective_lockstep_gate_pass": False},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "independent verification"):
                runner.report(path, Path(temporary) / "report.json")


if __name__ == "__main__":
    unittest.main()
