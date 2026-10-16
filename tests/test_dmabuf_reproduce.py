# SPDX-License-Identifier: MIT
"""Offline orchestration checks; no GPU, kernel installation or storage DMA."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "tools/reproduce/dmabuf/run.py"
SPEC = importlib.util.spec_from_file_location("dmabuf_reproduce", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ReproductionTests(unittest.TestCase):
    def test_cpu_preflight_also_hides_gpus_and_disables_hardware_opt_ins(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = runner.Workflow({"WORKDIR": directory}, {})
            workflow.source = lambda name: Path(directory)
            calls = []

            def execute(label, argv, **kwargs):
                calls.append(kwargs["env"])
                if label == "pd-mp-test":
                    workflow.results.mkdir()
                    (workflow.results / "pd-mp-test.xml").write_text(
                        "<testsuite><testcase name='control'/></testsuite>"
                    )

            workflow.execute = execute
            with patch.dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0",
                LMCACHE_RUN_HOST_DMABUF_IO="1",
                PYTHONPATH=os.pathsep.join(
                    [str(workflow.root / "src/vllm"), directory]
                ),
            ):
                workflow.tests("pd-mp-test")
            self.assertEqual(len(calls), 2)
            for env in calls:
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES",
                ):
                    self.assertEqual(env[key], "")
                self.assertEqual(env["LMCACHE_RUN_HOST_DMABUF_IO"], "0")
                self.assertNotIn(
                    str(workflow.root / "src/vllm"), env["PYTHONPATH"].split(os.pathsep)
                )
                self.assertIn(directory, env["PYTHONPATH"].split(os.pathsep))

    def test_fetch_exact_revision_and_refuse_edits_or_wrong_pin(self):
        with tempfile.TemporaryDirectory(prefix="dmabuf-test-") as directory:
            root = Path(directory)
            source = root / "source with spaces"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            (source / "payload").write_text("known bytes\n")
            subprocess.run(["git", "-C", str(source), "add", "payload"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            revision = runner.git(source, "rev-parse", "HEAD")
            destination = root / "destination with spaces"
            runner.fetch_source(destination, {"url": str(source), "commit": revision})
            runner.verify_source(destination, revision)
            cache = root / "cache"
            (cache / "src").mkdir(parents=True)
            (cache / "src/lmcache").symlink_to(destination, target_is_directory=True)
            workflow = runner.Workflow(
                {"WORKDIR": str(cache)}, {"lmcache": {"commit": revision}}
            )
            self.assertEqual(workflow.source("lmcache"), destination.resolve())
            with self.assertRaises(RuntimeError):
                runner.verify_source(destination, "0" * 40)
            (destination / "payload").write_text("local change\n")
            with self.assertRaises(RuntimeError):
                runner.fetch_source(
                    destination, {"url": str(source), "commit": revision}
                )
            self.assertEqual((destination / "payload").read_text(), "local change\n")

    def test_no_empty_or_skipped_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "checks.xml"
            for content in [
                "<testsuite/>",
                "<testsuite><testcase><skipped/></testcase></testsuite>",
                "<testsuite><testcase><error/></testcase></testsuite>",
            ]:
                report.write_text(content)
                with self.assertRaises(RuntimeError):
                    runner.check_junit(report)
            report.write_text("<testsuite><testcase name='actual'/></testsuite>")
            runner.check_junit(report)

    def test_plan_is_nonmutating_and_install_needs_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            workflow = runner.Workflow({"WORKDIR": str(root)}, {})
            workflow.plan()
            self.assertFalse(root.exists())
            with self.assertRaises(RuntimeError):
                workflow.kernel_install()
            self.assertFalse(root.exists())

    def test_config_paths_are_data_not_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".config"
            value = str(Path(directory) / "literal $(no-command) `also-literal`")
            path.write_text("CONFIG_KNLP_DMABUF_WORKDIR=" + json.dumps(value) + "\n")
            self.assertEqual(runner.read_config(path)["WORKDIR"], value)

    def test_install_rejects_unrecorded_or_changed_build(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = runner.Workflow(
                {"WORKDIR": directory}, {"linux": {"commit": "a" * 40}}
            )
            workflow.build.mkdir(parents=True)
            (workflow.build / "vmlinux").write_bytes(b"incomplete build")
            with self.assertRaises(RuntimeError):
                workflow.kernel_install()
            (workflow.build / ".config").write_text("CONFIG_IO_URING=y\n")
            workflow.results.mkdir()
            receipt = {
                "source": "a" * 40,
                "config": runner.digest(workflow.build / ".config"),
                "vmlinux": runner.digest(workflow.build / "vmlinux"),
            }
            (workflow.results / "kernel-build.json").write_text(json.dumps(receipt))
            (workflow.build / ".config").write_text("CONFIG_IO_URING=n\n")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                workflow.kernel_install()


if __name__ == "__main__":
    unittest.main()
