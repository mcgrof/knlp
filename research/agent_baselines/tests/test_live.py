# SPDX-License-Identifier: MIT
"""Validate workload freezing, pinned templates, and task failure containment."""

import contextlib
import copy
import io
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from research.agent_baselines import live
from research.agent_baselines.recording_model import CallRecorder


def row(iid="repo__issue-1"):
    return {
        "instance_id": iid,
        "problem_statement": "Fix the issue",
        "base_commit": "b" * 40,
        "repo": "owner/repo",
        "test_patch": "grading-only test patch",
        "version": "1.0",
        "FAIL_TO_PASS": '["test_issue"]',
        "PASS_TO_PASS": "[]",
    }


def selection(rows=None):
    return {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": "a" * 40,
        "split": "test",
        "instances": rows if rows is not None else [row()],
    }


def server_info():
    return {
        "model_id": "test/model",
        "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "kv_cache_dtype": "auto",
        "prefix_caching": True,
        "server_version": "0.13.0",
        "hardware": "test fixture",
        "gpu_memory_utilization": 0.8,
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
        "cache_start_state": "cold",
        "chat_template_sha256": "b" * 64,
        "tool_call_parser": "qwen3_coder",
    }


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return path

    def test_frozen_rows_keep_grading_data_and_order(self):
        rows = [row("repo__z-1"), row("repo__a-2")]
        metadata, selected = live.read_instances(
            self.write("selection.json", selection(rows))
        )
        self.assertEqual(selected, rows)
        self.assertEqual(metadata["dataset_revision"], "a" * 40)
        self.assertNotIn("instances", metadata)

    def test_invalid_selection_rejected_before_execution(self):
        for data in (
            selection([]),
            selection([row(), row()]),
            selection([None]),
            selection([row("../outside")]),
            selection([row(12)]),
            {**selection(), "dataset_revision": "main"},
            {**selection(), "instances": {"id": row()}},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                live.read_instances(self.write("selection.json", data))

    def test_missing_grading_fields_and_custom_images_rejected(self):
        for change in (
            {"test_patch": None},
            {"FAIL_TO_PASS": "{}"},
            {"PASS_TO_PASS": [12]},
            {"image": "custom:latest"},
            {"docker_image": "custom:latest"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                live.read_instances(
                    self.write("selection.json", selection([{**row(), **change}]))
                )

    def test_server_placeholders_and_template_hash_rejected(self):
        info = server_info()
        self.assertEqual(live.server_provenance(self.write("server.json", info)), info)
        for change in (
            {"model_revision": "REPLACE_ME"},
            {"chat_template_sha256": "abc"},
            {"prefix_caching": "true"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                live.server_provenance(self.write("server.json", {**info, **change}))

    def test_checkout_rejects_untracked_code(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        (checkout / "tracked.py").write_text("pass\n")
        subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "Fixture",
            ],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(live.check_checkout(checkout, revision), checkout)
        (checkout / "module_override.py").write_text("pass\n")
        with self.assertRaises(ValueError):
            live.check_checkout(checkout, revision)

    def select_args(self, *extra):
        return live.parser().parse_args(
            [
                "select",
                "--revision",
                "a" * 40,
                "--out",
                str(self.root / "frozen.json"),
                *extra,
            ]
        )

    def test_select_dry_run_does_not_load_or_write(self):
        loader = Mock(side_effect=AssertionError("Unexpected dataset download"))
        with patch.dict(
            "sys.modules", {"datasets": types.SimpleNamespace(load_dataset=loader)}
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(live.select(self.select_args()), 0)
        self.assertEqual(json.loads(output.getvalue())["limit"], 2)
        loader.assert_not_called()
        self.assertFalse((self.root / "frozen.json").exists())

    def test_select_pins_dataset_and_preserves_all_grading_fields(self):
        rows = [row("repo__z-1"), row("repo__a-2"), row("repo__b-3")]
        loader = Mock(return_value=rows)
        with patch.dict(
            "sys.modules", {"datasets": types.SimpleNamespace(load_dataset=loader)}
        ):
            live.select(self.select_args("--execute"))
        loader.assert_called_once_with(
            "princeton-nlp/SWE-bench_Verified", split="test", revision="a" * 40
        )
        frozen = json.loads((self.root / "frozen.json").read_text())
        self.assertEqual(frozen["instances"], [rows[1], rows[2]])

    def test_explicit_selection_order_and_unknown_ids(self):
        rows = [row("repo__a-1"), row("repo__b-2")]
        loader = Mock(return_value=rows)
        with patch.dict(
            "sys.modules", {"datasets": types.SimpleNamespace(load_dataset=loader)}
        ):
            live.select(
                self.select_args("--ids", "repo__b-2", "repo__a-1", "--execute")
            )
        frozen = json.loads((self.root / "frozen.json").read_text())
        self.assertEqual(frozen["instances"], rows[::-1])
        (self.root / "frozen.json").unlink()
        with patch.dict(
            "sys.modules", {"datasets": types.SimpleNamespace(load_dataset=loader)}
        ), self.assertRaises(ValueError):
            live.select(self.select_args("--ids", "repo__missing-3", "--execute"))
        self.assertFalse((self.root / "frozen.json").exists())

    def task(self, *, fail=False, cleanup_fail=False, finalize_fail=False):
        response = {
            "prompt_token_ids": [1, 2],
            "choices": [{"token_ids": [3]}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_kwargs={}),
            abort_exceptions=[],
            _query=Mock(return_value=response),
        )
        environment = Mock()
        if cleanup_fail:
            environment.cleanup.side_effect = OSError("private error details")
        received = {}

        class Agent:
            def __init__(self, model, env, **kwargs):
                self.model = model
                received["config"] = kwargs

            def run(self, task):
                received["task"] = task
                self.model._query([{"role": "user", "content": task}])
                if fail:
                    raise RuntimeError("private exception details")
                return {"exit_status": "Submitted", "submission": "diff --git a/f b/f"}

        config = {"model": {"model_name": "openai/test"}, "agent": {"step_limit": 2}}
        original = copy.deepcopy(config)
        model_factory = Mock(return_value=model)
        environment_factory = Mock(return_value=environment)
        finalize = (
            patch.object(
                CallRecorder, "finish", side_effect=OSError("private disk error")
            )
            if finalize_fail
            else contextlib.nullcontext()
        )
        with finalize:
            result = live.run_task(
                row(), config, self.root, model_factory, environment_factory, Agent
            )
        self.assertEqual(config, original)
        environment.cleanup.assert_called_once()
        self.assertEqual(received["task"], "Fix the issue")
        self.assertNotIn("test_patch", received["config"])
        self.assertEqual(
            received["config"]["output_path"],
            self.root / "trajectories/repo__issue-1.json",
        )
        return result

    def test_task_success_records_exact_calls_and_finishes(self):
        result = self.task()
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertIsNone(result["error_type"])
        task = json.loads((self.root / "tasks/repo__issue-1.json").read_text())
        self.assertTrue(task["recording_complete"])
        self.assertEqual(task["verified_calls"], 1)

    def test_task_exception_cleans_up_and_records_failure(self):
        result = self.task(fail=True)
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["exit_status"], "InfrastructureError")
        self.assertNotIn("private", json.dumps(result))
        task = json.loads((self.root / "tasks/repo__issue-1.json").read_text())
        self.assertTrue(task["recording_complete"])

    def test_cleanup_failure_does_not_skip_recorder_finish(self):
        result = self.task(cleanup_fail=True)
        self.assertEqual(result["cleanup_error_type"], "OSError")
        task = json.loads((self.root / "tasks/repo__issue-1.json").read_text())
        self.assertTrue(task["recording_complete"])

    def test_recording_finalize_failure_is_contained(self):
        result = self.task(finalize_fail=True)
        self.assertEqual(result["recording_error_type"], "OSError")
        self.assertNotIn("private", json.dumps(result))

    def test_evaluation_uses_frozen_rows_and_creates_no_dry_run_files(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        for name, value in [
            ("manifest.json", {}),
            ("instances.json", [row()]),
            ("summary.json", {}),
        ]:
            (run_dir / name).write_text(json.dumps(value))
        (run_dir / "predictions.jsonl").write_text(
            json.dumps({"instance_id": "repo__issue-1", "model_patch": ""}) + "\n"
        )
        args = live.parser().parse_args(
            [
                "evaluate",
                "--checkout",
                str(self.root / "swebench"),
                "--run",
                str(run_dir),
                "--out",
                str(self.root / "evaluation"),
            ]
        )
        with patch.object(
            live, "check_checkout", return_value=args.checkout
        ) as check, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(live.evaluate(args), 0)
        check.assert_called_once_with(args.checkout, live.SWEBENCH_REVISION)
        plan = json.loads(output.getvalue())
        command = plan["argv"]
        self.assertEqual(
            command[command.index("--dataset_name") + 1],
            str(run_dir / "instances.json"),
        )
        self.assertEqual(
            command[command.index("--predictions_path") + 1],
            str(run_dir / "predictions.jsonl"),
        )
        self.assertEqual(
            plan["instances_sha256"], live.digest(run_dir / "instances.json")
        )
        self.assertFalse(args.out.exists())

        relative_python = self.root / "venv/bin/python"
        relative_python.parent.mkdir(parents=True)
        fake_python = self.root / "python-stub"
        fake_python.write_text("#!/bin/sh\nexit 0\n")
        fake_python.chmod(0o700)
        relative_python.symlink_to(fake_python)
        args.python = os.path.relpath(relative_python, Path.cwd())
        with patch.object(
            live, "check_checkout", return_value=args.checkout
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            live.evaluate(args)
        self.assertEqual(json.loads(output.getvalue())["argv"][0], str(relative_python))
        args.execute = True
        with patch.object(live, "check_checkout", return_value=args.checkout):
            self.assertEqual(live.evaluate(args), 0)
        self.assertEqual(
            json.loads((args.out / "exit.json").read_text())["returncode"], 0
        )
        self.assertTrue((args.out / "evaluation.log").exists())


@unittest.skipUnless(
    os.getenv("MINI_SWE_AGENT_CHECKOUT"),
    "Set MINI_SWE_AGENT_CHECKOUT for pinned-template integration",
)
class PinnedTemplateTests(unittest.TestCase):
    def test_real_stock_and_stable_prefix_keep_identical_blocks(self):
        import yaml
        from jinja2 import Template

        checkout = live.check_checkout(
            Path(os.environ["MINI_SWE_AGENT_CHECKOUT"]), live.MINI_REVISION
        )
        config_path = checkout / "src/minisweagent/config/benchmarks/swebench.yaml"
        original = yaml.safe_load(config_path.read_text())
        arguments = live.parser().parse_args(
            [
                "run",
                "--checkout",
                str(checkout),
                "--instances",
                "selection.json",
                "--server-info",
                "server.json",
                "--model",
                "openai/test",
                "--endpoint",
                "http://localhost:8000/v1",
                "--out",
                "out",
            ]
        )
        stock = live.make_config(arguments, checkout)
        self.assertEqual(stock["model"]["model_kwargs"]["max_retries"], 0)
        arguments.arm = "stable-prefix"
        stable = live.make_config(arguments, checkout)
        stock_template = stock["agent"]["instance_template"]
        stable_template = stable["agent"]["instance_template"]
        self.assertEqual(stock_template, original["agent"]["instance_template"])
        self.assertEqual(
            stock["agent"]["system_template"], stable["agent"]["system_template"]
        )
        stable_start = stock_template.index("<instructions>")
        task_block = stock_template[:stable_start].rstrip()
        instruction_block = stock_template[stable_start:].rstrip()
        self.assertTrue(stable_template.startswith(instruction_block))
        self.assertEqual(stable_template.count(task_block), 1)
        self.assertEqual(stable_template.count(instruction_block), 1)
        self.assertEqual(
            Template(stable_template).render(task="UNIQUE TASK").count("UNIQUE TASK"), 1
        )
        self.assertEqual(stock["model"], stable["model"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments.instances = root / "instances.json"
            arguments.instances.write_text(json.dumps(selection()))
            arguments.server_info = root / "server.json"
            arguments.server_info.write_text(json.dumps(server_info()))
            arguments.out = root / "run"
            arguments.metrics_url = "http://localhost:8000/metrics"
            with patch(
                "research.agent_baselines.metrics.snapshot"
            ) as scrape, contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(live.run(arguments), 0)
            manifest = json.loads(output.getvalue())
            self.assertEqual(manifest["config"], stable)
            self.assertEqual(len(manifest["knlp_revision"]), 40)
            self.assertIsInstance(manifest["knlp_dirty"], bool)
            self.assertEqual(manifest["task_order"], ["repo__issue-1"])
            self.assertFalse(arguments.out.exists())
            scrape.assert_not_called()


if __name__ == "__main__":
    unittest.main()
