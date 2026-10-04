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
from research.agent_baselines.sglang_adapter import SGLANG_REVISION


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
        "backend": "vllm",
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
            {"backend": "other"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                live.server_provenance(self.write("server.json", {**info, **change}))

    def test_sglang_provenance_is_pinned_and_builds_capability_receipt(self):
        info = {
            **server_info(),
            "backend": "sglang",
            "server_revision": SGLANG_REVISION,
            "cache_scope": "test:model:no-adapter:campaign",
            "enabled_capabilities": [
                "openai_chat",
                "sglext_token_ids",
                "cache_source_report",
            ],
        }
        provenance = live.server_provenance(self.write("sglang.json", info))
        self.assertEqual(
            provenance["capability_receipt"]["source_revision"], SGLANG_REVISION
        )
        self.assertTrue(
            provenance["capability_receipt"]["features"]["sglext_token_ids"]["enabled"]
        )
        for change in (
            {"server_revision": "a" * 40},
            {"enabled_capabilities": ["openai_chat"]},
            {"cache_scope": None},
            {"cache_scope": "has spaces"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                live.server_provenance(self.write("invalid.json", {**info, **change}))

    def test_vllm_rejects_an_inapplicable_cache_scope(self):
        with self.assertRaisesRegex(ValueError, "only valid for the sglang"):
            live.server_provenance(
                self.write("vllm.json", {**server_info(), "cache_scope": "scope"})
            )

    def test_sglang_config_requests_exact_ids_and_cache_sources(self):
        info = {
            **server_info(),
            "backend": "sglang",
            "server_revision": SGLANG_REVISION,
            "cache_scope": "test:model:no-adapter:campaign",
            "enabled_capabilities": [
                "openai_chat",
                "sglext_token_ids",
                "cache_source_report",
            ],
        }
        provenance = live.server_provenance(self.write("sglang.json", info))
        args = types.SimpleNamespace(
            arm="stock",
            steps=2,
            task_seconds=10,
            model="openai/test",
            endpoint="http://localhost:30000/v1",
            seed=0,
            max_tokens=32,
            request_seconds=5,
        )
        checkout = self.root / "mini"
        template = checkout / "src/minisweagent/config/benchmarks"
        template.mkdir(parents=True)
        (template / "swebench.yaml").write_text(
            "agent:\n  instance_template: task\n"
            "model:\n  observation_template: result\n  model_kwargs: {}\n"
        )
        config = live.make_config(args, checkout, provenance)
        extra = config["model"]["model_kwargs"]["extra_body"]
        self.assertTrue(extra["return_input_ids_in_sglext"])
        self.assertTrue(extra["return_output_ids_in_sglext"])
        self.assertTrue(extra["return_cached_tokens_details"])
        self.assertNotIn("return_token_ids", extra)

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
        self.assertIsNone(frozen["limit"])
        self.assertEqual(frozen["requested_task_count"], 2)
        self.assertEqual(frozen["effective_task_count"], 2)
        (self.root / "frozen.json").unlink()
        with patch.dict(
            "sys.modules", {"datasets": types.SimpleNamespace(load_dataset=loader)}
        ), self.assertRaises(ValueError):
            live.select(self.select_args("--ids", "repo__missing-3", "--execute"))
        self.assertFalse((self.root / "frozen.json").exists())

    def task(
        self,
        *,
        fail=False,
        cleanup_fail=False,
        finalize_fail=False,
        environment_fatal=False,
        terminal_capture_failure=None,
        exit_status="Submitted",
        submission="diff --git a/f b/f",
        head_diff="diff --git a/source.py b/source.py\n",
        terminal_export_protocol=None,
        validation_returncodes=None,
        validation_responses=None,
        base_response=None,
    ):
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
        terminal_outputs = {
            "git status --short --untracked-files=all": " M source.py\n?? scratch.py\n",
            "git diff --name-only -- .": "source.py\n",
            "git ls-files --others --exclude-standard": "scratch.py\n",
            "git diff --no-ext-diff --binary -- .": (
                "diff --git a/source.py b/source.py\n"
            ),
            "git diff --cached --name-only -- .": "staged.py\n",
            "git diff --cached --no-ext-diff --binary -- .": (
                "diff --git a/staged.py b/staged.py\n"
            ),
            "git diff HEAD --no-ext-diff --binary -- .": head_diff,
            "git rev-parse --verify HEAD^{commit}": f"{'c' * 40}\n",
            f"git diff {'b' * 40} --no-ext-diff --binary -- .": head_diff,
        }
        validation_returncodes = list(validation_returncodes or [])
        validation_responses = list(validation_responses or [])
        if base_response is None:
            base_response = {
                "output": f"{'b' * 40}\n",
                "returncode": 0,
                "exception_info": "",
            }

        def execute(action, **kwargs):
            if action["command"] == f"git rev-parse --verify {'b' * 40}^{{commit}}":
                if isinstance(base_response, BaseException):
                    raise base_response
                return base_response
            if action["command"].startswith("terminal_dir="):
                if validation_responses:
                    response = validation_responses.pop(0)
                    if isinstance(response, BaseException):
                        raise response
                    return response
                returncode = validation_returncodes.pop(0)
                return {
                    "output": (
                        "error: patch failed: source.py:1\n"
                        "error: source.py: patch does not apply\n"
                        if returncode == 1
                        else ""
                    ),
                    "returncode": returncode,
                    "exception_info": "",
                }
            result = {
                "output": terminal_outputs[action["command"]],
                "returncode": 0,
                "exception_info": "",
            }
            if action["command"] == live.TERMINAL_CAPTURE_COMMANDS["status"]:
                if terminal_capture_failure == "nonzero":
                    result["returncode"] = 128
                elif terminal_capture_failure == "timeout":
                    result.update(returncode=-1, exception_info="timeout exception")
                elif terminal_capture_failure == "missing":
                    result.pop("returncode")
            return result

        environment.execute.side_effect = execute
        if cleanup_fail:
            environment.cleanup.side_effect = OSError("private error details")
        received = {}

        class Agent:
            def __init__(self, model, env, **kwargs):
                self.model = model
                self.messages = []
                received["config"] = kwargs

            def run(self, task):
                received["task"] = task
                self.model._query([{"role": "user", "content": task}])
                self.messages.extend(
                    [
                        {
                            "role": "assistant",
                            "extra": {
                                "actions": [
                                    {
                                        "command": "python -m pytest -q",
                                        "tool_call_id": "verification-call",
                                    }
                                ]
                            },
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "verification-call",
                            "content": "<returncode>0</returncode>\n1 passed",
                        },
                    ]
                )
                if fail:
                    raise RuntimeError("private exception details")
                if environment_fatal:
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": (
                                "<returncode>128</returncode>\n"
                                "OCI runtime exec failed: write /tmp/runc-process1: "
                                "no space left on device"
                            ),
                        }
                    )
                return {"exit_status": exit_status, "submission": submission}

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
                row(),
                config,
                self.root,
                model_factory,
                environment_factory,
                Agent,
                terminal_export_protocol=terminal_export_protocol,
            )
        self.assertEqual(config, original)
        environment.cleanup.assert_called_once()
        if received:
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
        capture = json.loads(
            (self.root / "terminal_captures/repo__issue-1.json").read_text()
        )
        self.assertTrue(capture["submitted"])
        self.assertEqual(capture["commands"]["changed_paths"]["output"], "source.py\n")
        self.assertEqual(capture["capture_status"], "complete")
        self.assertEqual(capture["failures"], [])
        self.assertEqual(capture["terminal_head"], "c" * 40)
        self.assertEqual(capture["commands"]["staged_paths"]["output"], "staged.py\n")
        self.assertEqual(
            capture["command_scopes"]["source_diff"], "unstaged tracked changes"
        )
        self.assertEqual(
            capture["last_verification_result"]["command"], "python -m pytest -q"
        )

    def test_nonzero_terminal_capture_is_infrastructure_failure(self):
        result = self.task(terminal_capture_failure="nonzero")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["model_patch"], "diff --git a/f b/f")
        self.assertIsNone(result["error_type"])
        self.assertEqual(result["terminal_capture_error_type"], "TerminalCaptureError")
        capture = json.loads(
            (self.root / "terminal_captures/repo__issue-1.json").read_text()
        )
        self.assertEqual(capture["capture_status"], "failed")
        self.assertEqual(capture["commands"]["status"]["returncode"], 128)
        self.assertIn("nonzero_returncode", capture["failures"][0]["reasons"])

    def test_timeout_terminal_capture_is_infrastructure_failure(self):
        result = self.task(terminal_capture_failure="timeout")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["model_patch"], "diff --git a/f b/f")
        self.assertIsNone(result["error_type"])
        self.assertEqual(result["terminal_capture_error_type"], "TerminalCaptureError")
        capture = json.loads(
            (self.root / "terminal_captures/repo__issue-1.json").read_text()
        )
        self.assertEqual(capture["capture_status"], "failed")
        self.assertEqual(capture["commands"]["status"]["returncode"], -1)
        self.assertIn("exception_info_present", capture["failures"][0]["reasons"])

    def test_missing_terminal_capture_status_is_infrastructure_failure(self):
        result = self.task(terminal_capture_failure="missing")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["model_patch"], "diff --git a/f b/f")
        self.assertEqual(result["terminal_capture_error_type"], "TerminalCaptureError")
        capture = json.loads(
            (self.root / "terminal_captures/repo__issue-1.json").read_text()
        )
        self.assertEqual(capture["capture_status"], "failed")
        self.assertIsNone(capture["commands"]["status"]["returncode"])
        self.assertIn("missing_returncode", capture["failures"][0]["reasons"])

    def test_terminal_export_preserves_valid_explicit_submission(self):
        result = self.task(
            terminal_export_protocol="terminal-export-v1",
            validation_returncodes=[0],
        )
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["model_patch"], "diff --git a/f b/f")
        self.assertEqual(result["terminal_export_decision"], "preserve_explicit")
        receipt = json.loads(
            (self.root / "terminal_exports/repo__issue-1.json").read_text()
        )
        self.assertFalse(receipt["exported"])
        self.assertEqual(receipt["original_exit_status"], "Submitted")
        self.assertEqual(receipt["task_base"]["resolved_base"], "b" * 40)
        self.assertEqual(receipt["terminal_head"], "c" * 40)
        self.assertEqual(receipt["model_calls"], 1)

    def test_terminal_export_recovers_final_diff_after_limit(self):
        final_diff = "diff --git a/source.py b/source.py\n+fixed\n"
        result = self.task(
            exit_status="LimitsExceeded",
            submission="",
            head_diff=final_diff,
            terminal_export_protocol="terminal-export-v1",
            validation_returncodes=[0],
        )
        self.assertEqual(result["exit_status"], "LimitsExceeded")
        self.assertEqual(result["model_patch"], final_diff)
        self.assertEqual(result["terminal_export_decision"], "export_final_tracked")

    def test_terminal_export_replaces_malformed_submission_not_valid_one(self):
        final_diff = "diff --git a/source.py b/source.py\n+fixed\n"
        result = self.task(
            submission="bare source text",
            head_diff=final_diff,
            terminal_export_protocol="terminal-export-v1",
            validation_returncodes=[0],
        )
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["model_patch"], final_diff)
        receipt = json.loads(
            (self.root / "terminal_exports/repo__issue-1.json").read_text()
        )
        self.assertEqual(receipt["original_submission"], "bare source text")
        self.assertEqual(receipt["decision"], "export_final_tracked")

    def test_terminal_export_rejects_empty_and_invalid_final_diffs(self):
        empty = self.task(
            exit_status="LimitsExceeded",
            submission="",
            head_diff="",
            terminal_export_protocol="terminal-export-v1",
        )
        self.assertEqual(empty["model_patch"], "")
        self.assertEqual(empty["terminal_export_decision"], "no_candidate")
        self.root = self.root / "invalid"
        self.root.mkdir()
        invalid = self.task(
            exit_status="TimeExceeded",
            submission="",
            head_diff="diff --git a/source.py b/source.py\n+bad\n",
            terminal_export_protocol="terminal-export-v1",
            validation_returncodes=[1],
        )
        self.assertEqual(invalid["exit_status"], "TimeExceeded")
        self.assertEqual(invalid["model_patch"], "")
        self.assertEqual(invalid["terminal_export_decision"], "no_candidate")

    def test_terminal_export_uses_combined_head_diff_and_excludes_untracked(self):
        combined = (
            "diff --git a/staged.py b/staged.py\n+staged\n"
            "diff --git a/source.py b/source.py\n+unstaged\n"
        )
        result = self.task(
            exit_status="LimitsExceeded",
            submission="",
            head_diff=combined,
            terminal_export_protocol="terminal-export-v1",
            validation_returncodes=[0],
        )
        self.assertEqual(result["model_patch"], combined)
        self.assertNotIn("scratch.py", result["model_patch"])

    def test_terminal_export_does_not_export_capture_failure(self):
        result = self.task(
            exit_status="LimitsExceeded",
            submission="",
            terminal_capture_failure="nonzero",
            terminal_export_protocol="terminal-export-v1",
        )
        self.assertEqual(result["model_patch"], "")
        self.assertEqual(result["terminal_export_decision"], "unavailable")
        self.assertEqual(result["terminal_capture_error_type"], "TerminalCaptureError")

    def test_terminal_export_rejects_nonexportable_terminal_reason(self):
        result = self.task(
            exit_status="RepeatedFormatError",
            submission="",
            terminal_export_protocol="terminal-export-v1",
        )
        self.assertEqual(result["exit_status"], "RepeatedFormatError")
        self.assertEqual(result["model_patch"], "")
        self.assertEqual(result["terminal_export_decision"], "not_exportable")

    def test_terminal_export_preserves_explicit_on_validation_timeout(self):
        original = "diff --git a/source.py b/source.py\n+explicit\n"
        result = self.task(
            submission=original,
            head_diff="diff --git a/source.py b/source.py\n+different\n",
            terminal_export_protocol="terminal-export-v1",
            validation_responses=[TimeoutError("validation timed out")],
        )
        self.assertEqual(result["model_patch"], original)
        self.assertEqual(result["terminal_export_decision"], "unavailable")
        receipt = json.loads(
            (self.root / "terminal_exports/repo__issue-1.json").read_text()
        )
        self.assertEqual(receipt["explicit_validation"]["state"], "unavailable")
        self.assertNotIn("final_validation", receipt)

    def test_terminal_export_reports_unavailable_final_validation(self):
        original = "not a Git diff"
        result = self.task(
            exit_status="Submitted",
            submission=original,
            head_diff="diff --git a/source.py b/source.py\n+candidate\n",
            terminal_export_protocol="terminal-export-v1",
            validation_responses=[TimeoutError("validation timed out")],
        )
        self.assertEqual(result["model_patch"], original)
        self.assertEqual(result["terminal_export_decision"], "unavailable")
        receipt = json.loads(
            (self.root / "terminal_exports/repo__issue-1.json").read_text()
        )
        self.assertEqual(receipt["final_validation"]["state"], "unavailable")

    def test_terminal_export_requires_verified_base_before_agent(self):
        result = self.task(
            exit_status="LimitsExceeded",
            submission="",
            terminal_export_protocol="terminal-export-v1",
            base_response={
                "returncode": 128,
                "output": "unknown revision",
                "exception_info": "",
            },
        )
        self.assertEqual(result["error_type"], "TerminalBaseError")
        self.assertEqual(result["model_patch"], "")
        self.assertEqual(result["terminal_export_decision"], "unavailable")
        receipt = json.loads(
            (self.root / "terminal_exports/repo__issue-1.json").read_text()
        )
        self.assertEqual(receipt["task_base"]["state"], "unavailable")
        self.assertEqual(receipt["model_calls"], 0)

    def test_terminal_validation_has_three_states(self):
        base = {"state": "valid", "resolved_base": "b" * 40}

        class Environment:
            def __init__(self, response):
                self.response = response

            def execute(self, action, **kwargs):
                if isinstance(self.response, BaseException):
                    raise self.response
                return self.response

        candidate = "diff --git a/source.py b/source.py\n+candidate\n"
        cases = [
            (
                {"returncode": 0, "output": "", "exception_info": ""},
                "valid",
                "valid",
            ),
            (
                {
                    "returncode": 1,
                    "output": (
                        "error: patch failed: source.py:1\n"
                        "error: source.py: patch does not apply\n"
                    ),
                    "exception_info": "",
                },
                "invalid",
                "patch_content_rejected",
            ),
            (
                {
                    "returncode": 128,
                    "output": (
                        "error: No valid patches in input "
                        '(allow with "--allow-empty")\n'
                    ),
                    "exception_info": "",
                },
                "invalid",
                "patch_content_rejected",
            ),
            (
                {
                    "returncode": 128,
                    "output": "fatal: cannot create temporary file\n",
                    "exception_info": "",
                },
                "unavailable",
                "ambiguous_patch_check_failure",
            ),
            (
                {"returncode": 200, "output": "", "exception_info": ""},
                "unavailable",
                "clean_index_setup_failed",
            ),
            (
                {
                    "returncode": 201,
                    "output": "missing blob deadbeef\n",
                    "exception_info": "",
                },
                "unavailable",
                "base_objects_unavailable",
            ),
            (
                {"returncode": 126, "output": "", "exception_info": ""},
                "unavailable",
                "validation_returncode_unavailable",
            ),
            (
                {"returncode": True, "output": "", "exception_info": ""},
                "unavailable",
                "validation_response_unavailable",
            ),
            (
                {"returncode": 0, "exception_info": ""},
                "unavailable",
                "validation_response_unavailable",
            ),
            (
                {"returncode": 0, "output": None, "exception_info": ""},
                "unavailable",
                "validation_response_unavailable",
            ),
            (
                {"returncode": 0, "output": "", "exception_info": "timeout"},
                "unavailable",
                "validation_response_unavailable",
            ),
            (
                TimeoutError("timeout"),
                "unavailable",
                "validation_exception",
            ),
        ]
        for response, state, reason in cases:
            with self.subTest(response=response):
                receipt = live._validate_terminal_patch(
                    Environment(response), candidate, base
                )
                self.assertEqual(receipt["state"], state)
                self.assertEqual(receipt["reason"], reason)

    def test_terminal_export_unavailable_is_infrastructure_failure(self):
        self.assertTrue(
            live._task_infrastructure_failed(
                {
                    "error_type": None,
                    "terminal_export_unavailable": True,
                }
            )
        )
        self.assertFalse(
            live._task_infrastructure_failed(
                {
                    "error_type": None,
                    "terminal_export_unavailable": False,
                }
            )
        )

    def test_terminal_export_uses_final_diff_only_after_proven_invalid(self):
        final_diff = "diff --git a/source.py b/source.py\n+fixed\n"
        result = self.task(
            submission="diff --git a/missing b/missing\n+bad\n",
            head_diff=final_diff,
            terminal_export_protocol="terminal-export-v1",
            validation_returncodes=[1, 0],
        )
        self.assertEqual(result["model_patch"], final_diff)
        self.assertEqual(result["terminal_export_decision"], "export_final_tracked")
        receipt = json.loads(
            (self.root / "terminal_exports/repo__issue-1.json").read_text()
        )
        self.assertEqual(receipt["explicit_validation"]["state"], "invalid")
        self.assertEqual(receipt["final_validation"]["state"], "valid")

    def test_terminal_patch_validation_uses_clean_head_without_mutation(self):
        repository = self.root / "repo"
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=repository, check=True
        )
        source = repository / "source.py"
        source.write_text("old = 1\n")
        subprocess.run(["git", "add", "source.py"], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "base"], cwd=repository, check=True
        )
        source.write_text("new = 2\n")
        candidate = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--", "."],
            cwd=repository,
            text=True,
        )

        class LocalEnvironment:
            def execute(self, action, **kwargs):
                completed = subprocess.run(
                    action["command"],
                    cwd=repository,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                return {
                    "returncode": completed.returncode,
                    "output": completed.stdout,
                    "exception_info": "",
                }

        base = {
            "state": "valid",
            "resolved_base": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip(),
        }
        receipt = live._validate_terminal_patch(LocalEnvironment(), candidate, base)
        self.assertEqual(receipt["state"], "valid")
        self.assertEqual(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=repository, text=True
            ),
            " M source.py\n",
        )
        invalid = candidate.replace("old = 1", "not present")
        self.assertEqual(
            live._validate_terminal_patch(LocalEnvironment(), invalid, base)["state"],
            "invalid",
        )

    def test_runner_preserves_explicit_patch_when_base_blob_is_missing(self):
        repository = self.root / "missing-object-repo"
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repository,
            check=True,
        )
        (repository / "a").write_text("a0\n")
        (repository / "b").write_text("b0\n")
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "base"],
            cwd=repository,
            check=True,
        )
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        missing_blob = subprocess.check_output(
            ["git", "rev-parse", "HEAD:a"], cwd=repository, text=True
        ).strip()
        blob_path = repository / ".git/objects" / missing_blob[:2] / missing_blob[2:]
        blob_contents = blob_path.read_bytes()

        class LocalEnvironment:
            def execute(self, action, **kwargs):
                if action["command"].startswith("terminal_dir="):
                    blob_path.unlink(missing_ok=True)
                completed = subprocess.run(
                    action["command"],
                    cwd=repository,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                return {
                    "returncode": completed.returncode,
                    "output": completed.stdout,
                    "exception_info": "",
                }

            def cleanup(self):
                if not blob_path.exists():
                    blob_path.parent.mkdir(parents=True, exist_ok=True)
                    blob_path.write_bytes(blob_contents)

        explicit = {}

        class Agent:
            def __init__(self, model, env, **kwargs):
                self.messages = []

            def run(self, task):
                (repository / "a").write_text("a1\n")
                explicit["patch"] = subprocess.check_output(
                    ["git", "diff", "HEAD", "--binary", "--", "a"],
                    cwd=repository,
                    text=True,
                )
                (repository / "a").write_text("a0\n")
                (repository / "b").write_text("b1\n")
                return {"exit_status": "Submitted", "submission": explicit["patch"]}

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_kwargs={}),
            abort_exceptions=[],
            _query=Mock(),
        )
        result = live.run_task(
            {**row(), "base_commit": base},
            {"model": {"model_name": "openai/test"}, "agent": {}},
            self.root / "missing-object-output",
            Mock(return_value=model),
            Mock(return_value=LocalEnvironment()),
            Agent,
            terminal_export_protocol="terminal-export-v1",
            dataset={"dataset": "test/dataset", "dataset_revision": "a" * 40},
        )
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["model_patch"], explicit["patch"])
        self.assertEqual(result["terminal_export_decision"], "unavailable")
        self.assertEqual(
            result["terminal_export_reason"], "explicit_validation_unavailable"
        )
        self.assertEqual(result["terminal_export_error"], "base_objects_unavailable")
        self.assertTrue(result["terminal_export_unavailable"])
        self.assertTrue(live._task_infrastructure_failed(result))
        receipt = json.loads(
            (
                self.root / "missing-object-output/terminal_exports/repo__issue-1.json"
            ).read_text()
        )
        self.assertEqual(receipt["explicit_validation"]["state"], "unavailable")
        self.assertEqual(
            receipt["explicit_validation"]["reason"], "base_objects_unavailable"
        )
        self.assertEqual(receipt["explicit_validation"]["base_apply_returncode"], 201)
        self.assertNotIn("final_validation", receipt)
        self.assertTrue(blob_path.exists())

    def test_runner_exports_frozen_base_after_committed_and_dirty_edits(self):
        repository = self.root / "runner-repo"
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repository,
            check=True,
        )
        (repository / "source.py").write_text("value = 1\n")
        (repository / "committed.py").write_text("value = 1\n")
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "base"],
            cwd=repository,
            check=True,
        )
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()

        class LocalEnvironment:
            cleanup_status = None

            def execute(self, action, **kwargs):
                completed = subprocess.run(
                    action["command"],
                    cwd=repository,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                return {
                    "returncode": completed.returncode,
                    "output": completed.stdout,
                    "exception_info": "",
                }

            def cleanup(self):
                self.cleanup_status = subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=all"],
                    cwd=repository,
                    text=True,
                )

        environment = LocalEnvironment()
        status_before_return = {}

        class Agent:
            def __init__(self, model, env, **kwargs):
                self.messages = []

            def run(self, task):
                (repository / "source.py").write_text("value = 2\n")
                (repository / "committed.py").write_text("value = 2\n")
                subprocess.run(["git", "add", "."], cwd=repository, check=True)
                subprocess.run(
                    ["git", "commit", "-q", "-m", "agent commit"],
                    cwd=repository,
                    check=True,
                )
                (repository / "source.py").write_text("value = 3\n")
                subprocess.run(["git", "add", "source.py"], cwd=repository, check=True)
                (repository / "source.py").write_text("value = 4\n")
                (repository / "helper.py").write_text("untracked = True\n")
                status_before_return["value"] = subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=all"],
                    cwd=repository,
                    text=True,
                )
                return {"exit_status": "LimitsExceeded", "submission": ""}

        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_kwargs={}),
            abort_exceptions=[],
            _query=Mock(),
        )
        result = live.run_task(
            {**row(), "base_commit": base},
            {"model": {"model_name": "openai/test"}, "agent": {}},
            self.root / "runner-output",
            Mock(return_value=model),
            Mock(return_value=environment),
            Agent,
            terminal_export_protocol="terminal-export-v1",
            dataset={"dataset": "test/dataset", "dataset_revision": "a" * 40},
        )
        self.assertEqual(result["exit_status"], "LimitsExceeded")
        self.assertEqual(result["terminal_export_decision"], "export_final_tracked")
        self.assertIn("+value = 4", result["model_patch"])
        self.assertIn("diff --git a/committed.py b/committed.py", result["model_patch"])
        self.assertNotIn("helper.py", result["model_patch"])
        self.assertEqual(environment.cleanup_status, status_before_return["value"])

        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "index"
            index_environment = {**os.environ, "GIT_INDEX_FILE": str(index)}
            subprocess.run(
                ["git", "read-tree", base],
                cwd=repository,
                env=index_environment,
                check=True,
            )
            subprocess.run(
                ["git", "apply", "--cached", "--check", "--whitespace=nowarn", "-"],
                cwd=repository,
                env=index_environment,
                input=result["model_patch"],
                text=True,
                check=True,
            )

        receipt = json.loads(
            (
                self.root / "runner-output/terminal_exports/repo__issue-1.json"
            ).read_text()
        )
        self.assertEqual(receipt["task_base"]["resolved_base"], base)
        self.assertNotEqual(receipt["terminal_head"], base)
        self.assertEqual(receipt["original_exit_status"], "LimitsExceeded")
        self.assertEqual(receipt["model_calls"], 0)
        self.assertEqual(receipt["dataset"]["dataset"], "test/dataset")

    def test_task_exception_cleans_up_and_records_failure(self):
        result = self.task(fail=True)
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertEqual(result["exit_status"], "InfrastructureError")
        self.assertNotIn("private", json.dumps(result))
        task = json.loads((self.root / "tasks/repo__issue-1.json").read_text())
        self.assertTrue(task["recording_complete"])

    def test_fatal_container_observation_invalidates_task(self):
        result = self.task(environment_fatal=True)
        self.assertEqual(result["exit_status"], "InfrastructureError")
        self.assertEqual(result["error_type"], "FatalEnvironmentError")
        self.assertEqual(result["infrastructure_reason"], "container_runtime_no_space")
        self.assertEqual(result["model_patch"], "")

    def test_normal_command_failure_is_not_infrastructure_failure(self):
        self.assertIsNone(
            live.fatal_environment_error(
                [{"role": "tool", "content": "<returncode>1</returncode>\nfailed"}]
            )
        )

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
        arguments.arm = "clear"
        clear = live.make_config(arguments, checkout)
        arguments.arm = "clear-stable-prefix"
        clear_stable = live.make_config(arguments, checkout)
        clear_template = clear["agent"]["instance_template"]
        clear_stable_template = clear_stable["agent"]["instance_template"]
        self.assertLess(
            clear_template.index("<pr_description>"),
            clear_template.index("<instructions>"),
        )
        self.assertLess(
            clear_stable_template.index("<instructions>"),
            clear_stable_template.index("<pr_description>"),
        )
        for template in (clear_template, clear_stable_template):
            self.assertEqual(template.count("{{task}}"), 1)
            self.assertIn(
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt",
                template,
            )
        self.assertEqual(clear["model"], clear_stable["model"])
        arguments.arm = "stable-prefix"
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
