# SPDX-License-Identifier: MIT
"""CPU tests for exact capture, task isolation, and mini's transport hook."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.agent_baselines.recording_model import (
    TokenCaptureError,
    attach_recorder,
    extract_token_ids,
    summarize_calls,
)


def response(prompt=None, output=None, *, tool=True):
    prompt = [1, 2, 3] if prompt is None else prompt
    output = [4, 5] if output is None else output
    message = {"role": "assistant", "content": "I will inspect the files."}
    if tool:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ]
    return {
        "id": "chatcmpl-local-test",
        "object": "chat.completion",
        "created": 1,
        "model": "local-test-model",
        "prompt_token_ids": prompt,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
                "token_ids": output,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(output),
            "total_tokens": len(prompt) + len(output),
        },
    }


class TestRecording(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def model(self, transport):
        return SimpleNamespace(
            _query=transport,
            config=SimpleNamespace(model_kwargs={"api_key": "secret-never-log"}),
            abort_exceptions=[],
        )

    def rows(self, recorder):
        return [json.loads(line) for line in recorder.log_path.read_text().splitlines()]

    def test_missing_or_invalid_ids_fail_closed(self):
        cases = []
        missing_prompt = response()
        del missing_prompt["prompt_token_ids"]
        cases.append(missing_prompt)
        missing_output = response()
        del missing_output["choices"][0]["token_ids"]
        cases.append(missing_output)
        mismatched = response()
        mismatched["usage"]["prompt_tokens"] += 1
        cases.append(mismatched)
        cases.extend(
            [response(prompt=[True]), response(output=[]), response(output=[-1])]
        )
        for index, data in enumerate(cases):
            with self.subTest(index=index):
                model = self.model(lambda messages, **kwargs: data)
                recorder = attach_recorder(model, self.root, f"task-{index}")
                with self.assertRaises(TokenCaptureError):
                    model._query([])
                recorder.finish()
                self.assertIs(model.abort_exceptions[-1], TokenCaptureError)
                row = self.rows(recorder)[0]
                self.assertFalse(row["token_identity_verified"])
                self.assertEqual(row["error"], "TokenCaptureError")
                self.assertNotIn("generated_token_ids", row)

    def test_errors_preserve_retry_timing_without_secrets(self):
        count = 0
        sent = []

        def transport(messages, **kwargs):
            nonlocal count
            count += 1
            sent.append(kwargs)
            if count == 1:
                raise TimeoutError("secret-never-log bearer credential")
            return response()

        model = self.model(transport)
        model.config.model_kwargs["extra_body"] = {"chat_template_kwargs": {"x": 1}}
        recorder = attach_recorder(model, self.root, "task", trial_id="trial")
        with self.assertRaises(TimeoutError):
            model._query([])
        result = model._query([], extra_body={"seed": 9})
        recorder.finish(exit_status="Submitted")
        self.assertEqual(extract_token_ids(result), ([1, 2, 3], [4, 5]))
        self.assertEqual(
            sent[-1]["extra_body"],
            {"chat_template_kwargs": {"x": 1}, "seed": 9, "return_token_ids": True},
        )
        rows = self.rows(recorder)
        self.assertEqual([row["request_sequence"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["error"], "TimeoutError")
        self.assertLessEqual(rows[0]["end_wall_ns"], rows[1]["start_wall_ns"])
        self.assertGreater(rows[1]["client_elapsed_ns"], 0)
        self.assertNotIn("secret-never-log", recorder.log_path.read_text())
        self.assertNotIn("api_key", recorder.task_path.read_text())
        task = json.loads(recorder.task_path.read_text())
        self.assertTrue(task["recording_complete"])
        self.assertEqual(task["verified_calls"], 1)
        self.assertEqual(task["calls_total"], 2)

    def test_cached_count_is_observed_only_when_present_and_valid(self):
        for index, cached in enumerate((2, None, -1, 4, True)):
            data = response()
            data["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
            model = self.model(lambda messages, **kwargs: data)
            recorder = attach_recorder(model, self.root, f"task-{index}")
            model._query([])
            recorder.finish()
            usage = self.rows(recorder)[0]["usage"]
            self.assertEqual("prompt_tokens_details" in usage, index == 0)
        self.assertEqual(usage["prompt_tokens"], 3)

    def test_litellm_provider_ids_must_not_conflict(self):
        data = response()
        data["choices"][0]["provider_specific_fields"] = {"token_ids": [99]}
        with self.assertRaises(TokenCaptureError):
            extract_token_ids(data)

    def test_two_task_models_remain_isolated(self):
        barrier = threading.Barrier(2)
        recorders = []
        models = []
        for index in range(2):

            def transport(messages, *, token=index, **kwargs):
                barrier.wait(timeout=5)
                return response(prompt=[token + 10], output=[token + 20])

            model = self.model(transport)
            models.append(model)
            recorders.append(attach_recorder(model, self.root, f"task-{index}"))
        errors = []

        def run(model):
            try:
                model._query([])
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=run, args=(model,)) for model in models]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        for index, recorder in enumerate(recorders):
            recorder.finish()
            row = self.rows(recorder)[0]
            self.assertEqual(row["prompt_token_ids"], [index + 10])
            self.assertEqual(row["trial_id"], recorder.trial_id)

    def test_reuse_and_unsafe_task_ids_are_rejected(self):
        model = self.model(lambda messages, **kwargs: response())
        recorder = attach_recorder(model, self.root, "task")
        with self.assertRaises(ValueError):
            attach_recorder(model, self.root, "other")
        with self.assertRaises(FileExistsError):
            attach_recorder(self.model(model._query), self.root, "task")
        with self.assertRaises(ValueError):
            attach_recorder(self.model(model._query), self.root, "../outside")
        recorder.finish()
        with self.assertRaises(RuntimeError):
            model._query([])

    def test_streaming_and_multiple_choices_are_rejected(self):
        for index, settings in enumerate(({"stream": True}, {"n": 2})):
            model = self.model(lambda messages, **kwargs: self.fail("transport used"))
            model.config.model_kwargs.update(settings)
            recorder = attach_recorder(model, self.root, f"task-{index}")
            with self.assertRaises(TokenCaptureError):
                model._query([])
            recorder.finish()

    def test_summary_separates_attempts_and_preserves_zero_cached_tokens(self):
        model = self.model(lambda messages, **kwargs: response())
        recorder = attach_recorder(model, self.root, "summary")
        model._query([])
        model._query([])
        recorder.finish()
        rows = self.rows(recorder)
        for index, row in enumerate(rows):
            row["client_elapsed_ns"] = (index + 1) * 1_000_000_000
            row["usage"]["prompt_tokens_details"] = {"cached_tokens": 0}
        rows.extend(
            [
                {"error": "TimeoutError", "client_elapsed_ns": 99_000_000_000},
                {"error": "TokenCaptureError"},
            ]
        )
        recorder.log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = summarize_calls(self.root)
        self.assertEqual(summary["total_attempts"], 4)
        self.assertEqual(summary["successful_captured_calls"], 2)
        self.assertEqual(summary["transport_failures"], 1)
        self.assertEqual(summary["capture_errors"], 1)
        self.assertEqual(summary["prompt_tokens"], 6)
        self.assertEqual(summary["completion_tokens"], 4)
        self.assertEqual(summary["reported_cached_tokens"], 0)
        self.assertEqual(summary["cached_token_reporting_calls"], 2)
        self.assertEqual(summary["latency_measurement"], "full_nonstreaming_call")
        self.assertEqual(summary["client_latency_s"]["mean"], 1.5)
        self.assertEqual(summary["client_latency_s"]["p50"], 1.5)
        self.assertAlmostEqual(summary["client_latency_s"]["p95"], 1.95)
        del rows[1]["usage"]["prompt_tokens_details"]
        rows[0]["usage"]["prompt_tokens_details"]["cached_tokens"] = 2
        recorder.log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary = summarize_calls(self.root)
        self.assertIsNone(summary["reported_cached_tokens"])
        self.assertEqual(summary["cached_token_reporting_calls"], 1)

    def test_empty_summary_has_unknown_cache_counts_and_latency(self):
        summary = summarize_calls(self.root)
        self.assertEqual(summary["total_attempts"], 0)
        self.assertIsNone(summary["reported_cached_tokens"])
        self.assertEqual(summary["client_latency_s"]["count"], 0)
        self.assertIsNone(summary["client_latency_s"]["p50"])
        self.assertIsNone(summary["client_latency_s"]["p95"])


_HAS_MINI = importlib.util.find_spec("minisweagent") is not None
_HAS_LITELLM = importlib.util.find_spec("litellm") is not None


@unittest.skipUnless(
    _HAS_MINI and _HAS_LITELLM, "install pinned mini for API contract tests"
)
class TestMiniContract(unittest.TestCase):
    setUp = TestRecording.setUp
    rows = TestRecording.rows

    def _exercise_http_transport(self, *, fail_first):
        # Exercise the real model and LiteLLM conversion against an HTTP fixture.
        # No model/server is needed, and the fixture never calls an external API.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import httpx
        import minisweagent
        import openai
        from minisweagent.models.litellm_model import LitellmModel
        from research.agent_baselines import live

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                status = 500 if fail_first and len(requests) == 1 else 200
                body = (
                    {"error": {"message": "fixture failure", "type": "server_error"}}
                    if status == 500
                    else response()
                )
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        http_client = httpx.Client(trust_env=False)
        self.addCleanup(http_client.close)
        client = openai.OpenAI(
            api_key="local-test-only",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            http_client=http_client,
        )
        checkout = Path(minisweagent.__file__).resolve().parents[2]
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
                "openai/local-test-model",
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--out",
                "unused-output",
            ]
        )
        model_config = live.make_config(arguments, checkout)["model"]
        model_config.pop("model_class")
        model_config["model_kwargs"]["client"] = client
        model = LitellmModel(**model_config)
        recorder = attach_recorder(model, self.root, "real-mini")
        result = model.query([{"role": "user", "content": "test"}])
        recorder.finish()
        self.assertEqual(result["extra"]["actions"][0]["command"], "ls")
        self.assertTrue(requests[0]["return_token_ids"])
        self.assertEqual(self.rows(recorder)[-1]["generated_token_ids"], [[4, 5]])
        return requests, self.rows(recorder)

    def test_actual_mini_transport_and_litellm_preserve_ids(self):
        requests, rows = self._exercise_http_transport(fail_first=False)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["error"])

    def test_production_config_records_each_http_retry(self):
        requests, rows = self._exercise_http_transport(fail_first=True)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(rows), 2)
        self.assertIsNotNone(rows[0]["error"])
        self.assertIsNone(rows[1]["error"])
        self.assertFalse(rows[0]["token_identity_verified"])
        self.assertTrue(rows[1]["token_identity_verified"])
        self.assertEqual([row["request_sequence"] for row in rows], [1, 2])

    def test_format_error_generation_is_captured_before_parsing(self):
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm
        from minisweagent.exceptions import FormatError
        from minisweagent.models.litellm_model import LitellmModel

        model = LitellmModel(
            model_name="openai/local-test-model", cost_tracking="ignore_errors"
        )
        recorder = attach_recorder(model, self.root, "bad-format")
        with patch.object(
            litellm,
            "completion",
            return_value=litellm.ModelResponse(**response(tool=False)),
        ):
            with self.assertRaises(FormatError):
                model.query([{"role": "user", "content": "test"}])
        recorder.finish()
        row = self.rows(recorder)[0]
        self.assertIsNone(row["error"])
        self.assertTrue(row["token_identity_verified"])

    def test_mini_retries_transport_error_but_aborts_missing_ids(self):
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm
        from minisweagent.models.litellm_model import LitellmModel

        model = LitellmModel(
            model_name="openai/local-test-model", cost_tracking="ignore_errors"
        )
        recorder = attach_recorder(model, self.root, "retry")
        with patch.object(
            litellm,
            "completion",
            side_effect=[
                TimeoutError("fixture timeout"),
                litellm.ModelResponse(**response()),
            ],
        ) as completion:
            model.query([{"role": "user", "content": "test"}])
            self.assertEqual(completion.call_count, 2)
        recorder.finish()
        self.assertEqual(
            [r["error"] for r in self.rows(recorder)], ["TimeoutError", None]
        )

        invalid = response()
        del invalid["prompt_token_ids"]
        model = LitellmModel(
            model_name="openai/local-test-model", cost_tracking="ignore_errors"
        )
        recorder = attach_recorder(model, self.root, "abort")
        with patch.object(
            litellm, "completion", return_value=litellm.ModelResponse(**invalid)
        ) as completion:
            with self.assertRaises(TokenCaptureError):
                model.query([{"role": "user", "content": "test"}])
            self.assertEqual(completion.call_count, 1)
        recorder.finish()


if __name__ == "__main__":
    unittest.main()
