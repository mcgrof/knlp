# SPDX-License-Identifier: MIT
"""Record engine token IDs at mini-SWE-agent's per-attempt transport hook.

Create one model per task, call ``attach_recorder`` before ``agent.run``,
and call ``recorder.finish`` in the task's finally block. The instance hook
leaves mini's retries and parsing intact. Logs contain token IDs and timing,
not configuration, request headers, API keys, or exception messages.

vLLM must return ``prompt_token_ids`` and ``choices[0].token_ids`` when
``return_token_ids`` is requested. Missing IDs abort capture; decoded text
is never retokenized and presented as an exact engine trace. Captured token
IDs still contain the task content, so treat logs as sensitive workload data.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any


class TokenCaptureError(RuntimeError):
    """An API response cannot establish an exact engine token sequence."""


def _mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        result = value.model_dump()
        if isinstance(result, dict):
            return result
    raise TokenCaptureError("The transport did not return a response object")


def _token_ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TokenCaptureError(f"Missing engine {name}")
    if any(type(token) is not int or token < 0 for token in value):
        raise TokenCaptureError(f"Invalid engine {name}")
    return value


def extract_token_ids(response: Any) -> tuple[list[int], list[int]]:
    """Read only server-supplied IDs and check them against API usage."""
    data = _mapping(response)
    prompt = _token_ids(data.get("prompt_token_ids"), "prompt_token_ids")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise TokenCaptureError("Exact capture requires one response choice")
    choice = _mapping(choices[0])
    # LiteLLM relocates unknown choice fields here during API conversion.
    # These are still server-returned IDs, not a text-tokenization fallback.
    extra = choice.get("provider_specific_fields") or {}
    nested_ids = _mapping(extra).get("token_ids")
    direct_ids = choice.get("token_ids")
    if direct_ids is not None and nested_ids is not None and direct_ids != nested_ids:
        raise TokenCaptureError("Conflicting engine completion token IDs")
    output = _token_ids(
        direct_ids if direct_ids is not None else nested_ids, "completion token_ids"
    )
    usage = _mapping(data.get("usage"))
    for key, expected in (
        ("prompt_tokens", len(prompt)),
        ("completion_tokens", len(output)),
    ):
        if type(usage.get(key)) is not int or usage[key] != expected:
            raise TokenCaptureError(f"Engine IDs disagree with {key}")
    return prompt, output


def _usage_record(response: Any, prompt: list[int], output: list[int]) -> dict:
    """Retain only token-count fields, never arbitrary provider metadata."""
    usage = {"prompt_tokens": len(prompt), "completion_tokens": len(output)}
    details = _mapping(_mapping(response)["usage"]).get("prompt_tokens_details")
    if details is not None:
        cached = _mapping(details).get("cached_tokens")
        if type(cached) is int and 0 <= cached <= len(prompt):
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


class CallRecorder:
    """Write one task's exact call records in EfficientAgent input format."""

    def __init__(
        self,
        run_dir: Path | str,
        task_id: str,
        *,
        trial_id: str | None = None,
        assigned_ns: int | None = None,
    ):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id):
            raise ValueError("task_id must be a safe filename component")
        self.task_id = task_id
        self.trial_id = trial_id or uuid.uuid4().hex
        self.assigned_ns = assigned_ns if assigned_ns is not None else time.time_ns()
        self.sequence = 0
        self.verified_calls = 0
        self.capture_errors = 0
        self.finished = False
        self.run_dir = Path(run_dir)
        self.task_path = self.run_dir / "tasks" / f"{task_id}.json"
        self.log_path = (
            self.run_dir
            / "attempts"
            / task_id
            / "attempt_00"
            / "telemetry"
            / "llm_requests.jsonl"
        )
        self.task_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation rejects stale attempts instead of silently appending.
        with self.task_path.open("x", encoding="utf-8") as stream:
            json.dump(self._task_record(), stream)
        with self.log_path.open("x", encoding="utf-8"):
            pass

    def _task_record(self, *, finished_ns=None, exit_status=None) -> dict:
        return {
            "attempt_index": 0,
            "trial_id": self.trial_id,
            "task_timing": {
                "assigned_ns": self.assigned_ns,
                "finished_ns": finished_ns,
            },
            "recording_complete": finished_ns is not None,
            "calls_total": self.sequence,
            "verified_calls": self.verified_calls,
            "capture_errors": self.capture_errors,
            "exit_status": exit_status,
        }

    def finish(self, *, finished_ns: int | None = None, exit_status=None) -> None:
        if self.finished:
            raise RuntimeError("Task recording is already finished")
        ended = finished_ns if finished_ns is not None else time.time_ns()
        if ended < self.assigned_ns:
            raise ValueError("Task finish precedes assignment")
        record = self._task_record(finished_ns=ended, exit_status=exit_status)
        temporary = self.task_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.task_path)
        self.finished = True

    def query(self, transport, messages, config_kwargs: dict, **kwargs):
        if self.finished:
            raise RuntimeError("Cannot record calls after task finish")
        parameters = config_kwargs | kwargs
        if parameters.get("stream", False) or parameters.get("n", 1) != 1:
            raise TokenCaptureError("Exact capture requires nonstreaming n=1")
        extra = dict(config_kwargs.get("extra_body") or {})
        extra.update(kwargs.get("extra_body") or {})
        extra["return_token_ids"] = True
        kwargs["extra_body"] = extra
        self.sequence += 1
        started = time.time_ns()
        monotonic_start = time.monotonic_ns()
        row = {
            "trial_id": self.trial_id,
            "request_sequence": self.sequence,
            "start_wall_ns": started,
            "error": None,
            "token_identity_verified": False,
            "token_source": "vllm_engine_response",
        }
        try:
            response = transport(messages, **kwargs)
            data = _mapping(response)
            prompt, output = extract_token_ids(data)
            row.update(
                token_identity_verified=True,
                prompt_token_ids=prompt,
                generated_token_ids=[output],
                usage=_usage_record(data, prompt, output),
            )
            self.verified_calls += 1
            return response
        except BaseException as error:
            # Exception text can contain URLs, authorization, or request bodies.
            row["error"] = type(error).__name__
            if isinstance(error, TokenCaptureError):
                self.capture_errors += 1
            raise
        finally:
            row["end_wall_ns"] = time.time_ns()
            row["client_elapsed_ns"] = time.monotonic_ns() - monotonic_start
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def attach_recorder(
    model,
    run_dir: Path | str,
    task_id: str,
    *,
    trial_id: str | None = None,
    assigned_ns: int | None = None,
) -> CallRecorder:
    """Attach to a fresh mini LitellmModel instance owned by one task.

    mini calls ``_query`` once per retry attempt, after preparing API
    messages and before parsing tool calls. Hooking it preserves successful
    generations that later raise FormatError, and logs failed attempts so
    their time remains in replay gaps. No global factory is patched.
    """
    if getattr(model, "_knlp_call_recorder", None) is not None:
        raise ValueError("A model instance can belong to only one recorded task")
    if not callable(getattr(model, "_query", None)):
        raise TypeError("Recording requires mini's LitellmModel transport hook")
    if not isinstance(getattr(model.config, "model_kwargs", None), dict):
        raise TypeError("Recording requires model.config.model_kwargs")
    if not isinstance(getattr(model, "abort_exceptions", None), list):
        raise TypeError("Recording requires mini's retry abort_exceptions list")
    recorder = CallRecorder(
        run_dir, task_id, trial_id=trial_id, assigned_ns=assigned_ns
    )
    transport = model._query

    def recorded_query(messages, **kwargs):
        return recorder.query(transport, messages, model.config.model_kwargs, **kwargs)

    model._query = recorded_query
    # Copy the upstream class list rather than modifying it for other tasks.
    model.abort_exceptions = [*model.abort_exceptions, TokenCaptureError]
    model._knlp_call_recorder = recorder
    return recorder


def summarize_calls(run_dir: Path | str) -> dict:
    """Aggregate recorded attempts without treating missing cache counts as zero.

    Failed transport attempts remain in attempt counts. Token and latency
    statistics cover successful exact captures only. These client durations
    include complete nonstreaming responses; they do not measure TTFT.
    Percentiles linearly interpolate between adjacent sorted observations.
    """
    result = {
        "total_attempts": 0,
        "successful_captured_calls": 0,
        "transport_failures": 0,
        "capture_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reported_cached_tokens": None,
        "cached_token_reporting_calls": 0,
        "latency_measurement": "full_nonstreaming_call",
    }
    durations = []
    cached_total = 0
    for path in sorted(
        Path(run_dir).glob("attempts/*/attempt_*/telemetry/llm_requests.jsonl")
    ):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                result["total_attempts"] += 1
                error = row.get("error")
                if error == "TokenCaptureError" or (
                    error is None and row.get("token_identity_verified") is not True
                ):
                    result["capture_errors"] += 1
                    continue
                if error is not None:
                    result["transport_failures"] += 1
                    continue
                prompt = _token_ids(row.get("prompt_token_ids"), "prompt_token_ids")
                outputs = row.get("generated_token_ids")
                if not isinstance(outputs, list) or len(outputs) != 1:
                    raise TokenCaptureError("Summary requires one generated sequence")
                output = _token_ids(outputs[0], "generated_token_ids[0]")
                elapsed = row.get("client_elapsed_ns")
                if type(elapsed) is not int or elapsed < 0:
                    raise ValueError("Recorded client_elapsed_ns must be nonnegative")
                durations.append(elapsed / 1e9)
                result["successful_captured_calls"] += 1
                result["prompt_tokens"] += len(prompt)
                result["completion_tokens"] += len(output)
                details = (row.get("usage") or {}).get("prompt_tokens_details") or {}
                cached = details.get("cached_tokens")
                if type(cached) is int and 0 <= cached <= len(prompt):
                    result["cached_token_reporting_calls"] += 1
                    cached_total += cached
    if (
        result["successful_captured_calls"] > 0
        and result["cached_token_reporting_calls"]
        == result["successful_captured_calls"]
    ):
        result["reported_cached_tokens"] = cached_total
    durations.sort()

    def percentile(fraction):
        if not durations:
            return None
        position = (len(durations) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return durations[low] + (durations[high] - durations[low]) * (position - low)

    result["client_latency_s"] = {
        "count": len(durations),
        "mean": statistics.mean(durations) if durations else None,
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }
    return result
