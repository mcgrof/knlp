# SPDX-License-Identifier: Apache-2.0
"""Crash-safe post-forward K/V and raw-logits capture for the bounded gate.

Only the ordinary forward batch, serial oracle, and forward collective arms are
instrumented.  The hook runs in MRv2's single worker process.  It copies the
four request-owned prompt caches at ``GPUModelRunner.sample_tokens`` entry,
before ``execute_model_state`` is cleared and before scheduler-owned blocks can
be freed, and captures logits at the GPU sampler entry before sampling mutates
them.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import json
import os
import threading
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAPTURE_SCHEMA = "kmesh.post_pic_kv_capture.v1"
CAPTURE_SCHEMA_VERSION = 1
CAPTURE_RELATIVE_PATH = "post-pic-capture-v1/family-manifest.json"
CAPTURE_BOUNDARY = (
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample_tokens:"
    "entry_before_execute_model_state_clear_and_before_scheduler_block_free"
)
CAPTURED_ARMS = {
    "ordinary": "ordinary_batch_forward",
    "serial_oracle": "serial_oracle",
    "collective_forward": "collective_forward",
}
EXPECTED_SEQUENCE_INDICES = (1, 2, 3, 4)
EXPECTED_LAYER_NAMES = tuple(
    f"model.layers.{index}.self_attn.attn" for index in range(24)
)
EXPECTED_BLOCK_SIZE = 32
EXPECTED_PROMPT_TOKENS = 512
EXPECTED_PHYSICAL_BLOCKS = 16
EXPECTED_KV_HEADS = 2
EXPECTED_HEAD_SIZE = 64
EXPECTED_KV_HIDDEN = EXPECTED_KV_HEADS * EXPECTED_HEAD_SIZE
EXPECTED_PACKED_HEAD = 2 * EXPECTED_HEAD_SIZE
EXPECTED_VOCAB_SIZE = 151936
EXPECTED_KV_SHAPE = (24, 512, 128)
INTERNAL_REQUEST_NONCE_HEX_LENGTH = 8
LOWER_HEX = frozenset("0123456789abcdef")
REQUEST_ID_CONTRACT = {
    "trace_domain": "trace_request_id",
    "public_domain": "external_request_id",
    "worker_domain": "engine_request_id",
    "lookup_domain": "lookup_id",
    "internal_format": "{external_request_id}-{8_lower_hex}",
    "randomization_source": (
        "vllm.v1.engine.input_processor.InputProcessor.assign_request_id"
    ),
    "lookup_identity_rule": "lookup IDs are transport-local, never request IDs",
    "layer_api_identity_rule": (
        "store_layer req_id is engine_request_id; retrieve_layer req_id is lookup_id"
    ),
}


class NumericalCaptureError(RuntimeError):
    """The bounded numerical-capture contract was violated."""


@dataclass(frozen=True)
class _RecordBinding:
    sequence_index: int
    trace_request_id: str
    trace_id: str
    prompt_token_ids_sha256: str
    raw: dict[str, Any]


@dataclass
class _CapturedMember:
    binding: _RecordBinding
    engine_request_id: str
    capture_batch_row: int
    batch_engine_request_ids: tuple[str, ...]
    physical_block_ids: tuple[int, ...]
    slots: tuple[int, ...]
    cache_shapes: tuple[tuple[int, ...], ...]
    cache_strides: tuple[tuple[int, ...], ...]
    tensor_path: Path
    logits_path: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise NumericalCaptureError(f"refusing to replace capture artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise NumericalCaptureError(f"stale capture temporary exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_bytes(path, payload)


def _strict_value_equal(actual: Any, expected: Any) -> bool:
    """Compare a frozen runtime DTO without Python's bool/int aliases."""
    if type(actual) is not type(expected):
        return False
    if is_dataclass(expected) and not isinstance(expected, type):
        if (
            hasattr(expected, "__dict__")
            and vars(actual).keys() != vars(expected).keys()
        ):
            return False
        return all(
            _strict_value_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
            for field in fields(expected)
        )
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(
            _strict_value_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    if type(expected) is dict:
        if len(actual) != len(expected):
            return False
        unmatched = list(actual.items())
        for expected_key, expected_value in expected.items():
            matches = [
                index
                for index, (actual_key, _actual_value) in enumerate(unmatched)
                if _strict_value_equal(actual_key, expected_key)
            ]
            if len(matches) != 1:
                return False
            _actual_key, actual_value = unmatched.pop(matches[0])
            if not _strict_value_equal(actual_value, expected_value):
                return False
        return not unmatched
    if type(expected) in (set, frozenset):
        if len(actual) != len(expected):
            return False
        unmatched = list(actual)
        for expected_item in expected:
            matches = [
                index
                for index, actual_item in enumerate(unmatched)
                if _strict_value_equal(actual_item, expected_item)
            ]
            if len(matches) != 1:
                return False
            unmatched.pop(matches[0])
        return not unmatched
    if hasattr(type(expected), "__struct_fields__"):
        import msgspec

        return _strict_value_equal(
            msgspec.to_builtins(actual),
            msgspec.to_builtins(expected),
        )
    result = actual == expected
    return type(result) is bool and result


def _atomic_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    if path.exists():
        raise NumericalCaptureError(f"refusing to replace capture tensor: {path}")
    if not tensors:
        raise NumericalCaptureError("safetensors capture cannot be empty")
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise NumericalCaptureError(
                f"capture tensor {name} must be contiguous CPU storage"
            )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise NumericalCaptureError(f"stale capture temporary exists: {temporary}")
    try:
        save_file(tensors, temporary)
        os.chmod(temporary, 0o644)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _file_binding(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    capture_root = root.resolve()
    if not resolved.is_file() or capture_root not in resolved.parents:
        raise NumericalCaptureError(f"capture file is absent or escapes root: {path}")
    return {
        "path": resolved.relative_to(capture_root).as_posix(),
        "sha256": _sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def _prompt_digest(tokens: object) -> str:
    if type(tokens) not in (list, tuple):
        raise NumericalCaptureError("prompt token IDs must be a plain list or tuple")
    normalized: list[int] = []
    for token in tokens:
        if type(token) is not int or not 0 <= token < 2**31:
            raise NumericalCaptureError(
                "prompt token IDs must be plain nonnegative ints"
            )
        normalized.append(token)
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _internal_request_id_matches(internal: object, external: object) -> bool:
    if type(internal) is not str or type(external) is not str or not external:
        return False
    prefix, separator, nonce = internal.rpartition("-")
    return (
        separator == "-"
        and prefix == external
        and len(nonce) == INTERNAL_REQUEST_NONCE_HEX_LENGTH
        and all(character in LOWER_HEX for character in nonce)
    )


def _plain_json_copy(value: object, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise NumericalCaptureError(f"{label} is not plain JSON evidence") from error


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise NumericalCaptureError(f"{label} must be a nonempty string")
    return value


def _make_capture_directories(output_dir: Path) -> Path:
    root = output_dir / "post-pic-capture-v1"
    root.mkdir(mode=0o755, exist_ok=False)
    members = root / "members"
    members.mkdir(mode=0o755)
    for sequence_index in EXPECTED_SEQUENCE_INDICES:
        (members / f"{sequence_index:03d}").mkdir(mode=0o755)
    _fsync_directory(members)
    _fsync_directory(root)
    _fsync_directory(output_dir)
    return root


@dataclass(frozen=True)
class DisabledNumericalInstrumentation:
    """No-op for arms outside the exact three-arm numerical comparison."""

    output_dir: Path
    arm: str

    def finish(self, **_evidence: Any) -> dict[str, Any]:
        return {
            "enabled": False,
            "status": "disabled",
            "arm": self.arm,
            "claim_boundary": (
                "Numerical capture is enabled only for ordinary, serial_oracle, "
                "and collective_forward."
            ),
        }


class NumericalInstrumentation:
    """Own the exact MRv2 hooks and one four-member capture family."""

    def __init__(
        self,
        *,
        ledger: Any,
        arm: str,
        output_dir: Path,
        records: tuple[dict[str, Any], ...],
        source: dict[str, Any],
    ) -> None:
        self.ledger = ledger
        self.arm = arm
        self.phase = CAPTURED_ARMS[arm]
        self.output_dir = output_dir.resolve()
        self.source = self._validate_source(source)
        self.capture_root = _make_capture_directories(self.output_dir)
        self._lock = threading.RLock()
        self._finished = False
        self._restored = False
        self._kernel_warmup_seen = False
        self._engine_to_sequence: dict[str, int] = {}
        self._sequence_to_engine: dict[int, str] = {}
        self._captured: dict[int, _CapturedMember] = {}
        self._records_by_sequence, self._records_by_digest = self._bind_records(records)
        self._hook_owners: list[tuple[type[Any], str, Any, Any]] = []
        self._install_hooks()
        self.ledger.emit(
            "gate.numerical.install",
            arm=self.arm,
            capture_phase=self.phase,
            capture_boundary=CAPTURE_BOUNDARY,
            target_sequence_indices=list(EXPECTED_SEQUENCE_INDICES),
        )

    @staticmethod
    def _validate_source(source: dict[str, Any]) -> dict[str, Any]:
        import collective_gate_common as common

        if type(source) is not dict:
            raise NumericalCaptureError("capture source must be a plain mapping")
        expected = {
            "vllm": (common.EXPECTED_VLLM_COMMIT, common.EXPECTED_VLLM_TREE),
            "lmcache": (common.EXPECTED_LMCACHE_COMMIT, common.EXPECTED_LMCACHE_TREE),
        }
        for name, (commit, tree) in expected.items():
            identity = source.get(name)
            if (
                type(identity) is not dict
                or identity.get("commit") != commit
                or identity.get("tree") != tree
                or identity.get("status_porcelain") != ""
            ):
                raise NumericalCaptureError(
                    f"capture source.{name} is not the exact clean source object"
                )
        return _plain_json_copy(source, "capture source")

    @staticmethod
    def _bind_records(
        records: tuple[dict[str, Any], ...],
    ) -> tuple[dict[int, _RecordBinding], dict[str, _RecordBinding]]:
        if type(records) is not tuple or len(records) != 5:
            raise NumericalCaptureError("capture requires the exact five-record trace")
        by_sequence: dict[int, _RecordBinding] = {}
        by_digest: dict[str, _RecordBinding] = {}
        trace_ids: set[str] = set()
        for expected_sequence, record in enumerate(records):
            if (
                type(record) is not dict
                or record.get("sequence_index") != expected_sequence
            ):
                raise NumericalCaptureError(
                    "capture records are not canonically ordered"
                )
            trace_request_id = _nonempty_string(
                record.get("request_id"), "trace request ID"
            )
            trace_id = _nonempty_string(record.get("trace_id"), "trace ID")
            tokens = record.get("prompt_token_ids")
            if type(tokens) is not list or len(tokens) != EXPECTED_PROMPT_TOKENS:
                raise NumericalCaptureError("every capture prompt must have 512 tokens")
            digest = _prompt_digest(tokens)
            binding = _RecordBinding(
                expected_sequence,
                trace_request_id,
                trace_id,
                digest,
                copy.deepcopy(record),
            )
            if digest in by_digest or expected_sequence in by_sequence:
                raise NumericalCaptureError("capture prompt identities are not unique")
            by_sequence[expected_sequence] = binding
            by_digest[digest] = binding
            trace_ids.add(trace_id)
        if len(trace_ids) != 1:
            raise NumericalCaptureError("capture records do not share one trace ID")
        return by_sequence, by_digest

    def _install_hooks(self) -> None:
        import torch
        from vllm.v1.worker.gpu import model_runner as model_runner_module
        from vllm.v1.worker.gpu.sample.sampler import Sampler

        gpu_model_runner = model_runner_module.GPUModelRunner
        original_add = gpu_model_runner.add_requests
        original_sample_tokens = gpu_model_runner.sample_tokens
        original_sampler_call = Sampler.__call__
        exact_signatures = (
            (original_add, ("self", "scheduler_output"), "GPUModelRunner.add_requests"),
            (
                original_sample_tokens,
                ("self", "grammar_output"),
                "GPUModelRunner.sample_tokens",
            ),
            (
                original_sampler_call,
                ("self", "logits", "input_batch"),
                "Sampler.__call__",
            ),
        )
        for function, expected, label in exact_signatures:
            if tuple(inspect.signature(function).parameters) != expected:
                raise NumericalCaptureError(
                    f"committed MRv2 signature drifted: {label}"
                )
            if getattr(function, "_kmesh_numerical_capture_v1", False):
                raise NumericalCaptureError(
                    f"numerical hook is already installed: {label}"
                )

        source_root = Path(self.source["vllm"]["path"]).resolve()
        for owner in (gpu_model_runner, Sampler):
            source_file = inspect.getsourcefile(owner)
            if source_file is None or not Path(source_file).resolve().is_relative_to(
                source_root
            ):
                raise NumericalCaptureError(
                    f"numerical hook class is not loaded from {source_root}: {owner}"
                )

        instrumentation = self

        @functools.wraps(original_add)
        def add_requests(runner: Any, scheduler_output: Any) -> None:
            instrumentation._observe_new_requests(runner, scheduler_output)
            return original_add(runner, scheduler_output)

        @functools.wraps(original_sample_tokens)
        @torch.inference_mode()
        def sample_tokens(runner: Any, grammar_output: Any) -> Any:
            instrumentation._capture_kv_at_sample_entry(runner, grammar_output)
            return original_sample_tokens(runner, grammar_output)

        @functools.wraps(original_sampler_call)
        @torch.inference_mode()
        def sampler_call(sampler: Any, logits: Any, input_batch: Any) -> Any:
            instrumentation._capture_raw_logits(logits, input_batch)
            return original_sampler_call(sampler, logits, input_batch)

        for wrapper in (add_requests, sample_tokens, sampler_call):
            wrapper._kmesh_numerical_capture_v1 = True

        gpu_model_runner.add_requests = add_requests
        gpu_model_runner.sample_tokens = sample_tokens
        Sampler.__call__ = sampler_call
        self._hook_owners = [
            (gpu_model_runner, "add_requests", original_add, add_requests),
            (
                gpu_model_runner,
                "sample_tokens",
                original_sample_tokens,
                sample_tokens,
            ),
            (Sampler, "__call__", original_sampler_call, sampler_call),
        ]

    def _restore_hooks(self) -> None:
        with self._lock:
            if self._restored:
                return
            for owner, name, original, wrapper in reversed(self._hook_owners):
                if getattr(owner, name) is not wrapper:
                    raise NumericalCaptureError(
                        f"numerical hook ownership drifted before restore: {owner}.{name}"
                    )
                setattr(owner, name, original)
            self._restored = True

    def _is_exact_kernel_warmup_batch(
        self,
        runner: Any,
        scheduler_output: Any,
        new_requests: list[Any],
    ) -> bool:
        """Recognize only MRv2's frozen sampler-kernel warmup prefill."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
        from vllm.v1.worker.gpu.kv_connector import (
            NO_OP_KV_CONNECTOR,
            ActiveKVConnector,
        )

        expected_ids = tuple(f"_warmup_{index}_" for index in range(4))
        connector = getattr(runner, "kv_connector", None)
        connector_is_disabled = connector is NO_OP_KV_CONNECTOR or (
            type(connector) is ActiveKVConnector
            and getattr(connector, "_disabled", None) is True
        )
        if (
            getattr(self.ledger, "phase", None) != "construction"
            or self._kernel_warmup_seen is not False
            or self._engine_to_sequence
            or self._sequence_to_engine
            or self._captured
            or getattr(runner, "decode_query_len", None) != 1
            or getattr(getattr(runner, "scheduler_config", None), "max_num_seqs", None)
            != 4
            or getattr(
                getattr(runner, "scheduler_config", None),
                "max_num_batched_tokens",
                None,
            )
            != 2048
            or not connector_is_disabled
        ):
            return False
        expected = SchedulerOutput.make_empty()
        expected.scheduled_new_reqs = [
            NewRequestData(
                req_id=request_id,
                prompt_token_ids=[0, 1],
                prefill_token_ids=[0, 1],
                mm_features=[],
                sampling_params=SamplingParams.for_sampler_warmup(),
                pooling_params=None,
                block_ids=([index],),
                num_computed_tokens=0,
                lora_request=None,
                prompt_embeds=None,
                prompt_is_token_ids=None,
            )
            for index, request_id in enumerate(expected_ids, start=1)
        ]
        expected.num_scheduled_tokens = {request_id: 2 for request_id in expected_ids}
        expected.total_num_scheduled_tokens = 8
        expected.num_common_prefix_blocks = [0]
        return _strict_value_equal(scheduler_output, expected) and _strict_value_equal(
            new_requests,
            expected.scheduled_new_reqs,
        )

    def _observe_new_requests(self, runner: Any, scheduler_output: Any) -> None:
        new_requests = getattr(scheduler_output, "scheduled_new_reqs", None)
        if type(new_requests) is not list:
            raise NumericalCaptureError("scheduled_new_reqs is not the committed list")
        unknown_requests = [
            request
            for request in new_requests
            if _prompt_digest(getattr(request, "prompt_token_ids", None))
            not in self._records_by_digest
        ]
        if unknown_requests:
            with self._lock:
                if len(unknown_requests) != len(new_requests) or not (
                    self._is_exact_kernel_warmup_batch(
                        runner,
                        scheduler_output,
                        new_requests,
                    )
                ):
                    raise NumericalCaptureError(
                        "runtime request prompt is absent from the frozen "
                        "five-record trace"
                    )
                self._kernel_warmup_seen = True
                self.ledger.emit(
                    "gate.numerical.kernel_warmup_ignored",
                    request_ids=[request.req_id for request in new_requests],
                    prompt_token_ids=[0, 1],
                    total_num_scheduled_tokens=8,
                )
            return
        for request in new_requests:
            engine_request_id = _nonempty_string(
                getattr(request, "req_id", None), "engine request ID"
            )
            prompt_token_ids = getattr(request, "prompt_token_ids", None)
            digest = _prompt_digest(prompt_token_ids)
            binding = self._records_by_digest.get(digest)
            if binding is None:
                raise AssertionError("unknown capture prompt escaped batch validation")
            with self._lock:
                previous_sequence = self._engine_to_sequence.get(engine_request_id)
                previous_engine = self._sequence_to_engine.get(binding.sequence_index)
                if previous_sequence not in (None, binding.sequence_index):
                    raise NumericalCaptureError("engine request ID changed trace owner")
                if previous_engine not in (None, engine_request_id):
                    raise NumericalCaptureError("trace request acquired two engine IDs")
                self._engine_to_sequence[engine_request_id] = binding.sequence_index
                self._sequence_to_engine[binding.sequence_index] = engine_request_id
            self.ledger.emit(
                "gate.numerical.request_bound",
                engine_request_id=engine_request_id,
                trace_request_id=binding.trace_request_id,
                sequence_index=binding.sequence_index,
                prompt_token_ids_sha256=digest,
            )

    def _target_rows(self, input_batch: Any) -> list[tuple[int, str, int]]:
        req_ids = getattr(input_batch, "req_ids", None)
        if type(req_ids) is not list or any(
            type(req_id) is not str for req_id in req_ids
        ):
            raise NumericalCaptureError("InputBatch.req_ids is not a plain string list")
        rows = [
            (row, req_id, self._engine_to_sequence[req_id])
            for row, req_id in enumerate(req_ids)
            if self._engine_to_sequence.get(req_id) in EXPECTED_SEQUENCE_INDICES
        ]
        if not rows:
            return []
        if (
            len(req_ids) != 4
            or getattr(input_batch, "num_reqs", None) != 4
            or len(rows) != 4
            or len(set(req_ids)) != 4
            or {row for row, _, _ in rows} != {0, 1, 2, 3}
            or {sequence for _, _, sequence in rows} != set(EXPECTED_SEQUENCE_INDICES)
        ):
            raise NumericalCaptureError(
                "target prompts did not execute as one exact four-request batch"
            )
        if getattr(input_batch, "num_draft_tokens", None) != 0:
            raise NumericalCaptureError("numerical capture forbids speculative tokens")
        cu_num_logits = getattr(input_batch, "cu_num_logits_np", None)
        if cu_num_logits is None or cu_num_logits.tolist() != [0, 1, 2, 3, 4]:
            raise NumericalCaptureError(
                "capture batch does not have one logit row/request"
            )
        if getattr(input_batch, "has_structured_output_reqs", None) is not False:
            raise NumericalCaptureError(
                "numerical capture forbids grammar-mutated logits"
            )
        prefill_lengths = getattr(input_batch, "prefill_len_np", None)
        is_prefilling = getattr(input_batch, "is_prefilling_np", None)
        if prefill_lengths is None or is_prefilling is None:
            raise NumericalCaptureError("capture InputBatch lacks prefill ownership")
        for row, _req_id, _sequence in rows:
            if int(prefill_lengths[row]) != EXPECTED_PROMPT_TOKENS or not bool(
                is_prefilling[row]
            ):
                raise NumericalCaptureError(
                    "capture is not the exact 512-token prefill"
                )
        return rows

    @staticmethod
    def _validate_runner(runner: Any) -> None:
        import torch

        parallel = getattr(runner, "parallel_config", None)
        if any(
            getattr(parallel, field, None) != 1
            for field in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise NumericalCaptureError("capture requires TP/PP/DCP/PCP size one")
        if (
            getattr(runner, "dcp_size", None) != 1
            or getattr(runner, "pcp_manager", None) is not None
        ):
            raise NumericalCaptureError(
                "capture requires unpartitioned prompt ownership"
            )
        if getattr(runner, "is_last_pp_rank", None) is not True:
            raise NumericalCaptureError("capture requires the last pipeline rank")
        attention = getattr(runner.vllm_config, "attention_config", None)
        if getattr(getattr(attention, "backend", None), "name", None) != "TRITON_ATTN":
            raise NumericalCaptureError("capture requires explicit TRITON_ATTN")

        block_tables = getattr(runner, "block_tables", None)
        if (
            getattr(block_tables, "num_kv_cache_groups", None) != 1
            or getattr(block_tables, "block_sizes", None) != [EXPECTED_BLOCK_SIZE]
            or getattr(block_tables, "kernel_block_sizes", None)
            != [EXPECTED_BLOCK_SIZE]
            or getattr(block_tables, "blocks_per_kv_block", None) != [1]
            or len(getattr(block_tables, "input_block_tables", ())) != 1
        ):
            raise NumericalCaptureError("capture block-table geometry drifted")

        cache_groups = getattr(runner.kv_cache_config, "kv_cache_groups", None)
        if type(cache_groups) is not list or len(cache_groups) != 1:
            raise NumericalCaptureError("capture requires exactly one KV cache group")
        cache_group = cache_groups[0]
        if tuple(getattr(cache_group, "layer_names", ())) != EXPECTED_LAYER_NAMES:
            raise NumericalCaptureError("capture KV layer names or order drifted")
        spec = getattr(cache_group, "kv_cache_spec", None)
        if getattr(spec, "block_size", None) != EXPECTED_BLOCK_SIZE:
            raise NumericalCaptureError("capture KV-cache spec block size drifted")

        caches = getattr(runner, "kv_caches", None)
        if type(caches) is not list or len(caches) != len(EXPECTED_LAYER_NAMES):
            raise NumericalCaptureError("capture runner does not expose 24 KV tensors")
        for cache in caches:
            if (
                type(cache) is not torch.Tensor
                or cache.dtype is not torch.bfloat16
                or cache.ndim != 4
                or tuple(cache.shape[1:])
                != (EXPECTED_KV_HEADS, EXPECTED_BLOCK_SIZE, EXPECTED_PACKED_HEAD)
            ):
                raise NumericalCaptureError("capture KV tensor layout or dtype drifted")

        backend_names = {
            group.backend.get_name()
            for groups in getattr(runner, "attn_groups", ())
            for group in groups
        }
        if backend_names != {"TRITON_ATTN"}:
            raise NumericalCaptureError("runtime attention groups are not exact Triton")

    def _capture_kv_at_sample_entry(self, runner: Any, grammar_output: Any) -> None:
        import torch

        state = getattr(runner, "execute_model_state", None)
        if state is None:
            return
        input_batch = state.input_batch
        rows = self._target_rows(input_batch)
        if not rows:
            return
        if grammar_output is not None:
            raise NumericalCaptureError(
                "capture target unexpectedly has grammar output"
            )
        self._validate_runner(runner)
        with self._lock:
            if self._captured:
                raise NumericalCaptureError(
                    "target K/V family was captured more than once"
                )

            torch.cuda.synchronize(runner.device)
            table = runner.block_tables.input_block_tables[0]
            if table.dtype is not torch.int32 or table.ndim != 2:
                raise NumericalCaptureError("input block table dtype or rank drifted")
            batch_engine_ids = tuple(input_batch.req_ids)
            for row, engine_request_id, sequence_index in rows:
                block_ids_tensor = table[row, :EXPECTED_PHYSICAL_BLOCKS]
                physical_block_ids = tuple(
                    int(value)
                    for value in block_ids_tensor.to(
                        device="cpu", dtype=torch.int64
                    ).tolist()
                )
                if (
                    len(physical_block_ids) != EXPECTED_PHYSICAL_BLOCKS
                    or any(block < 0 for block in physical_block_ids)
                    or len(set(physical_block_ids)) != EXPECTED_PHYSICAL_BLOCKS
                ):
                    raise NumericalCaptureError(
                        "capture request does not own 16 unique physical blocks"
                    )
                slots = tuple(
                    block * EXPECTED_BLOCK_SIZE + offset
                    for block in physical_block_ids
                    for offset in range(EXPECTED_BLOCK_SIZE)
                )
                block_indices = torch.tensor(
                    physical_block_ids,
                    dtype=torch.int64,
                    device=runner.device,
                ).repeat_interleave(EXPECTED_BLOCK_SIZE)
                offsets = torch.arange(
                    EXPECTED_BLOCK_SIZE,
                    dtype=torch.int64,
                    device=runner.device,
                ).repeat(EXPECTED_PHYSICAL_BLOCKS)
                key_layers = []
                value_layers = []
                cache_shapes = []
                cache_strides = []
                for cache in runner.kv_caches:
                    if max(physical_block_ids) >= cache.shape[0]:
                        raise NumericalCaptureError(
                            "block table addresses beyond a runner KV cache"
                        )
                    packed = cache[block_indices, :, offsets, :]
                    if tuple(packed.shape) != (
                        EXPECTED_PROMPT_TOKENS,
                        EXPECTED_KV_HEADS,
                        EXPECTED_PACKED_HEAD,
                    ):
                        raise NumericalCaptureError("logical KV gather shape drifted")
                    key, value = packed.split(EXPECTED_HEAD_SIZE, dim=-1)
                    key_layers.append(
                        key.reshape(EXPECTED_PROMPT_TOKENS, EXPECTED_KV_HIDDEN)
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    value_layers.append(
                        value.reshape(EXPECTED_PROMPT_TOKENS, EXPECTED_KV_HIDDEN)
                        .to(device="cpu", dtype=torch.bfloat16)
                        .contiguous()
                    )
                    cache_shapes.append(tuple(int(size) for size in cache.shape))
                    cache_strides.append(
                        tuple(int(stride) for stride in cache.stride())
                    )
                key_tensor = torch.stack(key_layers, dim=0).contiguous()
                value_tensor = torch.stack(value_layers, dim=0).contiguous()
                if (
                    tuple(key_tensor.shape) != EXPECTED_KV_SHAPE
                    or tuple(value_tensor.shape) != EXPECTED_KV_SHAPE
                    or key_tensor.dtype is not torch.bfloat16
                    or value_tensor.dtype is not torch.bfloat16
                ):
                    raise NumericalCaptureError("canonical K/V tensor contract drifted")
                member_root = self.capture_root / "members" / f"{sequence_index:03d}"
                tensor_path = member_root / "kv.safetensors"
                _atomic_safetensors(
                    tensor_path,
                    {"key": key_tensor, "value": value_tensor},
                )
                binding = self._records_by_sequence[sequence_index]
                self._captured[sequence_index] = _CapturedMember(
                    binding=binding,
                    engine_request_id=engine_request_id,
                    capture_batch_row=row,
                    batch_engine_request_ids=batch_engine_ids,
                    physical_block_ids=physical_block_ids,
                    slots=slots,
                    cache_shapes=tuple(cache_shapes),
                    cache_strides=tuple(cache_strides),
                    tensor_path=tensor_path,
                )
                self.ledger.emit(
                    "gate.numerical.kv_captured",
                    arm=self.arm,
                    sequence_index=sequence_index,
                    trace_request_id=binding.trace_request_id,
                    engine_request_id=engine_request_id,
                    capture_batch_row=row,
                    physical_block_ids=list(physical_block_ids),
                    tensor_path=str(tensor_path.relative_to(self.output_dir)),
                    tensor_sha256=_sha256_file(tensor_path),
                )
            torch.cuda.synchronize(runner.device)

    def _capture_raw_logits(self, logits: Any, input_batch: Any) -> None:
        import torch

        rows = self._target_rows(input_batch)
        if not rows:
            return
        with self._lock:
            if set(self._captured) != set(EXPECTED_SEQUENCE_INDICES):
                raise NumericalCaptureError(
                    "logits arrived before the complete K/V capture"
                )
            if (
                type(logits) is not torch.Tensor
                or logits.ndim != 2
                or tuple(logits.shape) != (4, EXPECTED_VOCAB_SIZE)
            ):
                raise NumericalCaptureError("raw logits are not exact [4, 151936]")
            raw_logits = (
                logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
            if not bool(torch.isfinite(raw_logits).all().item()):
                raise NumericalCaptureError("raw logits contain a non-finite value")
            for row, engine_request_id, sequence_index in rows:
                member = self._captured[sequence_index]
                if (
                    member.engine_request_id != engine_request_id
                    or member.capture_batch_row != row
                    or member.logits_path is not None
                ):
                    raise NumericalCaptureError("logits request ownership drifted")
                logits_path = member.tensor_path.with_name("logits.safetensors")
                member_logits = raw_logits[row].clone().contiguous()
                _atomic_safetensors(
                    logits_path,
                    {"raw_logits": member_logits},
                )
                member.logits_path = logits_path
                self.ledger.emit(
                    "gate.numerical.logits_captured",
                    arm=self.arm,
                    sequence_index=sequence_index,
                    trace_request_id=member.binding.trace_request_id,
                    engine_request_id=engine_request_id,
                    capture_batch_row=row,
                    vocab_size=member_logits.numel(),
                    logits_path=str(logits_path.relative_to(self.output_dir)),
                    logits_sha256=_sha256_file(logits_path),
                )

    def _validate_finish_evidence(
        self,
        *,
        arm: str,
        observations: Any,
        recovery_evidence: Any,
        source: Any,
        runtime: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        import collective_gate_common as common

        if arm != self.arm:
            raise NumericalCaptureError("finish arm differs from installed arm")
        finish_source = self._validate_source(source)
        if finish_source != self.source:
            raise NumericalCaptureError("finish source differs from install source")
        if type(runtime) is not dict:
            raise NumericalCaptureError("finish runtime evidence must be a mapping")
        if (
            runtime.get("image") != common.EXPECTED_IMAGE
            or runtime.get("target_volume") != common.EXPECTED_TARGET_VOLUME
            or runtime.get("accelerator_count") != 1
            or runtime.get("device_arch") != common.EXPECTED_RUNTIME["device_arch"]
        ):
            raise NumericalCaptureError(
                "finish runtime is not the pinned single-device runtime"
            )
        runtime_copy = _plain_json_copy(runtime, "capture runtime")

        if type(observations) is not list:
            raise NumericalCaptureError("finish observations must be a list")
        expected_phases = [
            "ordinary_prime" if self.arm == "ordinary" else "prime",
            *([self.phase] * 4),
        ]
        if (
            len(observations) != 5
            or any(type(item) is not dict for item in observations)
            or [item.get("sequence_index") for item in observations] != list(range(5))
            or [item.get("external_request_id") for item in observations]
            != [str(index) for index in range(5)]
            or [item.get("phase") for item in observations] != expected_phases
            or any(
                item.get("trace_request_id")
                != self._records_by_sequence[index].trace_request_id
                for index, item in enumerate(observations)
            )
        ):
            raise NumericalCaptureError(
                "observations do not bind the exact five-request execution"
            )
        selected = observations[1:]
        if (
            [item.get("sequence_index") for item in selected]
            != list(EXPECTED_SEQUENCE_INDICES)
            or [item.get("external_request_id") for item in selected]
            != [str(index) for index in EXPECTED_SEQUENCE_INDICES]
            or any(item.get("phase") != self.phase for item in selected)
            or any(
                item.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS for item in selected
            )
            or any(len(item.get("output_token_ids", ())) != 1 for item in selected)
        ):
            raise NumericalCaptureError(
                "observations do not bind the target four-request batch"
            )
        for item in selected:
            sequence_index = item["sequence_index"]
            member = self._captured.get(sequence_index)
            if (
                member is None
                or item.get("trace_request_id") != member.binding.trace_request_id
                or not _internal_request_id_matches(
                    member.engine_request_id,
                    item.get("external_request_id"),
                )
            ):
                raise NumericalCaptureError(
                    "observation and captured request IDs differ"
                )
            cached = item.get("num_cached_tokens")
            if self.arm == "ordinary" and cached != 0:
                raise NumericalCaptureError(
                    "ordinary numerical batch reports cached tokens"
                )
            if self.arm != "ordinary" and (type(cached) is not int or cached <= 0):
                raise NumericalCaptureError(
                    "PIC numerical batch did not advertise a hit"
                )

        if type(recovery_evidence) is not dict:
            raise NumericalCaptureError("finish recovery evidence must be a mapping")
        recovery_copy = _plain_json_copy(recovery_evidence, "recovery evidence")
        request_id_bindings = recovery_copy.get("request_id_bindings")
        expected_binding_action = (
            "gate.numerical.request_bound"
            if self.arm == "ordinary"
            else "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
        )
        if not isinstance(request_id_bindings, list) or len(request_id_bindings) != len(
            observations
        ):
            raise NumericalCaptureError(
                "recovery request-ID binding coverage is incomplete"
            )
        for sequence, (observation, binding) in enumerate(
            zip(observations, request_id_bindings, strict=True)
        ):
            if (
                type(binding) is not dict
                or binding.get("trace_request_id") != observation["trace_request_id"]
                or binding.get("external_request_id")
                != observation["external_request_id"]
                or binding.get("engine_request_id")
                != self._sequence_to_engine.get(sequence)
                or binding.get("phase") != observation["phase"]
                or binding.get("binding_event_action") != expected_binding_action
            ):
                raise NumericalCaptureError(
                    "recovery request-ID binding differs from numerical ownership"
                )
        return selected, recovery_copy, runtime_copy

    def _recovery_contract(
        self,
        recovery: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
        groups = recovery.get("groups")
        serial_members = recovery.get("serial_members")
        if type(groups) is not list or type(serial_members) is not list:
            raise NumericalCaptureError(
                "recovery groups or serial members are malformed"
            )
        trace_id = self._records_by_sequence[0].trace_id

        if self.arm == "ordinary":
            if groups or serial_members:
                raise NumericalCaptureError(
                    "ordinary arm unexpectedly has PIC recovery"
                )
            family = {
                "plan_id": f"ordinary-batch-forward:{trace_id}",
                "reuse_group_id": f"ordinary-no-reuse:{trace_id}",
                "profile": "ordinary_batch_forward",
                "member_count": 4,
                "all_members_successful": True,
            }
            member_evidence = {
                sequence: {
                    "plan_member_index": sequence - 1,
                    "outcome": "ordinary_forward",
                    "fallback_reason": None,
                    "pic_executed": False,
                }
                for sequence in EXPECTED_SEQUENCE_INDICES
            }
            return family, member_evidence

        if self.arm == "serial_oracle":
            if groups or len(serial_members) != 4:
                raise NumericalCaptureError(
                    "serial oracle recovery membership is not exact"
                )
            by_trace = {
                member.get("trace_request_id"): member
                for member in serial_members
                if type(member) is dict
            }
            if len(by_trace) != 4:
                raise NumericalCaptureError(
                    "serial oracle trace membership is not unique"
                )
            for sequence in EXPECTED_SEQUENCE_INDICES:
                captured = self._captured[sequence]
                recovered = by_trace.get(captured.binding.trace_request_id)
                if (
                    type(recovered) is not dict
                    or recovered.get("engine_request_id") != captured.engine_request_id
                    or not isinstance(recovered.get("selected_spans"), dict)
                    or not isinstance(recovered.get("retrieval_mask"), dict)
                ):
                    raise NumericalCaptureError(
                        "serial oracle recovery binding drifted"
                    )
            context = self._records_by_sequence[1].raw.get("kmesh_context")
            if type(context) is not dict:
                raise NumericalCaptureError("serial oracle lacks KMesh context")
            family = {
                "plan_id": f"serial-oracle:{trace_id}",
                "reuse_group_id": _nonempty_string(
                    context.get("group_id"), "serial reuse group ID"
                ),
                "profile": "serial_pic_v1",
                "member_count": 4,
                "all_members_successful": True,
            }
            member_evidence = {
                sequence: {
                    "plan_member_index": sequence - 1,
                    "outcome": "recovered",
                    "fallback_reason": None,
                    "pic_executed": True,
                }
                for sequence in EXPECTED_SEQUENCE_INDICES
            }
            return family, member_evidence

        if self.arm != "collective_forward" or len(groups) != 1 or serial_members:
            raise NumericalCaptureError(
                "collective forward recovery family is not exact"
            )
        group = groups[0]
        if (
            type(group) is not dict
            or group.get("call_index") != 0
            or group.get("num_layers") != 24
            or type(group.get("members")) is not list
            or len(group["members"]) != 4
        ):
            raise NumericalCaptureError("collective recovery group geometry drifted")
        plan_id = _nonempty_string(group.get("plan_id"), "collective plan ID")
        reuse_group_id = _nonempty_string(
            group.get("reuse_group_id"), "collective reuse group ID"
        )
        engine_order = group.get("engine_request_order")
        expected_engine_order = [
            self._captured[sequence].engine_request_id
            for sequence in EXPECTED_SEQUENCE_INDICES
        ]
        if engine_order != expected_engine_order:
            raise NumericalCaptureError(
                "collective recovery order differs from trace/capture order"
            )
        by_trace = {
            member.get("trace_request_id"): member
            for member in group["members"]
            if type(member) is dict
        }
        exact_counts = {
            "serial_rope_operations": 0,
            "serial_diff_operations": 0,
            "collective_rope_operations": 24,
            "collective_diff_operations": 1,
            "position_recovery_serial_operations": 0,
            "position_recovery_collective_operations": 24,
        }
        member_evidence: dict[int, dict[str, Any]] = {}
        for sequence in EXPECTED_SEQUENCE_INDICES:
            captured = self._captured[sequence]
            recovered = by_trace.get(captured.binding.trace_request_id)
            if (
                type(recovered) is not dict
                or recovered.get("engine_request_id") != captured.engine_request_id
                or recovered.get("outcome") != "recovered"
                or recovered.get("error") is not None
                or recovered.get("operation_counts") != exact_counts
            ):
                raise NumericalCaptureError("collective recovered member is not exact")
            member_evidence[sequence] = {
                "plan_member_index": engine_order.index(captured.engine_request_id),
                "outcome": "recovered",
                "fallback_reason": None,
                "pic_executed": True,
            }
        family = {
            "plan_id": plan_id,
            "reuse_group_id": reuse_group_id,
            "profile": "collective_batched_qkv_v1",
            "member_count": 4,
            "all_members_successful": True,
        }
        return family, member_evidence

    @staticmethod
    def _tensor_contract() -> dict[str, Any]:
        return {
            "format": "safetensors",
            "dtype": "bfloat16",
            "shape": list(EXPECTED_KV_SHAPE),
            "axis_order": ["layer", "prompt_token", "kv_hidden"],
            "logical_block_shape": [16, 24, 32, 128],
            "block_size": EXPECTED_BLOCK_SIZE,
            "prompt_token_count": EXPECTED_PROMPT_TOKENS,
            "kv_heads": EXPECTED_KV_HEADS,
            "head_dim": EXPECTED_HEAD_SIZE,
            "tensor_keys": ["key", "value"],
            "tensors": {
                "key": {"dtype": "bfloat16", "shape": list(EXPECTED_KV_SHAPE)},
                "value": {
                    "dtype": "bfloat16",
                    "shape": list(EXPECTED_KV_SHAPE),
                },
            },
            "layer_names": list(EXPECTED_LAYER_NAMES),
        }

    @staticmethod
    def _validate_capture_batch_ownership(
        captured: dict[int, _CapturedMember],
    ) -> tuple[str, ...]:
        if type(captured) is not dict or set(captured) != set(
            EXPECTED_SEQUENCE_INDICES
        ):
            raise NumericalCaptureError("capture does not contain sequences 1..4")
        captured_members = [
            captured[sequence] for sequence in EXPECTED_SEQUENCE_INDICES
        ]
        for sequence, member in zip(
            EXPECTED_SEQUENCE_INDICES, captured_members, strict=True
        ):
            binding_sequence = getattr(
                getattr(member, "binding", None), "sequence_index", None
            )
            if type(binding_sequence) is not int or binding_sequence != sequence:
                raise NumericalCaptureError(
                    "capture member differs from its trace-sequence owner"
                )
        capture_rows = [member.capture_batch_row for member in captured_members]
        if (
            any(type(row) is not int for row in capture_rows)
            or set(capture_rows) != {0, 1, 2, 3}
            or len(capture_rows) != len(set(capture_rows))
        ):
            raise NumericalCaptureError(
                "capture rows are not one exact four-row bijection"
            )
        batch_identities = [
            member.batch_engine_request_ids for member in captured_members
        ]
        if any(type(identity) is not tuple for identity in batch_identities) or any(
            identity != batch_identities[0] for identity in batch_identities[1:]
        ):
            raise NumericalCaptureError("capture batch identity differs by member")
        batch_engine_request_ids = batch_identities[0]
        captured_engine_request_ids = [
            member.engine_request_id for member in captured_members
        ]
        if (
            len(batch_engine_request_ids) != 4
            or any(
                type(engine_request_id) is not str or not engine_request_id
                for engine_request_id in batch_engine_request_ids
            )
            or len(set(batch_engine_request_ids)) != 4
            or any(
                type(engine_request_id) is not str or not engine_request_id
                for engine_request_id in captured_engine_request_ids
            )
            or set(batch_engine_request_ids) != set(captured_engine_request_ids)
        ):
            raise NumericalCaptureError("capture batch identity is not bijective")
        for member in captured_members:
            if (
                batch_engine_request_ids[member.capture_batch_row]
                != member.engine_request_id
            ):
                raise NumericalCaptureError(
                    "capture row does not own its engine request ID"
                )
        return batch_engine_request_ids

    def _finish(
        self,
        *,
        arm: str,
        observations: Any,
        recovery_evidence: Any,
        source: Any,
        runtime: Any,
    ) -> dict[str, Any]:
        import collective_gate_common as common

        with self._lock:
            if self._finished:
                raise NumericalCaptureError("numerical capture finish called twice")
            selected, recovery, runtime_copy = self._validate_finish_evidence(
                arm=arm,
                observations=observations,
                recovery_evidence=recovery_evidence,
                source=source,
                runtime=runtime,
            )
            self._validate_capture_batch_ownership(self._captured)
            for sequence in EXPECTED_SEQUENCE_INDICES:
                member = self._captured[sequence]
                if member.logits_path is None or not member.logits_path.is_file():
                    raise NumericalCaptureError("capture member lacks raw logits")
            all_slots = [
                slot
                for sequence in EXPECTED_SEQUENCE_INDICES
                for slot in self._captured[sequence].slots
            ]
            if len(all_slots) != 4 * EXPECTED_PROMPT_TOKENS or len(
                set(all_slots)
            ) != len(all_slots):
                raise NumericalCaptureError("destination slots overlap across requests")

            family, member_evidence = self._recovery_contract(recovery)
            selected_by_sequence = {item["sequence_index"]: item for item in selected}
            members = []
            for sequence in EXPECTED_SEQUENCE_INDICES:
                captured = self._captured[sequence]
                evidence = member_evidence[sequence]
                observation = selected_by_sequence[sequence]
                tensor_file = _file_binding(captured.tensor_path, self.capture_root)
                logits_file = _file_binding(captured.logits_path, self.capture_root)
                members.append(
                    {
                        "sequence_index": sequence,
                        "trace_request_id": captured.binding.trace_request_id,
                        "external_request_id": observation["external_request_id"],
                        "engine_request_id": captured.engine_request_id,
                        "plan_member_index": evidence["plan_member_index"],
                        "capture_batch_row": captured.capture_batch_row,
                        "prompt_token_ids_sha256": (
                            captured.binding.prompt_token_ids_sha256
                        ),
                        "plan_id": family["plan_id"],
                        "reuse_group_id": family["reuse_group_id"],
                        "outcome": evidence["outcome"],
                        "fallback_reason": evidence["fallback_reason"],
                        "pic_executed": evidence["pic_executed"],
                        "prompt_token_count": EXPECTED_PROMPT_TOKENS,
                        "generated_token_count": 0,
                        "slots": list(captured.slots),
                        "physical_block_ids": list(captured.physical_block_ids),
                        "tensor_file": tensor_file,
                        "logits_file": logits_file,
                        "batch_engine_request_ids": list(
                            captured.batch_engine_request_ids
                        ),
                        "source_cache_shapes": [
                            list(shape) for shape in captured.cache_shapes
                        ],
                        "source_cache_strides": [
                            list(stride) for stride in captured.cache_strides
                        ],
                    }
                )

            expected_files = {
                self._captured[sequence].tensor_path.resolve()
                for sequence in EXPECTED_SEQUENCE_INDICES
            } | {
                self._captured[sequence].logits_path.resolve()
                for sequence in EXPECTED_SEQUENCE_INDICES
                if self._captured[sequence].logits_path is not None
            }
            present_files = {
                path.resolve() for path in self.capture_root.rglob("*.safetensors")
            }
            if present_files != expected_files:
                raise NumericalCaptureError(
                    "capture has missing or unlisted safetensors"
                )

            trace_binding = self._records_by_sequence[0]
            manifest = {
                "schema": CAPTURE_SCHEMA,
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "arm": self.arm,
                "capture_phase": self.phase,
                "capture_boundary": CAPTURE_BOUNDARY,
                "source": self.source,
                "model": {
                    "model_id": common.MODEL_PATH,
                    "revision": common.MODEL_REVISION,
                    "attention_backend": "TRITON_ATTN",
                },
                "trace": {
                    "sha256": common.EXPECTED_TRACE_SHA256,
                    "trace_id": trace_binding.trace_id,
                },
                "runtime": runtime_copy,
                "collective": family,
                "tensor_contract": self._tensor_contract(),
                "request_id_contract": REQUEST_ID_CONTRACT,
                "members": members,
                "comparison_contract": {
                    "serial_vs_collective": {
                        "kv": {"mode": "bitwise_exact", "atol": 0.0, "rtol": 0.0},
                        "logits": {"atol": 1e-5, "rtol": 1e-4},
                    },
                    "ordinary_vs_pic": {
                        "kv": {"atol": 0.05, "rtol": 0.05},
                        "logits": {"atol": 0.05, "rtol": 0.05},
                        "require_exact_top1_and_generated_token": True,
                    },
                },
                "observation_bindings": [
                    {
                        "sequence_index": item["sequence_index"],
                        "trace_request_id": item["trace_request_id"],
                        "external_request_id": item["external_request_id"],
                        "engine_request_id": self._captured[
                            item["sequence_index"]
                        ].engine_request_id,
                    }
                    for item in selected
                ],
                "claim_boundary": (
                    "Synchronized correctness evidence only. Capture overhead is "
                    "excluded from every latency or performance claim."
                ),
            }
            manifest_path = self.capture_root / "family-manifest.json"
            _atomic_json(manifest_path, manifest)
            manifest_digest = _sha256_file(manifest_path)
            self._finished = True
            self.ledger.emit(
                "gate.numerical.finish",
                arm=self.arm,
                status="complete",
                manifest_path=str(manifest_path.relative_to(self.output_dir)),
                manifest_sha256=manifest_digest,
                member_count=4,
            )
            return {
                "enabled": True,
                "status": "complete",
                "arm": self.arm,
                "capture_phase": self.phase,
                "manifest_path": CAPTURE_RELATIVE_PATH,
                "manifest_sha256": manifest_digest,
                "member_count": 4,
                "capture_boundary": CAPTURE_BOUNDARY,
            }

    def finish(
        self,
        *,
        arm: str,
        observations: Any,
        recovery_evidence: Any,
        source: Any,
        runtime: Any,
    ) -> dict[str, Any]:
        try:
            return self._finish(
                arm=arm,
                observations=observations,
                recovery_evidence=recovery_evidence,
                source=source,
                runtime=runtime,
            )
        finally:
            self._restore_hooks()


def install(
    *,
    ledger: Any,
    arm: str,
    output_dir: Path,
    records: tuple[dict[str, Any], ...],
    source: dict[str, Any],
) -> Any:
    """Install exact MRv2 numerical wrappers before engine construction."""
    if arm not in CAPTURED_ARMS:
        return DisabledNumericalInstrumentation(output_dir=output_dir, arm=arm)
    return NumericalInstrumentation(
        ledger=ledger,
        arm=arm,
        output_dir=output_dir,
        records=records,
        source=source,
    )
