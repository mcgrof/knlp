#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one crash-safe KMesh serial-PIC correctness arm on the pinned single-device profile."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SEED = 20260731
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_PATH = (
    "/scratch/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/" + MODEL_REVISION
)
EXPECTED_IMAGE = (
    "rocm/vllm-dev@sha256:"
    "9e531214b46ebabcfc0ee0888324533de0d90c28a828d5803f7b381179d8457c"
)
EXPECTED_TARGET_VOLUME = "kmesh-rocm723-target-lmcache-v2"
EXPECTED_VLLM_COMMIT = "18f90213f45783bad6b7acd7bc43cbba957a69f8"
EXPECTED_VLLM_TREE = "c9bed864a0b1082d9aba501a339de7d20cd29017"
EXPECTED_LMCACHE_COMMIT = "aa4984b57f78b9d1dce2e8d870ce7aa4716cd341"
EXPECTED_LMCACHE_TREE = "d2b3feeecae29d0b7e8366e817738966d2d267eb"
EXPECTED_EXTENSION_HASHES = {
    "_C": "e4959c2dde2fe8d2bf1511871f370284e22a94f97f4ba7e1d3b6e15ce7a68eeb",
    "_rocm_C": "e24cbcd7eb6f3a7e99783ab7fb84b1e49a389418f5fd8b93c9bc6a8164780461",
}
EXPECTED_TRACE_SHA256 = (
    "6b869dfafaab5216c5c41e6dcb5da676231243748ffb5b055098c761e5b404f3"
)
EXPECTED_MANIFEST_SHA256 = (
    "2d78e3f4124cff8e93bd3e2bec77ff3402c5eecd14a3b743a88fe0820041f3d0"
)
EXPECTED_TRACE_ID = "kmesh-5f23ea6874fafa9e"
EXPECTED_ORACLE_AGGREGATE = {
    "compatible_pair_count": 10,
    "eligible_prompt_aligned_full_blocks_total": 0,
    "eligible_segment_relative_full_blocks_total": 110,
    "eligible_shared_residual_tokens_total": 640,
    "eligible_structural_shared_tokens_total": 4160,
    "ordered_shared_prefix_tokens_total": 4160,
    "pair_count": 10,
    "raw_prompt_aligned_full_blocks_total": 0,
    "raw_segment_relative_full_blocks_total": 110,
    "raw_shared_residual_tokens_total": 640,
    "raw_structural_shared_tokens_total": 4160,
    "same_absolute_offset_shared_tokens_total": 0,
}
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.11.0+gitd0c8b1f",
    "hip": "7.2.53211",
    "transformers": "5.14.1",
    "tokenizers": "0.23.0rc0",
    "sortedcontainers": "2.4.0",
    "aiofile": "3.9.0",
    "caio": "0.9.25",
    "target_volume": EXPECTED_TARGET_VOLUME,
    "wheel_vllm_distribution": "0.26.1rc1.dev221+g454ea5b52.rocm723",
    "overlay_vllm_version": "0.26.1rc1.dev221+g454ea5b52",
    "accelerator_count": 1,
    "device_arch": "gfx1100",
    "device_total_memory": 48301604864,
}
TRACE_CONFIG = {
    "agents": 5,
    "alignment": "unaligned",
    "block_size": 32,
    "case": "same_order_different_offsets",
    "mode": "prefill_focus",
    "nominal_shared_fraction": 0.8,
    "qps": 2,
    "seed": SEED,
    "tokens_per_agent": 512,
}

# Runtime selection changes source and accelerator pins, never the workload
# or the numerical acceptance policy.
CURRENT_CUDA = os.environ.get("TOKENDANCE_PROFILE") in ("current-cuda", "rtx4090-cuda")
if os.environ.get("TOKENDANCE_PROFILE", "historical") not in (
    "historical",
    "current-cuda",
    "rtx4090-cuda",
):
    raise RuntimeError("unknown TokenDance runtime profile")
if CURRENT_CUDA:
    _profile = json.loads(
        Path(__file__).resolve().parents[1].joinpath("profile.json").read_text()
    )
    _source_pair = (
        "rtx4090" if os.environ["TOKENDANCE_PROFILE"] == "rtx4090-cuda" else "current"
    )
    _current = _profile["rtx4090_cuda" if _source_pair == "rtx4090" else "current_cuda"]
    EXPECTED_IMAGE = _current["image"]
    EXPECTED_TARGET_VOLUME = _current["target_volume"]
    EXPECTED_RUNTIME = _current["runtime"]
    EXPECTED_EXTENSION_HASHES = _current["extension_hashes"]
    _pins = _profile["source_pairs"][_source_pair]
    EXPECTED_VLLM_COMMIT = _pins["vllm"]["commit"]
    EXPECTED_VLLM_TREE = _pins["vllm"]["tree"]
    EXPECTED_LMCACHE_COMMIT = _pins["lmcache"]["commit"]
    EXPECTED_LMCACHE_TREE = _pins["lmcache"]["tree"]
    MODEL_PATH = os.environ["TOKENDANCE_MODEL_DIR"]


class GateError(RuntimeError):
    """A gate input or observation violated the frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if type(value).__module__ == "torch" and hasattr(value, "numel"):
        result = {
            "type": type(value).__qualname__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "numel": int(value.numel()),
        }
        if str(value.dtype) == "torch.bool":
            result["true_count"] = int(value.sum().item())
        return result
    return repr(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class EventLedger:
    """Append-only, fsync-per-transition evidence from the actual runtime path."""

    def __init__(self, path: Path):
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        _fsync_directory(path.parent)
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._sequence = 0
        self._phase = "construction"
        self._started_ns = time.monotonic_ns()
        self._closed = False

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
        self.emit("gate.phase", selected_phase=phase)

    def emit(self, action: str, **details: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("event ledger is closed")
            event = {
                "schema_version": SCHEMA_VERSION,
                "sequence": self._sequence,
                "monotonic_ns": time.monotonic_ns() - self._started_ns,
                "captured_at_utc": _utc_now(),
                "thread": threading.current_thread().name,
                "phase": self._phase,
                "action": action,
                **_jsonable(details),
            }
            self._sequence += 1
            self._events.append(event)
            self._stream.write(json.dumps(event, sort_keys=True) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True


def _git_output(tree: Path, *arguments: str) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={tree}",
            "-C",
            str(tree),
            *arguments,
        ],
        text=True,
    ).strip()


def _source_identity(
    tree: Path, expected_commit: str, expected_tree: str
) -> dict[str, Any]:
    identity = {
        "path": str(tree),
        "commit": _git_output(tree, "rev-parse", "HEAD"),
        "tree": _git_output(tree, "rev-parse", "HEAD^{tree}"),
        "status_porcelain": _git_output(tree, "status", "--porcelain"),
    }
    if identity["commit"] != expected_commit:
        raise GateError(f"unexpected commit for {tree}: {identity['commit']}")
    if identity["tree"] != expected_tree:
        raise GateError(f"unexpected tree for {tree}: {identity['tree']}")
    if identity["status_porcelain"]:
        raise GateError(f"source worktree is dirty: {tree}")
    return identity


def _runtime_identity(
    vllm: Any,
    lmcache: Any,
    torch: Any,
    transformers: Any,
    extension_paths: dict[str, Path],
    extension_hashes: dict[str, str],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    runtime = {
        "captured_at_utc": _utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
        "sortedcontainers": importlib.metadata.version("sortedcontainers"),
        "aiofile": importlib.metadata.version("aiofile"),
        "caio": importlib.metadata.version("caio"),
        "target_volume": os.environ.get("KMESH_TARGET_VOLUME"),
        "wheel_vllm_distribution": importlib.metadata.version("vllm"),
        "overlay_vllm_version": vllm.__version__,
        "overlay_vllm_module": vllm.__file__,
        "overlay_lmcache_module": lmcache.__file__,
        "accelerator_count": torch.cuda.device_count(),
        "device_arch": (
            f"sm_{properties.major}{properties.minor}"
            if torch.version.hip is None
            else properties.gcnArchName
        ),
        "device_total_memory": properties.total_memory,
        "image": EXPECTED_IMAGE,
        "compiled_extensions": {
            name: {"path": str(extension_paths[name]), "sha256": digest}
            for name, digest in extension_hashes.items()
        },
    }
    if CURRENT_CUDA:
        runtime["lmcache_compiled_extensions"] = {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name in ("lmcache_native", "cuda_ops")
            for path in [Path(importlib.import_module(f"lmcache.{name}").__file__)]
        }
    observed = {name: runtime.get(name) for name in EXPECTED_RUNTIME}
    if observed != EXPECTED_RUNTIME:
        raise GateError(
            "pinned single-device runtime identity drifted: "
            + json.dumps({"expected": EXPECTED_RUNTIME, "observed": observed})
        )
    return runtime


def _load_vllm_overlay(source: Path, wheel_package: Path) -> Any:
    source_package = source / "vllm"
    existing = sys.modules.get("vllm")
    if existing is not None:
        if (
            Path(existing.__file__).resolve()
            != (source_package / "__init__.py").resolve()
        ):
            raise GateError("a different vLLM package was imported before the overlay")
        return existing
    spec = importlib.util.spec_from_file_location(
        "vllm",
        source_package / "__init__.py",
        submodule_search_locations=[str(source_package), str(wheel_package)],
    )
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load vLLM overlay from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm"] = module
    spec.loader.exec_module(module)
    return module


def _load_trace(
    trace_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise GateError("manifest must be a JSON object")
    trace_hash = _sha256(trace_path)
    manifest_hash = _sha256(manifest_path)
    if trace_hash != EXPECTED_TRACE_SHA256:
        raise GateError("trace SHA-256 differs from the pinned gate input")
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise GateError("manifest SHA-256 differs from the pinned gate input")
    if manifest.get("trace_sha256") != trace_hash:
        raise GateError("trace SHA-256 differs from its manifest")
    if manifest.get("config") != TRACE_CONFIG:
        raise GateError("trace configuration differs from the frozen gate")
    aggregate = manifest.get("oracle", {}).get("aggregate", {})
    if aggregate != EXPECTED_ORACLE_AGGREGATE:
        raise GateError("trace structural oracle differs from the frozen gate")

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != 5:
        raise GateError("the frozen trace must contain five agents")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("sequence_index") != index:
            raise GateError("trace sequence is not canonical")
        if record.get("trace_id") != EXPECTED_TRACE_ID:
            raise GateError("trace record carries an unexpected trace ID")
        tokens = record.get("prompt_token_ids")
        context = record.get("kmesh_context")
        if not isinstance(tokens, list) or len(tokens) != 512:
            raise GateError("every frozen prompt must contain 512 tokens")
        if (
            not isinstance(context, dict)
            or context.get("segment_encoding") != "sideband"
        ):
            raise GateError("trace lacks canonical KMesh sideband metadata")
        spans = context.get("segments")
        if not isinstance(spans, list) or not any(
            span.get("shareable") is True for span in spans if isinstance(span, dict)
        ):
            raise GateError("trace request has no shareable span")
    return manifest, records


def _span_summary(value: Any) -> Any:
    if value is None:
        return None
    cached = getattr(value, "cached_span", None)
    spans = getattr(value, "spans", None)
    return {
        "token_count": getattr(value, "token_count", None),
        "cached_span": (
            [getattr(cached, "start", None), getattr(cached, "end", None)]
            if cached is not None
            else None
        ),
        "spans": (
            [
                [getattr(span, "start", None), getattr(span, "end", None)]
                for span in spans
            ]
            if spans is not None
            else None
        ),
    }


def _owner_state(owner: Any) -> dict[str, Any]:
    role = getattr(owner, "_role", None)
    state: dict[str, Any] = {
        "owner_type": type(owner).__qualname__,
        "role": getattr(role, "name", str(role) if role is not None else None),
    }
    for name in (
        "_kmesh_store_sessions",
        "_kmesh_outstanding_lookup_ids",
        "_kmesh_store_cleanup_errors",
        "_kmesh_failed_worker_requests",
        "_prealloc_lookup_ids",
        "_allocated_lookup_ids",
        "_worker_owned_lookup_requests",
        "lookup_cleanup_registered",
        "lookup_cleanup_pending",
        "lookup_pins",
    ):
        value = getattr(owner, name, None)
        if value is not None:
            try:
                state[f"{name}_count"] = len(value)
            except TypeError:
                state[name] = _jsonable(value)
    for name in ("generation", "completed_layers", "flushed", "closed"):
        if hasattr(owner, name):
            state[name] = _jsonable(getattr(owner, name))
    expected_layer_names = getattr(owner, "expected_layer_names", None)
    if expected_layer_names is not None:
        state["expected_layer_names_count"] = len(expected_layer_names)
    manager = getattr(owner, "_manager", None)
    if manager is not None:
        state["manager_services_stopped"] = getattr(manager, "services_stopped", None)
    return state


def _call_details(
    name: str, owner: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    details = _owner_state(owner)
    request = args[0] if args and hasattr(args[0], "request_id") else None
    if request is not None:
        details["request_id"] = str(request.request_id)
    if name == "prepare_request_lookup" and len(args) > 1:
        details["sidecar_count"] = len(args[1])
    elif name == "get_num_new_matched_tokens" and len(args) > 1:
        details["num_computed_tokens"] = args[1]
    elif name == "update_state_after_alloc" and len(args) > 1:
        details["num_external_tokens"] = args[1]
    elif name in {"save_kv_layer", "wait_for_layer_load"} and args:
        details["layer_name"] = args[0]
    elif name == "lookup":
        token_ids = args[0] if args else kwargs.get("token_ids")
        details["token_count"] = len(token_ids)
        details["lookup_id"] = args[1] if len(args) > 1 else kwargs.get("lookup_id")
        explicit = args[3] if len(args) > 3 else kwargs.get("explicit_spans")
        details["explicit_spans"] = _span_summary(explicit)
    elif name in {"transfer_lookups"} and args:
        details["lookup_ids"] = list(args[0])
    elif name in {"cancel_lookup", "finish_lookup"} and args:
        details["lookup_id"] = args[0]
    elif (
        name
        in {
            "register_lookup_cleanup",
            "schedule_lookup_unpin",
            "lookup_unpin",
        }
        and args
    ):
        details["lookup_id"] = args[0]
    elif name == "_record_kmesh_load_failure" and len(args) > 1:
        contract = args[0]
        details["request_id"] = getattr(contract, "request_id", None)
        details["reason"] = args[1]
        invalidate_full_prompt = kwargs.get("invalidate_full_prompt", False)
        details["invalidate_full_prompt"] = invalidate_full_prompt
        slot_mapping = tuple(getattr(contract, "slot_mapping", ()))
        details["contract_slot_count"] = len(slot_mapping)
        details["advertised_tokens"] = getattr(contract, "advertised_tokens", None)
        details["lookup_id"] = getattr(contract, "lookup_id", None)
    elif name in {"store_layer", "retrieve_layer"}:
        token_ids = args[0] if args else kwargs.get("tokens")
        if token_ids is not None:
            details["token_count"] = len(token_ids)
        details["explicit_spans"] = _span_summary(kwargs.get("explicit_spans"))
        details["request_id"] = kwargs.get("request_id") or kwargs.get("req_id")
    return details


def _return_summary(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str, set, frozenset)):
        return _jsonable(value)
    if type(value).__module__ == "torch" and hasattr(value, "numel"):
        return _jsonable(value)
    return {"type": type(value).__qualname__}


def _install_instrumentation(
    ledger: EventLedger, *, inject_rollback: bool
) -> dict[str, Any]:
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl,
        _KMeshLayerwiseStoreSession,
    )
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.compute.blend.blender import LMCBlender
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient

    def wrap(owner_type: type[Any], name: str) -> None:
        original = getattr(owner_type, name)
        action = f"{owner_type.__name__}.{name}"

        if name in {"store_layer", "retrieve_layer"}:

            @functools.wraps(original)
            def wrapped_generator(owner: Any, *args: Any, **kwargs: Any) -> Any:
                details = _call_details(name, owner, args, kwargs)
                ledger.emit(f"{action}.begin", **details)
                try:
                    result = yield from original(owner, *args, **kwargs)
                except GeneratorExit:
                    close_details = {**details, **_owner_state(owner)}
                    ledger.emit(f"{action}.closed", **close_details)
                    raise
                except BaseException as error:
                    ledger.emit(
                        f"{action}.error",
                        **details,
                        exception_type=type(error).__name__,
                        message=str(error),
                    )
                    raise
                end_details = {**details, **_owner_state(owner)}
                ledger.emit(
                    f"{action}.end",
                    **end_details,
                    result=_return_summary(result),
                )
                return result

            if not inspect.isgeneratorfunction(wrapped_generator):
                raise GateError(f"instrumented {action} is not a generator")
            setattr(owner_type, name, wrapped_generator)
            return

        @functools.wraps(original)
        def wrapped(owner: Any, *args: Any, **kwargs: Any) -> Any:
            details = _call_details(name, owner, args, kwargs)
            ledger.emit(f"{action}.begin", **details)
            try:
                result = original(owner, *args, **kwargs)
            except BaseException as error:
                ledger.emit(
                    f"{action}.error",
                    **details,
                    exception_type=type(error).__name__,
                    message=str(error),
                )
                raise
            end_details = {**details, **_owner_state(owner)}
            ledger.emit(
                f"{action}.end",
                **end_details,
                result=_return_summary(result),
            )
            return result

        setattr(owner_type, name, wrapped)

    for method in (
        "prepare_request_lookup",
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
        "wait_for_save",
        "request_finished",
        "_record_kmesh_load_failure",
        "get_block_ids_with_load_errors",
        "shutdown",
    ):
        wrap(LMCacheConnectorV1Impl, method)
    for method in (
        "lookup",
        "transfer_lookups",
        "cancel_lookup",
        "finish_lookup",
        "close",
    ):
        wrap(LMCacheLookupClient, method)
    for method in (
        "store_layer",
        "retrieve_layer",
        "drain_pending_store_cleanups",
        "register_lookup_cleanup",
        "schedule_lookup_unpin",
        "lookup_unpin",
    ):
        wrap(LMCacheEngine, method)
    wrap(_KMeshLayerwiseStoreSession, "finish")

    original_blend = LMCBlender.blend
    injection = {"enabled": inject_rollback, "count": 0}

    @functools.wraps(original_blend)
    def blend(owner: Any, *args: Any, **kwargs: Any) -> Any:
        token_ids = args[0] if args else kwargs.get("tokens")
        details = {
            "token_count": len(token_ids),
            "request_id": kwargs.get("request_id"),
            "lookup_id": kwargs.get("req_id"),
            "generation": kwargs.get("generation"),
            "explicit_spans": _span_summary(kwargs.get("explicit_spans")),
        }
        ledger.emit("LMCBlender.blend.begin", **details)
        try:
            original_result = original_blend(owner, *args, **kwargs)
        except BaseException as error:
            ledger.emit(
                "LMCBlender.blend.error",
                **details,
                exception_type=type(error).__name__,
                message=str(error),
            )
            raise
        original_true_count = int(original_result.sum().item())
        result = original_result
        injected = False
        if (
            inject_rollback
            and ledger.phase == "rollback_probe"
            and injection["count"] == 0
        ):
            import torch

            if (
                type(original_result) is not torch.Tensor
                or original_result.dtype != torch.bool
            ):
                raise GateError("rollback boundary did not return an exact bool tensor")
            slot_mapping = kwargs.get("slot_mapping")
            if (
                type(slot_mapping) is not torch.Tensor
                or slot_mapping.dtype != torch.long
                or slot_mapping.ndim != 1
                or len(slot_mapping) != len(token_ids)
                or bool(torch.any(slot_mapping < 0).item())
                or len(torch.unique(slot_mapping)) != len(slot_mapping)
            ):
                raise GateError("rollback boundary lost its destination slot mapping")
            result = torch.zeros_like(original_result)
            injection["count"] += 1
            injected = True
            injected_slot_mapping = [
                int(slot) for slot in slot_mapping.to(device="cpu").tolist()
            ]
        else:
            injected_slot_mapping = None
        ledger.emit(
            "LMCBlender.blend.end",
            **details,
            original_true_count=original_true_count,
            returned_true_count=int(result.sum().item()),
            injected_all_false=injected,
            injected_slot_mapping=injected_slot_mapping,
        )
        return result

    LMCBlender.blend = blend
    return injection


def _kv_transfer_config() -> dict[str, Any]:
    config = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "engine_id": "kmesh-serial-pic-v1",
        "kv_buffer_device": "cuda",
        "kv_buffer_size": 1e9,
        "kv_rank": None,
        "kv_parallel_size": 1,
        "kv_ip": "127.0.0.1",
        "kv_port": 14579,
        "kv_connector_module_path": None,
        "enable_permute_local_kv": False,
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "kmesh_target_profile": "serial_pic_v1",
            "connectors": [
                {
                    "kv_connector": "KMeshConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "kmesh": {
                            "enabled": True,
                            "mode": "serial_pic",
                            "segment_encoding": "sideband",
                            "pic_backend": "lmcache",
                            "check_layer": 1,
                            "recompute_ratio": 0.15,
                            "max_group_size": 10,
                            "max_group_wait_ms": 0,
                            "require_same_active_length": True,
                            "fail_open": False,
                            "strict_validation": True,
                            "master_selection": "min_total_changed_blocks",
                            "storage_mode": "none",
                        }
                    },
                },
                {
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"use_native": False},
                },
            ],
        },
    }
    if CURRENT_CUDA:
        for field in ("kv_buffer_size", "kv_rank", "kv_parallel_size", "kv_ip"):
            del config[field]
    return config


def _engine_kwargs(*, connectors: bool) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": MODEL_PATH,
        "tokenizer": MODEL_PATH,
        "served_model_name": MODEL_PATH,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "seed": SEED,
        "enforce_eager": True,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.3,
        "kv_cache_dtype": "auto",
        "block_size": 32,
        "enable_prefix_caching": False,
        "disable_cascade_attn": True,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
        "disable_hybrid_kv_cache_manager": True,
        "distributed_executor_backend": "uni",
        "disable_custom_all_reduce": True,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 4,
        "trust_remote_code": False,
        "disable_log_stats": False,
    }
    if connectors:
        options["kv_transfer_config"] = _kv_transfer_config()
    return options


def _baseline_outputs(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("arm") != "ordinary" or value.get("status") != "complete":
        raise GateError("baseline result is not a completed ordinary arm")
    if value.get("checks", {}).get("all_passed") is not True:
        raise GateError("baseline result did not pass its checks")
    outputs = {}
    for observation in value.get("observations", []):
        outputs[observation["trace_request_id"]] = observation["output_token_ids"]
    if len(outputs) != 5:
        raise GateError("baseline result does not cover all five trace requests")
    return outputs


def _run_request(
    llm: Any,
    record: dict[str, Any],
    phase: str,
    ledger: EventLedger,
    *,
    connectors: bool,
) -> dict[str, Any]:
    from vllm import SamplingParams, TokensPrompt

    ledger.set_phase(phase)
    sampling_options: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 1,
        "seed": SEED,
    }
    if connectors:
        sampling_options["extra_args"] = {"kmesh": record["kmesh_context"]}
    started = time.perf_counter()
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=record["prompt_token_ids"])],
        SamplingParams(**sampling_options),
        use_tqdm=False,
    )[0]
    elapsed = time.perf_counter() - started
    completion = output.outputs[0]
    observation = {
        "phase": phase,
        "trace_request_id": record["request_id"],
        "sequence_index": record["sequence_index"],
        "prompt_tokens": len(output.prompt_token_ids),
        "num_cached_tokens": output.num_cached_tokens,
        "output_token_ids": list(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "elapsed_seconds": elapsed,
    }
    ledger.emit("gate.request.complete", **observation)
    return observation


def _events(
    ledger_events: list[dict[str, Any]], action: str, phase: str | None = None
) -> list[dict[str, Any]]:
    return [
        event
        for event in ledger_events
        if event.get("action") == action
        and (phase is None or event.get("phase") == phase)
    ]


def _events_with_phase_prefix(
    ledger_events: list[dict[str, Any]], action: str, phase_prefix: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in ledger_events
        if event.get("action") == action
        and str(event.get("phase", "")).startswith(phase_prefix)
    ]


def _check_parity(
    observations: list[dict[str, Any]], baseline: dict[str, list[int]]
) -> bool:
    return all(
        observation["output_token_ids"] == baseline.get(observation["trace_request_id"])
        for observation in observations
    )


def _evaluate(
    arm: str,
    observations: list[dict[str, Any]],
    events: list[dict[str, Any]],
    baseline: dict[str, list[int]],
    injection: dict[str, Any] | None,
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "observations_present": bool(observations),
        "one_generated_token_per_observation": all(
            len(observation["output_token_ids"]) == 1 for observation in observations
        ),
        "no_instrumented_runtime_errors": not any(
            str(event.get("action", "")).endswith(".error") for event in events
        ),
    }
    construction = _events(events, "gate.construction.end")

    def clean_prime() -> bool:
        prime_observations = [
            observation
            for observation in observations
            if observation.get("phase") == "prime"
        ]
        prime_lookups = _events(events, "LMCacheLookupClient.lookup.end", "prime")
        prime_finishes = _events(
            events, "_KMeshLayerwiseStoreSession.finish.end", "prime"
        )
        return (
            len(prime_observations) == 1
            and not prime_observations[0].get("num_cached_tokens")
            and len(prime_lookups) == 1
            and prime_lookups[0].get("result") == 0
            and not _events(events, "LMCBlender.blend.end", "prime")
            and len(
                _events(
                    events,
                    "LMCacheConnectorV1Impl.wait_for_save.end",
                    "prime",
                )
            )
            == 1
            and len(prime_finishes) == 1
            and prime_finishes[0].get("result") is True
            and prime_finishes[0].get("flushed") is True
            and prime_finishes[0].get("closed") is True
            and prime_finishes[0].get("completed_layers")
            == prime_finishes[0].get("expected_layer_names_count")
        )

    if arm == "ordinary":
        checks.update(
            {
                "all_five_trace_requests_executed": len(observations) == 5,
                "resolved_connectors_are_disabled": len(construction) == 1
                and construction[0].get("kv_transfer_config_present") is False,
                "ordinary_cached_tokens_are_zero": all(
                    not observation["num_cached_tokens"] for observation in observations
                ),
            }
        )
    else:
        checks["exact_output_parity"] = _check_parity(observations, baseline)
        checks["resolved_target_connector_is_present"] = len(construction) == 1 and (
            construction[0].get("kv_transfer_config_present") is True
        )
        shutdowns = _events(events, "LMCacheConnectorV1Impl.shutdown.end")
        worker_shutdowns = [
            event for event in shutdowns if event.get("role") == "WORKER"
        ]
        scheduler_shutdowns = [
            event for event in shutdowns if event.get("role") == "SCHEDULER"
        ]
        checks["both_connector_roles_shut_down"] = bool(worker_shutdowns) and bool(
            scheduler_shutdowns
        )
        checks["all_managers_report_services_stopped"] = bool(shutdowns) and all(
            event.get("manager_services_stopped") is True for event in shutdowns
        )
        checks["worker_shutdown_has_no_residue"] = bool(worker_shutdowns) and all(
            all(
                event.get(field) == 0
                for field in (
                    "_kmesh_store_sessions_count",
                    "_kmesh_outstanding_lookup_ids_count",
                    "_kmesh_store_cleanup_errors_count",
                    "_kmesh_failed_worker_requests_count",
                )
            )
            for event in worker_shutdowns
        )
        checks["scheduler_shutdown_has_no_ownership_residue"] = bool(
            scheduler_shutdowns
        ) and all(
            all(
                event.get(field) == 0
                for field in (
                    "_prealloc_lookup_ids_count",
                    "_allocated_lookup_ids_count",
                    "_worker_owned_lookup_requests_count",
                )
            )
            for event in scheduler_shutdowns
        )
        transfers = _events(events, "LMCacheLookupClient.transfer_lookups.end")
        transferred_ids = [
            lookup_id
            for event in _events(events, "LMCacheLookupClient.transfer_lookups.begin")
            for lookup_id in event.get("lookup_ids", [])
        ]
        checks["lookup_transfers_are_unique"] = bool(transfers) and len(
            transferred_ids
        ) == len(set(transferred_ids))

    if arm == "cold_miss":
        lookup_ends = _events(events, "LMCacheLookupClient.lookup.end", "cold_miss")
        checks.update(
            {
                "one_probe_executed": len(observations) == 1,
                "scheduler_observed_zero_hit": bool(lookup_ends)
                and all(event.get("result") == 0 for event in lookup_ends),
                "request_reported_zero_cached_tokens": all(
                    not observation["num_cached_tokens"] for observation in observations
                ),
                "worker_cacheblend_not_called": not _events(
                    events, "LMCBlender.blend.end", "cold_miss"
                ),
            }
        )
    elif arm == "positive_hit":
        probe_phases = [f"probe_{index:03d}" for index in range(1, 5)]
        probes = [item for item in observations if item.get("phase") in probe_phases]
        blends = _events_with_phase_prefix(events, "LMCBlender.blend.end", "probe_")
        invalidations = _events_with_phase_prefix(
            events,
            "LMCacheConnectorV1Impl._record_kmesh_load_failure.end",
            "probe_",
        )
        probe_lookup_ids: list[str] = []
        per_probe_lifecycle: list[bool] = []
        for phase in probe_phases:
            phase_observations = [item for item in probes if item.get("phase") == phase]
            lookups = _events(events, "LMCacheLookupClient.lookup.end", phase)
            lookup_id = lookups[0].get("lookup_id") if len(lookups) == 1 else None
            if isinstance(lookup_id, str):
                probe_lookup_ids.append(lookup_id)
            transfers = _events(
                events, "LMCacheLookupClient.transfer_lookups.begin", phase
            )
            phase_blends = _events(events, "LMCBlender.blend.end", phase)
            registrations = _events(
                events, "LMCacheEngine.register_lookup_cleanup.end", phase
            )
            cleanups = _events(events, "LMCacheEngine.schedule_lookup_unpin.end", phase)
            finishes = _events(events, "_KMeshLayerwiseStoreSession.finish.end", phase)
            per_probe_lifecycle.append(
                len(phase_observations) == 1
                and len(lookups) == 1
                and isinstance(lookup_id, str)
                and lookups[0].get("result", 0) > 0
                and bool((lookups[0].get("explicit_spans") or {}).get("spans"))
                and len(transfers) == 1
                and transfers[0].get("lookup_ids") == [lookup_id]
                and len(
                    _events(events, "LMCacheConnectorV1Impl.start_load_kv.end", phase)
                )
                == 1
                and len(phase_blends) == 1
                and phase_blends[0].get("lookup_id") == lookup_id
                and phase_blends[0].get("original_true_count", 0) > 0
                and phase_blends[0].get("injected_all_false") is False
                and len(
                    _events(
                        events,
                        "LMCacheConnectorV1Impl.wait_for_save.end",
                        phase,
                    )
                )
                == 1
                and len(finishes) == 1
                and finishes[0].get("result") is True
                and any(event.get("lookup_id") == lookup_id for event in registrations)
                and any(
                    event.get("lookup_id") == lookup_id
                    and event.get("lookup_cleanup_registered_count") == 0
                    and event.get("lookup_cleanup_pending_count") == 0
                    and event.get("lookup_pins_count") == 0
                    for event in cleanups
                )
            )
        checks.update(
            {
                "one_prime_and_four_off_position_probes": len(observations) == 5
                and len(probes) == 4,
                "prime_is_a_clean_zero_hit_store": clean_prime(),
                "every_probe_advertised_a_hit": bool(probes)
                and all(item["num_cached_tokens"] > 0 for item in probes),
                "each_probe_has_one_correlated_online_lifecycle": len(
                    per_probe_lifecycle
                )
                == 4
                and all(per_probe_lifecycle),
                "probe_lookup_ids_are_unique": len(probe_lookup_ids) == 4
                and len(set(probe_lookup_ids)) == 4,
                "no_probe_mask_was_injected": all(
                    event.get("injected_all_false") is False for event in blends
                ),
                "no_probe_load_invalidation": not invalidations,
            }
        )
    elif arm == "rollback":
        rollback_observations = [
            item for item in observations if item["phase"] == "rollback_probe"
        ]
        retry_observations = [
            item for item in observations if item["phase"] == "rollback_retry"
        ]
        injected = [
            event
            for event in _events(events, "LMCBlender.blend.end", "rollback_probe")
            if event.get("injected_all_false") is True
        ]
        failures = _events(
            events,
            "LMCacheConnectorV1Impl._record_kmesh_load_failure.end",
            "rollback_probe",
        )
        invalid_blocks = _events(
            events,
            "LMCacheConnectorV1Impl.get_block_ids_with_load_errors.end",
            "rollback_probe",
        )
        rollback_advertised_tokens = (
            rollback_observations[0].get("num_cached_tokens", 0)
            if len(rollback_observations) == 1
            else 0
        )
        injected_slot_mapping = (
            injected[0].get("injected_slot_mapping") if len(injected) == 1 else None
        )
        independent_boundary_valid = (
            type(rollback_advertised_tokens) is int
            and rollback_advertised_tokens > 0
            and isinstance(injected_slot_mapping, list)
            and len(rollback_observations) == 1
            and len(injected_slot_mapping) == rollback_observations[0]["prompt_tokens"]
            and all(type(slot) is int and slot >= 0 for slot in injected_slot_mapping)
            and len(set(injected_slot_mapping)) == len(injected_slot_mapping)
            and rollback_advertised_tokens <= len(injected_slot_mapping)
        )
        expected_invalid_block_ids = {
            int(slot) // 32
            for slot in (
                injected_slot_mapping[:rollback_advertised_tokens]
                if independent_boundary_valid
                else []
            )
        }
        observed_invalid_block_ids = {
            block_id
            for event in invalid_blocks
            for block_id in (event.get("result") or [])
        }
        nonempty_invalid_blocks = [
            event for event in invalid_blocks if event.get("result")
        ]
        exact_nonempty_invalidation = (
            len(nonempty_invalid_blocks) == 1
            and isinstance(nonempty_invalid_blocks[0].get("result"), list)
            and len(nonempty_invalid_blocks[0]["result"])
            == len(expected_invalid_block_ids)
            and set(nonempty_invalid_blocks[0]["result"]) == expected_invalid_block_ids
        )
        invalidation_followed_failure = (
            len(failures) == 1
            and len(nonempty_invalid_blocks) == 1
            and type(failures[0].get("sequence")) is int
            and type(nonempty_invalid_blocks[0].get("sequence")) is int
            and nonempty_invalid_blocks[0]["sequence"] > failures[0]["sequence"]
        )
        retry_blends = _events(events, "LMCBlender.blend.end", "rollback_retry")
        checks.update(
            {
                "prime_is_a_clean_zero_hit_store": clean_prime(),
                "one_rollback_and_one_later_retry": len(rollback_observations) == 1
                and len(retry_observations) == 1,
                "exactly_one_all_false_injection": injection is not None
                and injection.get("count") == 1
                and len(injected) == 1,
                "injection_followed_a_real_positive_retrieve": len(injected) == 1
                and injected[0].get("original_true_count", 0) > 0
                and injected[0].get("returned_true_count") == 0,
                "rollback_probe_advertised_a_hit": len(rollback_observations) == 1
                and type(rollback_observations[0].get("num_cached_tokens")) is int
                and rollback_observations[0]["num_cached_tokens"] > 0,
                "one_narrow_injected_failure_was_recorded": len(failures) == 1
                and len(injected) == 1
                and failures[0].get("request_id") == injected[0].get("request_id")
                and failures[0].get("lookup_id") == injected[0].get("lookup_id")
                and failures[0].get("invalidate_full_prompt") is False
                and failures[0].get("reason")
                == "explicit lookup/retrieve race produced an all-false mask"
                and len(rollback_observations) == 1
                and failures[0].get("advertised_tokens")
                == rollback_observations[0].get("num_cached_tokens"),
                "exactly_all_advertised_blocks_were_invalidated": bool(
                    expected_invalid_block_ids
                )
                and independent_boundary_valid
                and all(
                    isinstance(event.get("result"), list) for event in invalid_blocks
                )
                and exact_nonempty_invalidation
                and invalidation_followed_failure
                and observed_invalid_block_ids == expected_invalid_block_ids,
                "affected_request_store_was_suppressed": not _events(
                    events, "LMCacheEngine.store_layer.begin", "rollback_probe"
                ),
                "later_retry_consumed_cacheblend_tokens": len(retry_blends) == 1
                and retry_blends[0].get("original_true_count", 0) > 0
                and retry_blends[0].get("injected_all_false") is False
                and len(retry_observations) == 1
                and retry_observations[0].get("num_cached_tokens", 0) > 0,
            }
        )
    checks["all_passed"] = all(checks.values())
    return checks


def _cleanup_llm(llm: Any, ledger: EventLedger) -> None:
    ledger.set_phase("shutdown")
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    ledger.emit("gate.shutdown.complete")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=("ordinary", "cold_miss", "positive_hit", "rollback"),
    )
    parser.add_argument("--vllm-source", required=True, type=Path)
    parser.add_argument("--lmcache-source", required=True, type=Path)
    parser.add_argument("--wheel-vllm-package", required=True, type=Path)
    parser.add_argument("--lmcache-config", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-result", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.arm != "ordinary" and args.baseline_result is None:
        raise GateError("connector arms require --baseline-result")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise GateError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "arm": args.arm,
            "started_at_utc": _utc_now(),
            "pid": os.getpid(),
        },
    )
    ledger = EventLedger(args.output_dir / "events.jsonl")
    llm: Any = None
    cleaned = False
    ledger_closed = False
    try:
        if os.environ.get("KMESH_IMAGE_DIGEST") != EXPECTED_IMAGE:
            raise GateError("container image digest evidence is absent or unexpected")
        allowed_lmcache = {"LMCACHE_CONFIG_FILE"}
        unexpected_lmcache = sorted(
            name
            for name in os.environ
            if name.startswith("LMCACHE_") and name not in allowed_lmcache
        )
        if unexpected_lmcache:
            raise GateError(
                "unreviewed LMCache environment overrides: "
                + ", ".join(unexpected_lmcache)
            )
        os.environ["LMCACHE_CONFIG_FILE"] = str(args.lmcache_config)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        os.environ.setdefault("PYTHONHASHSEED", "0")

        source = {
            "vllm": _source_identity(
                args.vllm_source, EXPECTED_VLLM_COMMIT, EXPECTED_VLLM_TREE
            ),
            "lmcache": _source_identity(
                args.lmcache_source, EXPECTED_LMCACHE_COMMIT, EXPECTED_LMCACHE_TREE
            ),
        }
        manifest, records = _load_trace(args.trace, args.manifest)
        baseline = _baseline_outputs(args.baseline_result)
        sys.path.insert(0, str(args.lmcache_source))
        vllm = _load_vllm_overlay(args.vllm_source, args.wheel_vllm_package)
        import lmcache
        import torch
        import transformers
        from lmcache.v1.config import LMCacheEngineConfig
        from vllm import LLM

        extension_paths = {
            name: Path(importlib.import_module(f"vllm.{name}").__file__)
            for name in EXPECTED_EXTENSION_HASHES
        }
        extension_hashes = {
            name: _sha256(path) for name, path in extension_paths.items()
        }
        if extension_hashes != EXPECTED_EXTENSION_HASHES:
            raise GateError("compiled extension hashes differ from the pinned wheel")
        runtime = _runtime_identity(
            vllm,
            lmcache,
            torch,
            transformers,
            extension_paths,
            extension_hashes,
        )
        resolved_lmcache = LMCacheEngineConfig.from_file(args.lmcache_config)
        injection = (
            _install_instrumentation(ledger, inject_rollback=args.arm == "rollback")
            if args.arm != "ordinary"
            else None
        )
        ledger.emit("gate.construction.begin", arm=args.arm)
        started = time.perf_counter()
        llm = LLM(**_engine_kwargs(connectors=args.arm != "ordinary"))
        engine_startup_seconds = time.perf_counter() - started
        from vllm.v1.kmesh.config import validate_kmesh_target_profile

        resolved_profile = validate_kmesh_target_profile(llm.llm_engine.vllm_config)
        expected_profile = None if args.arm == "ordinary" else "serial_pic_v1"
        if resolved_profile != expected_profile:
            raise GateError(
                f"resolved target profile {resolved_profile!r} != {expected_profile!r}"
            )
        kv_transfer_config_present = (
            llm.llm_engine.vllm_config.kv_transfer_config is not None
        )
        ledger.emit(
            "gate.construction.end",
            arm=args.arm,
            engine_startup_seconds=engine_startup_seconds,
            resolved_target_profile=resolved_profile,
            kv_transfer_config_present=kv_transfer_config_present,
        )

        observations: list[dict[str, Any]] = []
        if args.arm == "ordinary":
            for record in records:
                observations.append(
                    _run_request(llm, record, "ordinary", ledger, connectors=False)
                )
        elif args.arm == "cold_miss":
            observations.append(
                _run_request(llm, records[-1], "cold_miss", ledger, connectors=True)
            )
        elif args.arm == "positive_hit":
            observations.append(
                _run_request(llm, records[0], "prime", ledger, connectors=True)
            )
            for record in records[1:]:
                observations.append(
                    _run_request(
                        llm,
                        record,
                        f"probe_{record['sequence_index']:03d}",
                        ledger,
                        connectors=True,
                    )
                )
        else:
            observations.append(
                _run_request(llm, records[0], "prime", ledger, connectors=True)
            )
            observations.append(
                _run_request(llm, records[1], "rollback_probe", ledger, connectors=True)
            )
            observations.append(
                _run_request(llm, records[1], "rollback_retry", ledger, connectors=True)
            )

        _cleanup_llm(llm, ledger)
        llm = None
        cleaned = True
        events = ledger.snapshot()
        ledger.close()
        ledger_closed = True
        checks = _evaluate(args.arm, observations, events, baseline, injection)
        if not checks["all_passed"]:
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise GateError("arm checks failed: " + ", ".join(failed))

        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "arm": args.arm,
            "checks": checks,
            "observations": observations,
            "source": source,
            "runtime": runtime,
            "inputs": {
                "runner": {
                    "path": str(Path(__file__)),
                    "sha256": _sha256(Path(__file__)),
                },
                "lmcache_config": {
                    "path": str(args.lmcache_config),
                    "sha256": _sha256(args.lmcache_config),
                    "resolved": _jsonable(resolved_lmcache),
                },
                "trace": {"path": str(args.trace), "sha256": _sha256(args.trace)},
                "manifest": {
                    "path": str(args.manifest),
                    "sha256": _sha256(args.manifest),
                    "trace_id": records[0]["trace_id"],
                    "oracle": manifest.get("oracle"),
                },
                "baseline_result": (
                    {
                        "path": str(args.baseline_result),
                        "sha256": _sha256(args.baseline_result),
                    }
                    if args.baseline_result is not None
                    else None
                ),
            },
            "engine": {
                "startup_seconds": engine_startup_seconds,
                "resolved_target_profile": resolved_profile,
                "kwargs": _engine_kwargs(connectors=args.arm != "ordinary"),
            },
            "event_ledger": {
                "path": str(ledger.path),
                "sha256": _sha256(ledger.path),
                "event_count": len(events),
            },
            "claim_boundary": (
                "Correctness evidence only: no performance, collective, "
                "Master-Mirror, capacity, or production claim."
            ),
        }
        _atomic_json(args.output_dir / "result.json", result)
        _atomic_json(
            args.output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "arm": args.arm,
                "completed_at_utc": _utc_now(),
                "result_sha256": _sha256(args.output_dir / "result.json"),
            },
        )
        print(json.dumps({"arm": args.arm, "checks": checks}, sort_keys=True))
    except BaseException as error:
        original_traceback = traceback.format_exc()
        cleanup_error = None
        if llm is not None and not cleaned:
            try:
                _cleanup_llm(llm, ledger)
                cleaned = True
            except BaseException as caught:
                cleanup_error = {
                    "exception_type": type(caught).__name__,
                    "message": str(caught),
                    "traceback": traceback.format_exc(),
                }
        if not ledger_closed:
            try:
                ledger.emit(
                    "gate.failure",
                    exception_type=type(error).__name__,
                    message=str(error),
                    cleanup_error=cleanup_error,
                )
            finally:
                ledger.close()
                ledger_closed = True
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "arm": args.arm,
            "failed_at_utc": _utc_now(),
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": original_traceback,
            "cleanup_error": cleanup_error,
            "events_sha256": _sha256(ledger.path),
        }
        _atomic_json(args.output_dir / "failure.json", failure)
        _atomic_json(
            args.output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "arm": args.arm,
                "failed_at_utc": failure["failed_at_utc"],
                "failure_sha256": _sha256(args.output_dir / "failure.json"),
            },
        )
        raise


if __name__ == "__main__":
    main()
