#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned common inputs for the KMesh collective-lockstep single-device gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

SERIAL_GATE_SHA256 = "214e81fec0aff22b1517373ca73847fe99ec0f4127c5148ac6dd3277af66fd6b"
EXPECTED_VLLM_COMMIT = "18f90213f45783bad6b7acd7bc43cbba957a69f8"
EXPECTED_VLLM_TREE = "c9bed864a0b1082d9aba501a339de7d20cd29017"
EXPECTED_LMCACHE_COMMIT = "aa4984b57f78b9d1dce2e8d870ce7aa4716cd341"
EXPECTED_LMCACHE_TREE = "d2b3feeecae29d0b7e8366e817738966d2d267eb"
COLLECTIVE_PROFILE = "collective_batched_qkv_v1"
COLLECTIVE_MODE = "collective_batched"
SERIAL_PROFILE = "serial_pic_v1"
SERIAL_MODE = "serial_pic"
ENGINE_ID = "kmesh-collective-qkv-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serial_gate_path() -> Path:
    return Path(__file__).with_name("run_online_serial_pic_gate.py")


def _load_serial_gate() -> ModuleType:
    path = _serial_gate_path()
    if not path.is_file():
        raise RuntimeError(f"accepted serial gate asset is absent: {path}")
    observed = _sha256(path)
    if observed != SERIAL_GATE_SHA256:
        raise RuntimeError(
            f"accepted serial gate asset drifted: {observed} != {SERIAL_GATE_SHA256}"
        )
    name = "_kmesh_accepted_serial_gate"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import accepted serial gate asset: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serial = _load_serial_gate()
if serial.CURRENT_CUDA:
    EXPECTED_VLLM_COMMIT = serial.EXPECTED_VLLM_COMMIT
    EXPECTED_VLLM_TREE = serial.EXPECTED_VLLM_TREE
    EXPECTED_LMCACHE_COMMIT = serial.EXPECTED_LMCACHE_COMMIT
    EXPECTED_LMCACHE_TREE = serial.EXPECTED_LMCACHE_TREE

# The trace, image, wheel extensions, runtime, and model are intentionally the
# accepted serial gate's exact immutable inputs. Only the two source descendants
# and the explicitly named target profile change.
SCHEMA_VERSION = serial.SCHEMA_VERSION
SEED = serial.SEED
MODEL_REVISION = serial.MODEL_REVISION
MODEL_PATH = serial.MODEL_PATH
EXPECTED_IMAGE = serial.EXPECTED_IMAGE
EXPECTED_TARGET_VOLUME = serial.EXPECTED_TARGET_VOLUME
EXPECTED_EXTENSION_HASHES = serial.EXPECTED_EXTENSION_HASHES
EXPECTED_TRACE_SHA256 = serial.EXPECTED_TRACE_SHA256
EXPECTED_MANIFEST_SHA256 = serial.EXPECTED_MANIFEST_SHA256
EXPECTED_TRACE_ID = serial.EXPECTED_TRACE_ID
EXPECTED_ORACLE_AGGREGATE = serial.EXPECTED_ORACLE_AGGREGATE
EXPECTED_RUNTIME = serial.EXPECTED_RUNTIME
TRACE_CONFIG = serial.TRACE_CONFIG


def engine_kwargs(
    *, connectors: bool, profile: str = COLLECTIVE_PROFILE
) -> dict[str, Any]:
    """Return the exact serial envelope with only sibling profile fields changed."""
    options = copy.deepcopy(serial._engine_kwargs(connectors=connectors))
    options["attention_config"] = {"backend": "TRITON_ATTN"}
    if not connectors:
        return options
    if profile == COLLECTIVE_PROFILE:
        mode = COLLECTIVE_MODE
    elif profile == SERIAL_PROFILE:
        mode = SERIAL_MODE
    else:
        raise ValueError(f"unsupported gate profile: {profile!r}")
    transfer = options["kv_transfer_config"]
    transfer["engine_id"] = ENGINE_ID
    extra = transfer["kv_connector_extra_config"]
    extra["kmesh_target_profile"] = profile
    connectors_config = extra["connectors"]
    kmesh = connectors_config[0]["kv_connector_extra_config"]["kmesh"]
    kmesh["mode"] = mode
    return options


def source_identity(vllm_source: Path, lmcache_source: Path) -> dict[str, Any]:
    """Validate and return the exact clean source identities."""
    return {
        "vllm": serial._source_identity(
            vllm_source,
            EXPECTED_VLLM_COMMIT,
            EXPECTED_VLLM_TREE,
        ),
        "lmcache": serial._source_identity(
            lmcache_source,
            EXPECTED_LMCACHE_COMMIT,
            EXPECTED_LMCACHE_TREE,
        ),
    }
