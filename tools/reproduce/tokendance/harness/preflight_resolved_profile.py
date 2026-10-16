#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resolve both pinned KMesh sibling profiles without loading model weights."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import collective_gate_common as common

base = common.serial
GateError = base.GateError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", required=True, type=Path)
    parser.add_argument("--lmcache-source", required=True, type=Path)
    parser.add_argument("--wheel-vllm-package", required=True, type=Path)
    parser.add_argument("--lmcache-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _children(children: list[tuple[type[Any], Any]]) -> list[dict[str, Any]]:
    return [
        {
            "connector_class": (
                f"{connector_type.__module__}.{connector_type.__qualname__}"
            ),
            "kv_connector": child.kv_transfer_config.kv_connector,
            "target_profile": child.kv_transfer_config.kv_connector_extra_config.get(
                "kmesh_target_profile"
            ),
        }
        for connector_type, child in children
    ]


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise GateError(f"refusing to replace preflight output: {args.output}")
    if os.environ.get("KMESH_IMAGE_DIGEST") != common.EXPECTED_IMAGE:
        raise GateError("container image digest evidence is absent or unexpected")
    os.environ["LMCACHE_CONFIG_FILE"] = str(args.lmcache_config)

    source = common.source_identity(args.vllm_source, args.lmcache_source)
    sys.path.insert(0, str(args.lmcache_source))
    vllm = base._load_vllm_overlay(args.vllm_source, args.wheel_vllm_package)
    import lmcache
    import lmcache.c_ops as c_ops
    import torch
    import transformers
    from lmcache.integration.vllm.target_profile import validate_target_profile
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl,
        _KMeshLayerwiseStoreSession,
    )
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.compute.blend.blender import (
        LMCBlendGroupMemberResult,
        LMCBlendGroupRequest,
        LMCBlendGroupResult,
        LMCBlendOperationCounts,
        LMCBlender,
    )
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupClient
    from lmcache.v1.platform import resolve_device_ops, torch_device_type
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.kmesh.config import validate_kmesh_target_profile

    extensions = {
        name: Path(importlib.import_module(f"vllm.{name}").__file__)
        for name in common.EXPECTED_EXTENSION_HASHES
    }
    extension_hashes = {name: base._sha256(path) for name, path in extensions.items()}
    if extension_hashes != common.EXPECTED_EXTENSION_HASHES:
        raise GateError("compiled extension hashes differ from the pinned wheel")
    runtime = base._runtime_identity(
        vllm,
        lmcache,
        torch,
        transformers,
        extensions,
        extension_hashes,
    )

    lmcache_config = LMCacheEngineConfig.from_file(args.lmcache_config)
    resolved: dict[str, Any] = {}
    for profile in (common.SERIAL_PROFILE, common.COLLECTIVE_PROFILE):
        config = EngineArgs(
            **common.engine_kwargs(connectors=True, profile=profile)
        ).create_engine_config()
        observed_profile = validate_kmesh_target_profile(config)
        if observed_profile != profile:
            raise GateError(
                f"resolved target profile {observed_profile!r} != {profile!r}"
            )
        if config.parallel_config.worker_cls != "vllm.v1.worker.gpu_worker.Worker":
            raise GateError("platform did not resolve the built-in GPU worker")
        attention_backend = getattr(config.attention_config.backend, "name", None)
        if attention_backend != "TRITON_ATTN":
            raise GateError("pinned source profile did not resolve TRITON_ATTN")
        if (
            config.scheduler_config.max_num_batched_tokens != 2048
            or config.scheduler_config.max_num_seqs != 4
        ):
            raise GateError("four-request batch geometry drifted")

        children = MultiConnector.inspect_connector_classes_and_configs(config)
        lmcache_children = [
            child
            for _, child in children
            if child.kv_transfer_config.kv_connector == "LMCacheConnectorV1"
        ]
        if len(children) != 2 or len(lmcache_children) != 1:
            raise GateError("resolved topology is not exact KMesh plus LMCache")
        for role in ("scheduler", "worker"):
            validate_target_profile(
                lmcache_config,
                profile,
                role=role,
                vllm_config=lmcache_children[0],
            )
        resolved[profile] = {
            "target_profile": observed_profile,
            "worker_cls": config.parallel_config.worker_cls,
            "attention_backend_config": attention_backend,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "children": _children(children),
            "lmcache_roles_validated": ["scheduler", "worker"],
        }

    instrumentation_classes = (
        LMCacheConnectorV1Impl,
        _KMeshLayerwiseStoreSession,
        LMCacheEngine,
        LMCBlender,
        LMCBlendGroupRequest,
        LMCBlendGroupResult,
        LMCBlendGroupMemberResult,
        LMCBlendOperationCounts,
        LMCacheLookupClient,
    )
    c_ops_symbols = (
        "EngineKVFormat",
        "alloc_pinned_ptr",
        "single_layer_kv_transfer",
    )
    missing_c_ops = [name for name in c_ops_symbols if not hasattr(c_ops, name)]
    if missing_c_ops:
        raise GateError("LMCache c_ops shim lacks: " + ", ".join(missing_c_ops))

    result = {
        "schema_version": common.SCHEMA_VERSION,
        "status": "complete",
        "captured_at_utc": base._utc_now(),
        "source": source,
        "runtime": runtime,
        "runner": {
            "path": str(Path(__file__)),
            "sha256": base._sha256(Path(__file__)),
            "common_sha256": base._sha256(Path(common.__file__)),
        },
        "lmcache_config": {
            "path": str(args.lmcache_config),
            "sha256": base._sha256(args.lmcache_config),
        },
        "resolved": resolved,
        "instrumentation_classes_imported": [
            f"{item.__module__}.{item.__qualname__}" for item in instrumentation_classes
        ],
        "c_ops_shim_symbols": list(c_ops_symbols),
        "device_ops": type(resolve_device_ops(torch_device_type)).__name__,
        "claim_boundary": (
            "No-weights configuration evidence only. Every arm pins the same "
            "TRITON_ATTN backend; runtime execution and any numerical capture "
            "remain separately proved and hash-bound."
        ),
    }
    base._atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": "complete"}))


if __name__ == "__main__":
    main()
