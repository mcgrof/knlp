#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one crash-safe KMesh collective-lockstep arm on the pinned single-device profile."""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib
import json
import math
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import collective_gate_common as common
import numerical_instrumentation

base = common.serial
GateError = base.GateError
EventLedger = base.EventLedger
ARMS = (
    "ordinary",
    "serial_oracle",
    "collective_forward",
    "collective_reverse",
    "cold_miss",
    "zero_sharing",
    "incompatible_partition",
    "sibling_failure",
)
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
KMESH_SIDECAR_PROVIDER = "vllm.kmesh.request_lookup.v1"
KMESH_SIDECAR_WIRE_TAG = "kmesh-request-lookup-v1"
ZERO_SHARING_CACHE_DOMAIN = "synthetic-cache-domain:a"
ZERO_SHARING_PROMPT_TOKENS = 512
ZERO_SHARING_FALLBACK = "no_shared_segments"
ZERO_SHARING_INTENT = "serial_pic"
ZERO_SHARING_SCHEDULER_STATE_FIELDS = (
    "_kmesh_prepared_lookups",
    "_kmesh_prepared_request_objects",
    "_kmesh_allocated_requests",
    "_unfinished_requests",
    "_request_trackers",
    "load_specs",
)
ZERO_SHARING_FORBIDDEN_ACTION_PREFIXES = (
    "LMCacheConnectorV1Impl._record_kmesh_load_failure.",
    "LMCacheEngine.store_layer.",
    "LMCacheEngine.retrieve_layer.",
    "LMCacheEngine.register_lookup_cleanup.",
    "LMCacheEngine.schedule_lookup_unpin.",
    "LMCacheEngine.lookup_unpin.",
    "LMCacheLookupClient.lookup.",
    "LMCacheLookupClient.transfer_lookups.",
    "LMCacheLookupClient.cancel_lookup.",
    "LMCacheLookupClient.finish_lookup.",
    "_KMeshLayerwiseStoreSession.finish.",
    "LMCBlender.blend.",
    "gate.collective.blend_group.",
)


def _external_trace_bindings(
    observations: list[dict[str, Any]],
) -> dict[str, str]:
    if type(observations) is not list or not observations:
        raise GateError("runtime observations must be a nonempty plain list")
    bindings: dict[str, str] = {}
    trace_ids: set[str] = set()
    for observation in observations:
        if type(observation) is not dict:
            raise GateError("runtime observation is not a plain mapping")
        external = observation.get("external_request_id")
        trace = observation.get("trace_request_id")
        if type(external) is not str or not external:
            raise GateError("runtime external request ID is invalid")
        if type(trace) is not str or not trace:
            raise GateError("runtime trace request ID is invalid")
        if external in bindings or trace in trace_ids:
            raise GateError("runtime request identity is not bijective")
        bindings[external] = trace
        trace_ids.add(trace)
    if list(bindings) != [str(index) for index in range(len(observations))]:
        raise GateError("runtime external request IDs are not the frozen sequence")
    return bindings


def _bind_internal_request_id(
    internal: object,
    external_to_trace: dict[str, str],
) -> tuple[str, str]:
    if type(internal) is not str:
        raise GateError("runtime internal request ID is not a plain string")
    external, separator, nonce = internal.rpartition("-")
    if (
        separator != "-"
        or len(nonce) != INTERNAL_REQUEST_NONCE_HEX_LENGTH
        or any(character not in LOWER_HEX for character in nonce)
        or external not in external_to_trace
    ):
        raise GateError("runtime internal request ID is not bound to an observation")
    return external, external_to_trace[external]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--vllm-source", required=True, type=Path)
    parser.add_argument("--lmcache-source", required=True, type=Path)
    parser.add_argument("--wheel-vllm-package", required=True, type=Path)
    parser.add_argument("--lmcache-config", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-result", type=Path)
    return parser.parse_args()


def _profile_for_arm(arm: str) -> str | None:
    if arm == "ordinary":
        return None
    if arm == "serial_oracle":
        return common.SERIAL_PROFILE
    return common.COLLECTIVE_PROFILE


def _sampling_params(record: dict[str, Any], *, connectors: bool) -> Any:
    from vllm import SamplingParams

    options: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 1,
        "seed": common.SEED,
    }
    if connectors:
        options["extra_args"] = {"kmesh": record["kmesh_context"]}
    return SamplingParams(**options)


def _run_batch(
    llm: Any,
    records: list[dict[str, Any]],
    phase: str,
    ledger: EventLedger,
    *,
    connectors: bool,
) -> list[dict[str, Any]]:
    """Submit one exact prompt batch and bind engine IDs back to trace IDs."""
    from vllm import TokensPrompt

    if not records:
        raise GateError("gate batch cannot be empty")
    ledger.set_phase(phase)
    submitted_trace_ids = [record["request_id"] for record in records]
    ledger.emit(
        "gate.batch.begin",
        batch_size=len(records),
        submitted_trace_request_ids=submitted_trace_ids,
        prompt_tokens=sum(len(record["prompt_token_ids"]) for record in records),
    )
    outputs = llm.generate(
        [
            TokensPrompt(prompt_token_ids=record["prompt_token_ids"])
            for record in records
        ],
        [_sampling_params(record, connectors=connectors) for record in records],
        use_tqdm=False,
    )
    if len(outputs) != len(records):
        raise GateError("vLLM returned a different number of batch outputs")

    observations: list[dict[str, Any]] = []
    for ordinal, (record, output) in enumerate(zip(records, outputs, strict=True)):
        if list(output.prompt_token_ids) != record["prompt_token_ids"]:
            raise GateError("vLLM output order or prompt identity drifted")
        if type(output.request_id) is not str:
            raise GateError("vLLM public request ID is not a plain string")
        completion = output.outputs[0]
        observation = {
            "phase": phase,
            "batch_ordinal": ordinal,
            "trace_request_id": record["request_id"],
            "external_request_id": output.request_id,
            "sequence_index": record["sequence_index"],
            "prompt_tokens": len(output.prompt_token_ids),
            "num_cached_tokens": output.num_cached_tokens,
            "output_token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
        }
        observations.append(observation)
        ledger.emit("gate.request.complete", **observation)
    ledger.emit(
        "gate.batch.end",
        batch_size=len(observations),
        submitted_trace_request_ids=submitted_trace_ids,
        returned_external_request_ids=[
            observation["external_request_id"] for observation in observations
        ],
    )
    return observations


def _without_sharing(record: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(record)
    for segment in changed["kmesh_context"]["segments"]:
        segment["shareable"] = False
    return changed


def _partition_b(record: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(record)
    changed["kmesh_context"]["cache_domain"] += ":partition-b"
    return changed


def _mask_evidence(mask: Any) -> dict[str, Any]:
    import torch

    if type(mask) is not torch.Tensor or mask.dtype is not torch.bool or mask.ndim != 1:
        raise GateError("retrieval mask is not an exact one-dimensional bool tensor")
    values = [bool(value) for value in mask.detach().to(device="cpu").tolist()]
    encoded = bytes(int(value) for value in values)
    return {
        "token_count": len(values),
        "true_count": sum(values),
        "true_indices": [index for index, value in enumerate(values) if value],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _operation_counts(value: Any) -> dict[str, int]:
    names = (
        "serial_rope_operations",
        "serial_diff_operations",
        "collective_rope_operations",
        "collective_diff_operations",
        "position_recovery_serial_operations",
        "position_recovery_collective_operations",
    )
    result = {name: getattr(value, name, None) for name in names}
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise GateError("collective member operation counts are not exact")
    return result


def _plain_identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise GateError(f"{label} is not an exact nonempty identifier")
    return value


def _canonical_lookup_id(value: object, label: str) -> str:
    lookup_id = _plain_identifier(value, label)
    try:
        parsed = uuid.UUID(lookup_id)
    except ValueError as error:
        raise GateError(f"{label} is not a canonical UUIDv4") from error
    if parsed.version != 4 or str(parsed) != lookup_id:
        raise GateError(f"{label} is not a canonical UUIDv4")
    return lookup_id


def _sidecar_payload_json(value: object) -> object:
    """Normalize the exact tuple-only KMesh wire value for JSON hashing."""
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is tuple:
        return [_sidecar_payload_json(item) for item in value]
    raise GateError("KMesh sidecar payload contains a noncanonical value")


def _sidecar_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_sidecar_evidence(value: object) -> list[dict[str, Any]]:
    """Reparse and hash every exact scheduler-side KMesh request sidecar."""
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRequestSidecar,
    )
    from vllm.v1.kmesh import (
        KMESH_REQUEST_SIDECAR_PROVIDER_ID,
        KMeshRequestLookupDescriptor,
    )

    if type(value) is not tuple:
        raise GateError("peer request sidecars are not an immutable tuple")
    if KMESH_REQUEST_SIDECAR_PROVIDER_ID != KMESH_SIDECAR_PROVIDER:
        raise GateError("KMesh request-sidecar provider constant drifted")
    result: list[dict[str, Any]] = []
    for index, sidecar in enumerate(value):
        if type(sidecar) is not KVConnectorRequestSidecar:
            raise GateError(f"peer sidecar {index} has a noncanonical exact type")
        if sidecar.provider_id != KMESH_SIDECAR_PROVIDER:
            raise GateError(f"peer sidecar {index} has an unexpected provider")
        try:
            descriptor = KMeshRequestLookupDescriptor.from_sidecar_payload(
                sidecar.payload
            )
        except (TypeError, ValueError) as error:
            raise GateError(f"peer sidecar {index} has an invalid payload") from error
        if descriptor.to_sidecar_payload() != sidecar.payload:
            raise GateError(f"peer sidecar {index} is not a canonical wire value")
        payload = _sidecar_payload_json(sidecar.payload)
        if type(payload) is not list or len(payload) != 8:
            raise GateError(f"peer sidecar {index} has an invalid wire arity")
        result.append(
            {
                "provider_id": sidecar.provider_id,
                "payload": payload,
                "payload_sha256": _sidecar_payload_sha256(payload),
                "wire_tag": payload[0],
                "intent": descriptor.lookup_intent.value,
                "descriptor_request_id": descriptor.request_id,
                "admission_generation": descriptor.admission_generation,
                "prompt_token_count": descriptor.prompt_token_count,
                "cache_domain": descriptor.cache_domain,
                "fallback": (
                    descriptor.admission_fallback.value
                    if descriptor.admission_fallback is not None
                    else None
                ),
                "span_count": len(descriptor.spans),
            }
        )
    return result


def _domain_aware_owner_state(
    original_owner_state: Any,
    owner: Any,
) -> dict[str, Any]:
    """Expose target scheduler maps needed to prove terminal zero ownership."""
    state = original_owner_state(owner)
    if type(state) is not dict:
        raise GateError("inherited owner state is not a plain mapping")
    for field in ZERO_SHARING_SCHEDULER_STATE_FIELDS:
        value = getattr(owner, field, None)
        if value is not None:
            try:
                state[f"{field}_count"] = len(value)
            except TypeError as error:
                raise GateError(f"scheduler state {field} is not countable") from error
    return state


def _domain_aware_layer_call_details(
    original_call_details: Any,
    name: str,
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Keep LMCache request IDs and transport lookup IDs disjoint."""
    details = original_call_details(name, owner, args, kwargs)
    if type(details) is not dict:
        raise GateError("inherited call details are not a plain mapping")
    if name == "prepare_request_lookup":
        if len(args) <= 1:
            raise GateError("prepare_request_lookup omitted positional peer sidecars")
        peer_sidecars = args[1]
        sidecars = _request_sidecar_evidence(peer_sidecars)
        if type(details.get("sidecar_count")) is not int or details.get(
            "sidecar_count"
        ) != len(sidecars):
            raise GateError("inherited request-sidecar count drifted")
        details["sidecars"] = sidecars
        return details
    if name not in {"store_layer", "retrieve_layer"}:
        return details

    explicit_request = kwargs.get("request_id")
    req_id = kwargs.get("req_id")
    inherited_request = explicit_request or req_id
    if details.get("request_id") != inherited_request:
        raise GateError("inherited layer-call identity behavior drifted")

    if explicit_request is not None:
        explicit_request = _plain_identifier(explicit_request, "layer-call request_id")
    if req_id is not None and name == "store_layer":
        req_id = _plain_identifier(req_id, "layer-call req_id")

    if name == "retrieve_layer":
        if explicit_request is not None:
            raise GateError("retrieve_layer carried an unexpected request ID")
        if req_id is None:
            raise GateError("retrieve_layer omitted its transport lookup ID")
        req_id = _canonical_lookup_id(req_id, "retrieve_layer lookup ID")
        if "lookup_id" in details and details["lookup_id"] != req_id:
            raise GateError("retrieve_layer lookup identity collided")
        details["lookup_id"] = req_id
        details.pop("request_id", None)
        return details

    if explicit_request is None and req_id is None:
        raise GateError("store_layer omitted its engine request ID")
    if (
        explicit_request is not None
        and req_id is not None
        and explicit_request != req_id
    ):
        raise GateError("store_layer carried conflicting request aliases")
    if "lookup_id" in details:
        raise GateError("store_layer inherited an unexpected lookup identity")
    details["request_id"] = explicit_request or req_id
    return details


def _install_instrumentation(
    ledger: EventLedger,
    *,
    inject_sibling_failure: bool,
) -> dict[str, Any]:
    """Instrument actual serial/group recovery without changing successful data."""
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
    from lmcache.v1.compute.blend.blender import (
        LMCBlendGroupRequest,
        LMCBlendGroupResult,
        LMCBlendSession,
        LMCBlender,
    )

    inherited_call_details = base._call_details
    if getattr(inherited_call_details, "_kmesh_domain_aware_v1", False):
        raise GateError("layer-call identity instrumentation was installed twice")
    inherited_owner_state = base._owner_state
    if getattr(inherited_owner_state, "_kmesh_domain_aware_v1", False):
        raise GateError("owner-state instrumentation was installed twice")

    @functools.wraps(inherited_owner_state)
    def domain_aware_owner_state(owner: Any) -> dict[str, Any]:
        return _domain_aware_owner_state(inherited_owner_state, owner)

    setattr(domain_aware_owner_state, "_kmesh_domain_aware_v1", True)
    base._owner_state = domain_aware_owner_state

    @functools.wraps(inherited_call_details)
    def domain_aware_call_details(
        name: str,
        owner: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return _domain_aware_layer_call_details(
            inherited_call_details, name, owner, args, kwargs
        )

    setattr(domain_aware_call_details, "_kmesh_domain_aware_v1", True)
    base._call_details = domain_aware_call_details
    base._install_instrumentation(ledger, inject_rollback=False)

    original_build_connector_meta = LMCacheConnectorV1Impl.build_connector_meta

    @functools.wraps(original_build_connector_meta)
    def build_connector_meta(owner: Any, *args: Any, **kwargs: Any) -> Any:
        state_before = base._owner_state(owner)
        ledger.emit("gate.lmcache_connector_meta.begin", **state_before)
        try:
            result = original_build_connector_meta(owner, *args, **kwargs)
        except BaseException as error:
            ledger.emit(
                "gate.lmcache_connector_meta.error",
                **state_before,
                exception_type=type(error).__name__,
                message=str(error),
            )
            raise
        requests = getattr(result, "requests", None)
        if type(requests) is not list:
            raise GateError("LMCache connector metadata has no exact request list")
        ledger.emit(
            "gate.lmcache_connector_meta.end",
            **base._owner_state(owner),
            request_count=len(requests),
        )
        return result

    LMCacheConnectorV1Impl.build_connector_meta = build_connector_meta
    state: dict[str, Any] = {
        "enabled": inject_sibling_failure,
        "target_request_id": None,
        "count": 0,
    }

    original_serial_blend = LMCBlender.blend

    @functools.wraps(original_serial_blend)
    def serial_blend(owner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_serial_blend(owner, *args, **kwargs)
        ledger.emit(
            "gate.serial_blend.evidence",
            request_id=kwargs.get("request_id"),
            lookup_id=kwargs.get("req_id"),
            explicit_spans=base._span_summary(kwargs.get("explicit_spans")),
            retrieval_mask=_mask_evidence(result),
        )
        return result

    LMCBlender.blend = serial_blend

    original_freeze = LMCacheConnectorV1Impl._freeze_kmesh_worker_plans

    @functools.wraps(original_freeze)
    def freeze_plans(owner: Any, *args: Any, **kwargs: Any) -> Any:
        plans = original_freeze(owner, *args, **kwargs)
        ledger.emit(
            "gate.collective.plan_projection",
            plans=[
                {
                    "plan_id": plan.plan_id,
                    "reuse_group_id": plan.reuse_group_id,
                    "members": [
                        {
                            "request_id": member.request_id,
                            "explicit_spans": base._span_summary(member.explicit_spans),
                        }
                        for member in plan.members
                    ],
                }
                for plan in plans
            ],
        )
        return plans

    LMCacheConnectorV1Impl._freeze_kmesh_worker_plans = freeze_plans

    original_step_retrieval = LMCBlendSession.step_retrieval

    @functools.wraps(original_step_retrieval)
    def step_retrieval(owner: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            state["enabled"]
            and state["count"] == 0
            and owner.request_id == state["target_request_id"]
        ):
            state["count"] += 1
            ledger.emit(
                "gate.sibling_failure.injected",
                request_id=owner.request_id,
                completed_serial_operations=_operation_counts(owner.operation_counts),
            )
            raise RuntimeError("KMesh gate injected one request-local sibling failure")
        return original_step_retrieval(owner, *args, **kwargs)

    LMCBlendSession.step_retrieval = step_retrieval
    original_blend_group = LMCBlender.blend_group

    @functools.wraps(original_blend_group)
    def blend_group(owner: Any, requests: Any, **blend_kwargs: Any) -> Any:
        request_snapshot = tuple(requests)
        if any(
            type(request) is not LMCBlendGroupRequest for request in request_snapshot
        ):
            raise GateError("blend_group received a noncanonical request member")
        if state["enabled"] and state["target_request_id"] is None:
            if len(request_snapshot) < 2:
                raise GateError("sibling-failure arm did not reach a real group")
            state["target_request_id"] = request_snapshot[1].request_id
        begin_members = [
            {
                "request_id": request.request_id,
                "generation": request.generation,
                "token_count": len(request.tokens),
                "explicit_spans": base._span_summary(
                    request.retrieve_kwargs.get("explicit_spans")
                ),
            }
            for request in request_snapshot
        ]
        ledger.emit(
            "gate.collective.blend_group.begin",
            num_layers=owner.num_layers,
            members=begin_members,
        )
        try:
            result = original_blend_group(owner, request_snapshot, **blend_kwargs)
        except BaseException as error:
            ledger.emit(
                "gate.collective.blend_group.error",
                exception_type=type(error).__name__,
                message=str(error),
                members=begin_members,
            )
            raise
        if type(result) is not LMCBlendGroupResult:
            raise GateError("blend_group returned a noncanonical result")
        result_members = []
        for member in result.members:
            result_members.append(
                {
                    "request_id": member.request_id,
                    "generation": member.generation,
                    "outcome": member.outcome.value,
                    "retrieval_mask": (
                        _mask_evidence(member.retrieval_mask)
                        if member.retrieval_mask is not None
                        else None
                    ),
                    "operation_counts": _operation_counts(member.operation_counts),
                    "error": (
                        {
                            "type": type(member.error).__name__,
                            "message": str(member.error),
                        }
                        if member.error is not None
                        else None
                    ),
                }
            )
        ledger.emit(
            "gate.collective.blend_group.end",
            num_layers=owner.num_layers,
            input_members=begin_members,
            result_members=result_members,
        )
        return result

    LMCBlender.blend_group = blend_group
    return state


def _events(
    events: list[dict[str, Any]],
    action: str,
    phase: str | None = None,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("action") == action
        and (phase is None or event.get("phase") == phase)
    ]


def _projection_payload_signature(value: object) -> tuple[object, ...]:
    """Return a type-sensitive signature for one plain JSON payload."""
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise GateError("projected plan payload contains a non-finite float")
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if type(value) is list:
        return (
            "list",
            tuple(_projection_payload_signature(item) for item in value),
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise GateError("projected plan payload has a non-string key")
        return (
            "dict",
            tuple(
                (key, _projection_payload_signature(value[key]))
                for key in sorted(value)
            ),
        )
    raise GateError("projected plan payload is not plain JSON")


def _expected_projection_phases(arm: str) -> tuple[str, ...]:
    """Return the exact ordered phase for each validate/bind round."""
    if arm in {"ordinary", "serial_oracle"}:
        return ()
    if arm in {
        "collective_forward",
        "collective_reverse",
        "incompatible_partition",
    }:
        return ("prime", arm)
    if arm == "sibling_failure":
        return ("prime", "sibling_failure", "sibling_failure")
    if arm in {"cold_miss", "zero_sharing"}:
        return (arm,)
    raise GateError(f"unknown arm for projection receipts: {arm}")


def _validate_sibling_projection_causality(
    events: list[dict[str, Any]],
    projection_events: list[dict[str, Any]],
    sibling_rounds: list[dict[str, Any]],
) -> None:
    """Bind the empty sibling retry round to the completed failure lifecycle."""

    def selected(action: str) -> list[dict[str, Any]]:
        return [
            event
            for event in events
            if event.get("action") == action and event.get("phase") == "sibling_failure"
        ]

    def sequence(event: dict[str, Any]) -> int:
        value = event.get("sequence")
        if type(value) is not int:
            raise GateError("sibling-failure causal event has an invalid sequence")
        return value

    def request_finish_owners(
        finish_events: list[dict[str, Any]],
        terminal: str,
    ) -> dict[str, dict[str, Any]]:
        if len(finish_events) != 4:
            raise GateError(
                f"sibling-failure request-finish {terminal} coverage is incomplete"
            )
        owners: dict[str, dict[str, Any]] = {}
        for event in finish_events:
            request_id = event.get("request_id")
            if (
                type(request_id) is not str
                or event.get("owner_type") != "LMCacheConnectorV1Impl"
                or event.get("role") != "SCHEDULER"
                or request_id in owners
            ):
                raise GateError(
                    f"sibling-failure request-finish {terminal} owner is invalid"
                )
            sequence(event)
            if terminal == "end" and (
                type(event.get("result")) is not dict
                or event["result"] != {"type": "tuple"}
                or type(event["result"].get("type")) is not str
            ):
                raise GateError("sibling-failure request-finish result is invalid")
            owners[request_id] = event
        return owners

    injections = selected("gate.sibling_failure.injected")
    group_begins = selected("gate.collective.blend_group.begin")
    group_ends = selected("gate.collective.blend_group.end")
    failure_begins = selected("LMCacheConnectorV1Impl._record_kmesh_load_failure.begin")
    failure_ends = selected("LMCacheConnectorV1Impl._record_kmesh_load_failure.end")
    finish_begins = selected("LMCacheConnectorV1Impl.request_finished.begin")
    finish_ends = selected("LMCacheConnectorV1Impl.request_finished.end")
    if not (
        len(injections) == 1
        and len(group_begins) == 1
        and len(group_ends) == 1
        and len(failure_begins) == 1
        and len(failure_ends) == 1
    ):
        raise GateError("sibling-failure retry lifecycle coverage is incomplete")

    result_members = group_ends[0].get("result_members")
    if type(result_members) is not list or any(
        type(member) is not dict for member in result_members
    ):
        raise GateError("sibling-failure group result is malformed")
    failed_ids = [
        member.get("request_id")
        for member in result_members
        if member.get("outcome") == "failed"
    ]
    recovered_ids = [
        member.get("request_id")
        for member in result_members
        if member.get("outcome") == "recovered"
    ]
    if (
        len(failed_ids) != 1
        or len(recovered_ids) != 3
        or any(type(request_id) is not str for request_id in failed_ids)
        or any(type(request_id) is not str for request_id in recovered_ids)
        or len(set([*failed_ids, *recovered_ids])) != 4
    ):
        raise GateError(
            "sibling-failure group outcomes are not one failed and three recovered"
        )
    failed_id = failed_ids[0]
    if any(
        event.get("request_id") != failed_id
        for event in (injections[0], failure_begins[0], failure_ends[0])
    ):
        raise GateError("sibling-failure causal events disagree on the failed owner")

    finish_begins_by_id = request_finish_owners(finish_begins, "begin")
    finish_ends_by_id = request_finish_owners(finish_ends, "end")
    expected_finish_ids = set([failed_id, *recovered_ids])
    if (
        set(finish_begins_by_id) != expected_finish_ids
        or set(finish_ends_by_id) != expected_finish_ids
    ):
        raise GateError("sibling-failure request finishes have ambiguous ownership")
    for request_id in expected_finish_ids:
        if sequence(finish_begins_by_id[request_id]) >= sequence(
            finish_ends_by_id[request_id]
        ):
            raise GateError("sibling-failure request-finish pair is not ordered")

    initial_bind_sequence = sequence(sibling_rounds[0])
    group_begin_sequence = sequence(group_begins[0])
    injection_sequence = sequence(injections[0])
    group_end_sequence = sequence(group_ends[0])
    failure_begin_sequence = sequence(failure_begins[0])
    failure_end_sequence = sequence(failure_ends[0])
    retry_validate_sequence = sequence(projection_events[-2])
    retry_bind_sequence = sequence(sibling_rounds[1])
    recovered_finish_end_sequences = sorted(
        sequence(finish_ends_by_id[request_id]) for request_id in recovered_ids
    )
    failed_finish_begin_sequence = sequence(finish_begins_by_id[failed_id])
    failed_finish_end_sequence = sequence(finish_ends_by_id[failed_id])
    if not (
        initial_bind_sequence
        < group_begin_sequence
        < injection_sequence
        < group_end_sequence
        < failure_begin_sequence
        < failure_end_sequence
        < recovered_finish_end_sequences[0]
    ):
        raise GateError("sibling-failure retry is not causally after the failure")

    retry_meta_begins = [
        event
        for event in selected("gate.lmcache_connector_meta.begin")
        if failure_end_sequence < sequence(event)
    ]
    retry_meta_ends = [
        event
        for event in selected("gate.lmcache_connector_meta.end")
        if failure_end_sequence < sequence(event)
    ]
    if (
        len(retry_meta_begins) != 1
        or len(retry_meta_ends) != 1
        or any(
            event.get("owner_type") != "LMCacheConnectorV1Impl"
            or event.get("role") != "SCHEDULER"
            for event in (retry_meta_begins[0], retry_meta_ends[0])
        )
        or type(retry_meta_ends[0].get("request_count")) is not int
        or retry_meta_ends[0].get("request_count") != 0
    ):
        raise GateError("sibling-failure retry metadata is not exactly empty")
    retry_meta_begin_sequence = sequence(retry_meta_begins[0])
    retry_meta_end_sequence = sequence(retry_meta_ends[0])
    if not (
        recovered_finish_end_sequences[-1]
        < retry_meta_begin_sequence
        < retry_meta_end_sequence
        < retry_validate_sequence
        < retry_bind_sequence
        < failed_finish_begin_sequence
        < failed_finish_end_sequence
    ):
        raise GateError("sibling-failure retry lifecycle is not ledger ordered")


def _canonical_projection_events(
    events: list[dict[str, Any]],
    expected_round_phases: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Require the exact ordered sequence of validate/bind projection pairs."""
    projection_events = _events(events, "gate.collective.plan_projection")
    expected_receipt_phases = tuple(
        phase
        for round_phase in expected_round_phases
        for phase in (round_phase, round_phase)
    )
    if len(projection_events) != len(expected_receipt_phases):
        raise GateError("projected plan validate/bind receipt coverage is incomplete")

    canonical: list[dict[str, Any]] = []
    validate_signature: tuple[object, ...] | None = None
    previous_sequence: int | None = None
    for receipt_index, (event, expected_phase) in enumerate(
        zip(projection_events, expected_receipt_phases, strict=True)
    ):
        phase = event.get("phase")
        plans = event.get("plans")
        sequence = event.get("sequence")
        if (
            type(phase) is not str
            or type(plans) is not list
            or type(sequence) is not int
        ):
            raise GateError("projected plan event is malformed")
        if phase != expected_phase:
            raise GateError("projected plan receipts have an unexpected phase order")
        if previous_sequence is not None and sequence <= previous_sequence:
            raise GateError("projected plan receipts are not ledger ordered")
        previous_sequence = sequence
        signature = _projection_payload_signature(plans)
        if receipt_index % 2 == 0:
            validate_signature = signature
        else:
            if signature != validate_signature:
                raise GateError(
                    "projected plan replay differs from its canonical payload"
                )
            canonical.append(event)

    if expected_round_phases == (
        "prime",
        "sibling_failure",
        "sibling_failure",
    ):
        sibling_rounds = [
            event for event in canonical if event["phase"] == "sibling_failure"
        ]
        if (
            len(sibling_rounds) != 2
            or not sibling_rounds[0]["plans"]
            or sibling_rounds[1]["plans"] != []
        ):
            raise GateError(
                "sibling-failure projection rounds are not nonempty then empty"
            )
        _validate_sibling_projection_causality(
            events,
            projection_events,
            sibling_rounds,
        )
    return canonical


def _zero_sharing_sidecar_evidence(
    observations: list[dict[str, Any]],
    request_id_bindings: list[dict[str, str]],
    events: list[dict[str, Any]],
) -> int:
    """Prove the exact no-op descriptor and its scheduler lifecycle."""
    prepare_begins = _events(
        events, "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
    )
    prepare_ends = _events(events, "LMCacheConnectorV1Impl.prepare_request_lookup.end")
    matched_begins = _events(
        events, "LMCacheConnectorV1Impl.get_num_new_matched_tokens.begin"
    )
    matched_ends = _events(
        events, "LMCacheConnectorV1Impl.get_num_new_matched_tokens.end"
    )
    update_begins = _events(
        events, "LMCacheConnectorV1Impl.update_state_after_alloc.begin"
    )
    update_ends = _events(events, "LMCacheConnectorV1Impl.update_state_after_alloc.end")
    finish_begins = _events(events, "LMCacheConnectorV1Impl.request_finished.begin")
    finish_ends = _events(events, "LMCacheConnectorV1Impl.request_finished.end")
    meta_begins = _events(events, "gate.lmcache_connector_meta.begin")
    meta_ends = _events(events, "gate.lmcache_connector_meta.end")
    meta_errors = _events(events, "gate.lmcache_connector_meta.error")
    forbidden = [
        event
        for event in events
        if type(event.get("action")) is str
        and (
            event["action"].startswith(ZERO_SHARING_FORBIDDEN_ACTION_PREFIXES)
            or event["action"] == "gate.serial_blend.evidence"
        )
    ]
    if (
        len(observations) != 4
        or len(request_id_bindings) != 4
        or any(
            len(receipts) != 4
            for receipts in (
                prepare_begins,
                prepare_ends,
                matched_begins,
                matched_ends,
                update_begins,
                update_ends,
                finish_begins,
                finish_ends,
            )
        )
        or not meta_begins
        or len(meta_begins) != len(meta_ends)
        or meta_errors
        or forbidden
    ):
        raise GateError(
            "zero-sharing scheduler or no-side-effect coverage is not exact"
        )

    previous_update_end = -1
    observed_request_ids: set[str] = set()
    observed_payload_digests: set[str] = set()
    for index, receipts in enumerate(
        zip(
            observations,
            request_id_bindings,
            prepare_begins,
            prepare_ends,
            matched_begins,
            matched_ends,
            update_begins,
            update_ends,
            finish_begins,
            finish_ends,
            strict=True,
        )
    ):
        (
            observation,
            binding,
            prepare_begin,
            prepare_end,
            matched_begin,
            matched_end,
            update_begin,
            update_end,
            finish_begin,
            finish_end,
        ) = receipts
        request_id = binding.get("engine_request_id")
        expected_payload = [
            KMESH_SIDECAR_WIRE_TAG,
            ZERO_SHARING_INTENT,
            request_id,
            index + 1,
            ZERO_SHARING_PROMPT_TOKENS,
            ZERO_SHARING_CACHE_DOMAIN,
            ZERO_SHARING_FALLBACK,
            [],
        ]
        expected_sidecar = {
            "provider_id": KMESH_SIDECAR_PROVIDER,
            "payload": expected_payload,
            "payload_sha256": _sidecar_payload_sha256(expected_payload),
            "wire_tag": KMESH_SIDECAR_WIRE_TAG,
            "intent": ZERO_SHARING_INTENT,
            "descriptor_request_id": request_id,
            "admission_generation": index + 1,
            "prompt_token_count": ZERO_SHARING_PROMPT_TOKENS,
            "cache_domain": ZERO_SHARING_CACHE_DOMAIN,
            "fallback": ZERO_SHARING_FALLBACK,
            "span_count": 0,
        }
        sequences = [
            event.get("sequence")
            for event in (
                prepare_begin,
                prepare_end,
                matched_begin,
                matched_end,
                update_begin,
                update_end,
            )
        ]
        common_receipts = (
            prepare_begin,
            prepare_end,
            matched_begin,
            matched_end,
            update_begin,
            update_end,
            finish_begin,
            finish_end,
        )
        if (
            observation.get("phase") != "zero_sharing"
            or observation.get("prompt_tokens") != ZERO_SHARING_PROMPT_TOKENS
            or binding.get("phase") != "zero_sharing"
            or binding.get("trace_request_id") != observation.get("trace_request_id")
            or binding.get("external_request_id")
            != observation.get("external_request_id")
            or binding.get("binding_event_action")
            != "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
            or type(request_id) is not str
            or request_id in observed_request_ids
            or any(event.get("phase") != "zero_sharing" for event in common_receipts)
            or any(event.get("request_id") != request_id for event in common_receipts)
            or any(
                event.get("owner_type") != "LMCacheConnectorV1Impl"
                for event in common_receipts
            )
            or any(event.get("role") != "SCHEDULER" for event in common_receipts)
            or type(prepare_begin.get("sidecar_count")) is not int
            or prepare_begin.get("sidecar_count") != 1
            or type(prepare_end.get("sidecar_count")) is not int
            or prepare_end.get("sidecar_count") != 1
            or _projection_payload_signature(prepare_begin.get("sidecars"))
            != _projection_payload_signature([expected_sidecar])
            or _projection_payload_signature(prepare_end.get("sidecars"))
            != _projection_payload_signature([expected_sidecar])
            or _projection_payload_signature(prepare_begin.get("sidecars"))
            != _projection_payload_signature(prepare_end.get("sidecars"))
            or "result" in prepare_begin
            or "result" not in prepare_end
            or prepare_end.get("result") is not None
            or type(matched_begin.get("num_computed_tokens")) is not int
            or matched_begin.get("num_computed_tokens") != 0
            or type(matched_end.get("num_computed_tokens")) is not int
            or matched_end.get("num_computed_tokens") != 0
            or "result" in matched_begin
            or type(matched_end.get("result")) is not int
            or matched_end.get("result") != 0
            or type(update_begin.get("num_external_tokens")) is not int
            or update_begin.get("num_external_tokens") != 0
            or type(update_end.get("num_external_tokens")) is not int
            or update_end.get("num_external_tokens") != 0
            or "result" in update_begin
            or "result" not in update_end
            or update_end.get("result") is not None
            or "result" in finish_begin
            or "result" not in finish_end
            or finish_end.get("result") != {"type": "tuple"}
            or any(type(sequence) is not int for sequence in sequences)
            or not (
                previous_update_end
                < sequences[0]
                < sequences[1]
                < sequences[2]
                < sequences[3]
                < sequences[4]
                < sequences[5]
            )
        ):
            raise GateError(
                f"zero-sharing request {index} lacks its exact no-op lifecycle"
            )
        payload_digest = expected_sidecar["payload_sha256"]
        if payload_digest in observed_payload_digests:
            raise GateError("zero-sharing descriptor payload digests are not unique")
        observed_request_ids.add(request_id)
        observed_payload_digests.add(payload_digest)
        previous_update_end = sequences[-1]

    meta_previous_end = previous_update_end
    for index, (begin, end) in enumerate(zip(meta_begins, meta_ends, strict=True)):
        begin_sequence = begin.get("sequence")
        end_sequence = end.get("sequence")
        if (
            begin.get("phase") != "zero_sharing"
            or end.get("phase") != "zero_sharing"
            or begin.get("owner_type") != "LMCacheConnectorV1Impl"
            or end.get("owner_type") != "LMCacheConnectorV1Impl"
            or begin.get("role") != "SCHEDULER"
            or end.get("role") != "SCHEDULER"
            or type(end.get("request_count")) is not int
            or end.get("request_count") != 0
            or type(begin_sequence) is not int
            or type(end_sequence) is not int
            or not (meta_previous_end < begin_sequence < end_sequence)
        ):
            raise GateError(
                f"zero-sharing connector metadata pair {index} is not an exact no-op"
            )
        meta_previous_end = end_sequence

    projections = _events(events, "gate.collective.plan_projection")
    projection_sequences = [event.get("sequence") for event in projections]
    if (
        len(projections) != 2
        or any(event.get("phase") != "zero_sharing" for event in projections)
        or any(event.get("plans") != [] for event in projections)
        or any(type(sequence) is not int for sequence in projection_sequences)
        or not (previous_update_end < projection_sequences[0] < projection_sequences[1])
    ):
        raise GateError("zero-sharing projection receipts are not post-admission")

    previous_finish_end = max(meta_previous_end, projection_sequences[-1])
    for index, (begin, end) in enumerate(zip(finish_begins, finish_ends, strict=True)):
        begin_sequence = begin.get("sequence")
        end_sequence = end.get("sequence")
        if (
            type(begin_sequence) is not int
            or type(end_sequence) is not int
            or not (previous_finish_end < begin_sequence < end_sequence)
        ):
            raise GateError(f"zero-sharing finish pair {index} is not ordered")
        previous_finish_end = end_sequence
    completions = _events(events, "gate.request.complete")
    if (
        len(completions) != 4
        or type(completions[0].get("sequence")) is not int
        or previous_finish_end >= completions[0]["sequence"]
    ):
        raise GateError("zero-sharing requests did not finish before publication")
    return len(prepare_begins)


def _baseline_outputs(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("arm") != "ordinary"
        or value.get("status") != "complete"
        or value.get("checks", {}).get("all_passed") is not True
    ):
        raise GateError("baseline result is not a completed passing ordinary arm")
    outputs = {
        observation["trace_request_id"]: observation["output_token_ids"]
        for observation in value.get("observations", [])
    }
    if len(outputs) != 5:
        raise GateError("ordinary result does not cover all five trace requests")
    return outputs


def _recovery_evidence(
    observations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind randomized internal IDs to public IDs and immutable trace owners."""
    external_to_trace = _external_trace_bindings(observations)
    observations_by_external = {
        observation["external_request_id"]: observation for observation in observations
    }
    ordinary = all(
        observation.get("phase") in {"ordinary_prime", "ordinary_batch_forward"}
        for observation in observations
    )
    if ordinary:
        recovery_arm = "ordinary"
    else:
        observed_phase_values = [
            observation.get("phase") for observation in observations
        ]
        if any(type(phase) is not str for phase in observed_phase_values):
            raise GateError("recovery observation phase is not a plain string")
        observed_phases = set(observed_phase_values)
        non_prime_phases = observed_phases - {"prime"}
        if len(non_prime_phases) != 1:
            raise GateError("recovery observations do not identify one exact arm")
        recovery_arm = next(iter(non_prime_phases))
    binding_action = (
        "gate.numerical.request_bound"
        if ordinary
        else "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
    )
    binding_events = _events(events, binding_action)
    if len(binding_events) != len(observations):
        raise GateError("canonical request-ID event coverage differs from observations")
    external_to_internal: dict[str, str] = {}
    request_id_bindings: list[dict[str, str]] = []
    for event in binding_events:
        internal = event.get("engine_request_id" if ordinary else "request_id")
        external, trace = _bind_internal_request_id(internal, external_to_trace)
        observation = observations_by_external[external]
        if (
            external in external_to_internal
            or event.get("phase") != observation.get("phase")
            or (
                ordinary
                and event.get("trace_request_id") != observation.get("trace_request_id")
            )
            or (
                not ordinary
                and (
                    event.get("owner_type") != "LMCacheConnectorV1Impl"
                    or event.get("role") != "SCHEDULER"
                )
            )
        ):
            raise GateError("canonical request-ID event is not an exact owner")
        external_to_internal[external] = internal
        request_id_bindings.append(
            {
                "trace_request_id": trace,
                "external_request_id": external,
                "engine_request_id": internal,
                "phase": observation["phase"],
                "binding_event_action": binding_action,
            }
        )
    if list(external_to_internal) != list(external_to_trace):
        raise GateError("canonical request-ID event order differs from observations")
    if recovery_arm == "zero_sharing":
        _zero_sharing_sidecar_evidence(
            observations,
            request_id_bindings,
            events,
        )

    def bind(internal: object, phase: object) -> tuple[str, str]:
        external, trace = _bind_internal_request_id(internal, external_to_trace)
        previous = external_to_internal.get(external)
        if (
            previous != internal
            or type(phase) is not str
            or observations_by_external[external].get("phase") != phase
        ):
            raise GateError("recovery request ID differs from its canonical binding")
        return external, trace

    projected_by_phase_and_members: dict[tuple[str, frozenset[str]], dict[str, Any]] = (
        {}
    )
    for event in _canonical_projection_events(
        events,
        _expected_projection_phases(recovery_arm),
    ):
        phase = event.get("phase")
        for plan in event["plans"]:
            member_ids = frozenset(
                member["request_id"] for member in plan.get("members", [])
            )
            for member_id in member_ids:
                bind(member_id, phase)
            owner = (phase, member_ids)
            if owner in projected_by_phase_and_members:
                raise GateError("projected plan has a duplicate phase/member owner")
            projected_by_phase_and_members[owner] = plan

    groups = []
    for call_index, event in enumerate(
        _events(events, "gate.collective.blend_group.end")
    ):
        event_phase = event.get("phase")
        input_members = event["input_members"]
        result_members = event["result_members"]
        if len(input_members) != len(result_members):
            raise GateError("blend_group input/result member counts differ")
        member_ids = frozenset(member["request_id"] for member in input_members)
        if len(member_ids) != len(input_members):
            raise GateError("blend_group input contains duplicate request IDs")
        plan = projected_by_phase_and_members.get((event_phase, member_ids))
        if plan is None:
            raise GateError("blend_group input has no exact projected-plan owner")
        projected_members = plan.get("members", [])
        projected_by_request_id = {
            member["request_id"]: member for member in projected_members
        }
        if (
            len(projected_by_request_id) != len(projected_members)
            or frozenset(projected_by_request_id) != member_ids
        ):
            raise GateError("projected plan has ambiguous blend_group ownership")
        for input_member in input_members:
            projected = projected_by_request_id[input_member["request_id"]]
            if projected.get("explicit_spans") != input_member.get("explicit_spans"):
                raise GateError(
                    "blend_group selected spans differ from its projected plan"
                )
        members = []
        for input_member, result_member in zip(
            input_members, result_members, strict=True
        ):
            if result_member.get("request_id") != input_member.get(
                "request_id"
            ) or result_member.get("generation") != input_member.get("generation"):
                raise GateError(
                    "blend_group result member differs from its input owner"
                )
            engine_request_id = result_member["request_id"]
            external_request_id, trace_request_id = bind(
                engine_request_id,
                event_phase,
            )
            members.append(
                {
                    "trace_request_id": trace_request_id,
                    "external_request_id": external_request_id,
                    "engine_request_id": engine_request_id,
                    "generation": result_member["generation"],
                    "outcome": result_member["outcome"],
                    "selected_spans": input_member["explicit_spans"],
                    "retrieval_mask": result_member["retrieval_mask"],
                    "operation_counts": result_member["operation_counts"],
                    "error": result_member["error"],
                }
            )
        groups.append(
            {
                "call_index": call_index,
                "plan_id": plan["plan_id"],
                "reuse_group_id": plan["reuse_group_id"],
                "num_layers": event["num_layers"],
                "engine_request_order": [
                    member["request_id"] for member in input_members
                ],
                "members": members,
            }
        )
    serial_members = []
    for event in _events(events, "gate.serial_blend.evidence"):
        engine_request_id = event["request_id"]
        external_request_id, trace_request_id = bind(
            engine_request_id,
            event.get("phase"),
        )
        serial_members.append(
            {
                "trace_request_id": trace_request_id,
                "external_request_id": external_request_id,
                "engine_request_id": engine_request_id,
                "selected_spans": event["explicit_spans"],
                "retrieval_mask": event["retrieval_mask"],
            }
        )
    return {
        "groups": groups,
        "serial_members": serial_members,
        "request_id_bindings": request_id_bindings,
        "trace_submission_order": [
            observation["trace_request_id"]
            for observation in observations
            if observation["phase"] not in {"prime", "ordinary_prime"}
        ],
    }


def _cleanup_checks(events: list[dict[str, Any]]) -> dict[str, bool]:
    shutdowns = _events(events, "LMCacheConnectorV1Impl.shutdown.end")
    workers = [event for event in shutdowns if event.get("role") == "WORKER"]
    schedulers = [event for event in shutdowns if event.get("role") == "SCHEDULER"]
    return {
        "both_connector_roles_shut_down": bool(workers) and bool(schedulers),
        "all_managers_report_services_stopped": bool(shutdowns)
        and all(event.get("manager_services_stopped") is True for event in shutdowns),
        "worker_shutdown_has_no_residue": bool(workers)
        and all(
            all(
                event.get(field) == 0
                for field in (
                    "_kmesh_store_sessions_count",
                    "_kmesh_outstanding_lookup_ids_count",
                    "_kmesh_store_cleanup_errors_count",
                    "_kmesh_failed_worker_requests_count",
                )
            )
            for event in workers
        ),
        "scheduler_shutdown_has_no_ownership_residue": bool(schedulers)
        and all(
            all(
                event.get(field) == 0
                for field in (
                    "_prealloc_lookup_ids_count",
                    "_allocated_lookup_ids_count",
                    "_worker_owned_lookup_requests_count",
                )
            )
            for event in schedulers
        ),
    }


def _clean_prime(
    observations: list[dict[str, Any]], events: list[dict[str, Any]]
) -> bool:
    prime = [item for item in observations if item["phase"] == "prime"]
    lookups = _events(events, "LMCacheLookupClient.lookup.end", "prime")
    finishes = _events(events, "_KMeshLayerwiseStoreSession.finish.end", "prime")
    return (
        len(prime) == 1
        and not prime[0]["num_cached_tokens"]
        and len(lookups) == 1
        and lookups[0].get("result") == 0
        and len(finishes) == 1
        and finishes[0].get("result") is True
        and finishes[0].get("flushed") is True
        and finishes[0].get("closed") is True
    )


def _recovered_counts_are_exact(group: dict[str, Any]) -> bool:
    num_layers = group.get("num_layers")
    if type(num_layers) is not int or num_layers <= 0:
        return False
    for member in group.get("members", []):
        counts = member.get("operation_counts", {})
        if counts.get("position_recovery_serial_operations") != 0:
            return False
        if member.get("outcome") == "recovered" and (
            counts.get("serial_rope_operations") != 0
            or counts.get("serial_diff_operations") != 0
            or counts.get("collective_rope_operations") != num_layers
            or counts.get("collective_diff_operations") != 1
            or counts.get("position_recovery_collective_operations") != num_layers
        ):
            return False
    return True


def _evaluate(
    arm: str,
    observations: list[dict[str, Any]],
    events: list[dict[str, Any]],
    baseline: dict[str, list[int]],
    injection: dict[str, Any] | None,
    recovery: dict[str, Any],
) -> dict[str, bool]:
    checks: dict[str, bool] = {
        "observations_present": bool(observations),
        "one_generated_token_per_observation": all(
            len(item["output_token_ids"]) == 1 for item in observations
        ),
        "no_instrumented_runtime_errors": not any(
            str(event.get("action", "")).endswith(".error") for event in events
        ),
    }
    construction = _events(events, "gate.construction.end")
    if arm == "ordinary":
        checks.update(
            {
                "all_five_trace_requests_executed": len(observations) == 5,
                "ordinary_uses_forward_four_request_batch": len(
                    _events(events, "gate.batch.begin", "ordinary_batch_forward")
                )
                == 1
                and _events(events, "gate.batch.begin", "ordinary_batch_forward")[
                    0
                ].get("batch_size")
                == 4,
                "resolved_connectors_are_disabled": len(construction) == 1
                and construction[0].get("kv_transfer_config_present") is False,
                "ordinary_cached_tokens_are_zero": all(
                    not item["num_cached_tokens"] for item in observations
                ),
            }
        )
        checks["all_passed"] = all(checks.values())
        return checks

    checks.update(
        {
            "exact_output_parity": all(
                item["output_token_ids"] == baseline.get(item["trace_request_id"])
                for item in observations
            ),
            "resolved_expected_profile": len(construction) == 1
            and construction[0].get("resolved_target_profile") == _profile_for_arm(arm),
            **_cleanup_checks(events),
        }
    )
    group_calls = recovery["groups"]
    serial_members = recovery["serial_members"]

    if arm == "serial_oracle":
        checks.update(
            {
                "prime_is_clean_zero_hit_store": _clean_prime(observations, events),
                "four_probes_share_one_generate_call": len(
                    _events(events, "gate.batch.begin", "serial_oracle")
                )
                == 1
                and _events(events, "gate.batch.begin", "serial_oracle")[0].get(
                    "batch_size"
                )
                == 4,
                "serial_oracle_has_no_group_call": not group_calls,
                "serial_oracle_has_four_positive_masks": len(serial_members) == 4
                and all(
                    member["retrieval_mask"]["true_count"] > 0
                    for member in serial_members
                ),
                "every_serial_probe_advertised_a_hit": all(
                    item["num_cached_tokens"] > 0
                    for item in observations
                    if item["phase"] == "serial_oracle"
                ),
            }
        )
    elif arm in {"collective_forward", "collective_reverse"}:
        expected_order = [1, 2, 3, 4] if arm == "collective_forward" else [4, 3, 2, 1]
        group = group_calls[0] if len(group_calls) == 1 else {}
        submitted_engine_order = [
            binding["engine_request_id"]
            for binding in recovery["request_id_bindings"]
            if binding["phase"] == arm
        ]
        checks.update(
            {
                "prime_is_clean_zero_hit_store": _clean_prime(observations, events),
                "four_probes_share_one_generate_call": len(
                    _events(events, "gate.batch.begin", arm)
                )
                == 1
                and _events(events, "gate.batch.begin", arm)[0].get("batch_size") == 4,
                "trace_submission_order_is_exact": [
                    item["sequence_index"]
                    for item in observations
                    if item["phase"] == arm
                ]
                == expected_order,
                "exactly_one_four_member_group_call": len(group_calls) == 1
                and len(group.get("members", [])) == 4,
                "every_collective_probe_advertised_a_hit": all(
                    item["num_cached_tokens"] > 0
                    for item in observations
                    if item["phase"] == arm
                ),
                "group_engine_order_matches_submission": bool(group)
                and group.get("engine_request_order") == submitted_engine_order,
                "every_member_recovered": bool(group)
                and all(
                    member.get("outcome") == "recovered"
                    and member.get("error") is None
                    and member.get("retrieval_mask", {}).get("true_count", 0) > 0
                    for member in group.get("members", [])
                ),
                "per_member_operation_counts_are_exact": bool(group)
                and _recovered_counts_are_exact(group),
                "collective_kernel_counts_are_explicitly_zero": bool(group)
                and all(
                    member["operation_counts"]["collective_rope_operations"]
                    == group.get("num_layers")
                    and member["operation_counts"]["collective_diff_operations"] == 1
                    for member in group.get("members", [])
                    if member.get("outcome") == "recovered"
                ),
            }
        )
    elif arm == "cold_miss":
        lookups = _events(events, "LMCacheLookupClient.lookup.end", "cold_miss")
        checks.update(
            {
                "one_cold_probe_executed": len(observations) == 1,
                "cold_lookup_is_zero": bool(lookups)
                and all(event.get("result") == 0 for event in lookups),
                "cold_probe_reports_zero_cached_tokens": all(
                    not item["num_cached_tokens"] for item in observations
                ),
                "cold_probe_has_no_recovery_call": not group_calls
                and not serial_members,
            }
        )
    elif arm == "zero_sharing":
        plan_events = [
            event
            for event in _canonical_projection_events(
                events,
                _expected_projection_phases(arm),
            )
            if event["phase"] == "zero_sharing"
        ]
        batch_begins = _events(events, "gate.batch.begin", arm)
        batch_ends = _events(events, "gate.batch.end", arm)
        zero_schedulers = [
            event
            for event in _events(events, "LMCacheConnectorV1Impl.shutdown.end")
            if event.get("role") == "SCHEDULER"
        ]
        checks.update(
            {
                "four_zero_sharing_requests_share_one_generate_call": (
                    len(observations) == 4
                    and len(batch_begins) == 1
                    and len(batch_ends) == 1
                    and batch_begins[0].get("batch_size") == 4
                    and batch_ends[0].get("batch_size") == 4
                ),
                "zero_sharing_noop_lifecycle_is_exact": True,
                "zero_sharing_connector_metadata_is_empty": True,
                "zero_sharing_has_no_data_plane_side_effect": True,
                "zero_sharing_forms_no_plan": len(plan_events) == 1
                and plan_events[0].get("plans") == [],
                "zero_sharing_performs_no_lookup_rpc": not _events(
                    events, "LMCacheLookupClient.lookup.begin"
                )
                and not _events(events, "LMCacheLookupClient.lookup.end"),
                "zero_sharing_has_no_recovery_call": not group_calls
                and not serial_members,
                "zero_sharing_reports_zero_cached_tokens": all(
                    type(item.get("num_cached_tokens")) is int
                    and item.get("num_cached_tokens") == 0
                    for item in observations
                ),
                "zero_sharing_scheduler_state_is_fully_drained": bool(zero_schedulers)
                and all(
                    all(
                        type(event.get(f"{field}_count")) is int
                        and event.get(f"{field}_count") == 0
                        for field in ZERO_SHARING_SCHEDULER_STATE_FIELDS
                    )
                    for event in zero_schedulers
                ),
            }
        )
    elif arm == "incompatible_partition":
        plan_events = [
            event
            for event in _canonical_projection_events(
                events,
                _expected_projection_phases(arm),
            )
            if event["phase"] == "incompatible_partition"
        ]
        plans = plan_events[0]["plans"] if plan_events else []
        group = group_calls[0] if len(group_calls) == 1 else {}
        trace_members = {
            member.get("trace_request_id") for member in group.get("members", [])
        }
        expected_a = {
            item["trace_request_id"]
            for item in observations
            if item["sequence_index"] in {1, 2}
        }
        checks.update(
            {
                "prime_is_clean_zero_hit_store": _clean_prime(observations, events),
                "planner_emits_two_disjoint_partitions": len(plans) == 2
                and sorted(len(plan.get("members", [])) for plan in plans) == [2, 2],
                "only_primed_partition_executes_collectively": len(group_calls) == 1
                and trace_members == expected_a,
                "unprimed_partition_reports_zero_cached_tokens": all(
                    not item["num_cached_tokens"]
                    for item in observations
                    if item["sequence_index"] in {3, 4}
                ),
                "partition_group_counts_are_exact": bool(group)
                and _recovered_counts_are_exact(group),
            }
        )
    elif arm == "sibling_failure":
        group = group_calls[0] if len(group_calls) == 1 else {}
        failed = [
            member
            for member in group.get("members", [])
            if member.get("outcome") == "failed"
        ]
        recovered = [
            member
            for member in group.get("members", [])
            if member.get("outcome") == "recovered"
        ]
        failures = _events(
            events,
            "LMCacheConnectorV1Impl._record_kmesh_load_failure.end",
            "sibling_failure",
        )
        checks.update(
            {
                "prime_is_clean_zero_hit_store": _clean_prime(observations, events),
                "one_sibling_failure_was_injected": injection is not None
                and injection.get("count") == 1,
                "one_failed_member_and_three_recovered_siblings": len(failed) == 1
                and len(recovered) == 3,
                "recovered_sibling_counts_are_exact": bool(group)
                and _recovered_counts_are_exact(group),
                "failed_member_is_invalidated": (
                    len(failures) == 1
                    and failures[0].get("request_id")
                    == failed[0].get("engine_request_id")
                    if len(failed) == 1
                    else False
                ),
                "failed_member_does_not_abort_group": len(observations) == 5,
                "every_sibling_probe_advertised_a_hit": all(
                    item["num_cached_tokens"] > 0
                    for item in observations
                    if item["phase"] == "sibling_failure"
                ),
            }
        )
    checks["all_passed"] = all(checks.values())
    return checks


def _validate_numerical_result(output_dir: Path, value: Any) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("enabled")) is not bool:
        raise GateError("numerical instrumentation returned an invalid binding")
    if not value["enabled"]:
        if value.get("status") != "disabled":
            raise GateError("disabled numerical instrumentation has invalid status")
        return value
    if value.get("status") != "complete":
        raise GateError("enabled numerical instrumentation did not complete")
    relative = value.get("manifest_path")
    digest = value.get("manifest_sha256")
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise GateError("numerical manifest path must be relative to the arm")
    if relative != "post-pic-capture-v1/family-manifest.json":
        raise GateError("numerical manifest path differs from capture schema v1")
    path = output_dir / relative
    if not path.resolve().is_relative_to(output_dir.resolve()):
        raise GateError("numerical manifest must remain inside the arm artifacts")
    if type(digest) is not str or digest != base._sha256(path):
        raise GateError("numerical manifest digest does not bind its bytes")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "kmesh.post_pic_kv_capture.v1":
        raise GateError("numerical manifest has an unexpected schema")
    return value


def _execute_arm(
    arm: str,
    llm: Any,
    records: list[dict[str, Any]],
    ledger: EventLedger,
) -> list[dict[str, Any]]:
    connectors = arm != "ordinary"
    observations: list[dict[str, Any]] = []
    if arm == "ordinary":
        observations.extend(
            _run_batch(
                llm,
                [records[0]],
                "ordinary_prime",
                ledger,
                connectors=False,
            )
        )
        observations.extend(
            _run_batch(
                llm,
                records[1:],
                "ordinary_batch_forward",
                ledger,
                connectors=False,
            )
        )
    elif arm in {
        "serial_oracle",
        "collective_forward",
        "collective_reverse",
        "incompatible_partition",
        "sibling_failure",
    }:
        observations.extend(
            _run_batch(llm, [records[0]], "prime", ledger, connectors=True)
        )
        probes = records[1:]
        if arm == "collective_reverse":
            probes = list(reversed(probes))
        elif arm == "incompatible_partition":
            probes = [
                _partition_b(record) if record["sequence_index"] in {3, 4} else record
                for record in probes
            ]
        observations.extend(_run_batch(llm, probes, arm, ledger, connectors=connectors))
    elif arm == "cold_miss":
        observations.extend(
            _run_batch(llm, [records[-1]], arm, ledger, connectors=True)
        )
    elif arm == "zero_sharing":
        observations.extend(
            _run_batch(
                llm,
                [_without_sharing(record) for record in records[1:]],
                arm,
                ledger,
                connectors=True,
            )
        )
    return observations


def main() -> None:
    args = _parse_args()
    if args.arm != "ordinary" and args.baseline_result is None:
        raise GateError("connector arms require --baseline-result")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise GateError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base._atomic_json(
        args.output_dir / "status.json",
        {
            "schema_version": common.SCHEMA_VERSION,
            "status": "running",
            "arm": args.arm,
            "started_at_utc": base._utc_now(),
            "pid": os.getpid(),
        },
    )
    ledger = EventLedger(args.output_dir / "events.jsonl")
    llm: Any = None
    cleaned = False
    ledger_closed = False
    try:
        if os.environ.get("KMESH_IMAGE_DIGEST") != common.EXPECTED_IMAGE:
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

        source = common.source_identity(args.vllm_source, args.lmcache_source)
        manifest, records = base._load_trace(args.trace, args.manifest)
        baseline = _baseline_outputs(args.baseline_result)
        sys.path.insert(0, str(args.lmcache_source))
        vllm = base._load_vllm_overlay(args.vllm_source, args.wheel_vllm_package)
        import lmcache
        import torch
        import transformers
        from lmcache.v1.config import LMCacheEngineConfig
        from vllm import LLM

        extensions = {
            name: Path(importlib.import_module(f"vllm.{name}").__file__)
            for name in common.EXPECTED_EXTENSION_HASHES
        }
        extension_hashes = {
            name: base._sha256(path) for name, path in extensions.items()
        }
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
        resolved_lmcache = LMCacheEngineConfig.from_file(args.lmcache_config)
        injection = (
            _install_instrumentation(
                ledger,
                inject_sibling_failure=args.arm == "sibling_failure",
            )
            if args.arm != "ordinary"
            else None
        )
        numerical = numerical_instrumentation.install(
            ledger=ledger,
            arm=args.arm,
            output_dir=args.output_dir,
            records=tuple(copy.deepcopy(records)),
            source=copy.deepcopy(source),
        )

        profile = _profile_for_arm(args.arm)
        engine_kwargs = common.engine_kwargs(
            connectors=profile is not None,
            profile=profile or common.COLLECTIVE_PROFILE,
        )
        ledger.emit("gate.construction.begin", arm=args.arm)
        llm = LLM(**engine_kwargs)
        from vllm.v1.kmesh.config import validate_kmesh_target_profile

        resolved_profile = validate_kmesh_target_profile(llm.llm_engine.vllm_config)
        if resolved_profile != profile:
            raise GateError(
                f"resolved target profile {resolved_profile!r} != {profile!r}"
            )
        attention_backend = getattr(
            llm.llm_engine.vllm_config.attention_config.backend,
            "name",
            None,
        )
        if attention_backend != "TRITON_ATTN":
            raise GateError("runtime did not resolve the exact TRITON_ATTN backend")
        ledger.emit(
            "gate.construction.end",
            arm=args.arm,
            resolved_target_profile=resolved_profile,
            resolved_attention_backend=attention_backend,
            kv_transfer_config_present=(
                llm.llm_engine.vllm_config.kv_transfer_config is not None
            ),
        )

        observations = _execute_arm(args.arm, llm, records, ledger)
        pre_cleanup_events = ledger.snapshot()
        recovery = _recovery_evidence(observations, pre_cleanup_events)
        numerical_result = _validate_numerical_result(
            args.output_dir,
            numerical.finish(
                arm=args.arm,
                observations=observations,
                recovery_evidence=recovery,
                source=source,
                runtime=runtime,
            ),
        )

        base._cleanup_llm(llm, ledger)
        llm = None
        cleaned = True
        events = ledger.snapshot()
        ledger.close()
        ledger_closed = True
        checks = _evaluate(
            args.arm,
            observations,
            events,
            baseline,
            injection,
            recovery,
        )
        if not checks["all_passed"]:
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise GateError("arm checks failed: " + ", ".join(failed))

        result = {
            "schema_version": common.SCHEMA_VERSION,
            "status": "complete",
            "arm": args.arm,
            "request_id_contract": REQUEST_ID_CONTRACT,
            "checks": checks,
            "observations": observations,
            "recovery_evidence": recovery,
            "numerical_evidence": numerical_result,
            "source": source,
            "runtime": runtime,
            "inputs": {
                "runner": {
                    "path": str(Path(__file__)),
                    "sha256": base._sha256(Path(__file__)),
                },
                "common": {
                    "path": str(Path(common.__file__)),
                    "sha256": base._sha256(Path(common.__file__)),
                },
                "accepted_serial_runner": {
                    "path": str(common._serial_gate_path()),
                    "sha256": common.SERIAL_GATE_SHA256,
                },
                "numerical_instrumentation": {
                    "path": str(Path(numerical_instrumentation.__file__)),
                    "sha256": base._sha256(Path(numerical_instrumentation.__file__)),
                },
                "lmcache_config": {
                    "path": str(args.lmcache_config),
                    "sha256": base._sha256(args.lmcache_config),
                    "resolved": base._jsonable(resolved_lmcache),
                },
                "trace": {
                    "path": str(args.trace),
                    "sha256": base._sha256(args.trace),
                },
                "manifest": {
                    "path": str(args.manifest),
                    "sha256": base._sha256(args.manifest),
                    "trace_id": records[0]["trace_id"],
                    "oracle": manifest.get("oracle"),
                },
                "baseline_result": (
                    {
                        "path": str(args.baseline_result),
                        "sha256": base._sha256(args.baseline_result),
                    }
                    if args.baseline_result is not None
                    else None
                ),
            },
            "engine": {
                "resolved_target_profile": resolved_profile,
                "resolved_attention_backend": attention_backend,
                "kwargs": engine_kwargs,
            },
            "event_ledger": {
                "path": str(ledger.path),
                "sha256": base._sha256(ledger.path),
                "event_count": len(events),
            },
            "claim_boundary": (
                "Correctness evidence only. The pinned profile batches cached-key "
                "recovery and model-side rotary/difference operations while "
                "keeping selection, attention and destinations request-local. "
                "No latency, throughput, "
                "Master-Mirror, capacity, generic-model, distributed, production, "
                "or performance claim follows. Ordinary numerical batch-shape "
                "comparability is forward-order only. "
                + (
                    "Raw numerical material is separately hash-bound but is not "
                    "interpreted by this correctness runner."
                    if numerical_result["enabled"]
                    else "This arm does not capture raw K/V tensors or logits."
                )
            ),
        }
        base._atomic_json(args.output_dir / "result.json", result)
        base._atomic_json(
            args.output_dir / "status.json",
            {
                "schema_version": common.SCHEMA_VERSION,
                "status": "complete",
                "arm": args.arm,
                "completed_at_utc": base._utc_now(),
                "result_sha256": base._sha256(args.output_dir / "result.json"),
            },
        )
        print(json.dumps({"arm": args.arm, "checks": checks}, sort_keys=True))
    except BaseException as error:
        original_traceback = traceback.format_exc()
        cleanup_error = None
        if llm is not None and not cleaned:
            try:
                base._cleanup_llm(llm, ledger)
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
            "schema_version": common.SCHEMA_VERSION,
            "status": "failed",
            "arm": args.arm,
            "failed_at_utc": base._utc_now(),
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": original_traceback,
            "cleanup_error": cleanup_error,
            "events_sha256": base._sha256(ledger.path),
        }
        base._atomic_json(args.output_dir / "failure.json", failure)
        base._atomic_json(
            args.output_dir / "status.json",
            {
                "schema_version": common.SCHEMA_VERSION,
                "status": "failed",
                "arm": args.arm,
                "failed_at_utc": failure["failed_at_utc"],
                "failure_sha256": base._sha256(args.output_dir / "failure.json"),
            },
        )
        raise


if __name__ == "__main__":
    main()
