#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify and bind independently completed KMesh collective-lockstep arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import profile, validate_current_runtime, validate_sources

import safetensors
import torch
from safetensors.torch import load_file

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
NUMERICAL_ARMS = ("ordinary", "serial_oracle", "collective_forward")
KV_SHAPE = (24, 512, 128)
VOCAB_SIZE = 151936
LOCKSTEP_LOGITS_ATOL = 1e-5
LOCKSTEP_LOGITS_RTOL = 1e-4
ORDINARY_KV_ATOL = 5e-2
ORDINARY_KV_RTOL = 5e-2
ORDINARY_LOGITS_ATOL = 5e-2
ORDINARY_LOGITS_RTOL = 5e-2
CAPTURE_BOUNDARY = (
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner.sample_tokens:"
    "entry_before_execute_model_state_clear_and_before_scheduler_block_free"
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
OBSERVATION_FIELDS = (
    "phase",
    "batch_ordinal",
    "trace_request_id",
    "external_request_id",
    "sequence_index",
    "prompt_tokens",
    "num_cached_tokens",
    "output_token_ids",
    "finish_reason",
)
OPERATION_COUNT_FIELDS = (
    "serial_rope_operations",
    "serial_diff_operations",
    "collective_rope_operations",
    "collective_diff_operations",
    "position_recovery_serial_operations",
    "position_recovery_collective_operations",
)
WORKER_RESIDUE_FIELDS = (
    "_kmesh_store_sessions_count",
    "_kmesh_outstanding_lookup_ids_count",
    "_kmesh_store_cleanup_errors_count",
    "_kmesh_failed_worker_requests_count",
)
SCHEDULER_RESIDUE_FIELDS = (
    "_prealloc_lookup_ids_count",
    "_allocated_lookup_ids_count",
    "_worker_owned_lookup_requests_count",
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _projection_payload_signature(
    path: Path,
    value: object,
) -> tuple[object, ...]:
    """Return a type-sensitive signature for one plain JSON payload."""
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{path} projection payload has a non-finite float")
        return ("float", value.hex())
    if type(value) is str:
        return ("str", value)
    if type(value) is list:
        return (
            "list",
            tuple(_projection_payload_signature(path, item) for item in value),
        )
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise RuntimeError(f"{path} projection payload has a non-string key")
        return (
            "dict",
            tuple(
                (key, _projection_payload_signature(path, value[key]))
                for key in sorted(value)
            ),
        )
    raise RuntimeError(f"{path} projection payload is not plain JSON")


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
    raise RuntimeError(f"unknown arm for projection receipts: {arm}")


def _validate_sibling_projection_causality(
    path: Path,
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
            raise RuntimeError(
                f"{path} sibling-failure causal event has an invalid sequence"
            )
        return value

    def request_finish_owners(
        finish_events: list[dict[str, Any]],
        terminal: str,
    ) -> dict[str, dict[str, Any]]:
        if len(finish_events) != 4:
            raise RuntimeError(
                f"{path} sibling-failure request-finish {terminal} coverage is "
                "incomplete"
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
                raise RuntimeError(
                    f"{path} sibling-failure request-finish {terminal} owner is invalid"
                )
            sequence(event)
            if terminal == "end" and (
                type(event.get("result")) is not dict
                or event["result"] != {"type": "tuple"}
                or type(event["result"].get("type")) is not str
            ):
                raise RuntimeError(
                    f"{path} sibling-failure request-finish result is invalid"
                )
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
        raise RuntimeError(
            f"{path} sibling-failure retry lifecycle coverage is incomplete"
        )

    result_members = group_ends[0].get("result_members")
    if type(result_members) is not list or any(
        type(member) is not dict for member in result_members
    ):
        raise RuntimeError(f"{path} sibling-failure group result is malformed")
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
        raise RuntimeError(
            f"{path} sibling-failure group outcomes are not one failed and three "
            "recovered"
        )
    failed_id = failed_ids[0]
    if any(
        event.get("request_id") != failed_id
        for event in (injections[0], failure_begins[0], failure_ends[0])
    ):
        raise RuntimeError(
            f"{path} sibling-failure causal events disagree on the failed owner"
        )

    finish_begins_by_id = request_finish_owners(finish_begins, "begin")
    finish_ends_by_id = request_finish_owners(finish_ends, "end")
    expected_finish_ids = set([failed_id, *recovered_ids])
    if (
        set(finish_begins_by_id) != expected_finish_ids
        or set(finish_ends_by_id) != expected_finish_ids
    ):
        raise RuntimeError(
            f"{path} sibling-failure request finishes have ambiguous ownership"
        )
    for request_id in expected_finish_ids:
        if sequence(finish_begins_by_id[request_id]) >= sequence(
            finish_ends_by_id[request_id]
        ):
            raise RuntimeError(
                f"{path} sibling-failure request-finish pair is not ordered"
            )

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
        raise RuntimeError(
            f"{path} sibling-failure retry is not causally after the failure"
        )

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
        raise RuntimeError(
            f"{path} sibling-failure retry metadata is not exactly empty"
        )
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
        raise RuntimeError(
            f"{path} sibling-failure retry lifecycle is not ledger ordered"
        )


def _canonical_projection_events(
    path: Path,
    events: list[dict[str, Any]],
    expected_round_phases: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Require the exact ordered sequence of validate/bind projection pairs."""
    projection_events = [
        event
        for event in events
        if event.get("action") == "gate.collective.plan_projection"
    ]
    expected_receipt_phases = tuple(
        phase
        for round_phase in expected_round_phases
        for phase in (round_phase, round_phase)
    )
    if len(projection_events) != len(expected_receipt_phases):
        raise RuntimeError(
            f"{path} projection validate/bind receipt coverage is incomplete"
        )

    canonical: list[dict[str, Any]] = []
    validate_signature: tuple[object, ...] | None = None
    previous_sequence: int | None = None
    for event_index, (event, expected_phase) in enumerate(
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
            raise RuntimeError(f"{path} projection event {event_index} is malformed")
        if phase != expected_phase:
            raise RuntimeError(
                f"{path} projection event {event_index} has unexpected phase order"
            )
        if previous_sequence is not None and sequence <= previous_sequence:
            raise RuntimeError(f"{path} projection receipts are not ledger ordered")
        previous_sequence = sequence
        signature = _projection_payload_signature(path, plans)
        if event_index % 2 == 0:
            validate_signature = signature
        else:
            if signature != validate_signature:
                raise RuntimeError(
                    f"{path} projection replay for {phase} differs from its "
                    "canonical payload"
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
            raise RuntimeError(
                f"{path} sibling-failure projection rounds are not nonempty then empty"
            )
        _validate_sibling_projection_causality(
            path,
            events,
            projection_events,
            sibling_rounds,
        )
    return canonical


def _sidecar_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_zero_sharing_sidecar_evidence(
    path: Path,
    observations: list[dict[str, Any]],
    request_id_bindings: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Reconstruct the exact no-op descriptor and scheduler lifecycle."""

    def select(action: str) -> list[dict[str, Any]]:
        return [event for event in events if event.get("action") == action]

    prepare_begins = select("LMCacheConnectorV1Impl.prepare_request_lookup.begin")
    prepare_ends = select("LMCacheConnectorV1Impl.prepare_request_lookup.end")
    matched_begins = select("LMCacheConnectorV1Impl.get_num_new_matched_tokens.begin")
    matched_ends = select("LMCacheConnectorV1Impl.get_num_new_matched_tokens.end")
    update_begins = select("LMCacheConnectorV1Impl.update_state_after_alloc.begin")
    update_ends = select("LMCacheConnectorV1Impl.update_state_after_alloc.end")
    finish_begins = select("LMCacheConnectorV1Impl.request_finished.begin")
    finish_ends = select("LMCacheConnectorV1Impl.request_finished.end")
    meta_begins = select("gate.lmcache_connector_meta.begin")
    meta_ends = select("gate.lmcache_connector_meta.end")
    meta_errors = select("gate.lmcache_connector_meta.error")
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
        raise RuntimeError(
            f"{path} zero-sharing scheduler or no-side-effect coverage is not exact"
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
        label = f"{path}:zero_sharing.request[{index}]"
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
            or _projection_payload_signature(path, prepare_begin.get("sidecars"))
            != _projection_payload_signature(path, [expected_sidecar])
            or _projection_payload_signature(path, prepare_end.get("sidecars"))
            != _projection_payload_signature(path, [expected_sidecar])
            or _projection_payload_signature(path, prepare_begin.get("sidecars"))
            != _projection_payload_signature(path, prepare_end.get("sidecars"))
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
            raise RuntimeError(f"{label} lacks its exact no-op lifecycle")
        payload_digest = expected_sidecar["payload_sha256"]
        if payload_digest in observed_payload_digests:
            raise RuntimeError(f"{path} zero-sharing payload digests are not unique")
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
            raise RuntimeError(
                f"{path} zero-sharing metadata pair {index} is not an exact no-op"
            )
        meta_previous_end = end_sequence

    projections = select("gate.collective.plan_projection")
    projection_sequences = [event.get("sequence") for event in projections]
    if (
        len(projections) != 2
        or any(event.get("phase") != "zero_sharing" for event in projections)
        or any(event.get("plans") != [] for event in projections)
        or any(type(sequence) is not int for sequence in projection_sequences)
        or not (previous_update_end < projection_sequences[0] < projection_sequences[1])
    ):
        raise RuntimeError(
            f"{path} zero-sharing projection receipts are not post-admission"
        )

    previous_finish_end = max(meta_previous_end, projection_sequences[-1])
    for index, (begin, end) in enumerate(zip(finish_begins, finish_ends, strict=True)):
        begin_sequence = begin.get("sequence")
        end_sequence = end.get("sequence")
        if (
            type(begin_sequence) is not int
            or type(end_sequence) is not int
            or not (previous_finish_end < begin_sequence < end_sequence)
        ):
            raise RuntimeError(f"{path} zero-sharing finish pair {index} is unordered")
        previous_finish_end = end_sequence
    completions = select("gate.request.complete")
    if (
        len(completions) != 4
        or type(completions[0].get("sequence")) is not int
        or previous_finish_end >= completions[0]["sequence"]
    ):
        raise RuntimeError(
            f"{path} zero-sharing requests did not finish before publication"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _plain_event_identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise RuntimeError(f"{label} is not an exact nonempty identifier")
    return value


def _verify_observation_events(
    path: Path,
    observations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Reconstruct every published observation from the ordered event ledger."""
    expected_fields = set(OBSERVATION_FIELDS)
    for index, observation in enumerate(observations):
        label = f"{path}:observations[{index}]"
        if type(observation) is not dict or set(observation) != expected_fields:
            raise RuntimeError(f"{label} has a noncanonical payload shape")
        if (
            type(observation["phase"]) is not str
            or not observation["phase"]
            or type(observation["batch_ordinal"]) is not int
            or observation["batch_ordinal"] < 0
            or type(observation["trace_request_id"]) is not str
            or not observation["trace_request_id"]
            or type(observation["external_request_id"]) is not str
            or not observation["external_request_id"]
            or type(observation["sequence_index"]) is not int
            or observation["sequence_index"] < 0
            or type(observation["prompt_tokens"]) is not int
            or observation["prompt_tokens"] <= 0
            or type(observation["num_cached_tokens"]) is not int
            or observation["num_cached_tokens"] < 0
            or type(observation["output_token_ids"]) is not list
            or any(
                type(token_id) is not int or token_id < 0
                for token_id in observation["output_token_ids"]
            )
            or type(observation["finish_reason"]) is not str
            or not observation["finish_reason"]
        ):
            raise RuntimeError(f"{label} has malformed runtime values")

    completion_events = [
        event for event in events if event.get("action") == "gate.request.complete"
    ]
    reconstructed = [
        {field: event.get(field) for field in OBSERVATION_FIELDS}
        for event in completion_events
    ]
    if reconstructed != observations:
        raise RuntimeError(
            f"{path} observations differ from ordered hash-bound completion events"
        )


def _verify_positive_mask(value: object, token_count: int, label: str) -> None:
    if type(value) is not dict or set(value) != {
        "token_count",
        "true_count",
        "true_indices",
        "sha256",
    }:
        raise RuntimeError(f"{label} is not canonical mask evidence")
    observed_token_count = value["token_count"]
    true_count = value["true_count"]
    true_indices = value["true_indices"]
    digest = value["sha256"]
    if (
        type(observed_token_count) is not int
        or observed_token_count != token_count
        or type(true_count) is not int
        or true_count <= 0
        or true_count > observed_token_count
        or type(true_indices) is not list
        or any(
            type(index) is not int or index < 0 or index >= observed_token_count
            for index in true_indices
        )
        or true_indices != sorted(set(true_indices))
        or len(true_indices) != true_count
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in LOWER_HEX for character in digest)
    ):
        raise RuntimeError(f"{label} mask geometry is malformed")
    true_index_set = set(true_indices)
    encoded = bytes(
        int(index in true_index_set) for index in range(observed_token_count)
    )
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise RuntimeError(f"{label} mask digest does not bind its true indices")


def _verify_span_evidence(
    value: object,
    label: str,
    retrieval_mask: object | None = None,
) -> None:
    if type(value) is not dict or set(value) != {
        "token_count",
        "cached_span",
        "spans",
    }:
        raise RuntimeError(f"{label} is not canonical span evidence")
    spans = value["spans"]
    cached_span = value["cached_span"]
    if (
        type(value["token_count"]) is not int
        or value["token_count"] != KV_SHAPE[1]
        or type(cached_span) is not list
        or len(cached_span) != 2
        or any(type(bound) is not int for bound in cached_span)
        or cached_span != [0, KV_SHAPE[1]]
        or type(spans) is not list
        or not spans
    ):
        raise RuntimeError(f"{label} span geometry is malformed")
    previous_end = 0
    selected_indices: list[int] = []
    for index, span in enumerate(spans):
        if (
            type(span) is not list
            or len(span) != 2
            or any(type(bound) is not int for bound in span)
        ):
            raise RuntimeError(f"{label}.spans[{index}] is not an integer range")
        start, end = span
        if start < previous_end or start < 0 or end <= start or end > KV_SHAPE[1]:
            raise RuntimeError(f"{label}.spans[{index}] is out of order or bounds")
        selected_indices.extend(range(start, end))
        previous_end = end
    if retrieval_mask is not None and (
        type(retrieval_mask) is not dict
        or retrieval_mask.get("true_indices") != selected_indices
    ):
        raise RuntimeError(f"{label} does not select the recovered mask indices")


def _operation_counts_match(value: object, expected: dict[str, int]) -> bool:
    return (
        type(value) is dict
        and set(value) == set(OPERATION_COUNT_FIELDS)
        and all(type(value[field]) is int for field in OPERATION_COUNT_FIELDS)
        and value == expected
    )


def _verify_observation_layout(
    path: Path,
    arm: str,
    observations: list[dict[str, Any]],
) -> None:
    expected_sequences = {
        "ordinary": [0, 1, 2, 3, 4],
        "serial_oracle": [0, 1, 2, 3, 4],
        "collective_forward": [0, 1, 2, 3, 4],
        "collective_reverse": [0, 4, 3, 2, 1],
        "cold_miss": [4],
        "zero_sharing": [1, 2, 3, 4],
        "incompatible_partition": [0, 1, 2, 3, 4],
        "sibling_failure": [0, 1, 2, 3, 4],
    }[arm]
    prime_phase = "ordinary_prime" if arm == "ordinary" else "prime"
    request_phase = "ordinary_batch_forward" if arm == "ordinary" else arm
    expected_phases = (
        [request_phase]
        if arm == "cold_miss"
        else (
            [request_phase] * 4
            if arm == "zero_sharing"
            else [prime_phase] + [request_phase] * 4
        )
    )
    sequences = [item.get("sequence_index") for item in observations]
    phases = [item.get("phase") for item in observations]
    if (
        any(type(sequence) is not int for sequence in sequences)
        or sequences != expected_sequences
        or phases != expected_phases
    ):
        raise RuntimeError(f"{path} has the wrong {arm} observation layout")


def _verify_event_id_domains(
    path: Path,
    events: list[dict[str, Any]],
    request_id_bindings: list[dict[str, Any]],
) -> None:
    engine_phase_owners = {
        binding["engine_request_id"]: binding.get("phase")
        for binding in request_id_bindings
    }
    external_phase_owners = {
        binding["external_request_id"]: binding.get("phase")
        for binding in request_id_bindings
    }
    trace_phase_owners = {
        binding["trace_request_id"]: binding.get("phase")
        for binding in request_id_bindings
    }
    if any(
        type(binding.get("phase")) is not str for binding in request_id_bindings
    ) or any(
        len(owners) != len(request_id_bindings)
        for owners in (
            engine_phase_owners,
            external_phase_owners,
            trace_phase_owners,
        )
    ):
        raise RuntimeError(f"{path} has ambiguous request-ID phase owners")
    engine_ids = set(engine_phase_owners)
    external_ids = set(external_phase_owners)
    trace_ids = set(trace_phase_owners)
    request_domain_values = engine_ids | external_ids | trace_ids

    def verify_lookup(value: object, label: str) -> None:
        lookup_id = _plain_event_identifier(value, label)
        if lookup_id in request_domain_values:
            raise RuntimeError(f"{label} collapsed into a request-ID domain")
        try:
            parsed = uuid.UUID(lookup_id)
        except (AttributeError, ValueError):
            raise RuntimeError(f"{label} is not a canonical UUIDv4") from None
        if (
            parsed.version != 4
            or parsed.variant != uuid.RFC_4122
            or str(parsed) != lookup_id
        ):
            raise RuntimeError(f"{label} is not a canonical UUIDv4")

    retrieve_actions = {
        f"LMCacheEngine.retrieve_layer.{suffix}"
        for suffix in ("begin", "end", "closed", "error")
    }
    store_actions = {
        f"LMCacheEngine.store_layer.{suffix}"
        for suffix in ("begin", "end", "closed", "error")
    }
    lookup_begins = [
        event
        for event in events
        if event.get("action") == "LMCacheLookupClient.lookup.begin"
    ]
    lookup_ends = [
        event
        for event in events
        if event.get("action") == "LMCacheLookupClient.lookup.end"
    ]
    begin_by_owner: dict[tuple[str, str], int] = {}
    end_by_owner: dict[tuple[str, str], int] = {}
    for label, source, destination in (
        ("begin", lookup_begins, begin_by_owner),
        ("end", lookup_ends, end_by_owner),
    ):
        for event in source:
            phase = event.get("phase")
            lookup_id = event.get("lookup_id")
            sequence = event.get("sequence")
            if type(phase) is not str or type(sequence) is not int:
                raise RuntimeError(f"{path} lookup {label} owner is malformed")
            verify_lookup(
                lookup_id,
                f"{path}:events[{sequence}].lookup_id",
            )
            owner = (phase, lookup_id)
            if owner in destination:
                raise RuntimeError(f"{path} lookup {label} owner is duplicated")
            destination[owner] = sequence
    if set(begin_by_owner) != set(end_by_owner) or any(
        begin_by_owner[owner] >= end_by_owner[owner] for owner in begin_by_owner
    ):
        raise RuntimeError(f"{path} lookup begin/end ownership drifted")
    if len({lookup_id for _phase, lookup_id in begin_by_owner}) != len(begin_by_owner):
        raise RuntimeError(f"{path} reused a lookup UUID across phases")
    canonical_lookup_owners = set(begin_by_owner)
    for event in events:
        sequence = event.get("sequence")
        label = f"{path}:events[{sequence}]"
        for field, phase_owners in (
            ("trace_request_id", trace_phase_owners),
            ("external_request_id", external_phase_owners),
            ("engine_request_id", engine_phase_owners),
            ("request_id", engine_phase_owners),
        ):
            if field in event and (
                event[field] not in phase_owners
                or event.get("phase") != phase_owners[event[field]]
            ):
                raise RuntimeError(f"{label}.{field} crossed identity or phase domains")
        if "lookup_id" in event:
            verify_lookup(event["lookup_id"], f"{label}.lookup_id")
            if (event.get("phase"), event["lookup_id"]) not in (
                canonical_lookup_owners
            ):
                raise RuntimeError(f"{label}.lookup_id has no same-phase owner")
        if "lookup_ids" in event:
            lookup_ids = event["lookup_ids"]
            if type(lookup_ids) is not list:
                raise RuntimeError(f"{label}.lookup_ids is not a plain list")
            for index, lookup_id in enumerate(lookup_ids):
                verify_lookup(lookup_id, f"{label}.lookup_ids[{index}]")
                if (event.get("phase"), lookup_id) not in canonical_lookup_owners:
                    raise RuntimeError(
                        f"{label}.lookup_ids[{index}] has no same-phase owner"
                    )

        action = event.get("action")
        if action in retrieve_actions:
            if (
                "request_id" in event
                or "lookup_id" not in event
                or (event.get("phase"), event.get("lookup_id"))
                not in canonical_lookup_owners
            ):
                raise RuntimeError(
                    f"{label} did not bind retrieve_layer to its lookup owner"
                )
        elif action in store_actions:
            if "request_id" not in event or "lookup_id" in event:
                raise RuntimeError(
                    f"{label} did not keep store_layer request identity separate"
                )

    def verify_layer_lifecycles(
        stem: str,
        identity_field: str,
    ) -> None:
        begin_action = f"{stem}.begin"
        terminal_actions = {f"{stem}.{suffix}" for suffix in ("end", "closed", "error")}
        begins: dict[tuple[str, str], int] = {}
        terminals: dict[tuple[str, str], int] = {}
        for event in events:
            action = event.get("action")
            if action != begin_action and action not in terminal_actions:
                continue
            phase = event.get("phase")
            identity = event.get(identity_field)
            sequence = event.get("sequence")
            if (
                type(phase) is not str
                or type(identity) is not str
                or type(sequence) is not int
            ):
                raise RuntimeError(f"{path} {stem} lifecycle owner is malformed")
            owner = (phase, identity)
            destination = begins if action == begin_action else terminals
            if owner in destination:
                raise RuntimeError(f"{path} {stem} lifecycle owner is duplicated")
            destination[owner] = sequence
        if set(begins) != set(terminals) or any(
            begins[owner] >= terminals[owner] for owner in begins
        ):
            raise RuntimeError(f"{path} {stem} lifecycle is incomplete")

    verify_layer_lifecycles("LMCacheEngine.retrieve_layer", "lookup_id")
    verify_layer_lifecycles("LMCacheEngine.store_layer", "request_id")


def _verify_recovery_ledger(
    path: Path,
    arm: str,
    observations: list[dict[str, Any]],
    recovery: dict[str, Any],
    request_id_bindings: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Reconstruct all recovery identity evidence from the hash-bound ledger."""
    _verify_observation_layout(path, arm, observations)
    if set(recovery) != {
        "groups",
        "serial_members",
        "request_id_bindings",
        "trace_submission_order",
    }:
        raise RuntimeError(f"{path} recovery evidence shape drifted")
    groups = recovery.get("groups")
    serial_members = recovery.get("serial_members")
    if type(groups) is not list or type(serial_members) is not list:
        raise RuntimeError(f"{path} recovery members are not plain lists")

    observations_by_trace = {
        observation["trace_request_id"]: observation for observation in observations
    }

    bindings_by_engine = {
        binding["engine_request_id"]: binding for binding in request_id_bindings
    }
    bindings_by_trace = {
        binding["trace_request_id"]: binding for binding in request_id_bindings
    }
    if len(bindings_by_engine) != len(request_id_bindings) or len(
        bindings_by_trace
    ) != len(request_id_bindings):
        raise RuntimeError(f"{path} recovery request-ID owners are ambiguous")
    if arm == "zero_sharing":
        _verify_zero_sharing_sidecar_evidence(
            path,
            observations,
            request_id_bindings,
            events,
        )

    def bind_engine(value: object, phase: object, label: str) -> dict[str, Any]:
        if type(value) is not str or type(phase) is not str:
            raise RuntimeError(f"{label} has a non-string request owner")
        binding = bindings_by_engine.get(value)
        if binding is None or binding.get("phase") != phase:
            raise RuntimeError(f"{label} differs from its canonical phase/owner")
        return binding

    expected_submission_order = [
        observation["trace_request_id"]
        for observation in observations
        if observation["phase"] not in {"prime", "ordinary_prime"}
    ]
    if recovery.get("trace_submission_order") != expected_submission_order:
        raise RuntimeError(f"{path} trace submission order drifted")

    projection_events = _canonical_projection_events(
        path,
        events,
        _expected_projection_phases(arm),
    )
    projected: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
    projected_member_sets: list[tuple[str, frozenset[str]]] = []
    projected_plan_ids: set[str] = set()
    projected_reuse_group_ids: set[str] = set()
    for event_index, event in enumerate(projection_events):
        phase = event.get("phase")
        plans = event.get("plans")
        if type(phase) is not str or type(plans) is not list:
            raise RuntimeError(f"{path} projection event {event_index} is malformed")
        event_plan_ids: list[str] = []
        for plan_index, plan in enumerate(plans):
            label = f"{path}:projection[{event_index}].plans[{plan_index}]"
            if type(plan) is not dict:
                raise RuntimeError(f"{label} is not a plain mapping")
            plan_id = _plain_event_identifier(plan.get("plan_id"), f"{label}.plan_id")
            reuse_group_id = _plain_event_identifier(
                plan.get("reuse_group_id"), f"{label}.reuse_group_id"
            )
            if plan_id in projected_plan_ids:
                raise RuntimeError(f"{label}.plan_id is not globally unique")
            if reuse_group_id in projected_reuse_group_ids:
                raise RuntimeError(f"{label}.reuse_group_id is not globally unique")
            projected_plan_ids.add(plan_id)
            projected_reuse_group_ids.add(reuse_group_id)
            event_plan_ids.append(plan_id)
            members = plan.get("members")
            if type(members) is not list or not members:
                raise RuntimeError(f"{label}.members is not a nonempty plain list")
            member_ids: list[str] = []
            for member_index, member in enumerate(members):
                member_label = f"{label}.members[{member_index}]"
                if type(member) is not dict:
                    raise RuntimeError(f"{member_label} is not a plain mapping")
                request_id = member.get("request_id")
                bind_engine(request_id, phase, member_label)
                _verify_span_evidence(
                    member.get("explicit_spans"), f"{member_label}.explicit_spans"
                )
                member_ids.append(request_id)
            member_set = frozenset(member_ids)
            if len(member_set) != len(member_ids) or member_ids != sorted(member_ids):
                raise RuntimeError(f"{label} request IDs are not unique and sorted")
            owner = (phase, member_set)
            if owner in projected:
                raise RuntimeError(f"{label} has a duplicate phase/member owner")
            projected[owner] = plan
            projected_member_sets.append(owner)
        if event_plan_ids != sorted(event_plan_ids):
            raise RuntimeError(
                f"{path} projection event {event_index} plan IDs are not sorted"
            )

    begin_events = [
        event
        for event in events
        if event.get("action") == "gate.collective.blend_group.begin"
    ]
    end_events = [
        event
        for event in events
        if event.get("action") == "gate.collective.blend_group.end"
    ]
    error_events = [
        event
        for event in events
        if event.get("action") == "gate.collective.blend_group.error"
    ]
    if error_events or len(begin_events) != len(end_events):
        raise RuntimeError(f"{path} blend_group begin/end evidence is incomplete")

    reconstructed_groups: list[dict[str, Any]] = []
    group_phases: list[str] = []
    for call_index, (begin, end) in enumerate(
        zip(begin_events, end_events, strict=True)
    ):
        label = f"{path}:blend_group[{call_index}]"
        phase = begin.get("phase")
        begin_members = begin.get("members")
        input_members = end.get("input_members")
        result_members = end.get("result_members")
        if (
            type(phase) is not str
            or end.get("phase") != phase
            or type(begin.get("sequence")) is not int
            or type(end.get("sequence")) is not int
            or begin["sequence"] >= end["sequence"]
            or type(begin_members) is not list
            or not begin_members
            or begin_members != input_members
            or type(result_members) is not list
            or len(result_members) != len(begin_members)
            or type(begin.get("num_layers")) is not int
            or begin["num_layers"] != KV_SHAPE[0]
            or end.get("num_layers") != begin["num_layers"]
        ):
            raise RuntimeError(f"{label} begin/end geometry drifted")
        member_ids = [member.get("request_id") for member in begin_members]
        member_set = frozenset(member_ids)
        if len(member_set) != len(member_ids):
            raise RuntimeError(f"{label} has duplicate request IDs")
        plan = projected.get((phase, member_set))
        if plan is None:
            raise RuntimeError(f"{label} has no exact projected-plan owner")
        projected_by_id = {
            member.get("request_id"): member for member in plan["members"]
        }
        projected_order = [member.get("request_id") for member in plan["members"]]
        if (
            len(projected_by_id) != len(plan["members"])
            or projected_order != member_ids
        ):
            raise RuntimeError(f"{label} projected ownership is ambiguous")

        normalized_members: list[dict[str, Any]] = []
        group_generation: int | None = None
        for member_index, (input_member, result_member) in enumerate(
            zip(input_members, result_members, strict=True)
        ):
            member_label = f"{label}.members[{member_index}]"
            if type(input_member) is not dict or type(result_member) is not dict:
                raise RuntimeError(f"{member_label} is not a plain mapping")
            request_id = input_member.get("request_id")
            binding = bind_engine(request_id, phase, member_label)
            observation = observations_by_trace.get(binding["trace_request_id"])
            projected_member = projected_by_id.get(request_id)
            generation = input_member.get("generation")
            outcome = result_member.get("outcome")
            if type(generation) is not int or generation <= 0:
                raise RuntimeError(f"{member_label} generation is not positive")
            if group_generation is None:
                group_generation = generation
            elif generation != group_generation:
                raise RuntimeError(f"{label} generations are not shared")
            if (
                observation is None
                or projected_member is None
                or projected_member.get("explicit_spans")
                != input_member.get("explicit_spans")
                or result_member.get("request_id") != request_id
                or result_member.get("generation") != generation
                or input_member.get("token_count") != KV_SHAPE[1]
                or observation.get("prompt_tokens") != KV_SHAPE[1]
                or input_member.get("token_count") != observation.get("prompt_tokens")
            ):
                raise RuntimeError(
                    f"{member_label} identity, spans, or token geometry drifted"
                )
            expected_operation_counts = {
                "serial_rope_operations": 0,
                "serial_diff_operations": 0,
                "collective_rope_operations": (
                    begin["num_layers"] if outcome == "recovered" else 0
                ),
                "collective_diff_operations": 1 if outcome == "recovered" else 0,
                "position_recovery_serial_operations": 0,
                "position_recovery_collective_operations": (
                    begin["num_layers"] if outcome == "recovered" else 0
                ),
            }
            operation_counts = result_member.get("operation_counts")
            retrieval_mask = result_member.get("retrieval_mask")
            error = result_member.get("error")
            if (
                outcome not in {"recovered", "failed"}
                or not _operation_counts_match(
                    operation_counts, expected_operation_counts
                )
                or (
                    outcome == "recovered"
                    and (type(retrieval_mask) is not dict or error is not None)
                )
                or (
                    outcome == "failed"
                    and (
                        retrieval_mask is not None
                        or type(error) is not dict
                        or set(error) != {"type", "message"}
                        or any(
                            type(error[field]) is not str or not error[field]
                            for field in ("type", "message")
                        )
                    )
                )
            ):
                raise RuntimeError(f"{member_label} recovery result is not exact")
            _verify_span_evidence(
                input_member.get("explicit_spans"),
                f"{member_label}.explicit_spans",
                retrieval_mask if outcome == "recovered" else None,
            )
            normalized_members.append(
                {
                    "trace_request_id": binding["trace_request_id"],
                    "external_request_id": binding["external_request_id"],
                    "engine_request_id": request_id,
                    "generation": generation,
                    "outcome": outcome,
                    "selected_spans": input_member.get("explicit_spans"),
                    "retrieval_mask": retrieval_mask,
                    "operation_counts": operation_counts,
                    "error": error,
                }
            )
        reconstructed_groups.append(
            {
                "call_index": call_index,
                "plan_id": plan["plan_id"],
                "reuse_group_id": plan["reuse_group_id"],
                "num_layers": begin["num_layers"],
                "engine_request_order": member_ids,
                "members": normalized_members,
            }
        )
        group_phases.append(phase)
    if groups != reconstructed_groups:
        raise RuntimeError(f"{path} normalized group evidence differs from its ledger")

    serial_events = [
        event for event in events if event.get("action") == "gate.serial_blend.evidence"
    ]
    reconstructed_serial: list[dict[str, Any]] = []
    for event_index, event in enumerate(serial_events):
        label = f"{path}:serial_blend[{event_index}]"
        binding = bind_engine(event.get("request_id"), event.get("phase"), label)
        _verify_span_evidence(
            event.get("explicit_spans"),
            f"{label}.explicit_spans",
            event.get("retrieval_mask"),
        )
        reconstructed_serial.append(
            {
                "trace_request_id": binding["trace_request_id"],
                "external_request_id": binding["external_request_id"],
                "engine_request_id": binding["engine_request_id"],
                "selected_spans": event.get("explicit_spans"),
                "retrieval_mask": event.get("retrieval_mask"),
            }
        )
    if serial_members != reconstructed_serial:
        raise RuntimeError(f"{path} normalized serial evidence differs from its ledger")

    bindings_for_sequence = {
        observation["sequence_index"]: bindings_by_trace[
            observation["trace_request_id"]
        ]["engine_request_id"]
        for observation in observations
    }
    all_probe_ids = [
        bindings_for_sequence[sequence]
        for sequence in (1, 2, 3, 4)
        if sequence in bindings_for_sequence
    ]
    expected_plan_sets: list[frozenset[str]] = []
    expected_group_orders: list[list[str]] = []
    expected_serial_order: list[str] = []
    if arm in {"collective_forward", "collective_reverse", "sibling_failure"}:
        expected_plan_sets = [frozenset(all_probe_ids)]
        expected_group_orders = [
            [
                binding["engine_request_id"]
                for observation, binding in zip(
                    observations, request_id_bindings, strict=True
                )
                if observation["phase"] == arm
            ]
        ]
    elif arm == "incompatible_partition":
        partition_a = [bindings_for_sequence[sequence] for sequence in (1, 2)]
        partition_b = [bindings_for_sequence[sequence] for sequence in (3, 4)]
        expected_plan_sets = [frozenset(partition_a), frozenset(partition_b)]
        expected_group_orders = [partition_a]
    elif arm == "serial_oracle":
        expected_serial_order = all_probe_ids

    actual_plan_sets = [member_set for phase, member_set in projected_member_sets]
    if (
        len(actual_plan_sets) != len(expected_plan_sets)
        or set(actual_plan_sets) != set(expected_plan_sets)
        or any(phase != arm for phase, _member_set in projected_member_sets)
    ):
        raise RuntimeError(f"{path} has the wrong {arm} projected-plan layout")
    if arm == "zero_sharing" and not any(
        event.get("phase") == "zero_sharing" for event in projection_events
    ):
        raise RuntimeError(f"{path} lacks the zero-sharing empty projection")
    actual_group_orders = [group["engine_request_order"] for group in groups]
    if actual_group_orders != expected_group_orders or group_phases != [arm] * len(
        expected_group_orders
    ):
        raise RuntimeError(f"{path} has the wrong {arm} group layout")
    if [member["engine_request_id"] for member in serial_members] != (
        expected_serial_order
    ):
        raise RuntimeError(f"{path} has the wrong {arm} serial layout")
    if arm == "sibling_failure":
        group = groups[0]
        outcomes = [member.get("outcome") for member in group["members"]]
        member_ids = [member.get("engine_request_id") for member in group["members"]]
        if (
            len(member_ids) != 4
            or member_ids != group["engine_request_order"]
            or outcomes != ["recovered", "failed", "recovered", "recovered"]
        ):
            raise RuntimeError(f"{path} has the wrong sibling-failure outcomes")
    elif any(
        member.get("outcome") != "recovered"
        for group in groups
        for member in group["members"]
    ):
        raise RuntimeError(f"{path} has a non-recovered control member")

    injection_events = [
        event
        for event in events
        if event.get("action") == "gate.sibling_failure.injected"
    ]
    failure_begins = [
        event
        for event in events
        if event.get("action")
        == "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin"
    ]
    failure_ends = [
        event
        for event in events
        if event.get("action")
        == "LMCacheConnectorV1Impl._record_kmesh_load_failure.end"
    ]
    if arm != "sibling_failure":
        if injection_events or failure_begins or failure_ends:
            raise RuntimeError(f"{path} has unexpected sibling-failure evidence")
    else:
        failed_member = groups[0]["members"][1]
        if (
            len(injection_events) != 1
            or len(failure_begins) != 1
            or len(failure_ends) != 1
        ):
            raise RuntimeError(f"{path} sibling-failure evidence is not singular")
        injection = injection_events[0]
        failure_begin = failure_begins[0]
        failure_end = failure_ends[0]
        failed_engine_id = failed_member["engine_request_id"]
        identity_fields = (
            "request_id",
            "reason",
            "invalidate_full_prompt",
            "contract_slot_count",
            "advertised_tokens",
            "lookup_id",
        )
        if (
            injection.get("phase") != arm
            or injection.get("request_id") != failed_engine_id
            or not _operation_counts_match(
                injection.get("completed_serial_operations"),
                {field: 0 for field in OPERATION_COUNT_FIELDS},
            )
            or any(
                failure_begin.get(field) != failure_end.get(field)
                for field in identity_fields
            )
            or failure_begin.get("phase") != arm
            or failure_end.get("phase") != arm
            or failure_begin.get("request_id") != failed_engine_id
            or failure_begin.get("owner_type") != "LMCacheConnectorV1Impl"
            or failure_end.get("owner_type") != "LMCacheConnectorV1Impl"
            or failure_begin.get("role") != "WORKER"
            or failure_end.get("role") != "WORKER"
            or failure_begin.get("invalidate_full_prompt") is not True
            or type(failure_begin.get("contract_slot_count")) is not int
            or failure_begin["contract_slot_count"] <= 0
            or type(failure_begin.get("advertised_tokens")) is not int
            or failure_begin["advertised_tokens"] <= 0
            or type(injection.get("sequence")) is not int
            or type(failure_begin.get("sequence")) is not int
            or type(failure_end.get("sequence")) is not int
            or not (
                begin_events[0]["sequence"]
                < injection["sequence"]
                < end_events[0]["sequence"]
                < failure_begin["sequence"]
                < failure_end["sequence"]
            )
        ):
            raise RuntimeError(f"{path} sibling-failure identity chain drifted")

    _verify_event_id_domains(path, events, request_id_bindings)


def _verify_lookup_claims(
    path: Path,
    arm: str,
    observations: list[dict[str, Any]],
    request_id_bindings: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, str]:
    def selected(action: str) -> list[dict[str, Any]]:
        return [event for event in events if event.get("action") == action]

    for observation in observations:
        sequence = observation["sequence_index"]
        cached = observation["num_cached_tokens"]
        expected_positive = (
            arm
            in {
                "serial_oracle",
                "collective_forward",
                "collective_reverse",
                "sibling_failure",
            }
            and sequence in {1, 2, 3, 4}
        ) or (arm == "incompatible_partition" and sequence in {1, 2})
        if (
            type(cached) is not int
            or (expected_positive and cached <= 0)
            or (not expected_positive and cached != 0)
        ):
            raise RuntimeError(f"{path} cached-token lookup matrix drifted")

    matched_begins = selected("LMCacheConnectorV1Impl.get_num_new_matched_tokens.begin")
    matched_ends = selected("LMCacheConnectorV1Impl.get_num_new_matched_tokens.end")
    lookup_begins = selected("LMCacheLookupClient.lookup.begin")
    lookup_ends = selected("LMCacheLookupClient.lookup.end")
    if arm == "ordinary":
        if matched_begins or matched_ends or lookup_begins or lookup_ends:
            raise RuntimeError(f"{path} ordinary arm unexpectedly performed lookup")
        return {}

    bindings_by_trace = {
        binding["trace_request_id"]: binding for binding in request_id_bindings
    }
    expected_lookup_count = 0 if arm == "zero_sharing" else len(observations)
    if (
        len(matched_begins) != len(observations)
        or len(matched_ends) != len(observations)
        or len(lookup_begins) != expected_lookup_count
        or len(lookup_ends) != expected_lookup_count
    ):
        raise RuntimeError(f"{path} lookup lifecycle count differs from its arm")
    lookup_by_engine: dict[str, str] = {}
    for index, (observation, begin, end) in enumerate(
        zip(observations, matched_begins, matched_ends, strict=True)
    ):
        label = f"{path}:matched_tokens[{index}]"
        binding = bindings_by_trace.get(observation["trace_request_id"])
        if (
            binding is None
            or begin.get("phase") != observation["phase"]
            or end.get("phase") != observation["phase"]
            or begin.get("request_id") != binding["engine_request_id"]
            or end.get("request_id") != binding["engine_request_id"]
            or begin.get("owner_type") != "LMCacheConnectorV1Impl"
            or end.get("owner_type") != "LMCacheConnectorV1Impl"
            or begin.get("role") != "SCHEDULER"
            or end.get("role") != "SCHEDULER"
            or type(begin.get("num_computed_tokens")) is not int
            or begin.get("num_computed_tokens") != 0
            or type(end.get("num_computed_tokens")) is not int
            or end.get("num_computed_tokens") != 0
            or type(end.get("result")) is not int
            or end.get("result") != observation["num_cached_tokens"]
            or type(begin.get("sequence")) is not int
            or type(end.get("sequence")) is not int
            or begin["sequence"] >= end["sequence"]
        ):
            raise RuntimeError(f"{label} is not request-owned exact evidence")
        nested_begins = [
            event
            for event in lookup_begins
            if begin["sequence"] < event["sequence"] < end["sequence"]
        ]
        nested_ends = [
            event
            for event in lookup_ends
            if begin["sequence"] < event["sequence"] < end["sequence"]
        ]
        if arm == "zero_sharing":
            if nested_begins or nested_ends:
                raise RuntimeError(f"{label} zero-sharing path performed lookup")
            continue
        if (
            len(nested_begins) != 1
            or len(nested_ends) != 1
            or nested_begins[0].get("phase") != observation["phase"]
            or nested_ends[0].get("phase") != observation["phase"]
            or nested_begins[0].get("owner_type") != "LMCacheLookupClient"
            or nested_ends[0].get("owner_type") != "LMCacheLookupClient"
            or nested_begins[0].get("lookup_id") != nested_ends[0].get("lookup_id")
            or type(nested_begins[0].get("token_count")) is not int
            or nested_begins[0].get("token_count") != KV_SHAPE[1]
            or type(nested_ends[0].get("token_count")) is not int
            or nested_ends[0].get("token_count") != KV_SHAPE[1]
            or type(nested_ends[0].get("result")) is not int
            or nested_ends[0].get("result") != observation["num_cached_tokens"]
            or not (
                begin["sequence"]
                < nested_begins[0]["sequence"]
                < nested_ends[0]["sequence"]
                < end["sequence"]
            )
        ):
            raise RuntimeError(f"{label} lookup owner/result drifted")
        lookup_by_engine[binding["engine_request_id"]] = nested_ends[0]["lookup_id"]
    return lookup_by_engine


def _verify_layer_operation_claims(
    path: Path,
    arm: str,
    observations: list[dict[str, Any]],
    request_id_bindings: list[dict[str, Any]],
    lookup_by_engine: dict[str, str],
    events: list[dict[str, Any]],
) -> None:
    def selected(action: str, phase: str | None = None) -> list[dict[str, Any]]:
        return [
            event
            for event in events
            if event.get("action") == action
            and (phase is None or event.get("phase") == phase)
        ]

    bindings_by_trace = {
        binding["trace_request_id"]: binding for binding in request_id_bindings
    }
    expected_retrieve_lookups = {
        lookup_by_engine[
            bindings_by_trace[observation["trace_request_id"]]["engine_request_id"]
        ]
        for observation in observations
        if observation["num_cached_tokens"] > 0
    }
    retrieve_begins = selected("LMCacheEngine.retrieve_layer.begin")
    retrieve_terminals = [
        event
        for event in events
        if event.get("action")
        in {
            "LMCacheEngine.retrieve_layer.end",
            "LMCacheEngine.retrieve_layer.closed",
        }
    ]
    retrieve_begin_by_lookup = {
        event.get("lookup_id"): event for event in retrieve_begins
    }
    retrieve_terminal_by_lookup = {
        event.get("lookup_id"): event for event in retrieve_terminals
    }
    if (
        len(retrieve_begins) != len(expected_retrieve_lookups)
        or len(retrieve_terminals) != len(expected_retrieve_lookups)
        or set(retrieve_begin_by_lookup) != expected_retrieve_lookups
        or set(retrieve_terminal_by_lookup) != expected_retrieve_lookups
        or any(
            type(retrieve_begin_by_lookup[lookup_id].get("sequence")) is not int
            or type(retrieve_terminal_by_lookup[lookup_id].get("sequence")) is not int
            or retrieve_begin_by_lookup[lookup_id]["sequence"]
            >= retrieve_terminal_by_lookup[lookup_id]["sequence"]
            for lookup_id in expected_retrieve_lookups
        )
    ):
        raise RuntimeError(f"{path} retrieve-layer operation-call ownership drifted")

    prime_arms = {
        "serial_oracle",
        "collective_forward",
        "collective_reverse",
        "incompatible_partition",
        "sibling_failure",
    }
    if arm not in prime_arms:
        return
    prime = [
        observation for observation in observations if observation["phase"] == "prime"
    ]
    prime_binding = (
        bindings_by_trace.get(prime[0]["trace_request_id"]) if len(prime) == 1 else None
    )
    finishes = selected("_KMeshLayerwiseStoreSession.finish.end", "prime")
    store_begins = selected("LMCacheEngine.store_layer.begin", "prime")
    store_terminals = [
        event
        for event in events
        if event.get("phase") == "prime"
        and event.get("action")
        in {"LMCacheEngine.store_layer.end", "LMCacheEngine.store_layer.closed"}
    ]
    if (
        prime_binding is None
        or len(finishes) != 1
        or len(store_begins) != 1
        or len(store_terminals) != 1
        or store_begins[0].get("request_id") != prime_binding["engine_request_id"]
        or store_terminals[0].get("request_id") != prime_binding["engine_request_id"]
        or type(store_begins[0].get("sequence")) is not int
        or type(store_terminals[0].get("sequence")) is not int
        or type(finishes[0].get("sequence")) is not int
        or not (
            store_begins[0]["sequence"]
            < store_terminals[0]["sequence"]
            < finishes[0]["sequence"]
        )
    ):
        raise RuntimeError(f"{path} prime store-layer operation call is incomplete")
    _verify_span_evidence(
        store_begins[0].get("explicit_spans"), f"{path}:prime_store.explicit_spans"
    )
    if store_terminals[0].get("explicit_spans") != store_begins[0].get(
        "explicit_spans"
    ):
        raise RuntimeError(f"{path} prime store-layer spans drifted at terminal")


def _verify_sibling_failure_lookup_claim(
    path: Path,
    observations: list[dict[str, Any]],
    members: list[dict[str, Any]],
    lookup_by_engine: dict[str, str],
    events: list[dict[str, Any]],
) -> None:
    failed = [member for member in members if member.get("outcome") == "failed"]
    observations_by_trace = {
        observation["trace_request_id"]: observation for observation in observations
    }
    failure_begins = [
        event
        for event in events
        if event.get("action")
        == "LMCacheConnectorV1Impl._record_kmesh_load_failure.begin"
        and event.get("phase") == "sibling_failure"
    ]
    failure_ends = [
        event
        for event in events
        if event.get("action")
        == "LMCacheConnectorV1Impl._record_kmesh_load_failure.end"
        and event.get("phase") == "sibling_failure"
    ]
    if len(failed) != 1:
        raise RuntimeError(f"{path} sibling failure member is not singular")
    failed_member = failed[0]
    failed_engine_id = failed_member.get("engine_request_id")
    failed_observation = observations_by_trace.get(
        failed_member.get("trace_request_id")
    )
    expected_lookup_id = lookup_by_engine.get(failed_engine_id)
    if (
        failed_observation is None
        or expected_lookup_id is None
        or len(failure_begins) != 1
        or len(failure_ends) != 1
        or failure_begins[0].get("request_id") != failed_engine_id
        or failure_ends[0].get("request_id") != failed_engine_id
        or failure_begins[0].get("lookup_id") != expected_lookup_id
        or failure_ends[0].get("lookup_id") != expected_lookup_id
        or type(failure_begins[0].get("contract_slot_count")) is not int
        or failure_begins[0].get("contract_slot_count") != KV_SHAPE[1]
        or type(failure_ends[0].get("contract_slot_count")) is not int
        or failure_ends[0].get("contract_slot_count") != KV_SHAPE[1]
        or type(failure_begins[0].get("advertised_tokens")) is not int
        or failure_begins[0].get("advertised_tokens")
        != failed_observation["num_cached_tokens"]
        or type(failure_ends[0].get("advertised_tokens")) is not int
        or failure_ends[0].get("advertised_tokens")
        != failed_observation["num_cached_tokens"]
    ):
        raise RuntimeError(f"{path} failed member lookup ownership drifted")


def _verify_arm_claims(
    path: Path,
    arm: str,
    observations: list[dict[str, Any]],
    recovery: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive claim-critical arm facts from hash-bound primary evidence."""

    def selected(action: str, phase: str | None = None) -> list[dict[str, Any]]:
        return [
            event
            for event in events
            if event.get("action") == action
            and (phase is None or event.get("phase") == phase)
        ]

    canonical_projection_events = _canonical_projection_events(
        path,
        events,
        _expected_projection_phases(arm),
    )

    def canonical_projections(phase: str) -> list[dict[str, Any]]:
        return [
            event for event in canonical_projection_events if event["phase"] == phase
        ]

    if any(
        observation["prompt_tokens"] != KV_SHAPE[1]
        or len(observation["output_token_ids"]) != 1
        for observation in observations
    ):
        raise RuntimeError(
            f"{path} did not execute the exact frozen prompt/output size"
        )
    if any(
        event.get("action") == "gate.failure"
        or str(event.get("action", "")).endswith(".error")
        for event in events
    ):
        raise RuntimeError(f"{path} contains an instrumented runtime error")

    expected_profile = {
        "ordinary": None,
        "serial_oracle": "serial_pic_v1",
        "collective_forward": "collective_batched_qkv_v1",
        "collective_reverse": "collective_batched_qkv_v1",
        "cold_miss": "collective_batched_qkv_v1",
        "zero_sharing": "collective_batched_qkv_v1",
        "incompatible_partition": "collective_batched_qkv_v1",
        "sibling_failure": "collective_batched_qkv_v1",
    }[arm]
    construction_begins = selected("gate.construction.begin", "construction")
    construction_ends = selected("gate.construction.end", "construction")
    connectors_expected = arm != "ordinary"
    if (
        len(construction_begins) != 1
        or len(construction_ends) != 1
        or construction_begins[0].get("arm") != arm
        or construction_ends[0].get("arm") != arm
        or construction_ends[0].get("resolved_target_profile") != expected_profile
        or construction_ends[0].get("resolved_attention_backend") != "TRITON_ATTN"
        or construction_ends[0].get("kv_transfer_config_present")
        is not connectors_expected
        or type(construction_begins[0].get("sequence")) is not int
        or type(construction_ends[0].get("sequence")) is not int
        or construction_begins[0]["sequence"] >= construction_ends[0]["sequence"]
    ):
        raise RuntimeError(f"{path} construction profile is not exact")

    expected_batches: list[tuple[str, list[dict[str, Any]]]]
    if arm == "ordinary":
        expected_batches = [
            ("ordinary_prime", observations[:1]),
            ("ordinary_batch_forward", observations[1:]),
        ]
    elif arm in {
        "serial_oracle",
        "collective_forward",
        "collective_reverse",
        "incompatible_partition",
        "sibling_failure",
    }:
        expected_batches = [("prime", observations[:1]), (arm, observations[1:])]
    else:
        expected_batches = [(arm, observations)]
    batch_begins = selected("gate.batch.begin")
    batch_ends = selected("gate.batch.end")
    completion_events = selected("gate.request.complete")
    if len(batch_begins) != len(expected_batches) or len(batch_ends) != len(
        expected_batches
    ):
        raise RuntimeError(f"{path} has the wrong generate-call layout")
    completion_by_phase: dict[str, list[dict[str, Any]]] = {}
    for event in completion_events:
        completion_by_phase.setdefault(event.get("phase"), []).append(event)
    for batch_index, ((phase, batch), begin, end) in enumerate(
        zip(expected_batches, batch_begins, batch_ends, strict=True)
    ):
        trace_ids = [observation["trace_request_id"] for observation in batch]
        external_ids = [observation["external_request_id"] for observation in batch]
        phase_completions = completion_by_phase.get(phase, [])
        if (
            not batch
            or [observation["batch_ordinal"] for observation in batch]
            != list(range(len(batch)))
            or begin.get("phase") != phase
            or end.get("phase") != phase
            or type(begin.get("batch_size")) is not int
            or type(end.get("batch_size")) is not int
            or begin.get("batch_size") != len(batch)
            or end.get("batch_size") != len(batch)
            or begin.get("submitted_trace_request_ids") != trace_ids
            or end.get("submitted_trace_request_ids") != trace_ids
            or end.get("returned_external_request_ids") != external_ids
            or begin.get("prompt_tokens")
            != sum(observation["prompt_tokens"] for observation in batch)
            or len(phase_completions) != len(batch)
            or type(begin.get("sequence")) is not int
            or type(end.get("sequence")) is not int
            or any(
                type(event.get("sequence")) is not int for event in phase_completions
            )
            or not (
                begin["sequence"]
                < phase_completions[0]["sequence"]
                <= phase_completions[-1]["sequence"]
                < end["sequence"]
            )
        ):
            raise RuntimeError(
                f"{path} batch {batch_index} differs from reconstructed observations"
            )

    lookup_by_engine = _verify_lookup_claims(
        path,
        arm,
        observations,
        recovery["request_id_bindings"],
        events,
    )
    _verify_layer_operation_claims(
        path,
        arm,
        observations,
        recovery["request_id_bindings"],
        lookup_by_engine,
        events,
    )

    shutdown_begins = selected("LMCacheConnectorV1Impl.shutdown.begin", "shutdown")
    shutdowns = selected("LMCacheConnectorV1Impl.shutdown.end", "shutdown")
    terminal_shutdowns = selected("gate.shutdown.complete")
    cleanup_verified = False
    if not connectors_expected:
        if (
            shutdown_begins
            or shutdowns
            or len(terminal_shutdowns) != 1
            or terminal_shutdowns[0].get("phase") != "shutdown"
        ):
            raise RuntimeError(f"{path} ordinary arm unexpectedly shut down connectors")
    else:
        begins_by_role = {event.get("role"): event for event in shutdown_begins}
        by_role = {event.get("role"): event for event in shutdowns}
        if (
            len(shutdown_begins) != 2
            or len(shutdowns) != 2
            or len(terminal_shutdowns) != 1
            or terminal_shutdowns[0].get("phase") != "shutdown"
            or type(terminal_shutdowns[0].get("sequence")) is not int
            or set(begins_by_role) != {"WORKER", "SCHEDULER"}
            or set(by_role) != {"WORKER", "SCHEDULER"}
            or any(
                begins_by_role[role].get("owner_type") != "LMCacheConnectorV1Impl"
                or type(begins_by_role[role].get("sequence")) is not int
                or type(by_role[role].get("sequence")) is not int
                or begins_by_role[role]["sequence"] >= by_role[role]["sequence"]
                or by_role[role]["sequence"]
                >= terminal_shutdowns[0].get("sequence", -1)
                for role in ("WORKER", "SCHEDULER")
            )
            or any(
                event.get("owner_type") != "LMCacheConnectorV1Impl"
                or event.get("manager_services_stopped") is not True
                for event in shutdowns
            )
            or any(
                type(by_role["WORKER"].get(field)) is not int
                or by_role["WORKER"].get(field) != 0
                for field in WORKER_RESIDUE_FIELDS
            )
            or any(
                type(by_role["SCHEDULER"].get(field)) is not int
                or by_role["SCHEDULER"].get(field) != 0
                for field in SCHEDULER_RESIDUE_FIELDS
            )
            or (
                arm == "zero_sharing"
                and any(
                    type(by_role["SCHEDULER"].get(f"{field}_count")) is not int
                    or by_role["SCHEDULER"].get(f"{field}_count") != 0
                    for field in ZERO_SHARING_SCHEDULER_STATE_FIELDS
                )
            )
        ):
            raise RuntimeError(f"{path} connector shutdown left ownership residue")
        cleanup_verified = True

    prime_arms = {
        "serial_oracle",
        "collective_forward",
        "collective_reverse",
        "incompatible_partition",
        "sibling_failure",
    }
    if arm in prime_arms:
        prime = [
            observation
            for observation in observations
            if observation["phase"] == "prime"
        ]
        prime_lookups = selected("LMCacheLookupClient.lookup.end", "prime")
        prime_finishes = selected("_KMeshLayerwiseStoreSession.finish.end", "prime")
        prime_waits = selected("LMCacheConnectorV1Impl.wait_for_save.end", "prime")
        if (
            len(prime) != 1
            or prime[0]["sequence_index"] != 0
            or prime[0]["num_cached_tokens"] != 0
            or len(prime_lookups) != 1
            or prime_lookups[0].get("result") != 0
            or len(prime_waits) != 1
            or len(prime_finishes) != 1
            or prime_finishes[0].get("result") is not True
            or prime_finishes[0].get("flushed") is not True
            or prime_finishes[0].get("closed") is not True
            or prime_finishes[0].get("completed_layers") != KV_SHAPE[0]
            or prime_finishes[0].get("expected_layer_names_count") != KV_SHAPE[0]
            or selected("gate.serial_blend.evidence", "prime")
            or selected("gate.collective.blend_group.begin", "prime")
            or selected("gate.collective.blend_group.end", "prime")
        ):
            raise RuntimeError(
                f"{path} prime lookup/store was not a clean zero-hit store"
            )

    observations_by_trace = {
        observation["trace_request_id"]: observation for observation in observations
    }

    def verify_member_mask(member: dict[str, Any], label: str) -> None:
        observation = observations_by_trace.get(member.get("trace_request_id"))
        if observation is None:
            raise RuntimeError(f"{label} has no observation owner")
        _verify_positive_mask(
            member.get("retrieval_mask"), observation["prompt_tokens"], label
        )

    groups = recovery["groups"]
    serial_members = recovery["serial_members"]
    probe_observations = [
        observation for observation in observations if observation["phase"] == arm
    ]
    if arm == "ordinary":
        if any(observation["num_cached_tokens"] != 0 for observation in observations):
            raise RuntimeError(f"{path} ordinary arm advertised cached tokens")
    elif arm == "serial_oracle":
        if (
            groups
            or len(serial_members) != 4
            or any(
                observation["num_cached_tokens"] <= 0
                for observation in probe_observations
            )
        ):
            raise RuntimeError(f"{path} serial positive-hit control failed")
        for index, member in enumerate(serial_members):
            verify_member_mask(member, f"{path}:serial_members[{index}]")
    elif arm in {"collective_forward", "collective_reverse"}:
        if (
            len(groups) != 1
            or len(groups[0]["members"]) != 4
            or serial_members
            or any(
                observation["num_cached_tokens"] <= 0
                for observation in probe_observations
            )
        ):
            raise RuntimeError(f"{path} collective positive-hit control failed")
        for index, member in enumerate(groups[0]["members"]):
            if member["outcome"] != "recovered":
                raise RuntimeError(f"{path} collective member did not recover")
            verify_member_mask(member, f"{path}:groups[0].members[{index}]")
    elif arm == "cold_miss":
        cold_lookups = selected("LMCacheLookupClient.lookup.end", arm)
        if (
            len(cold_lookups) != 1
            or cold_lookups[0].get("result") != 0
            or any(
                observation["num_cached_tokens"] != 0 for observation in observations
            )
            or groups
            or serial_members
        ):
            raise RuntimeError(f"{path} cold-miss control failed")
    elif arm == "zero_sharing":
        projections = canonical_projections(arm)
        if (
            len(projections) != 1
            or projections[0].get("plans") != []
            or any(
                observation["num_cached_tokens"] != 0 for observation in observations
            )
            or groups
            or serial_members
        ):
            raise RuntimeError(f"{path} zero-sharing control failed")
    elif arm == "incompatible_partition":
        projections = canonical_projections(arm)
        plans = projections[0].get("plans") if len(projections) == 1 else None
        plan_member_sets = (
            [
                {member.get("request_id") for member in plan.get("members", [])}
                for plan in plans
            ]
            if type(plans) is list
            else []
        )
        primed = [
            observation
            for observation in probe_observations
            if observation["sequence_index"] in {1, 2}
        ]
        unprimed = [
            observation
            for observation in probe_observations
            if observation["sequence_index"] in {3, 4}
        ]
        if (
            len(plan_member_sets) != 2
            or sorted(len(members) for members in plan_member_sets) != [2, 2]
            or plan_member_sets[0] & plan_member_sets[1]
            or len(groups) != 1
            or len(groups[0]["members"]) != 2
            or serial_members
            or len(primed) != 2
            or any(observation["num_cached_tokens"] <= 0 for observation in primed)
            or len(unprimed) != 2
            or any(observation["num_cached_tokens"] != 0 for observation in unprimed)
        ):
            raise RuntimeError(f"{path} incompatible-partition control failed")
        for index, member in enumerate(groups[0]["members"]):
            verify_member_mask(member, f"{path}:groups[0].members[{index}]")
    elif arm == "sibling_failure":
        members = groups[0]["members"] if len(groups) == 1 else []
        recovered = [member for member in members if member["outcome"] == "recovered"]
        failed = [member for member in members if member["outcome"] == "failed"]
        zero_counts = {field: 0 for field in OPERATION_COUNT_FIELDS}
        if (
            len(members) != 4
            or len(recovered) != 3
            or len(failed) != 1
            or failed[0].get("retrieval_mask") is not None
            or not _operation_counts_match(
                failed[0].get("operation_counts"), zero_counts
            )
            or serial_members
            or any(
                observation["num_cached_tokens"] <= 0
                for observation in probe_observations
            )
        ):
            raise RuntimeError(f"{path} sibling-failure isolation control failed")
        _verify_sibling_failure_lookup_claim(
            path, observations, members, lookup_by_engine, events
        )
        for index, member in enumerate(recovered):
            verify_member_mask(member, f"{path}:recovered_siblings[{index}]")

    return {
        "all_passed": True,
        "construction_profile": expected_profile,
        "attention_backend": "TRITON_ATTN",
        "connector_present": connectors_expected,
        "cleanup_verified": cleanup_verified,
        "observation_events_verified": len(observations),
        "batch_calls_verified": len(expected_batches),
        "arm_control_verified": True,
    }


def _verify_numerical_event_bindings(
    path: Path,
    arm: str,
    numerical: dict[str, Any],
    manifest: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> None:
    actions = {
        "kv": "gate.numerical.kv_captured",
        "logits": "gate.numerical.logits_captured",
        "finish": "gate.numerical.finish",
    }
    captured_events = {
        name: [event for event in events if event.get("action") == action]
        for name, action in actions.items()
    }
    if not numerical["enabled"]:
        if any(captured_events.values()):
            raise RuntimeError(f"{path} disabled arm contains numerical capture events")
        return
    if manifest is None:
        raise RuntimeError(f"{path} enabled arm lacks a numerical manifest")
    members = manifest.get("members")
    kv_events = captured_events["kv"]
    logits_events = captured_events["logits"]
    finish_events = captured_events["finish"]
    capture_phase = manifest.get("capture_phase")
    if (
        type(members) is not list
        or len(members) != 4
        or len(kv_events) != 4
        or len(logits_events) != 4
        or len(finish_events) != 1
    ):
        raise RuntimeError(f"{path} numerical event coverage is incomplete")

    expected_sequences = {1, 2, 3, 4}
    if (
        any(
            type(event.get("sequence_index")) is not int
            for event in (*kv_events, *logits_events)
        )
        or {event["sequence_index"] for event in kv_events} != expected_sequences
        or {event["sequence_index"] for event in logits_events} != expected_sequences
    ):
        raise RuntimeError(f"{path} numerical event sequence ownership drifted")
    kv_events_by_sequence = {event["sequence_index"]: event for event in kv_events}
    logits_events_by_sequence = {
        event["sequence_index"]: event for event in logits_events
    }

    observed_paths: set[str] = set()
    observed_capture_rows: set[int] = set()
    batch_engine_request_ids: list[str | None] = [None, None, None, None]
    for member_index, member in enumerate(members, start=1):
        label = f"{path}:numerical.members[{member_index - 1}]"
        if type(member) is not dict:
            raise RuntimeError(f"{label} is not a plain mapping")
        if (
            type(member.get("sequence_index")) is not int
            or member.get("sequence_index") != member_index
        ):
            raise RuntimeError(f"{label} sequence ownership drifted")
        kv_event = kv_events_by_sequence[member_index]
        logits_event = logits_events_by_sequence[member_index]
        tensor_file = member.get("tensor_file")
        logits_file = member.get("logits_file")
        if type(tensor_file) is not dict or type(logits_file) is not dict:
            raise RuntimeError(f"{label} lacks numerical file descriptors")
        capture_batch_row = member.get("capture_batch_row")
        engine_request_id = member.get("engine_request_id")
        if (
            type(capture_batch_row) is not int
            or capture_batch_row not in {0, 1, 2, 3}
            or capture_batch_row in observed_capture_rows
            or type(engine_request_id) is not str
            or not engine_request_id
        ):
            raise RuntimeError(f"{label} capture-row ownership drifted")
        observed_capture_rows.add(capture_batch_row)
        batch_engine_request_ids[capture_batch_row] = engine_request_id
        expected_tensor_relative = f"members/{member_index:03d}/kv.safetensors"
        expected_logits_relative = f"members/{member_index:03d}/logits.safetensors"
        expected_output_prefix = Path(numerical["manifest_path"]).parent.as_posix()
        expected_tensor_path = f"{expected_output_prefix}/{expected_tensor_relative}"
        expected_logits_path = f"{expected_output_prefix}/{expected_logits_relative}"
        for descriptor, expected_relative in (
            (tensor_file, expected_tensor_relative),
            (logits_file, expected_logits_relative),
        ):
            descriptor_path = descriptor.get("path")
            descriptor_sha = descriptor.get("sha256")
            if (
                descriptor_path != expected_relative
                or descriptor_path in observed_paths
                or type(descriptor_sha) is not str
                or len(descriptor_sha) != 64
                or any(character not in LOWER_HEX for character in descriptor_sha)
                or type(descriptor.get("nbytes")) is not int
                or descriptor["nbytes"] <= 0
            ):
                raise RuntimeError(f"{label} file descriptor ownership drifted")
            observed_paths.add(descriptor_path)

        common_event_fields = (
            kv_event.get("arm") == arm
            and logits_event.get("arm") == arm
            and kv_event.get("phase") == capture_phase
            and logits_event.get("phase") == capture_phase
            and type(kv_event.get("sequence_index")) is int
            and kv_event.get("sequence_index") == member_index
            and type(logits_event.get("sequence_index")) is int
            and logits_event.get("sequence_index") == member_index
            and kv_event.get("trace_request_id") == member.get("trace_request_id")
            and logits_event.get("trace_request_id") == member.get("trace_request_id")
            and kv_event.get("engine_request_id") == member.get("engine_request_id")
            and logits_event.get("engine_request_id") == member.get("engine_request_id")
            and type(kv_event.get("capture_batch_row")) is int
            and kv_event.get("capture_batch_row") == capture_batch_row
            and type(logits_event.get("capture_batch_row")) is int
            and logits_event.get("capture_batch_row") == capture_batch_row
        )
        if (
            not common_event_fields
            or kv_event.get("physical_block_ids") != member.get("physical_block_ids")
            or kv_event.get("tensor_path") != expected_tensor_path
            or kv_event.get("tensor_sha256") != tensor_file.get("sha256")
            or logits_event.get("vocab_size") != VOCAB_SIZE
            or type(logits_event.get("vocab_size")) is not int
            or logits_event.get("logits_path") != expected_logits_path
            or logits_event.get("logits_sha256") != logits_file.get("sha256")
            or (
                "nbytes" in kv_event
                and (
                    type(kv_event["nbytes"]) is not int
                    or kv_event["nbytes"] != tensor_file["nbytes"]
                )
            )
            or (
                "nbytes" in logits_event
                and (
                    type(logits_event["nbytes"]) is not int
                    or logits_event["nbytes"] != logits_file["nbytes"]
                )
            )
        ):
            raise RuntimeError(f"{label} differs from hash-bound capture events")

    if (
        observed_capture_rows != {0, 1, 2, 3}
        or any(
            engine_request_id is None for engine_request_id in batch_engine_request_ids
        )
        or len(set(batch_engine_request_ids)) != 4
    ):
        raise RuntimeError(f"{path} numerical capture rows are not bijective")
    for member_index, member in enumerate(members, start=1):
        if (
            type(member.get("batch_engine_request_ids")) is not list
            or member["batch_engine_request_ids"] != batch_engine_request_ids
            or batch_engine_request_ids[member["capture_batch_row"]]
            != member["engine_request_id"]
        ):
            raise RuntimeError(
                f"{path}:numerical.members[{member_index - 1}] batch ownership drifted"
            )

    finish = finish_events[0]
    if (
        finish.get("phase") != capture_phase
        or finish.get("arm") != arm
        or finish.get("status") != "complete"
        or finish.get("manifest_path") != numerical["manifest_path"]
        or finish.get("manifest_sha256") != numerical["manifest_sha256"]
        or type(finish.get("member_count")) is not int
        or finish.get("member_count") != 4
        or any(type(event.get("sequence")) is not int for event in events)
        or not (
            max(event["sequence"] for event in kv_events)
            < min(event["sequence"] for event in logits_events)
            <= max(event["sequence"] for event in logits_events)
            < finish["sequence"]
        )
    ):
        raise RuntimeError(f"{path} numerical finish event binding drifted")


def _load(path: Path, expected_arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("arm") != expected_arm:
        raise RuntimeError(f"{path} is not a completed {expected_arm} result")
    if value.get("checks", {}).get("all_passed") is not True:
        raise RuntimeError(f"{path} did not pass all arm checks")
    if value.get("request_id_contract") != REQUEST_ID_CONTRACT:
        raise RuntimeError(f"{path} request-ID contract drifted")
    observations = value.get("observations")
    if not isinstance(observations, list) or not observations:
        raise RuntimeError(f"{path} has no runtime observations")
    external_ids = [item.get("external_request_id") for item in observations]
    trace_ids = [item.get("trace_request_id") for item in observations]
    if (
        external_ids != [str(index) for index in range(len(observations))]
        or any(type(trace_id) is not str or not trace_id for trace_id in trace_ids)
        or len(set(trace_ids)) != len(trace_ids)
    ):
        raise RuntimeError(f"{path} request observations are not bijective")
    observations_by_trace = {item["trace_request_id"]: item for item in observations}
    recovery = value.get("recovery_evidence")
    if not isinstance(recovery, dict):
        raise RuntimeError(f"{path} has malformed recovery evidence")
    request_id_bindings = recovery.get("request_id_bindings")
    expected_binding_action = (
        "gate.numerical.request_bound"
        if expected_arm == "ordinary"
        else "LMCacheConnectorV1Impl.prepare_request_lookup.begin"
    )
    if not isinstance(request_id_bindings, list) or len(request_id_bindings) != len(
        observations
    ):
        raise RuntimeError(f"{path} request-ID binding coverage is incomplete")
    bindings_by_trace: dict[str, dict[str, Any]] = {}
    for observation, binding in zip(
        observations,
        request_id_bindings,
        strict=True,
    ):
        if (
            type(binding) is not dict
            or binding.get("trace_request_id") != observation["trace_request_id"]
            or binding.get("external_request_id") != observation["external_request_id"]
            or binding.get("phase") != observation.get("phase")
            or binding.get("binding_event_action") != expected_binding_action
            or not _internal_request_id_matches(
                binding.get("engine_request_id"),
                observation["external_request_id"],
            )
        ):
            raise RuntimeError(f"{path} canonical request-ID binding is invalid")
        bindings_by_trace[observation["trace_request_id"]] = binding
    if len(bindings_by_trace) != len(observations):
        raise RuntimeError(f"{path} canonical request-ID bindings are ambiguous")
    recovery_members = list(recovery.get("serial_members", []))
    for group in recovery.get("groups", []):
        recovery_members.extend(group.get("members", []))
    for member in recovery_members:
        if type(member) is not dict:
            raise RuntimeError(f"{path} recovery member is not a mapping")
        observation = observations_by_trace.get(member.get("trace_request_id"))
        binding = bindings_by_trace.get(member.get("trace_request_id"))
        if (
            observation is None
            or binding is None
            or member.get("external_request_id") != observation["external_request_id"]
            or member.get("engine_request_id") != binding["engine_request_id"]
        ):
            raise RuntimeError(f"{path} recovery request-ID binding is invalid")

    status_path = path.parent / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("status") != "complete"
        or status.get("arm") != expected_arm
        or status.get("result_sha256") != _sha256(path)
    ):
        raise RuntimeError(f"{status_path} does not publish this exact result")

    ledger_path = path.parent / "events.jsonl"
    ledger = value.get("event_ledger")
    if not isinstance(ledger, dict) or ledger.get("sha256") != _sha256(ledger_path):
        raise RuntimeError(f"{ledger_path} differs from the recorded digest")
    events = []
    for sequence, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines()
    ):
        if not line:
            raise RuntimeError(f"{ledger_path} contains a blank event")
        event = json.loads(line)
        if event.get("schema_version") != 1 or event.get("sequence") != sequence:
            raise RuntimeError(f"{ledger_path} has a noncanonical event sequence")
        events.append(event)
    if len(events) != ledger.get("event_count"):
        raise RuntimeError(f"{ledger_path} event count differs from the result")
    if not events or events[-1].get("action") != "gate.shutdown.complete":
        raise RuntimeError(f"{ledger_path} lacks terminal clean-shutdown evidence")
    canonical_events = [
        event for event in events if event.get("action") == expected_binding_action
    ]
    if len(canonical_events) != len(request_id_bindings):
        raise RuntimeError(f"{ledger_path} request-ID event coverage is incomplete")
    for binding, event in zip(
        request_id_bindings,
        canonical_events,
        strict=True,
    ):
        event_internal = event.get(
            "engine_request_id" if expected_arm == "ordinary" else "request_id"
        )
        if (
            event_internal != binding["engine_request_id"]
            or event.get("phase") != binding["phase"]
            or (
                expected_arm == "ordinary"
                and event.get("trace_request_id") != binding["trace_request_id"]
            )
            or (
                expected_arm != "ordinary"
                and (
                    event.get("owner_type") != "LMCacheConnectorV1Impl"
                    or event.get("role") != "SCHEDULER"
                )
            )
        ):
            raise RuntimeError(f"{ledger_path} request-ID event binding drifted")

    _verify_observation_events(path, observations, events)
    _verify_recovery_ledger(
        path,
        expected_arm,
        observations,
        recovery,
        request_id_bindings,
        events,
    )
    verified_arm_claims = _verify_arm_claims(
        path,
        expected_arm,
        observations,
        recovery,
        events,
    )

    numerical = value.get("numerical_evidence")
    if not isinstance(numerical, dict) or type(numerical.get("enabled")) is not bool:
        raise RuntimeError(f"{path} has malformed numerical evidence")
    manifest: dict[str, Any] | None = None
    if numerical["enabled"]:
        relative = numerical.get("manifest_path")
        digest = numerical.get("manifest_sha256")
        if type(relative) is not str or Path(relative).is_absolute():
            raise RuntimeError(f"{path} has an invalid numerical manifest path")
        if relative != "post-pic-capture-v1/family-manifest.json":
            raise RuntimeError(f"{path} numerical manifest path is not schema v1")
        manifest_path = path.parent / relative
        if not manifest_path.resolve().is_relative_to(path.parent.resolve()):
            raise RuntimeError(f"{path} numerical manifest escapes its arm")
        if numerical.get("status") != "complete" or digest != _sha256(manifest_path):
            raise RuntimeError(f"{path} numerical manifest binding failed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "kmesh.post_pic_kv_capture.v1":
            raise RuntimeError(f"{path} numerical manifest schema is unexpected")
    elif numerical.get("status") != "disabled":
        raise RuntimeError(f"{path} disabled numerical evidence has invalid status")
    _verify_numerical_event_bindings(
        path,
        expected_arm,
        numerical,
        manifest,
        events,
    )

    value["_verified_ledger"] = {
        "path": str(ledger_path),
        "sha256": _sha256(ledger_path),
        "event_count": len(events),
        "terminal_action": events[-1]["action"],
    }
    value["_verified_arm_claims"] = verified_arm_claims
    return value


def _bound_capture_file(root: Path, descriptor: object, label: str) -> Path:
    if not isinstance(descriptor, dict):
        raise RuntimeError(f"{label} is not a file descriptor")
    relative = descriptor.get("path")
    digest = descriptor.get("sha256")
    size = descriptor.get("nbytes")
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{label} has an invalid relative path")
    if ".." in Path(relative).parts:
        raise RuntimeError(f"{label} escapes its capture root")
    path = (root / relative).resolve()
    if path.parent != root.resolve() and root.resolve() not in path.parents:
        raise RuntimeError(f"{label} escapes its capture root")
    if (
        not path.is_file()
        or type(size) is not int
        or path.stat().st_size != size
        or type(digest) is not str
        or _sha256(path) != digest
    ):
        raise RuntimeError(f"{label} differs from its size or digest binding")
    return path


def _load_numerical_family(
    result_path: Path,
    result: dict[str, Any],
    arm: str,
) -> dict[str, dict[str, Any]]:
    evidence = result["numerical_evidence"]
    manifest_path = result_path.parent / evidence["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "kmesh.post_pic_kv_capture.v1"
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("arm") != arm
        or manifest.get("capture_boundary") != CAPTURE_BOUNDARY
    ):
        raise RuntimeError(f"{arm} numerical manifest identity is invalid")
    if manifest.get("request_id_contract") != REQUEST_ID_CONTRACT:
        raise RuntimeError(f"{arm} numerical request-ID contract drifted")
    model = manifest.get("model")
    engine_kwargs = result["engine"]["kwargs"]
    if (
        not isinstance(model, dict)
        or model.get("model_id") != engine_kwargs.get("model")
        or model.get("revision") != engine_kwargs.get("revision")
        or model.get("attention_backend") != "TRITON_ATTN"
    ):
        raise RuntimeError(f"{arm} numerical manifest did not use TRITON_ATTN")
    trace = manifest.get("trace")
    if (
        not isinstance(trace, dict)
        or trace.get("sha256") != result["inputs"]["trace"]["sha256"]
        or trace.get("trace_id") != result["inputs"]["manifest"]["trace_id"]
    ):
        raise RuntimeError(f"{arm} numerical trace identity drifted")
    if manifest.get("runtime") != result["runtime"]:
        raise RuntimeError(f"{arm} numerical runtime identity drifted")
    source = manifest.get("source")
    for name in ("vllm", "lmcache"):
        if not isinstance(source, dict) or not isinstance(source.get(name), dict):
            raise RuntimeError(f"{arm} numerical source.{name} is absent")
        if any(
            source[name].get(field) != result["source"][name].get(field)
            for field in ("commit", "tree")
        ):
            raise RuntimeError(f"{arm} numerical source.{name} drifted")

    tensor_contract = manifest.get("tensor_contract")
    expected_contract = {
        "format": "safetensors",
        "dtype": "bfloat16",
        "shape": list(KV_SHAPE),
        "axis_order": ["layer", "prompt_token", "kv_hidden"],
        "logical_block_shape": [16, 24, 32, 128],
        "block_size": 32,
        "prompt_token_count": 512,
        "kv_heads": 2,
        "head_dim": 64,
        "tensor_keys": ["key", "value"],
    }
    if not isinstance(tensor_contract, dict) or any(
        tensor_contract.get(field) != expected
        for field, expected in expected_contract.items()
    ):
        raise RuntimeError(f"{arm} numerical tensor contract is invalid")
    tensors_contract = tensor_contract.get("tensors")
    if tensors_contract != {
        "key": {"dtype": "bfloat16", "shape": list(KV_SHAPE)},
        "value": {"dtype": "bfloat16", "shape": list(KV_SHAPE)},
    }:
        raise RuntimeError(f"{arm} numerical tensor declarations are invalid")
    layer_names = tensor_contract.get("layer_names")
    if (
        not isinstance(layer_names, list)
        or len(layer_names) != KV_SHAPE[0]
        or len(set(layer_names)) != KV_SHAPE[0]
        or any(not isinstance(name, str) or not name for name in layer_names)
    ):
        raise RuntimeError(f"{arm} numerical layer names are invalid")

    expected_comparison_contract = {
        "serial_vs_collective": {
            "kv": {"mode": "bitwise_exact", "atol": 0.0, "rtol": 0.0},
            "logits": {
                "atol": LOCKSTEP_LOGITS_ATOL,
                "rtol": LOCKSTEP_LOGITS_RTOL,
            },
        },
        "ordinary_vs_pic": {
            "kv": {"atol": ORDINARY_KV_ATOL, "rtol": ORDINARY_KV_RTOL},
            "logits": {
                "atol": ORDINARY_LOGITS_ATOL,
                "rtol": ORDINARY_LOGITS_RTOL,
            },
            "require_exact_top1_and_generated_token": True,
        },
    }
    if manifest.get("comparison_contract") != expected_comparison_contract:
        raise RuntimeError(f"{arm} numerical comparison contract drifted")

    collective = manifest.get("collective")
    expected_profile = {
        "ordinary": "ordinary_batch_forward",
        "serial_oracle": "serial_pic_v1",
        "collective_forward": "collective_batched_qkv_v1",
    }[arm]
    if (
        not isinstance(collective, dict)
        or collective.get("profile") != expected_profile
        or collective.get("member_count") != 4
        or collective.get("all_members_successful") is not True
        or not isinstance(collective.get("plan_id"), str)
        or not collective["plan_id"]
        or not isinstance(collective.get("reuse_group_id"), str)
        or not collective["reuse_group_id"]
    ):
        raise RuntimeError(f"{arm} numerical family identity is invalid")
    if arm == "collective_forward":
        groups = result["recovery_evidence"]["groups"]
        if len(groups) != 1 or any(
            collective[field] != groups[0].get(field)
            for field in ("plan_id", "reuse_group_id")
        ):
            raise RuntimeError(f"{arm} numerical plan/group identity drifted")

    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 4:
        raise RuntimeError(f"{arm} numerical family must contain four members")
    observations = {
        item["trace_request_id"]: item
        for item in result["observations"]
        if item.get("sequence_index") in {1, 2, 3, 4}
    }
    request_id_bindings = {
        item["trace_request_id"]: item
        for item in result["recovery_evidence"]["request_id_bindings"]
    }
    loaded: dict[str, dict[str, Any]] = {}
    root = manifest_path.parent
    globally_owned_slots: set[int] = set()
    observed_capture_rows: set[int] = set()
    batch_engine_request_ids: list[str | None] = [None, None, None, None]
    for expected_sequence, member in enumerate(members, start=1):
        label = f"{arm}.members[{expected_sequence - 1}]"
        if not isinstance(member, dict):
            raise RuntimeError(f"{label} is not an object")
        trace_id = member.get("trace_request_id")
        capture_batch_row = member.get("capture_batch_row")
        if (
            member.get("sequence_index") != expected_sequence
            or type(member.get("sequence_index")) is not int
            or member.get("plan_member_index") != expected_sequence - 1
            or type(member.get("plan_member_index")) is not int
            or type(capture_batch_row) is not int
            or capture_batch_row not in {0, 1, 2, 3}
            or capture_batch_row in observed_capture_rows
            or type(trace_id) is not str
            or trace_id in loaded
            or member.get("plan_id") != collective["plan_id"]
            or member.get("reuse_group_id") != collective["reuse_group_id"]
        ):
            raise RuntimeError(f"{label} order or identity is invalid")
        observed_capture_rows.add(capture_batch_row)
        observation = observations.get(trace_id)
        request_id_binding = request_id_bindings.get(trace_id)
        if (
            observation is None
            or request_id_binding is None
            or member.get("external_request_id") != observation["external_request_id"]
            or member.get("engine_request_id")
            != request_id_binding["engine_request_id"]
            or type(member.get("prompt_token_count")) is not int
            or member.get("prompt_token_count") != 512
            or type(member.get("generated_token_count")) is not int
            or member.get("generated_token_count") != 0
        ):
            raise RuntimeError(f"{label} differs from its runtime observation")
        if arm == "ordinary":
            if (
                member.get("outcome") != "ordinary_forward"
                or member.get("pic_executed") is not False
            ):
                raise RuntimeError(f"{label} ordinary ownership is invalid")
        elif (
            member.get("outcome") != "recovered"
            or member.get("fallback_reason") is not None
            or member.get("pic_executed") is not True
        ):
            raise RuntimeError(f"{label} PIC ownership is invalid")
        prompt_digest = member.get("prompt_token_ids_sha256")
        blocks = member.get("physical_block_ids")
        slots = member.get("slots")
        if (
            type(prompt_digest) is not str
            or len(prompt_digest) != 64
            or any(character not in "0123456789abcdef" for character in prompt_digest)
            or not isinstance(blocks, list)
            or len(blocks) != 16
            or any(type(block) is not int or block < 0 for block in blocks)
            or len(set(blocks)) != 16
            or not isinstance(slots, list)
            or len(slots) != 512
            or any(type(slot) is not int or slot < 0 for slot in slots)
            or len(set(slots)) != 512
            or slots
            != [block * 32 + offset for block in blocks for offset in range(32)]
            or globally_owned_slots.intersection(slots)
        ):
            raise RuntimeError(f"{label} prompt-slot ownership is invalid")
        globally_owned_slots.update(slots)
        batch_engine_request_ids[capture_batch_row] = member["engine_request_id"]
        tensor_path = _bound_capture_file(root, member.get("tensor_file"), label)
        logits_path = _bound_capture_file(
            root,
            member.get("logits_file"),
            f"{label}.logits_file",
        )
        tensors = load_file(str(tensor_path), device="cpu")
        logits = load_file(str(logits_path), device="cpu")
        if set(tensors) != {"key", "value"} or set(logits) != {"raw_logits"}:
            raise RuntimeError(f"{label} safetensor keys are invalid")
        key = tensors["key"]
        value = tensors["value"]
        raw_logits = logits["raw_logits"]
        if any(
            tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != KV_SHAPE
            or not tensor.is_contiguous()
            for tensor in (key, value)
        ):
            raise RuntimeError(f"{label} K/V tensor contract is invalid")
        if (
            raw_logits.dtype != torch.float32
            or raw_logits.ndim != 1
            or raw_logits.numel() != 151936
            or not raw_logits.is_contiguous()
        ):
            raise RuntimeError(f"{label} raw-logit tensor contract is invalid")
        if not all(
            bool(torch.isfinite(tensor.float()).all().item())
            for tensor in (key, value, raw_logits)
        ):
            raise RuntimeError(f"{label} contains non-finite numerical material")
        loaded[trace_id] = {
            "key": key,
            "value": value,
            "raw_logits": raw_logits,
            "output_token_id": observation["output_token_ids"][0],
        }
    if (
        observed_capture_rows != {0, 1, 2, 3}
        or any(
            engine_request_id is None for engine_request_id in batch_engine_request_ids
        )
        or len(set(batch_engine_request_ids)) != 4
    ):
        raise RuntimeError(f"{arm} numerical capture rows are not bijective")
    for expected_sequence, member in enumerate(members, start=1):
        if (
            type(member.get("batch_engine_request_ids")) is not list
            or member["batch_engine_request_ids"] != batch_engine_request_ids
            or batch_engine_request_ids[member["capture_batch_row"]]
            != member["engine_request_id"]
        ):
            raise RuntimeError(
                f"{arm}.members[{expected_sequence - 1}] batch ownership drifted"
            )
    expected_bindings = [
        {
            "sequence_index": member["sequence_index"],
            "trace_request_id": member["trace_request_id"],
            "external_request_id": member["external_request_id"],
            "engine_request_id": member["engine_request_id"],
        }
        for member in members
    ]
    if manifest.get("observation_bindings") != expected_bindings:
        raise RuntimeError(f"{arm} numerical observation bindings drifted")
    return loaded


def _delta_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    delta = torch.abs(reference.float() - candidate.float()).reshape(-1)
    quantiles = torch.quantile(
        delta,
        torch.tensor([0.5, 0.95, 0.99, 0.999], dtype=torch.float32),
    )
    return {
        "element_count": delta.numel(),
        "mismatch_count": int(torch.count_nonzero(delta).item()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "p50_abs": float(quantiles[0].item()),
        "p95_abs": float(quantiles[1].item()),
        "p99_abs": float(quantiles[2].item()),
        "p999_abs": float(quantiles[3].item()),
    }


def _bf16_raw_bit_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    if (
        reference.dtype != torch.bfloat16
        or candidate.dtype != torch.bfloat16
        or tuple(reference.shape) != tuple(candidate.shape)
        or not reference.is_contiguous()
        or not candidate.is_contiguous()
    ):
        raise RuntimeError("raw-bit comparison requires like-shaped contiguous BF16")
    mismatch = reference.view(torch.int16) != candidate.view(torch.int16)
    element_count = reference.numel()
    mismatch_count = int(torch.count_nonzero(mismatch).item())
    return {
        "storage_dtype": "bfloat16",
        "element_count": element_count,
        "raw_bit_mismatch_count": mismatch_count,
        "raw_bit_mismatch_fraction": mismatch_count / element_count,
        "bitwise_equal": mismatch_count == 0,
    }


def _compare_numerical_families(
    ordinary: dict[str, dict[str, Any]],
    serial: dict[str, dict[str, Any]],
    collective: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(ordinary) != set(serial) or set(serial) != set(collective):
        raise RuntimeError("numerical families have different trace membership")
    members: dict[str, Any] = {}
    lockstep_primary_pass = True
    ordinary_pic_diagnostic_pass = True
    for trace_id in sorted(ordinary):
        arm_values = {
            "ordinary": ordinary[trace_id],
            "serial_oracle": serial[trace_id],
            "collective_forward": collective[trace_id],
        }
        comparisons: dict[str, Any] = {}
        for tensor_name in ("key", "value"):
            ordinary_tensor = ordinary[trace_id][tensor_name]
            serial_tensor = serial[trace_id][tensor_name]
            collective_tensor = collective[trace_id][tensor_name]
            serial_collective_raw_bits = _bf16_raw_bit_metrics(
                serial_tensor,
                collective_tensor,
            )
            serial_collective_exact = serial_collective_raw_bits["bitwise_equal"]
            ordinary_serial_close = torch.allclose(
                ordinary_tensor.float(),
                serial_tensor.float(),
                atol=ORDINARY_KV_ATOL,
                rtol=ORDINARY_KV_RTOL,
            )
            ordinary_collective_close = torch.allclose(
                ordinary_tensor.float(),
                collective_tensor.float(),
                atol=ORDINARY_KV_ATOL,
                rtol=ORDINARY_KV_RTOL,
            )
            comparisons[tensor_name] = {
                "serial_collective_exact": serial_collective_exact,
                "serial_collective_raw_bits": serial_collective_raw_bits,
                "serial_collective_delta": _delta_metrics(
                    serial_tensor, collective_tensor
                ),
                "ordinary_serial_within_tolerance": ordinary_serial_close,
                "ordinary_serial_delta": _delta_metrics(ordinary_tensor, serial_tensor),
                "ordinary_collective_within_tolerance": ordinary_collective_close,
                "ordinary_collective_delta": _delta_metrics(
                    ordinary_tensor, collective_tensor
                ),
            }
            lockstep_primary_pass = lockstep_primary_pass and serial_collective_exact
            ordinary_pic_diagnostic_pass = ordinary_pic_diagnostic_pass and all(
                (ordinary_serial_close, ordinary_collective_close)
            )

        ordinary_logits = ordinary[trace_id]["raw_logits"]
        serial_logits = serial[trace_id]["raw_logits"]
        collective_logits = collective[trace_id]["raw_logits"]
        serial_collective_logits_close = torch.allclose(
            serial_logits,
            collective_logits,
            atol=LOCKSTEP_LOGITS_ATOL,
            rtol=LOCKSTEP_LOGITS_RTOL,
        )
        ordinary_serial_logits_close = torch.allclose(
            ordinary_logits,
            serial_logits,
            atol=ORDINARY_LOGITS_ATOL,
            rtol=ORDINARY_LOGITS_RTOL,
        )
        ordinary_collective_logits_close = torch.allclose(
            ordinary_logits,
            collective_logits,
            atol=ORDINARY_LOGITS_ATOL,
            rtol=ORDINARY_LOGITS_RTOL,
        )
        top1 = {
            arm: int(values["raw_logits"].argmax().item())
            for arm, values in arm_values.items()
        }
        generated = {
            arm: values["output_token_id"] for arm, values in arm_values.items()
        }
        top1_exact = len(set(top1.values())) == 1 and top1 == generated
        lockstep_top1 = {
            arm: top1[arm] for arm in ("serial_oracle", "collective_forward")
        }
        lockstep_generated = {
            arm: generated[arm] for arm in ("serial_oracle", "collective_forward")
        }
        lockstep_top1_exact = (
            len(set(lockstep_top1.values())) == 1
            and lockstep_top1 == lockstep_generated
        )
        comparisons["raw_logits"] = {
            "serial_collective_within_tolerance": serial_collective_logits_close,
            "serial_collective_delta": _delta_metrics(serial_logits, collective_logits),
            "ordinary_serial_within_tolerance": ordinary_serial_logits_close,
            "ordinary_serial_delta": _delta_metrics(ordinary_logits, serial_logits),
            "ordinary_collective_within_tolerance": (ordinary_collective_logits_close),
            "ordinary_collective_delta": _delta_metrics(
                ordinary_logits, collective_logits
            ),
            "top1_by_arm": top1,
            "generated_token_by_arm": generated,
            "top1_and_generated_tokens_are_exact": top1_exact,
            "serial_collective_top1_and_generated_tokens_are_exact": (
                lockstep_top1_exact
            ),
        }
        lockstep_primary_pass = lockstep_primary_pass and all(
            (serial_collective_logits_close, lockstep_top1_exact)
        )
        ordinary_pic_diagnostic_pass = ordinary_pic_diagnostic_pass and all(
            (
                ordinary_serial_logits_close,
                ordinary_collective_logits_close,
                top1_exact,
            )
        )
        members[trace_id] = comparisons
    return {
        "status": "complete",
        "lockstep_primary_pass": lockstep_primary_pass,
        "ordinary_pic_diagnostic_pass": ordinary_pic_diagnostic_pass,
        "ordinary_pic_claim_accepted": False,
        "thresholds": {
            "serial_collective_kv": {"atol": 0.0, "rtol": 0.0},
            "serial_collective_logits": {
                "atol": LOCKSTEP_LOGITS_ATOL,
                "rtol": LOCKSTEP_LOGITS_RTOL,
            },
            "ordinary_pic_kv": {
                "atol": ORDINARY_KV_ATOL,
                "rtol": ORDINARY_KV_RTOL,
            },
            "ordinary_pic_logits": {
                "atol": ORDINARY_LOGITS_ATOL,
                "rtol": ORDINARY_LOGITS_RTOL,
            },
        },
        "members": members,
        "claim_boundary": (
            "Forward-order, same-TRITON_ATTN numerical correctness only. "
            "Serial-oracle/collective-forward BF16 storage-payload parity is "
            "terminal; ordinary/PIC parity is a separately reported nonterminal "
            "diagnostic. Device-wide synchronization and artifact I/O invalidate "
            "timing."
        ),
    }


def _identity(value: dict[str, Any]) -> dict[str, Any]:
    runtime = value["runtime"]
    shared_kwargs = {
        key: item
        for key, item in value["engine"]["kwargs"].items()
        if key != "kv_transfer_config"
    }
    return {
        "source": value["source"],
        "runner_sha256": value["inputs"]["runner"]["sha256"],
        "common_sha256": value["inputs"]["common"]["sha256"],
        "accepted_serial_runner_sha256": value["inputs"]["accepted_serial_runner"][
            "sha256"
        ],
        "numerical_instrumentation_sha256": value["inputs"][
            "numerical_instrumentation"
        ]["sha256"],
        "runtime": {
            field: runtime[field]
            for field in (
                "python",
                "torch",
                "hip",
                "transformers",
                "tokenizers",
                "sortedcontainers",
                "aiofile",
                "caio",
                "target_volume",
                "wheel_vllm_distribution",
                "overlay_vllm_version",
                "overlay_vllm_module",
                "overlay_lmcache_module",
                "accelerator_count",
                "device_arch",
                "device_total_memory",
                "image",
                "compiled_extensions",
            )
        },
        "shared_engine_kwargs": shared_kwargs,
        "trace_sha256": value["inputs"]["trace"]["sha256"],
        "manifest_sha256": value["inputs"]["manifest"]["sha256"],
        "lmcache_config_sha256": value["inputs"]["lmcache_config"]["sha256"],
    }


def _semantic_group(value: dict[str, Any]) -> dict[str, Any]:
    groups = value["recovery_evidence"]["groups"]
    if len(groups) != 1:
        raise RuntimeError(f"{value['arm']} does not have exactly one group")
    group = groups[0]
    members = sorted(
        (
            {
                "trace_request_id": member["trace_request_id"],
                "outcome": member["outcome"],
                "selected_spans": member["selected_spans"],
                "retrieval_mask": member["retrieval_mask"],
                "operation_counts": member["operation_counts"],
                "error": member["error"],
            }
            for member in group["members"]
        ),
        key=lambda member: member["trace_request_id"],
    )
    return {
        "call_index": group.get("call_index"),
        "num_layers": group.get("num_layers"),
        "trace_request_members": [member["trace_request_id"] for member in members],
        "members": members,
    }


def _mask_map_from_group(value: dict[str, Any]) -> dict[str, Any]:
    return {
        member["trace_request_id"]: {
            "selected_spans": member["selected_spans"],
            "retrieval_mask": member["retrieval_mask"],
        }
        for member in _semantic_group(value)["members"]
    }


def _mask_map_from_serial(value: dict[str, Any]) -> dict[str, Any]:
    return {
        member["trace_request_id"]: {
            "selected_spans": member["selected_spans"],
            "retrieval_mask": member["retrieval_mask"],
        }
        for member in value["recovery_evidence"]["serial_members"]
    }


def _verify_cross_arm_output_parity(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    canonical_trace_by_sequence = {
        observation["sequence_index"]: observation["trace_request_id"]
        for observation in results["ordinary"]["observations"]
    }
    if set(canonical_trace_by_sequence) != set(range(5)):
        raise RuntimeError("ordinary sequence-to-trace identity is not canonical")

    def canonical_outputs(arm: str) -> dict[str, list[int]]:
        outputs: dict[str, list[int]] = {}
        for observation in results[arm]["observations"]:
            sequence = observation["sequence_index"]
            trace_id = observation["trace_request_id"]
            if (
                trace_id != canonical_trace_by_sequence.get(sequence)
                or trace_id in outputs
            ):
                raise RuntimeError(
                    f"{arm} sequence-to-trace output identity is not canonical"
                )
            outputs[trace_id] = observation["output_token_ids"]
        return outputs

    outputs_by_arm = {arm: canonical_outputs(arm) for arm in results}
    ordinary_outputs = outputs_by_arm["ordinary"]
    serial_outputs = outputs_by_arm["serial_oracle"]
    lockstep_arms = (
        "serial_oracle",
        "collective_forward",
        "collective_reverse",
    )
    control_arms = (
        "cold_miss",
        "zero_sharing",
        "incompatible_partition",
        "sibling_failure",
    )
    lockstep_terminal_by_arm = {
        arm: outputs_by_arm[arm] == serial_outputs for arm in lockstep_arms
    }
    ordinary_pic_diagnostic_by_arm = {
        arm: outputs_by_arm[arm] == ordinary_outputs for arm in lockstep_arms
    }
    control_terminal_by_arm = {
        arm: all(
            outputs == ordinary_outputs.get(trace_id)
            for trace_id, outputs in outputs_by_arm[arm].items()
        )
        for arm in control_arms
    }
    lockstep_terminal_pass = all(lockstep_terminal_by_arm.values())
    control_terminal_pass = all(control_terminal_by_arm.values())
    if not lockstep_terminal_pass:
        mismatched = sorted(
            arm for arm, matches in lockstep_terminal_by_arm.items() if not matches
        )
        raise RuntimeError(
            "lockstep arm outputs differ from the serial oracle: "
            + ", ".join(mismatched)
        )
    if not control_terminal_pass:
        mismatched = sorted(
            arm for arm, matches in control_terminal_by_arm.items() if not matches
        )
        raise RuntimeError(
            "control arm outputs differ from ordinary: " + ", ".join(mismatched)
        )
    return {
        "lockstep_terminal_by_arm_against_serial_oracle": lockstep_terminal_by_arm,
        "lockstep_terminal_pass": lockstep_terminal_pass,
        "ordinary_pic_diagnostic_by_arm_against_ordinary": (
            ordinary_pic_diagnostic_by_arm
        ),
        "ordinary_pic_output_diagnostic_pass": all(
            ordinary_pic_diagnostic_by_arm.values()
        ),
        "control_terminal_by_arm_against_ordinary": control_terminal_by_arm,
        "control_terminal_pass": control_terminal_pass,
    }


def _build_claims(
    independently_verified: dict[str, bool],
    output_policy: dict[str, Any],
    *,
    order_stable: bool,
    serial_collective_mask_parity: bool,
    terminal_cleanup_pass: bool,
    numerical_comparison: dict[str, Any],
) -> dict[str, bool]:
    lockstep_outputs = output_policy["lockstep_terminal_by_arm_against_serial_oracle"]
    control_outputs = output_policy["control_terminal_by_arm_against_ordinary"]
    claims = {
        "ordinary_baseline_pass": independently_verified["ordinary"],
        "serial_oracle_pass": independently_verified["serial_oracle"]
        and lockstep_outputs["serial_oracle"],
        "collective_forward_pass": independently_verified["collective_forward"]
        and lockstep_outputs["collective_forward"],
        "collective_reverse_pass": independently_verified["collective_reverse"]
        and lockstep_outputs["collective_reverse"],
        "forward_reverse_order_stability_pass": order_stable,
        "serial_collective_mask_parity_pass": serial_collective_mask_parity,
        "lockstep_output_pass": output_policy["lockstep_terminal_pass"],
        "cold_miss_control_pass": independently_verified["cold_miss"]
        and control_outputs["cold_miss"],
        "zero_sharing_control_pass": independently_verified["zero_sharing"]
        and control_outputs["zero_sharing"],
        "incompatible_partition_control_pass": independently_verified[
            "incompatible_partition"
        ]
        and control_outputs["incompatible_partition"],
        "sibling_failure_isolation_pass": independently_verified["sibling_failure"]
        and control_outputs["sibling_failure"],
        "control_output_pass": output_policy["control_terminal_pass"],
        "terminal_cleanup_pass": terminal_cleanup_pass,
        "collective_kernel_batching_claim": True,
        "performance_claim": False,
        "numerical_capture_available": True,
        "numerical_forward_ordinary_comparison_available": True,
        "lockstep_numerical_pass": numerical_comparison["lockstep_primary_pass"],
        "ordinary_pic_output_diagnostic_pass": output_policy[
            "ordinary_pic_output_diagnostic_pass"
        ],
        "ordinary_pic_diagnostic_pass": numerical_comparison[
            "ordinary_pic_diagnostic_pass"
        ]
        and output_policy["ordinary_pic_output_diagnostic_pass"],
        "ordinary_pic_claim_accepted": False,
    }
    claims["full_collective_lockstep_gate_pass"] = all(
        claims[name]
        for name in (
            "ordinary_baseline_pass",
            "serial_oracle_pass",
            "collective_forward_pass",
            "collective_reverse_pass",
            "forward_reverse_order_stability_pass",
            "serial_collective_mask_parity_pass",
            "lockstep_output_pass",
            "cold_miss_control_pass",
            "zero_sharing_control_pass",
            "incompatible_partition_control_pass",
            "sibling_failure_isolation_pass",
            "control_output_pass",
            "terminal_cleanup_pass",
            "lockstep_numerical_pass",
        )
    )
    return claims


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(
            "--" + arm.replace("_", "-"),
            required=True,
            type=Path,
        )
    parser.add_argument(
        "--source-pair", choices=["dated", "reference", "current"], default="dated"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.checksums.exists():
        raise RuntimeError("refusing to overwrite verification outputs")
    paths = {arm: getattr(args, arm) for arm in ARMS}
    results = {arm: _load(path, arm) for arm, path in paths.items()}

    for result in results.values():
        validate_sources(result["source"], args.source_pair)
        if args.source_pair == "current":
            validate_current_runtime(result["runtime"])
    baseline_identity = _identity(results["ordinary"])
    if args.source_pair == "current":
        compiled = results["ordinary"]["runtime"]["lmcache_compiled_extensions"]
        if set(compiled) != {"lmcache_native", "cuda_ops"} or any(
            item["runtime"].get("lmcache_compiled_extensions") != compiled
            for item in results.values()
        ):
            raise RuntimeError("LMCache native extensions differ between CUDA arms")
    identity_match = {
        arm: _identity(value) == baseline_identity
        for arm, value in results.items()
        if arm != "ordinary"
    }
    if not all(identity_match.values()):
        mismatched = sorted(
            arm for arm, matches in identity_match.items() if not matches
        )
        raise RuntimeError("arm runtime identities differ: " + ", ".join(mismatched))

    ordinary_hash = _sha256(paths["ordinary"])
    baseline_binding = {
        arm: value["inputs"].get("baseline_result", {}).get("sha256") == ordinary_hash
        for arm, value in results.items()
        if arm != "ordinary"
    }
    if not all(baseline_binding.values()):
        raise RuntimeError("one or more arms are not bound to this ordinary result")

    output_policy = _verify_cross_arm_output_parity(results)

    forward_semantic = _semantic_group(results["collective_forward"])
    reverse_semantic = _semantic_group(results["collective_reverse"])
    order_stable = forward_semantic == reverse_semantic
    if not order_stable:
        raise RuntimeError("forward and reverse collective evidence differs")

    serial_masks = _mask_map_from_serial(results["serial_oracle"])
    forward_masks = _mask_map_from_group(results["collective_forward"])
    reverse_masks = _mask_map_from_group(results["collective_reverse"])
    serial_collective_mask_parity = (
        serial_masks == forward_masks == reverse_masks and len(serial_masks) == 4
    )
    if not serial_collective_mask_parity:
        raise RuntimeError("collective masks/spans differ from the serial oracle")

    numerical_enabled = {
        arm: result["numerical_evidence"]["enabled"] for arm, result in results.items()
    }
    expected_numerical_enabled = {arm: arm in NUMERICAL_ARMS for arm in ARMS}
    if numerical_enabled != expected_numerical_enabled:
        raise RuntimeError(
            "numerical instrumentation arm set differs from the frozen gate: "
            f"{numerical_enabled!r}"
        )
    numerical_families = {
        arm: _load_numerical_family(paths[arm], results[arm], arm)
        for arm in NUMERICAL_ARMS
    }
    numerical_comparison = _compare_numerical_families(
        numerical_families["ordinary"],
        numerical_families["serial_oracle"],
        numerical_families["collective_forward"],
    )
    if numerical_comparison["lockstep_primary_pass"] is not True:
        raise RuntimeError(
            "serial-oracle/collective-forward K/V, raw-logit, or output "
            "numerical parity failed"
        )

    independently_verified = {
        arm: value["_verified_arm_claims"].get("all_passed") is True
        for arm, value in results.items()
    }
    claims = _build_claims(
        independently_verified,
        output_policy,
        order_stable=order_stable,
        serial_collective_mask_parity=serial_collective_mask_parity,
        terminal_cleanup_pass=all(
            value["_verified_arm_claims"].get("cleanup_verified") is True
            for arm, value in results.items()
            if arm != "ordinary"
        ),
        numerical_comparison=numerical_comparison,
    )

    summary = {
        "schema_version": 2,
        "source_pair": args.source_pair,
        "source_pins": profile()["source_pairs"][args.source_pair],
        "acceptance_policy": "serial-pic-lockstep-primary-v2",
        "status": "complete",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "claims": claims,
        "identity_match": identity_match,
        "baseline_binding": baseline_binding,
        "output_policy": output_policy,
        "serial_collective_mask_parity": serial_collective_mask_parity,
        "numerical_verifier_runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
        },
        "numerical_comparison": numerical_comparison,
        "checksum_scope": "verified summary; arm ledgers and numerical manifests bind input artifacts",
        "forward_reverse_semantic_evidence": forward_semantic,
        "arm_results": {
            arm: {
                "path": str(paths[arm]),
                "sha256": _sha256(paths[arm]),
                "checks": value["checks"],
                "independently_verified_arm_claims": value["_verified_arm_claims"],
                "verified_event_ledger": value["_verified_ledger"],
                "numerical_evidence": value["numerical_evidence"],
            }
            for arm, value in results.items()
        },
        "claim_boundary": (
            "This proves a bounded single-device collective correctness reference, "
            "including exact request-local operation calls, mask/span agreement "
            "with the serial oracle, controls, cleanup, and forward-order raw "
            "serial-oracle/collective K/V, logit, and output parity under the "
            "frozen tolerances. Ordinary/PIC parity is retained as a nonterminal "
            "diagnostic and is not an accepted claim. Cached-key recovery and "
            "model-side rotary/difference calls batch under the pinned profile. "
            "This is not a latency, throughput, performance, capacity, "
            "Master-Mirror, generic-model, "
            "distributed, or production claim. Device synchronization and "
            "artifact I/O make the numerical arms invalid for timing."
        ),
    }
    _atomic_text(args.output, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    checksum_text = f"{_sha256(args.output)}  {args.output.name}\n"
    _atomic_text(args.checksums, checksum_text)
    print(json.dumps({"claims": claims, "summary": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
