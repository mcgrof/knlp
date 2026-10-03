# SPDX-License-Identifier: MIT
"""Backend-neutral workflow and cache-event contracts for agent experiments.

The contract deliberately stores hashes and structural metadata, not prompt
text.  Raw prompts and token IDs remain in the existing sensitive run capture.
Offline reuse analysis and physical engine events use different event sources
so a report cannot silently present prefix potential as an observed cache hit.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Iterable

WORKFLOW_SCHEMA = "knlp.agent_baselines.workflow.v1"
CACHE_EVENT_SCHEMA = "knlp.agent_baselines.cache_event.v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ROLES = {"system", "developer", "user", "assistant", "tool"}
_EVENT_PHASES = {"requested", "observed", "analysis"}
_EVENT_SOURCES = {"client", "engine", "offline"}
_CACHE_ACTIONS = {
    "retain",
    "release",
    "store",
    "evict",
    "retrieve",
    "prefetch",
    "transfer_complete",
    "skip",
    "potential_reuse",
}
_CACHE_TIERS = {"none", "gpu", "host", "storage"}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value with a stable, whitespace-free representation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    tokens = list(token_ids)
    _tokens(tokens, "token_ids", allow_empty=False)
    return canonical_sha256(tokens)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a stable identifier")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _REVISION.fullmatch(value):
        raise ValueError(f"{name} must be an immutable 40-character revision")
    return value


def _tokens(value: Any, name: str, *, allow_empty: bool) -> list[int]:
    if not isinstance(value, list) or (not value and not allow_empty):
        requirement = "possibly empty" if allow_empty else "nonempty"
        raise ValueError(f"{name} must be a {requirement} list")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return value


def prompt_segment(
    segment_id: str,
    *,
    role: str,
    origin: str,
    content: str,
    token_ids: Iterable[int] | None = None,
    inputs: Iterable[str] = (),
    movable: bool = False,
) -> dict:
    """Build a prompt-segment receipt without retaining its sensitive text."""
    _identifier(segment_id, "segment_id")
    if role not in _ROLES:
        raise ValueError(f"role must be one of {sorted(_ROLES)}")
    _identifier(origin, "origin")
    dependencies = list(inputs)
    if len(set(dependencies)) != len(dependencies):
        raise ValueError("segment inputs must be unique")
    for dependency in dependencies:
        _identifier(dependency, "segment input")
    if type(movable) is not bool:
        raise ValueError("movable must be a boolean")
    tokens = None if token_ids is None else list(token_ids)
    if tokens is not None:
        _tokens(tokens, "token_ids", allow_empty=True)
    return {
        "segment_id": segment_id,
        "role": role,
        "origin": origin,
        "content_sha256": text_sha256(content),
        "token_ids_sha256": canonical_sha256(tokens) if tokens is not None else None,
        "token_count": len(tokens) if tokens is not None else None,
        "inputs": dependencies,
        "movable": movable,
    }


def prompt_layout_sha256(segment_ids: Iterable[str], segments: Iterable[dict]) -> str:
    """Bind segment order, roles, and content identities into one layout hash."""
    by_id = {segment.get("segment_id"): segment for segment in segments}
    ordered = []
    for segment_id in segment_ids:
        _identifier(segment_id, "segment_id")
        if segment_id not in by_id:
            raise ValueError(f"Unknown prompt segment: {segment_id}")
        segment = by_id[segment_id]
        ordered.append(
            {
                "segment_id": segment_id,
                "role": segment.get("role"),
                "content_sha256": segment.get("content_sha256"),
                "token_ids_sha256": segment.get("token_ids_sha256"),
            }
        )
    if not ordered:
        raise ValueError("A prompt layout needs at least one segment")
    return canonical_sha256(ordered)


def workflow_request(
    request_id: str,
    *,
    workflow_id: str,
    branch_id: str,
    segment_ids: Iterable[str],
    segments: Iterable[dict],
    prompt_token_ids: Iterable[int],
    engine_settings: dict,
    depends_on: Iterable[str] = (),
    tool_state_version: str = "initial",
    ready_ns: int | None = None,
) -> dict:
    """Build one request receipt with explicit causal and cache identities."""
    _identifier(request_id, "request_id")
    _identifier(workflow_id, "workflow_id")
    _identifier(branch_id, "branch_id")
    _identifier(tool_state_version, "tool_state_version")
    dependencies = list(depends_on)
    if request_id in dependencies or len(set(dependencies)) != len(dependencies):
        raise ValueError("request dependencies must be unique and non-self")
    for dependency in dependencies:
        _identifier(dependency, "request dependency")
    if ready_ns is not None and (type(ready_ns) is not int or ready_ns < 0):
        raise ValueError("ready_ns must be a nonnegative integer or null")
    ordered_segments = list(segment_ids)
    tokens = list(prompt_token_ids)
    _tokens(tokens, "prompt_token_ids", allow_empty=False)
    if not isinstance(engine_settings, dict):
        raise ValueError("engine_settings must be an object")
    return {
        "request_id": request_id,
        "workflow_id": workflow_id,
        "branch_id": branch_id,
        "depends_on": dependencies,
        "segment_ids": ordered_segments,
        "tool_state_version": tool_state_version,
        "ready_ns": ready_ns,
        "prompt": {
            "token_count": len(tokens),
            "token_ids_sha256": token_ids_sha256(tokens),
            "layout_sha256": prompt_layout_sha256(ordered_segments, segments),
            "engine_settings_sha256": canonical_sha256(engine_settings),
        },
    }


def workflow_manifest(
    workflow_id: str,
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    chat_template_sha256: str,
    segments: Iterable[dict],
    requests: Iterable[dict],
    tool_effects: Iterable[dict] = (),
) -> dict:
    manifest = {
        "schema": WORKFLOW_SCHEMA,
        "workflow_id": workflow_id,
        "model": {
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_sha256": chat_template_sha256,
        },
        "segments": list(segments),
        "requests": list(requests),
        "tool_effects": list(tool_effects),
    }
    return validate_workflow_manifest(manifest)


def validate_workflow_manifest(value: Any) -> dict:
    """Validate graph references, acyclicity, and cache identity receipts."""
    if not isinstance(value, dict) or value.get("schema") != WORKFLOW_SCHEMA:
        raise ValueError(f"Expected workflow schema {WORKFLOW_SCHEMA}")
    workflow_id = _identifier(value.get("workflow_id"), "workflow_id")
    model = value.get("model")
    if (
        not isinstance(model, dict)
        or not isinstance(model.get("model_id"), str)
        or not model["model_id"]
    ):
        raise ValueError("Workflow model identity is required")
    _revision(model.get("model_revision"), "model_revision")
    _revision(model.get("tokenizer_revision"), "tokenizer_revision")
    _hash(model.get("chat_template_sha256"), "chat_template_sha256")

    segments = value.get("segments")
    requests = value.get("requests")
    effects = value.get("tool_effects")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Workflow requires prompt segments")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Workflow requires requests")
    if not isinstance(effects, list):
        raise ValueError("tool_effects must be a list")

    segment_ids = set()
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Each prompt segment must be an object")
        segment_id = _identifier(segment.get("segment_id"), "segment_id")
        if segment_id in segment_ids:
            raise ValueError("Duplicate segment_id")
        segment_ids.add(segment_id)
        if segment.get("role") not in _ROLES:
            raise ValueError("Invalid prompt segment role")
        _identifier(segment.get("origin"), "segment origin")
        _hash(segment.get("content_sha256"), "segment content_sha256")
        token_hash = segment.get("token_ids_sha256")
        token_count = segment.get("token_count")
        if (token_hash is None) != (token_count is None):
            raise ValueError("Segment token hash and count must be present together")
        if token_hash is not None:
            _hash(token_hash, "segment token_ids_sha256")
            if type(token_count) is not int or token_count < 0:
                raise ValueError("Segment token_count must be nonnegative")
        if type(segment.get("movable")) is not bool:
            raise ValueError("Segment movable must be a boolean")
        inputs = segment.get("inputs")
        if not isinstance(inputs, list):
            raise ValueError("Segment inputs must be a unique list")
        for item in inputs:
            _identifier(item, "segment input")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Segment inputs must be a unique list")

    request_ids = set()
    dependencies: dict[str, list[str]] = {}
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError("Each workflow request must be an object")
        request_id = _identifier(request.get("request_id"), "request_id")
        if request_id in request_ids:
            raise ValueError("Duplicate request_id")
        request_ids.add(request_id)
        if request.get("workflow_id") != workflow_id:
            raise ValueError("Request workflow_id does not match manifest")
        _identifier(request.get("branch_id"), "branch_id")
        _identifier(request.get("tool_state_version"), "tool_state_version")
        ready_ns = request.get("ready_ns")
        if ready_ns is not None and (type(ready_ns) is not int or ready_ns < 0):
            raise ValueError("ready_ns must be a nonnegative integer or null")
        ordered_segments = request.get("segment_ids")
        if not isinstance(ordered_segments, list) or not ordered_segments:
            raise ValueError("Request segment_ids must be nonempty")
        for item in ordered_segments:
            _identifier(item, "request segment_id")
        if any(item not in segment_ids for item in ordered_segments):
            raise ValueError("Request references an unknown prompt segment")
        deps = request.get("depends_on")
        if not isinstance(deps, list):
            raise ValueError("Request dependencies must be a unique list")
        for dependency in deps:
            _identifier(dependency, "request dependency")
        if len(set(deps)) != len(deps):
            raise ValueError("Request dependencies must be a unique list")
        dependencies[request_id] = deps
        prompt = request.get("prompt")
        if not isinstance(prompt, dict):
            raise ValueError("Request prompt identity is required")
        if type(prompt.get("token_count")) is not int or prompt["token_count"] <= 0:
            raise ValueError("Request prompt token_count must be positive")
        for name in (
            "token_ids_sha256",
            "layout_sha256",
            "engine_settings_sha256",
        ):
            _hash(prompt.get(name), f"request prompt {name}")
        expected_layout = prompt_layout_sha256(ordered_segments, segments)
        if prompt["layout_sha256"] != expected_layout:
            raise ValueError("Request prompt layout hash does not match its segments")

    for request_id, deps in dependencies.items():
        if request_id in deps or any(dep not in request_ids for dep in deps):
            raise ValueError("Request dependency references an unknown or self request")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(request_id: str) -> None:
        if request_id in visiting:
            raise ValueError("Workflow request graph contains a cycle")
        if request_id in visited:
            return
        visiting.add(request_id)
        for dependency in dependencies[request_id]:
            visit(dependency)
        visiting.remove(request_id)
        visited.add(request_id)

    for request_id in request_ids:
        visit(request_id)

    effect_ids = set()
    for effect in effects:
        if not isinstance(effect, dict):
            raise ValueError("Each tool effect must be an object")
        effect_id = _identifier(effect.get("effect_id"), "effect_id")
        if effect_id in effect_ids:
            raise ValueError("Duplicate effect_id")
        effect_ids.add(effect_id)
        if effect.get("request_id") not in request_ids:
            raise ValueError("Tool effect references an unknown request")
        if effect.get("kind") not in {"read", "write", "network", "other"}:
            raise ValueError("Invalid tool effect kind")
        _identifier(effect.get("resource"), "tool effect resource")
        if type(effect.get("isolated")) is not bool:
            raise ValueError("Tool effect isolated must be a boolean")
        for name in ("before_version", "after_version"):
            if effect.get(name) is not None:
                _identifier(effect[name], f"tool effect {name}")
    return deepcopy(value)


def validate_cache_event(value: Any) -> dict:
    """Validate one event and keep analysis distinct from physical evidence."""
    if not isinstance(value, dict) or value.get("schema") != CACHE_EVENT_SCHEMA:
        raise ValueError(f"Expected cache-event schema {CACHE_EVENT_SCHEMA}")
    for name in ("event_id", "workflow_id", "request_id", "backend"):
        _identifier(value.get(name), name)
    phase = value.get("phase")
    source = value.get("source")
    action = value.get("action")
    tier = value.get("tier")
    if phase not in _EVENT_PHASES or source not in _EVENT_SOURCES:
        raise ValueError("Invalid cache-event phase or source")
    if action not in _CACHE_ACTIONS or tier not in _CACHE_TIERS:
        raise ValueError("Invalid cache-event action or tier")
    if type(value.get("timestamp_ns")) is not int or value["timestamp_ns"] < 0:
        raise ValueError("timestamp_ns must be nonnegative")
    for name in ("token_count", "byte_count"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    pressure = value.get("pressure")
    if pressure is not None and type(pressure) is not bool:
        raise ValueError("pressure must be a boolean or null")
    enforced = value.get("enforced")
    if phase == "analysis":
        if (
            source != "offline"
            or action != "potential_reuse"
            or tier != "none"
            or enforced is not None
        ):
            raise ValueError("Offline analysis cannot claim an enforced cache action")
    elif phase == "requested":
        if source != "client" or action == "potential_reuse" or enforced is not None:
            raise ValueError(
                "Requested actions need client source and unknown enforcement"
            )
    elif (
        source != "engine" or action == "potential_reuse" or type(enforced) is not bool
    ):
        raise ValueError(
            "Observed actions require engine source and physical enforcement state"
        )
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Cache events require nonempty provenance")
    return deepcopy(value)
