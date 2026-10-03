# SPDX-License-Identifier: MIT
"""Pinned SGLang capabilities used by the PrefixPlan experiment adapter.

This module records what was found in a specific source revision.  A source
capability is not proof that a running server enabled or exercised it; the
receipt keeps those states separate and live responses remain the authority.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable

SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang"
SGLANG_REVISION = "cbe070b6bfce18936620f406b9b0d71bdc17452b"
CAPABILITY_SCHEMA = "knlp.agent_baselines.sglang_capabilities.v1"

# Evidence paths are relative to SGLang's repository at SGLANG_REVISION.
SOURCE_CAPABILITIES = {
    "openai_chat": {
        "available": True,
        "evidence": "python/sglang/srt/entrypoints/openai/protocol.py:858",
    },
    "sglext_token_ids": {
        "available": True,
        "evidence": "python/sglang/srt/entrypoints/openai/serving_chat.py:2189",
    },
    "cache_source_report": {
        "available": True,
        "evidence": "python/sglang/srt/entrypoints/openai/utils.py:295",
    },
    "input_ids_bypass": {
        "available": True,
        "evidence": "python/sglang/srt/entrypoints/openai/protocol.py:963",
    },
    "frontend_ir": {
        "available": True,
        "evidence": "python/sglang/lang/ir.py",
    },
    "fork_join": {
        "available": True,
        "evidence": "python/sglang/lang/interpreter.py",
    },
    "hicache_host": {
        "available": True,
        "evidence": "python/sglang/srt/server_args.py",
    },
    "prompt_reordering_compiler": {
        "available": False,
        "evidence": "not found in the audited revision",
    },
    "explicit_request_residency": {
        "available": False,
        "evidence": "no per-request retain/release/prefetch API found",
    },
    "workflow_scheduler_hints": {
        "available": False,
        "evidence": "no dependency/readiness scheduling API found",
    },
}

_REQUIRED_LIVE_FEATURES = {"openai_chat", "sglext_token_ids"}
_CACHE_SCOPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def capability_receipt(enabled: Iterable[str]) -> dict:
    """Build an immutable source receipt plus operator-declared runtime state."""
    if isinstance(enabled, (str, bytes)):
        raise ValueError("SGLang enabled capabilities must be an iterable of names")
    enabled_values = list(enabled)
    if any(not isinstance(name, str) for name in enabled_values):
        raise ValueError("SGLang capability names must be strings")
    if len(set(enabled_values)) != len(enabled_values):
        raise ValueError("SGLang enabled capabilities must be unique")
    enabled = set(enabled_values)
    unknown = enabled - SOURCE_CAPABILITIES.keys()
    if unknown:
        raise ValueError(f"Unknown SGLang capabilities: {sorted(unknown)}")
    unavailable = {
        name for name in enabled if not SOURCE_CAPABILITIES[name]["available"]
    }
    if unavailable:
        raise ValueError(f"Unavailable SGLang capabilities: {sorted(unavailable)}")
    features = {
        name: {**details, "enabled": name in enabled}
        for name, details in SOURCE_CAPABILITIES.items()
    }
    return {
        "schema": CAPABILITY_SCHEMA,
        "implementation": "sglang",
        "repository": SGLANG_REPOSITORY,
        "source_revision": SGLANG_REVISION,
        "features": features,
    }


def validate_capability_receipt(value: Any, *, require_live: bool = False) -> dict:
    """Reject drift, unsupported claims, and incomplete live token capture."""
    if not isinstance(value, dict) or value.get("schema") != CAPABILITY_SCHEMA:
        raise ValueError(f"Expected capability schema {CAPABILITY_SCHEMA}")
    for key, expected in (
        ("implementation", "sglang"),
        ("repository", SGLANG_REPOSITORY),
        ("source_revision", SGLANG_REVISION),
    ):
        if value.get(key) != expected:
            raise ValueError(f"SGLang capability receipt has unexpected {key}")
    features = value.get("features")
    if not isinstance(features, dict) or set(features) != set(SOURCE_CAPABILITIES):
        raise ValueError("SGLang capability receipt feature set has drifted")
    for name, expected in SOURCE_CAPABILITIES.items():
        feature = features[name]
        if not isinstance(feature, dict):
            raise ValueError(f"Invalid SGLang capability entry: {name}")
        if feature.get("available") is not expected["available"]:
            raise ValueError(f"SGLang source availability mismatch: {name}")
        if feature.get("evidence") != expected["evidence"]:
            raise ValueError(f"SGLang source evidence mismatch: {name}")
        if type(feature.get("enabled")) is not bool:
            raise ValueError(f"SGLang runtime state must be boolean: {name}")
        if feature["enabled"] and not feature["available"]:
            raise ValueError(f"Unavailable SGLang capability claimed enabled: {name}")
    if require_live:
        missing = {
            name for name in _REQUIRED_LIVE_FEATURES if not features[name]["enabled"]
        }
        if missing:
            raise ValueError(
                f"SGLang live capture requires enabled capabilities: {sorted(missing)}"
            )
    return deepcopy(value)


def validate_cache_scope(value: Any) -> str:
    """Validate an operator-selected cache namespace shared by trusted tasks."""
    if not isinstance(value, str) or not _CACHE_SCOPE_PATTERN.fullmatch(value):
        raise ValueError(
            "SGLang cache_scope must be a 1-128 character namespace identifier"
        )
    return value


def request_extensions(receipt: Any, *, request_id: str, cache_scope: str) -> dict:
    """Return only source-verified OpenAI request extensions for a live call."""
    checked = validate_capability_receipt(receipt, require_live=True)
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("SGLang requests require a request identity")
    cache_scope = validate_cache_scope(cache_scope)
    features = checked["features"]
    result = {
        "rid": request_id,
        "extra_key": cache_scope,
        "return_input_ids_in_sglext": True,
        "return_output_ids_in_sglext": True,
    }
    if features["cache_source_report"]["enabled"]:
        result["return_cached_tokens_details"] = True
    return result
