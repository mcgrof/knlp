#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate deterministic synthetic KMesh All-Gather traces."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KMESH_CONTEXT_SCHEMA_VERSION = 1
GENERATOR_VERSION = "1.2"
DEFAULT_SEED = 20260731
TRACE_FILE = "trace.jsonl"
MANIFEST_FILE = "manifest.json"

CASES = (
    "same_order_different_offsets",
    "different_shared_order",
    "missing_shared_segment",
    "different_private_history_lengths",
    "incompatible_round_ids",
    "mixed_cache_domains",
    "cot_branch_tree",
)
AGENT_COUNTS = (1, 3, 5, 10)
SHARED_FRACTIONS = (0.0, 0.5, 0.8, 0.9, 0.95)
ALIGNMENTS = ("aligned", "unaligned")
QPS_VALUES = (1, 2, 4)
MODES = ("prefill_focus", "round_e2e")

CASE_DESCRIPTIONS = {
    "same_order_different_offsets": (
        "Shared updates keep the same logical order but begin at different "
        "absolute prompt offsets."
    ),
    "different_shared_order": (
        "Every request has the same shared updates, with agent-dependent order."
    ),
    "missing_shared_segment": (
        "The final agent replaces one shared update with private filler tokens."
    ),
    "different_private_history_lengths": (
        "Private histories grow by agent while shared update order stays fixed."
    ),
    "incompatible_round_ids": (
        "Token-identical requests are split across incompatible round IDs."
    ),
    "mixed_cache_domains": (
        "Token-identical requests alternate between two cache domains."
    ),
    "cot_branch_tree": (
        "Sibling CoT branches share an ancestor and carry branch-local thoughts."
    ),
}


def _require_exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise TypeError(
            f"{name} must be exactly {expected.__name__}, not {type(value).__name__}"
        )


@dataclasses.dataclass(frozen=True)
class TraceConfig:
    """Configuration for one deterministic controlled trace."""

    case: str
    agents: int = 5
    shared_fraction: float = 0.8
    alignment: str = "aligned"
    block_size: int = 32
    qps: int = 1
    mode: str = "prefill_focus"
    seed: int = DEFAULT_SEED
    tokens_per_agent: int = 512

    def __post_init__(self) -> None:
        _require_exact_type("case", self.case, str)
        _require_exact_type("agents", self.agents, int)
        _require_exact_type("shared_fraction", self.shared_fraction, float)
        _require_exact_type("alignment", self.alignment, str)
        _require_exact_type("block_size", self.block_size, int)
        _require_exact_type("qps", self.qps, int)
        _require_exact_type("mode", self.mode, str)
        _require_exact_type("seed", self.seed, int)
        _require_exact_type("tokens_per_agent", self.tokens_per_agent, int)
        if self.case not in CASES:
            raise ValueError(f"unknown case: {self.case}")
        if self.agents not in AGENT_COUNTS:
            raise ValueError(f"agents must be one of {AGENT_COUNTS}")
        if self.shared_fraction not in SHARED_FRACTIONS:
            raise ValueError(f"shared_fraction must be one of {SHARED_FRACTIONS}")
        if self.alignment not in ALIGNMENTS:
            raise ValueError(f"alignment must be one of {ALIGNMENTS}")
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")
        if self.qps not in QPS_VALUES:
            raise ValueError(f"qps must be one of {QPS_VALUES}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.tokens_per_agent < self.block_size * 8:
            raise ValueError("tokens_per_agent must cover at least eight blocks")


@dataclasses.dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    kind: str
    share_scope: str
    shareable: bool
    source_id: str
    token_ids: tuple[int, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _content_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_json(list(token_ids)).encode()).hexdigest()


def _derived_rng(seed: int, *parts: object) -> random.Random:
    material = "|".join((str(seed), *(str(part) for part in parts)))
    digest = hashlib.sha256(material.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _make_tokens(seed: int, key: str, length: int) -> tuple[int, ...]:
    rng = _derived_rng(seed, key)
    return tuple(rng.randrange(128, 65536) for _ in range(length))


def _task_length(block_size: int) -> int:
    return max(4, block_size // 4)


def _distribute_blocks(total_blocks: int, segments: int) -> list[int]:
    quotient, remainder = divmod(total_blocks, segments)
    return [quotient + (index < remainder) for index in range(segments)]


def _shared_lengths(config: TraceConfig) -> list[int]:
    if config.shared_fraction == 0.0:
        return []

    num_segments = 4
    private_reserve = 8
    if config.case == "same_order_different_offsets":
        private_reserve = config.agents + 8
    max_shared_tokens = (
        config.tokens_per_agent - _task_length(config.block_size) - private_reserve
    )
    target_tokens = round(config.tokens_per_agent * config.shared_fraction)
    target_blocks = round(target_tokens / config.block_size)
    max_blocks = max_shared_tokens // config.block_size
    target_blocks = max(num_segments, min(target_blocks, max_blocks))
    block_counts = _distribute_blocks(target_blocks, num_segments)
    lengths = [count * config.block_size for count in block_counts]

    if config.alignment == "unaligned":
        shift = max(1, config.block_size // 6)
        residuals = (-shift, -1, 1, shift)
        lengths = [length + residual for length, residual in zip(lengths, residuals)]

    if any(length <= 0 for length in lengths):
        raise ValueError("trace configuration produced an empty shared segment")
    if any(
        (length % config.block_size == 0) != (config.alignment == "aligned")
        for length in lengths
    ):
        raise AssertionError("shared segment alignment invariant failed")
    return lengths


def _shared_specs(config: TraceConfig) -> list[SegmentSpec]:
    return [
        SegmentSpec(
            segment_id=f"shared_update:O_{index + 1}",
            kind="shared_update",
            share_scope="group",
            shareable=True,
            source_id="all_gather_coordinator",
            token_ids=_make_tokens(config.seed, f"shared_update:O_{index + 1}", length),
        )
        for index, length in enumerate(_shared_lengths(config))
    ]


def _private_spec(
    config: TraceConfig,
    agent_id: str,
    suffix: str,
    length: int,
    *,
    kind: str = "private_history",
    content_key: str | None = None,
) -> SegmentSpec:
    identity_key = f"{agent_id}:{suffix}"
    return SegmentSpec(
        segment_id=f"{kind}:{identity_key}",
        kind=kind,
        share_scope="request" if kind != "thought" else "branch",
        shareable=False,
        source_id=agent_id,
        token_ids=_make_tokens(config.seed, content_key or identity_key, length),
    )


def _task_spec(config: TraceConfig) -> SegmentSpec:
    return SegmentSpec(
        segment_id="task:Q_1",
        kind="task",
        share_scope="group",
        shareable=False,
        source_id="all_gather_coordinator",
        token_ids=_make_tokens(
            config.seed, "task:Q_1", _task_length(config.block_size)
        ),
    )


def _agent_round(config: TraceConfig, agent_index: int) -> str:
    if config.case != "incompatible_round_ids" or config.agents == 1:
        return "round-000"
    split = (config.agents + 1) // 2
    return "round-000" if agent_index < split else "round-001"


def _cache_domain(config: TraceConfig, agent_index: int) -> str:
    if config.case == "mixed_cache_domains" and agent_index % 2:
        return "synthetic-cache-domain:b"
    return "synthetic-cache-domain:a"


def _branch_lineage(
    config: TraceConfig, agent_index: int
) -> tuple[str, str | None, list[str]]:
    if config.case != "cot_branch_tree":
        return "root", None, ["root"]
    branch_id = f"branch-{agent_index:03d}"
    return branch_id, "root", ["root", branch_id]


def _ordered_shared_specs(
    config: TraceConfig,
    agent_index: int,
    shared_specs: Sequence[SegmentSpec],
) -> list[SegmentSpec]:
    ordered = list(shared_specs)
    if config.case == "different_shared_order" and ordered:
        offset = agent_index % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]
    if (
        config.case == "missing_shared_segment"
        and agent_index == config.agents - 1
        and ordered
    ):
        missing = ordered[-1]
        ordered[-1] = _private_spec(
            config,
            f"agent-{agent_index:03d}",
            "missing_shared_filler",
            len(missing.token_ids),
        )
    return ordered


def _agent_segment_specs(
    config: TraceConfig,
    agent_index: int,
    shared_specs: Sequence[SegmentSpec],
) -> list[SegmentSpec]:
    agent_id = f"agent-{agent_index:03d}"
    task_spec = _task_spec(config)
    shared_tokens = sum(len(segment.token_ids) for segment in shared_specs)
    private_budget = config.tokens_per_agent - shared_tokens - len(task_spec.token_ids)
    if private_budget <= 0:
        raise ValueError("trace configuration leaves no private-history tokens")

    ordered_shared = _ordered_shared_specs(config, agent_index, shared_specs)
    if config.case == "same_order_different_offsets":
        prefix_length = max(1, private_budget // 4)
        padding_length = agent_index + 1
        tail_length = private_budget - prefix_length - padding_length
        if tail_length <= 0:
            raise ValueError("not enough private tokens for offset control")
        return [
            _private_spec(config, agent_id, "history_prefix", prefix_length),
            _private_spec(config, agent_id, "offset_padding", padding_length),
            *ordered_shared,
            _private_spec(config, agent_id, "history_tail", tail_length),
            task_spec,
        ]

    history_length = private_budget
    if config.case == "different_private_history_lengths":
        history_length += agent_index * max(1, config.block_size // 4)

    content_key = None
    if config.case in {"incompatible_round_ids", "mixed_cache_domains"}:
        content_key = "compatibility_control:history"
    specs = [
        _private_spec(
            config,
            agent_id,
            "history",
            history_length,
            content_key=content_key,
        ),
        *ordered_shared,
    ]
    if config.case == "cot_branch_tree":
        specs.append(
            _private_spec(
                config,
                agent_id,
                "branch_thought",
                max(4, config.block_size // 2),
                kind="thought",
            )
        )
    specs.append(task_spec)
    return specs


def _block_geometry(start: int, end: int, block_size: int) -> dict[str, int]:
    first_full_block = ((start + block_size - 1) // block_size) * block_size
    prompt_full_blocks = max(0, (end - first_full_block) // block_size)
    length = end - start
    return {
        "block_size": block_size,
        "start_offset": start % block_size,
        "end_offset": end % block_size,
        "segment_relative_full_blocks": length // block_size,
        "prompt_full_blocks": prompt_full_blocks,
        "residual_tokens": length % block_size,
    }


def _materialize_segments(
    specs: Sequence[SegmentSpec], round_id: str, branch_id: str, block_size: int
) -> tuple[list[int], list[dict[str, Any]]]:
    prompt_token_ids: list[int] = []
    segments: list[dict[str, Any]] = []
    for logical_order, spec in enumerate(specs):
        start = len(prompt_token_ids)
        prompt_token_ids.extend(spec.token_ids)
        end = len(prompt_token_ids)
        source_branch_id = "root" if spec.share_scope == "group" else branch_id
        segments.append(
            {
                "segment_id": spec.segment_id,
                "kind": spec.kind,
                "token_start": start,
                "token_end": end,
                "length": end - start,
                "logical_order": logical_order,
                "source_id": spec.source_id,
                "share_scope": spec.share_scope,
                "shareable": spec.shareable,
                "reuse_eligible": spec.shareable,
                "lineage_id": f"{round_id}:{source_branch_id}:{spec.segment_id}",
                "source_lineage": {
                    "round_id": round_id,
                    "branch_id": source_branch_id,
                    "source_id": spec.source_id,
                },
                "content_sha256": _content_sha256(spec.token_ids),
                "block_geometry": _block_geometry(start, end, block_size),
            }
        )
    return prompt_token_ids, segments


def _wire_segment(segment: dict[str, Any]) -> dict[str, Any]:
    """Select only fields accepted by strict sideband metadata parsing."""
    return {
        "segment_id": segment["segment_id"],
        "kind": segment["kind"],
        "token_start": segment["token_start"],
        "token_end": segment["token_end"],
        "logical_order": segment["logical_order"],
        "source_id": segment["source_id"],
        "shareable": segment["shareable"],
        "lineage_id": segment["lineage_id"],
    }


def _kmesh_context(
    trace_id: str,
    group_kind: str,
    round_id: str,
    branch_id: str,
    parent_branch_id: str | None,
    cache_domain: str,
    segments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": KMESH_CONTEXT_SCHEMA_VERSION,
        "enabled": True,
        "execution_id": f"execution:{trace_id}",
        "group_kind": group_kind,
        "group_id": f"group:{trace_id}",
        "round_id": round_id,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "state_kind": "kv_cache",
        "cache_domain": cache_domain,
        "segment_encoding": "sideband",
        "segments": [_wire_segment(segment) for segment in segments],
    }


def _compatibility_key(
    config: TraceConfig,
    trace_id: str,
    round_id: str,
    cache_domain: str,
    prompt_length: int,
) -> dict[str, Any]:
    return {
        "execution_id": f"execution:{trace_id}",
        "group_id": f"group:{trace_id}",
        "round_id": round_id,
        "model_id": "synthetic:qwen2.5",
        "model_revision": "trace-generator-v1",
        "tokenizer_id_or_revision": "synthetic-token-ids-v1",
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "kv_layout": "paged-kv",
        "attention_kind": "full_attention",
        "rope_configuration": "qwen2.5-default",
        "attention_backend": "reference",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "lora_adapter_identity": None,
        "cache_salt": None,
        "cache_domain": cache_domain,
        "active_prompt_length": prompt_length,
        "cached_span": [0, 0],
        "prefill_chunk_span": [0, prompt_length],
    }


def _trace_id(config: TraceConfig) -> str:
    digest = hashlib.sha256(_canonical_json(dataclasses.asdict(config)).encode())
    return f"kmesh-{digest.hexdigest()[:16]}"


def _build_record(
    config: TraceConfig,
    trace_id: str,
    agent_index: int,
    shared_specs: Sequence[SegmentSpec],
) -> dict[str, Any]:
    agent_id = f"agent-{agent_index:03d}"
    round_id = _agent_round(config, agent_index)
    cache_domain = _cache_domain(config, agent_index)
    branch_id, parent_branch_id, branch_path = _branch_lineage(config, agent_index)
    specs = _agent_segment_specs(config, agent_index, shared_specs)
    prompt_token_ids, segments = _materialize_segments(
        specs, round_id, branch_id, config.block_size
    )
    request_id = f"{trace_id}:{agent_id}"
    group_kind = "cot_branch" if config.case == "cot_branch_tree" else "agent_round"
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "request_id": request_id,
        "case": config.case,
        "seed": config.seed,
        "sequence_index": agent_index,
        "agent_id": agent_id,
        "lineage": {
            "execution_id": f"execution:{trace_id}",
            "group_kind": group_kind,
            "group_id": f"group:{trace_id}",
            "round_id": round_id,
            "agent_id": agent_id,
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "branch_path": branch_path,
            "state_kind": "kv_cache",
            "segment_encoding": "sideband",
        },
        "kmesh_context": _kmesh_context(
            trace_id,
            group_kind,
            round_id,
            branch_id,
            parent_branch_id,
            cache_domain,
            segments,
        ),
        "compatibility_key": _compatibility_key(
            config, trace_id, round_id, cache_domain, len(prompt_token_ids)
        ),
        "workload": {
            "mode": config.mode,
            "qps": config.qps,
            "max_new_tokens": 1 if config.mode == "prefill_focus" else 32,
        },
        "private_history_ids": [
            segment["segment_id"]
            for segment in segments
            if segment["kind"] == "private_history"
        ],
        "shared_update_ids": [
            segment["segment_id"]
            for segment in segments
            if segment["kind"] == "shared_update"
        ],
        "task_query": {
            "segment_id": "task:Q_1",
            "label": "synthetic all-gather task query",
        },
        "prompt_token_ids": prompt_token_ids,
        "segments": segments,
    }


def _assert_compatibility_control_isolation(
    config: TraceConfig, records: Sequence[dict[str, Any]]
) -> None:
    if config.case not in {"incompatible_round_ids", "mixed_cache_domains"}:
        return
    first_prompt = records[0]["prompt_token_ids"]
    if any(record["prompt_token_ids"] != first_prompt for record in records[1:]):
        raise AssertionError(f"{config.case} must keep all prompt token IDs identical")
    controlled_field = (
        "round_id" if config.case == "incompatible_round_ids" else "cache_domain"
    )
    baseline = {
        key: value
        for key, value in records[0]["compatibility_key"].items()
        if key != controlled_field
    }
    for record in records[1:]:
        comparison = {
            key: value
            for key, value in record["compatibility_key"].items()
            if key != controlled_field
        }
        if comparison != baseline:
            raise AssertionError(
                f"{config.case} changed compatibility fields other than "
                f"{controlled_field}"
            )


def _validate_record(record: dict[str, Any], block_size: int) -> None:
    prompt = record["prompt_token_ids"]
    cursor = 0
    seen_ids: set[str] = set()
    for logical_order, segment in enumerate(record["segments"]):
        start = segment["token_start"]
        end = segment["token_end"]
        if start != cursor or end <= start or end > len(prompt):
            raise ValueError(f"invalid segment span in {record['request_id']}")
        if segment["logical_order"] != logical_order:
            raise ValueError(f"invalid segment order in {record['request_id']}")
        if segment["segment_id"] in seen_ids:
            raise ValueError(f"duplicate segment id in {record['request_id']}")
        seen_ids.add(segment["segment_id"])
        tokens = prompt[start:end]
        if segment["length"] != len(tokens):
            raise ValueError(f"invalid segment length in {record['request_id']}")
        if segment["content_sha256"] != _content_sha256(tokens):
            raise ValueError(f"invalid segment digest in {record['request_id']}")
        expected_geometry = _block_geometry(start, end, block_size)
        if segment["block_geometry"] != expected_geometry:
            raise ValueError(f"invalid block geometry in {record['request_id']}")
        cursor = end
    if cursor != len(prompt):
        raise ValueError(f"segments do not cover prompt in {record['request_id']}")
    expected_wire_segments = [_wire_segment(segment) for segment in record["segments"]]
    if record["kmesh_context"]["segments"] != expected_wire_segments:
        raise ValueError(f"wire segments diverge in {record['request_id']}")


def _eligible_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [segment for segment in record["segments"] if segment["reuse_eligible"]]


def _segment_tokens(record: dict[str, Any], segment: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        record["prompt_token_ids"][segment["token_start"] : segment["token_end"]]
    )


def _segment_map(
    record: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], tuple[int, ...]]]:
    return {
        segment["segment_id"]: (segment, _segment_tokens(record, segment))
        for segment in _eligible_segments(record)
    }


def _prompt_aligned_blocks(
    record: dict[str, Any], segment: dict[str, Any], block_size: int
) -> set[tuple[int, tuple[int, ...]]]:
    start = segment["token_start"]
    end = segment["token_end"]
    first = ((start + block_size - 1) // block_size) * block_size
    prompt = record["prompt_token_ids"]
    return {
        (position - start, tuple(prompt[position : position + block_size]))
        for position in range(first, end - block_size + 1, block_size)
    }


def _matching_segments(
    left: dict[str, Any], right: dict[str, Any]
) -> list[tuple[str, dict[str, Any], dict[str, Any], tuple[int, ...]]]:
    left_map = _segment_map(left)
    right_map = _segment_map(right)
    matching = []
    for segment_id in sorted(left_map.keys() & right_map.keys()):
        left_segment, left_tokens = left_map[segment_id]
        right_segment, right_tokens = right_map[segment_id]
        if left_tokens == right_tokens:
            matching.append((segment_id, left_segment, right_segment, left_tokens))
    return matching


def _ordered_shared_prefix_tokens(left: dict[str, Any], right: dict[str, Any]) -> int:
    total = 0
    for left_segment, right_segment in zip(
        _eligible_segments(left), _eligible_segments(right)
    ):
        if left_segment["segment_id"] != right_segment["segment_id"]:
            break
        left_tokens = _segment_tokens(left, left_segment)
        right_tokens = _segment_tokens(right, right_segment)
        if left_tokens != right_tokens:
            break
        total += len(left_tokens)
    return total


def _pairwise_overlap(
    left: dict[str, Any], right: dict[str, Any], block_size: int
) -> dict[str, Any]:
    matching = _matching_segments(left, right)
    raw_tokens = sum(len(tokens) for _, _, _, tokens in matching)
    relative_blocks = sum(len(tokens) // block_size for *_, tokens in matching)
    residual_tokens = sum(len(tokens) % block_size for *_, tokens in matching)
    absolute_offset_tokens = sum(
        len(tokens)
        for _, left_segment, right_segment, tokens in matching
        if left_segment["token_start"] == right_segment["token_start"]
    )
    prompt_aligned_blocks = sum(
        len(
            _prompt_aligned_blocks(left, left_segment, block_size)
            & _prompt_aligned_blocks(right, right_segment, block_size)
        )
        for _, left_segment, right_segment, _ in matching
    )
    compatible = left["compatibility_key"] == right["compatibility_key"]
    eligible_tokens = raw_tokens if compatible else 0
    denominator = min(len(left["prompt_token_ids"]), len(right["prompt_token_ids"]))
    return {
        "left_request_id": left["request_id"],
        "right_request_id": right["request_id"],
        "compatible": compatible,
        "shared_segment_ids": [segment_id for segment_id, *_ in matching],
        "raw_structural_shared_tokens": raw_tokens,
        "eligible_structural_shared_tokens": eligible_tokens,
        "raw_structural_shared_fraction": raw_tokens / denominator,
        "eligible_structural_shared_fraction": eligible_tokens / denominator,
        "ordered_shared_prefix_tokens": _ordered_shared_prefix_tokens(left, right),
        "raw_segment_relative_full_blocks": relative_blocks,
        "eligible_segment_relative_full_blocks": (relative_blocks if compatible else 0),
        "raw_prompt_aligned_full_blocks": prompt_aligned_blocks,
        "eligible_prompt_aligned_full_blocks": (
            prompt_aligned_blocks if compatible else 0
        ),
        "raw_shared_residual_tokens": residual_tokens,
        "eligible_shared_residual_tokens": residual_tokens if compatible else 0,
        "same_absolute_offset_shared_tokens": absolute_offset_tokens,
    }


def _compatibility_groups(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: dict[str, dict[str, Any]] = {}
    for record in records:
        canonical_key = _canonical_json(record["compatibility_key"])
        grouped[canonical_key].append(record)
        keys[canonical_key] = record["compatibility_key"]
    groups = []
    for canonical_key in sorted(grouped):
        request_ids = sorted(record["request_id"] for record in grouped[canonical_key])
        group_hash = hashlib.sha256(canonical_key.encode()).hexdigest()[:16]
        groups.append(
            {
                "compatibility_group_id": f"compat:{group_hash}",
                "compatibility_key": keys[canonical_key],
                "request_ids": request_ids,
            }
        )
    return groups


def compute_structural_overlap(
    records: Sequence[dict[str, Any]], block_size: int
) -> dict[str, Any]:
    """Compute a segment-structural oracle from emitted records."""
    for record in records:
        _validate_record(record, block_size)
    pairwise = [
        _pairwise_overlap(records[left], records[right], block_size)
        for left in range(len(records))
        for right in range(left + 1, len(records))
    ]
    aggregate_fields = (
        "raw_structural_shared_tokens",
        "eligible_structural_shared_tokens",
        "ordered_shared_prefix_tokens",
        "raw_segment_relative_full_blocks",
        "eligible_segment_relative_full_blocks",
        "raw_prompt_aligned_full_blocks",
        "eligible_prompt_aligned_full_blocks",
        "raw_shared_residual_tokens",
        "eligible_shared_residual_tokens",
        "same_absolute_offset_shared_tokens",
    )
    aggregate = {
        "pair_count": len(pairwise),
        "compatible_pair_count": sum(pair["compatible"] for pair in pairwise),
    }
    aggregate.update(
        {
            f"{field}_total": sum(pair[field] for pair in pairwise)
            for field in aggregate_fields
        }
    )
    return {
        "method": "segment_identity_and_prompt_slices_v1",
        "definition": {
            "raw": (
                "Same segment_id and exact emitted token slice; independent of KV "
                "values and absolute prompt position."
            ),
            "eligible": (
                "Raw structural overlap gated by exact compatibility-key equality."
            ),
            "segment_relative_full_blocks": (
                "Full block-sized chunks anchored at each matching segment start."
            ),
            "prompt_aligned_full_blocks": (
                "Matching full blocks wholly inside both absolute prompt spans."
            ),
        },
        "compatibility_groups": _compatibility_groups(records),
        "pairwise": pairwise,
        "aggregate": aggregate,
    }


def _observer_expectation(
    records: Sequence[dict[str, Any]], block_size: int
) -> dict[str, Any]:
    """Describe what the current bounded observer is expected to admit."""
    supported_records = []
    required_fallback_counts: dict[str, int] = defaultdict(int)
    for record in records:
        context = record["kmesh_context"]
        if context["group_kind"] != "agent_round":
            required_fallback_counts["group_kind_unsupported"] += 1
        elif context["state_kind"] != "kv_cache":
            required_fallback_counts["state_kind_unsupported"] += 1
        else:
            supported_records.append(record)

    supported_oracle = compute_structural_overlap(supported_records, block_size)
    supported_aggregate = supported_oracle["aggregate"]
    supported_overlap = (
        supported_aggregate["compatible_pair_count"] > 0
        and supported_aggregate["eligible_structural_shared_tokens_total"] > 0
    )
    return {
        "contract": "kmesh_observer_supported_scope_v1",
        "supported_group_kinds": ["agent_round"],
        "supported_state_kinds": ["kv_cache"],
        "supported_request_count": len(supported_records),
        "required_fallback_counts": dict(sorted(required_fallback_counts.items())),
        "supported_compatible_pair_count": supported_aggregate["compatible_pair_count"],
        "supported_eligible_structural_shared_tokens_total": supported_aggregate[
            "eligible_structural_shared_tokens_total"
        ],
        "batch_group_presence_expected": supported_overlap,
        "serial_paced_group_presence_expected": False,
    }


def _case_expectations(
    config: TraceConfig,
    records: Sequence[dict[str, Any]],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    shared_sequences = [tuple(record["shared_update_ids"]) for record in records]
    shared_id_sets = [frozenset(sequence) for sequence in shared_sequences]
    starts_by_id: dict[str, set[int]] = defaultdict(set)
    for record in records:
        for segment in record["segments"]:
            if segment["kind"] == "shared_update":
                starts_by_id[segment["segment_id"]].add(segment["token_start"])
    private_lengths = [
        sum(
            segment["length"]
            for segment in record["segments"]
            if segment["kind"] == "private_history"
        )
        for record in records
    ]
    checks: dict[str, bool]
    if config.case == "same_order_different_offsets":
        checks = {
            "shared_order_is_identical": len(set(shared_sequences)) == 1,
            "shared_offsets_differ": any(
                len(starts) > 1 for starts in starts_by_id.values()
            ),
            "total_prompt_lengths_match": len(
                {len(record["prompt_token_ids"]) for record in records}
            )
            == 1,
        }
    elif config.case == "different_shared_order":
        checks = {
            "shared_orders_differ": len(set(shared_sequences)) > 1,
            "shared_segment_sets_match": len(set(shared_id_sets)) == 1,
        }
    elif config.case == "missing_shared_segment":
        checks = {
            "shared_segment_sets_differ": len(set(shared_id_sets)) > 1,
            "total_prompt_lengths_match": len(
                {len(record["prompt_token_ids"]) for record in records}
            )
            == 1,
        }
    elif config.case == "different_private_history_lengths":
        checks = {
            "private_history_lengths_differ": len(set(private_lengths)) > 1,
            "shared_order_is_identical": len(set(shared_sequences)) == 1,
        }
    elif config.case == "incompatible_round_ids":
        compatibility_differences = {
            key
            for key in records[0]["compatibility_key"]
            if len(
                {
                    _canonical_json(record["compatibility_key"][key])
                    for record in records
                }
            )
            > 1
        }
        checks = {
            "round_ids_differ": len(
                {record["lineage"]["round_id"] for record in records}
            )
            > 1,
            "shared_order_is_identical": len(set(shared_sequences)) == 1,
            "prompts_are_token_identical": len(
                {_content_sha256(record["prompt_token_ids"]) for record in records}
            )
            == 1,
            "only_round_id_compatibility_field_differs": (
                compatibility_differences == {"round_id"}
            ),
        }
    elif config.case == "mixed_cache_domains":
        compatibility_differences = {
            key
            for key in records[0]["compatibility_key"]
            if len(
                {
                    _canonical_json(record["compatibility_key"][key])
                    for record in records
                }
            )
            > 1
        }
        checks = {
            "cache_domains_differ": len(
                {record["compatibility_key"]["cache_domain"] for record in records}
            )
            > 1,
            "shared_order_is_identical": len(set(shared_sequences)) == 1,
            "prompts_are_token_identical": len(
                {_content_sha256(record["prompt_token_ids"]) for record in records}
            )
            == 1,
            "only_cache_domain_compatibility_field_differs": (
                compatibility_differences == {"cache_domain"}
            ),
        }
    else:
        checks = {
            "group_kind_is_cot_branch": all(
                record["lineage"]["group_kind"] == "cot_branch" for record in records
            ),
            "sibling_branches_differ": len(
                {record["lineage"]["branch_id"] for record in records}
            )
            > 1,
            "branches_have_root_parent": all(
                record["lineage"]["parent_branch_id"] == "root" for record in records
            ),
        }
    metadata_exercised = all(checks.values())
    aggregate = oracle["aggregate"]
    overlap_gate_exercised = (
        aggregate["pair_count"] > 0
        and aggregate["raw_structural_shared_tokens_total"] > 0
    )
    reasons = []
    if not metadata_exercised:
        reasons.append("case-specific metadata control is degenerate")
    if aggregate["pair_count"] == 0:
        reasons.append("overlap gate requires at least two requests")
    elif aggregate["raw_structural_shared_tokens_total"] == 0:
        reasons.append("overlap gate requires structurally shared tokens")
    return {
        "description": CASE_DESCRIPTIONS[config.case],
        "checks": checks,
        "metadata_exercised": metadata_exercised,
        "overlap_gate_exercised": overlap_gate_exercised,
        "fully_exercised": metadata_exercised and overlap_gate_exercised,
        "degenerate_reason": "; ".join(reasons) if reasons else None,
    }


def _branch_tree(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    request_ids_by_branch: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, str | None] = {}
    for record in records:
        lineage = record["lineage"]
        branch_id = lineage["branch_id"]
        request_ids_by_branch[branch_id].append(record["request_id"])
        parents[branch_id] = lineage["parent_branch_id"]
        if lineage["parent_branch_id"] is not None:
            parents.setdefault(lineage["parent_branch_id"], None)
    return [
        {
            "branch_id": branch_id,
            "parent_branch_id": parents[branch_id],
            "request_ids": sorted(request_ids_by_branch[branch_id]),
        }
        for branch_id in sorted(parents)
    ]


def _serialize_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(f"{_canonical_json(record)}\n" for record in records)


def _mean(values: Sequence[int | float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fraction_bucket(value: float) -> str:
    bucket = min(9, max(0, int(value * 10)))
    closing = "]" if bucket == 9 else ")"
    return f"[{bucket / 10:.1f},{(bucket + 1) / 10:.1f}{closing}"


def generate_trace(
    config: TraceConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate records and their deterministic manifest."""
    trace_id = _trace_id(config)
    shared_specs = _shared_specs(config)
    records = [
        _build_record(config, trace_id, agent_index, shared_specs)
        for agent_index in range(config.agents)
    ]
    _assert_compatibility_control_isolation(config, records)
    oracle = compute_structural_overlap(records, config.block_size)
    trace_text = _serialize_jsonl(records)
    prompt_lengths = [len(record["prompt_token_ids"]) for record in records]
    shareable_tokens = [
        sum(
            segment["length"]
            for segment in record["segments"]
            if segment["reuse_eligible"]
        )
        for record in records
    ]
    shareable_fractions = [
        tokens / prompt_length
        for tokens, prompt_length in zip(shareable_tokens, prompt_lengths)
    ]
    shareable_fraction_by_request = {
        record["request_id"]: fraction
        for record, fraction in zip(records, shareable_fractions)
    }
    shareable_fraction_bucket_counts: dict[str, int] = defaultdict(int)
    for fraction in shareable_fractions:
        shareable_fraction_bucket_counts[_fraction_bucket(fraction)] += 1
    manifest_config = dataclasses.asdict(config)
    manifest_config["nominal_shared_fraction"] = manifest_config.pop("shared_fraction")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "benchmarks/kmesh/generate_all_gather_trace.py",
        "generator_version": GENERATOR_VERSION,
        "trace_file": TRACE_FILE,
        "trace_sha256": hashlib.sha256(trace_text.encode()).hexdigest(),
        "config": manifest_config,
        "case_expectations": _case_expectations(config, records, oracle),
        "counts": {
            "requests": len(records),
            "records": len(records),
            "segments": sum(len(record["segments"]) for record in records),
            "prompt_tokens": sum(prompt_lengths),
            "shareable_segment_instances": sum(
                len(_eligible_segments(record)) for record in records
            ),
            "unique_shared_segment_ids": len(
                {
                    segment["segment_id"]
                    for record in records
                    for segment in _eligible_segments(record)
                }
            ),
        },
        "observed": {
            "prompt_tokens_min": min(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
            "prompt_tokens_mean": _mean(prompt_lengths),
            "shareable_fraction_min": min(shareable_fractions),
            "shareable_fraction_max": max(shareable_fractions),
            "shareable_fraction_mean": _mean(shareable_fractions),
            "shareable_fraction_by_request": shareable_fraction_by_request,
            "shareable_fraction_bucket_counts": dict(
                sorted(shareable_fraction_bucket_counts.items())
            ),
            "round_ids": sorted({record["lineage"]["round_id"] for record in records}),
            "cache_domains": sorted(
                {record["compatibility_key"]["cache_domain"] for record in records}
            ),
            "branch_ids": sorted(
                {record["lineage"]["branch_id"] for record in records}
            ),
        },
        "branch_tree": _branch_tree(records),
        "oracle": oracle,
        "observer_expectation": _observer_expectation(records, config.block_size),
    }
    return records, manifest


def write_trace(config: TraceConfig, output_dir: Path) -> tuple[Path, Path]:
    """Write one trace and manifest, replacing only their fixed file names."""
    records, manifest = generate_trace(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / TRACE_FILE
    manifest_path = output_dir / MANIFEST_FILE
    trace_path.write_text(_serialize_jsonl(records), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trace_path, manifest_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--agents", type=int, choices=AGENT_COUNTS, default=5)
    parser.add_argument(
        "--shared-fraction",
        type=float,
        choices=SHARED_FRACTIONS,
        default=0.8,
    )
    parser.add_argument("--alignment", choices=ALIGNMENTS, default="aligned")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--qps", type=int, choices=QPS_VALUES, default=1)
    parser.add_argument("--mode", choices=MODES, default="prefill_focus")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tokens-per-agent", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = TraceConfig(
        case=args.case,
        agents=args.agents,
        shared_fraction=args.shared_fraction,
        alignment=args.alignment,
        block_size=args.block_size,
        qps=args.qps,
        mode=args.mode,
        seed=args.seed,
        tokens_per_agent=args.tokens_per_agent,
    )
    trace_path, manifest_path = write_trace(config, args.output_dir)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "trace": str(trace_path),
                "trace_id": _trace_id(config),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
