# SPDX-License-Identifier: MIT
"""Reference completion and policy identity precede worker protocol selection."""

import copy

import pytest

from tools.kv.decision.production import MODEL, REVISION, validate
from tools.kv.decision.production_prepare import protocol, reference_gate


def reports():
    analysis, evidence = {"models": {}}, {"models": {}}
    for name in ("qwen3", "qwen25"):
        common = {
            "model": MODEL if name == "qwen3" else "Qwen2.5",
            "revision": REVISION if name == "qwen3" else "revision",
            "source": {"commit": "source"},
            "input_sha256": name,
        }
        candidates = {
            arm: {
                "preservation_certificate": arm == "SD",
                "aggregate_math_certificate": arm in ("AF", "SD"),
            }
            for arm in ("AF", "RF", "SF", "AD", "RD", "SD")
        }
        analysis["models"][name] = {**common, "candidates": candidates}
        evidence["models"][name] = {
            **common,
            "retained_tokens_decode_exactly": True,
            "counts": {
                arm: {"nll": 128, "retrieval": 400, "gsm8k": 1319}
                for arm in ("N", "AF", "RF", "SF", "AD", "RD", "SD")
            },
        }
    return analysis, evidence


def test_dynamic_reference_success_remains_distinct_from_fixed_worker_support():
    selection = reference_gate(*reports(), "qwen3")
    assert selection["reference_candidates"]["SD"]["aggregate_math"]
    assert not selection["reference_candidates"]["SF"]["aggregate_math"]
    assert selection["optimized_symmetric_policy"] == "fixed"
    assert "separate engineering gap" in selection["scope"]


@pytest.mark.parametrize(
    "defect", ["model", "missing_items", "identity", "candidates", "tokens"]
)
def test_partial_or_inconsistent_reference_does_not_unlock_preparation(defect):
    analysis, evidence = reports()
    if defect == "model":
        del analysis["models"]["qwen25"]
    elif defect == "missing_items":
        evidence["models"]["qwen25"]["counts"]["SD"]["gsm8k"] = 1318
    elif defect == "identity":
        evidence["models"]["qwen3"]["revision"] = "changed"
    elif defect == "candidates":
        del analysis["models"]["qwen3"]["candidates"]["RF"]
    else:
        evidence["models"]["qwen3"]["retained_tokens_decode_exactly"] = False
    with pytest.raises(ValueError):
        reference_gate(analysis, evidence, "qwen3")


def test_generated_protocol_keeps_full_family_and_rejects_changed_objective():
    scales = {
        "model": MODEL,
        "revision": REVISION,
        "scales": [{"K": 0.1, "V": 0.2} for _ in range(36)],
    }
    data = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": "float16",
        "gsm_parser": "last_numeric_answer_label_v1",
        "samples": {
            "confirmation": [
                {"kind": kind}
                for kind, n in (("nll", 128), ("retrieval", 400), ("gsm8k", 1319))
                for _ in range(n)
            ]
        },
    }
    record = protocol(scales)
    validate(record, data, scales)
    changed = copy.deepcopy(record)
    changed["certificates"]["certificates_per_objective"] = 1
    with pytest.raises(ValueError, match="family"):
        validate(changed, data, scales)
    changed = copy.deepcopy(record)
    changed["serving_qualification_route"] = "preservation"
    with pytest.raises(ValueError, match="preselected"):
        validate(changed, data, scales)
    changed = copy.deepcopy(record)
    changed["kv_allocator_allowance_bytes"] = 0
    changed["engine"]["kv_cache_memory_bytes"] += record["kv_allocator_allowance_bytes"]
    with pytest.raises(ValueError, match="allocation rounding"):
        validate(changed, data, scales)


@pytest.mark.parametrize(
    "bytes_per_pair,near_capacity_requests", [(4, 35), (3, 46), (2, 70)]
)
def test_payload_request_leaves_room_for_unsplit_allocator_tails(
    bytes_per_pair, near_capacity_requests
):
    record = protocol(
        {"model": MODEL, "revision": REVISION, "scales": [{"K": 0.1, "V": 0.2}] * 36}
    )
    per_block = 16 * 8 * 128 * bytes_per_pair
    layers = 36
    blocks = record["engine"]["kv_cache_memory_bytes"] // (layers * per_block)
    # Torch 2.11 native should_split leaves <=1 MiB attached to a large
    # active allocation; scales occupy 512-byte small-pool blocks.
    upper = layers * (blocks * per_block + 1024**2 + 3 * 512)
    assert upper <= record["persistent_budget_bytes"]
    assert blocks >= near_capacity_requests * ((8192 + 32) // 16)
    # A fresh cudaMalloc rounds to 2 MiB. The previous scale-only request
    # left 256 KiB per layer unsplit and exceeded the declared 40 GiB.
    old_blocks = (record["persistent_budget_bytes"] - layers * 3 * 512) // (
        layers * per_block
    )
    old_request = old_blocks * per_block
    old_segment = ((old_request + 2 * 1024**2 - 1) // (2 * 1024**2)) * 2 * 1024**2
    assert 0 < old_segment - old_request <= 1024**2
    assert layers * (old_segment + 3 * 512) > record["persistent_budget_bytes"]
