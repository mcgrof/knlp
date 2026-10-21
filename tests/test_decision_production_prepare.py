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
