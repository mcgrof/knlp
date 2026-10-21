# SPDX-License-Identifier: MIT
"""Publication keeps complete endpoint families and excludes nonnumeric payloads."""

import copy
import json

import pytest

from tools.kv.decision.figures import ARMS, ENDPOINTS, MODELS, aggregate, sha


def binary(n):
    return {
        "n": n,
        "native_correct": n - 10,
        "candidate_correct": n - 9,
        "losses": 2,
        "gains": 3,
        "net_delta": 1 / n,
        "net_interval_95": [-0.004, 0.01],
        "net_one_sided_bounds": [-0.005, 0.02],
        "gross_loss_one_sided_bounds": [0.001, 0.03],
        "certificate_alpha": 0.025 / 12,
        "raw_request": "PRIVATE_SENTINEL",
    }


def reports():
    nll = {
        "n_documents": 64,
        "native_mean": 1.0,
        "candidate_mean": 1.001,
        "absolute_delta": 0.001,
        "relative_delta": 0.001,
        "absolute_one_sided_bounds": [-0.001, 0.003],
        "relative_one_sided_bounds": [-0.001, 0.003],
        "alpha": 0.025 / 12,
        "resamples": 100000,
        "seed": 20261020,
        "quantile": "linear",
        "document_text": "PRIVATE_SENTINEL",
    }
    endpoints = {
        name: copy.deepcopy(
            nll if name.startswith("nll-") else binary(1319 if name == "gsm8k" else 200)
        )
        for name in ENDPOINTS
    }
    analysis, evidence = {"alpha": 0.025 / 12, "models": {}}, {"models": {}}
    for name, (model, _) in MODELS.items():
        common = {
            "model": model,
            "revision": "a" * 40,
            "source": {"commit": "b" * 40, "operator_log": "PRIVATE_SENTINEL"},
            "input_sha256": "c" * 64,
        }
        analysis["models"][name] = {
            **copy.deepcopy(common),
            "candidates": {
                arm: {
                    **copy.deepcopy(endpoints),
                    "preservation_certificate": False,
                    "aggregate_math_certificate": True,
                }
                for arm in ARMS[1:]
            },
            "direct": [
                {
                    "baseline": control + policy,
                    "candidate": "A" + policy,
                    "endpoint": key,
                    "result": copy.deepcopy(value),
                    "p_two_sided": 0.5,
                    "holm8_p": 1.0,
                    "paired_interval_95": [-0.01, 0.01],
                    "bonferroni8_nll_interval": (
                        [-0.02, 0.02] if key.startswith("nll-") else None
                    ),
                    "test_method": "PRIVATE_SENTINEL",
                }
                for policy in ("F", "D")
                for control in ("R", "S")
                for key, value in endpoints.items()
            ],
            "memory": {
                arm: {
                    length: {"persistent_bytes": 1024, "scale_bytes": 8}
                    for length in ("1024", "8192")
                }
                for arm in ARMS
            },
            "private_runtime": "PRIVATE_SENTINEL",
        }
        evidence["models"][name] = {
            **copy.deepcopy(common),
            "retained_tokens_decode_exactly": True,
            "counts": {
                arm: {"nll": 128, "retrieval": 400, "gsm8k": 1319} for arm in ARMS
            },
            "raw_tokens": "PRIVATE_SENTINEL",
        }
    return analysis, evidence


def retained(tmp_path, analysis, evidence):
    paths = [tmp_path / name for name in ("analysis.json", "evidence.json")]
    for path, record in zip(paths, (analysis, evidence)):
        path.write_text(json.dumps(record))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"outputs": {p.name: sha(p) for p in paths}}))
    return *paths, receipt


def test_complete_endpoint_export_has_no_raw_fields(tmp_path):
    result = aggregate(*retained(tmp_path, *reports()))
    assert "PRIVATE_SENTINEL" not in json.dumps(result)
    for model in result["models"].values():
        assert len(model["direct"]) == 20
        for candidate in model["candidates"].values():
            assert set(ENDPOINTS) <= set(candidate)
            assert not candidate["preservation_certificate"]
            assert candidate["aggregate_math_certificate"]
        assert "named baseline" in model["direct_scope"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_direct",
        "repeated_direct",
        "nll",
        "retrieval",
        "payload",
        "nan",
        "memory_payload",
        "source_payload",
    ],
)
def test_incomplete_or_nonnumeric_data_cannot_be_exported(tmp_path, defect):
    analysis, evidence = reports()
    model = analysis["models"]["qwen3"]
    if defect == "missing_direct":
        model["direct"].pop()
    elif defect == "repeated_direct":
        model["direct"][-1] = copy.deepcopy(model["direct"][0])
    elif defect == "nll":
        model["candidates"]["SF"]["nll-8192"]["n_documents"] = 63
    elif defect == "retrieval":
        model["candidates"]["SD"]["retrieval-1024"]["n"] = 199
    elif defect == "payload":
        model["direct"][0]["result"]["net_delta"] = {"text": "PRIVATE_SENTINEL"}
    elif defect == "memory_payload":
        model["memory"]["N"]["1024"]["scale_bytes"] = "PRIVATE_SENTINEL"
    elif defect == "source_payload":
        model["source"]["commit"] = "PRIVATE_SENTINEL"
    else:
        model["candidates"]["AF"]["gsm8k"]["net_delta"] = float("nan")
    with pytest.raises(ValueError):
        aggregate(*retained(tmp_path, analysis, evidence))


def test_changed_report_requires_a_new_reader_receipt(tmp_path):
    paths = retained(tmp_path, *reports())
    paths[0].write_text(paths[0].read_text() + " ")
    with pytest.raises(ValueError, match="receipt"):
        aggregate(*paths)
