# SPDX-License-Identifier: MIT
"""A few passing probes must not stand in for the entire worker test matrix."""

import copy

import pytest

from tools.kv.decision.production_qualify import coverage, kernel_artifacts


def records():
    rows = []
    for kind in ("writer", "attention"):
        for phase in ("prefill", "mixed", "decode"):
            rows += [
                dict(
                    kind=kind,
                    phase=phase,
                    layer=f"layers.{i}",
                    exact=True,
                    passed=True,
                    query_dtype="torch.float16",
                )
                for i in range(36)
            ]
    rows += [
        dict(kind="graph", padded_tokens=n, cache_codes_exact=True, passed=True)
        for n in [1, 8, 32, 44, 45, 59, 60, 96]
    ]
    rows += [
        dict(kind="routing", requests=96, slots=[1, 2]),
        dict(
            kind="routing",
            requests=96,
            slots=[3, 4],
            phase="decode",
            positions=[8192] * 96,
        ),
    ]
    return rows


def test_full_numerical_matrix_has_explicit_coverage():
    result = coverage(records(), "AF")
    assert result["maximum_actual_requests"] == 96
    assert result["graph_shapes"] == [1, 8, 32, 44, 45, 59, 60, 96]


@pytest.mark.parametrize(
    "bad",
    [
        "layer",
        "phase",
        "graph",
        "query",
        "writer",
        "verdict",
        "cache",
        "slots",
        "long_context",
    ],
)
def test_missing_or_inconsistent_worker_checks_reject_qualification(bad):
    rows = copy.deepcopy(records())
    if bad == "layer":
        rows[0]["layer"] = "different-layer"
    elif bad == "phase":
        rows = [r for r in rows if r.get("phase") != "mixed"]
    elif bad == "graph":
        rows = [r for r in rows if r.get("padded_tokens") != 96]
    elif bad == "query":
        rows[110]["query_dtype"] = "torch.float8_e4m3fn"
    elif bad == "writer":
        del rows[0]["exact"]
    elif bad == "verdict":
        del rows[110]["passed"]
    elif bad == "cache":
        next(r for r in rows if r["kind"] == "graph")["cache_codes_exact"] = False
    elif bad == "slots":
        for r in rows:
            if r["kind"] == "routing":
                r["slots"] = [1, 2]
    else:
        for r in rows:
            if r["kind"] == "routing":
                r["positions"] = [1024] * r["requests"]
    with pytest.raises(ValueError):
        coverage(rows, "AF")


def dynamic_records():
    rows = records()
    for row in rows:
        if row["kind"] == "writer":
            row["records"] = [
                dict(inline_scale_bytes_checked=512, scale_policy="per_token_head")
            ]
    for i in range(36):
        for route in ("2D", "3D"):
            rows.append(
                dict(
                    kind="attention_kernel",
                    layer=f"layers.{i}",
                    phase="decode",
                    route=route,
                    query_fp8=False,
                    query_dtype="torch.float16",
                    generated_sha256={"ptx": "1" * 64, "cubin": "2" * 64},
                )
            )
        for phase in ("prefill", "mixed", "decode"):
            rows.append(
                dict(
                    kind="writer_kernel",
                    layer=f"layers.{i}",
                    phase=phase,
                    reference_fp8=True,
                    generated_sha256={"ptx": "3" * 64, "cubin": "4" * 64},
                )
            )
    return rows


def test_dynamic_requires_actual_launch_routes_and_scale_bytes():
    result = coverage(dynamic_records(), "SD")
    assert set(result["dynamic_kernel_routes"]) == {"2D", "3D"}


@pytest.mark.parametrize(
    "defect", ["route", "layer", "writer", "scale", "q", "binary", "capacity"]
)
def test_dynamic_missing_execution_evidence_fails(defect):
    rows = dynamic_records()
    if defect == "route":
        rows = [r for r in rows if r.get("route") != "3D"]
    elif defect == "layer":
        next(r for r in rows if r["kind"] == "attention_kernel")["layer"] = "different"
    elif defect == "writer":
        next(r for r in rows if r["kind"] == "writer_kernel")["reference_fp8"] = False
    elif defect == "scale":
        rows[0]["records"][0]["inline_scale_bytes_checked"] = 0
    elif defect == "q":
        next(r for r in rows if r["kind"] == "attention_kernel")["query_fp8"] = True
    elif defect == "binary":
        next(r for r in rows if r["kind"] == "attention_kernel")[
            "generated_sha256"
        ] = {}
    else:
        for r in rows:
            if r["kind"] == "routing" and r.get("phase") == "decode":
                r["requests"] = 67
    with pytest.raises(ValueError):
        coverage(rows, "SD")


def test_launch_artifact_bytes_are_rechecked(tmp_path):
    import hashlib

    data = {"ptx": b"synthetic PTX", "cubin": b"synthetic compiled bytes"}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in data.items()}
    for key, value in data.items():
        (tmp_path / (hashes[key] + "." + key)).write_bytes(value)
    rows = [dict(kind="attention_kernel", generated_sha256=hashes)]
    assert len(kernel_artifacts(tmp_path, rows)) == 2
    (tmp_path / (hashes["cubin"] + ".cubin")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        kernel_artifacts(tmp_path, rows)
