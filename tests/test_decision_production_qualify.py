# SPDX-License-Identifier: MIT
"""A few passing probes must not stand in for the entire worker test matrix."""

import copy

import pytest

from tools.kv.decision.production_qualify import coverage


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
