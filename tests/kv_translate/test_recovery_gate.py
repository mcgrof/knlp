# SPDX-License-Identifier: GPL-2.0
"""The recovery gate must refuse the evidence sets that previously advanced.

Every case here is one an external audit built and ran against the earlier
gate, which returned an advancing verdict on all of them. They are rebuilt and
run through the production command line rather than through a local helper,
because the earlier integrity tests asserted completeness themselves while the
gate under test never did, and passed for that reason.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "research" / "kv_translate" / "recovery_gate.py"
N = 16
DOCS = [f"doc{i:02d}" for i in range(N)]

R3_MATRIX = {
    "retrieval": ["cand", "cand|wrong", "target_native", "empty_cache", "anchor77"],
    "cloze": ["cand", "cand|wrong", "target_native", "empty_cache", "anchor77"],
    "free": ["cand", "target_native", "anchor77"],
}


def _free(cond, doc, healthy=1.0, **over):
    r = {
        "kind": "free",
        "condition": cond,
        "doc": doc,
        "manifest_version": "v2",
        "healthy": healthy,
        "empty": 0.0,
        "too_short": 0.0,
        "degenerate": 0.0,
        "stopped": 0.0,
        "cap_hit": 0.0,
    }
    r.update(over)
    return r


def _rows(tmp_path, name, mutate=None, scores=None):
    scores = scores or {}
    out = []
    for kind in ("retrieval", "cloze"):
        for cond in R3_MATRIX[kind]:
            for d in DOCS:
                v = scores.get(
                    (kind, cond),
                    1.0 if "wrong" not in cond and cond != "empty_cache" else 0.0,
                )
                if callable(v):
                    v = v(d)
                out.append(
                    {
                        "kind": kind,
                        "condition": cond,
                        "doc": d,
                        "manifest_version": "v2",
                        "primary": v,
                    }
                )
    for cond in R3_MATRIX["free"]:
        for d in DOCS:
            out.append(_free(cond, d))
    if mutate:
        out = mutate(out)
    p = tmp_path / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in out))
    return p


def _manifest(tmp_path, docs=DOCS):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps(
            {
                "version": "v2",
                "joint_sha256": "deadbeef",
                "documents": [{"doc": d} for d in docs],
            }
        )
    )
    return p


def _matrix(tmp_path):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(R3_MATRIX))
    return p


def _run(tmp_path, rows, stage="R3"):
    out = tmp_path / "gate.json"
    r = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--rows",
            str(rows),
            "--manifest",
            str(_manifest(tmp_path)),
            "--matrix",
            str(_matrix(tmp_path)),
            "--stage",
            stage,
            "--candidate",
            "cand",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    data = json.loads(out.read_text()) if out.exists() else {}
    return r.returncode, data, r.stdout + r.stderr


def test_a_complete_positive_fixture_still_advances(tmp_path):
    """The control. A gate that refuses everything is not a gate."""
    rc, d, _ = _run(tmp_path, _rows(tmp_path, "ok"))
    assert rc == 0, d
    assert d["valid_evidence"] is True
    assert d["advances"] is True, d.get("failed_requirements")


def test_one_candidate_document_of_sixteen_is_refused(tmp_path):
    """Previously advanced on a paired comparison of n=1."""

    def keep_one(rows):
        return [
            r for r in rows if not (r["condition"] == "cand" and r["doc"] != DOCS[0])
        ]

    rc, d, err = _run(tmp_path, _rows(tmp_path, "thin", keep_one))
    assert rc == 2
    assert d["verdict"] == "INVALID_EVIDENCE"
    assert d["advances"] is False
    assert "absent" in d["reason"]


def test_a_duplicated_cell_is_refused(tmp_path):
    def dup(rows):
        first = next(r for r in rows if r["condition"] == "cand")
        return rows + [dict(first)]

    rc, d, _ = _run(tmp_path, _rows(tmp_path, "dup", dup))
    assert rc == 2
    assert "more than once" in d["reason"]


def test_a_missing_gross_repetition_field_is_refused(tmp_path):
    """An absent field is not zero failures."""

    def drop(rows):
        for r in rows:
            if r["kind"] == "free" and r["condition"] == "cand":
                r.pop("degenerate", None)
        return rows

    rc, d, _ = _run(tmp_path, _rows(tmp_path, "nodegen", drop))
    assert rc == 2
    assert "degenerate" in d["reason"]


def test_healthy_while_empty_and_too_short_is_refused(tmp_path):
    """A derived label that contradicts the raw fields it summarises."""

    def contradict(rows):
        for r in rows:
            if r["kind"] == "free" and r["condition"] == "cand":
                r["empty"] = 1.0
                r["too_short"] = 1.0
        return rows

    rc, d, _ = _run(tmp_path, _rows(tmp_path, "contra", contradict))
    assert rc == 2
    assert "healthy" in d["reason"]


def test_sign_changing_bootstrap_draws_make_recovery_undefined(tmp_path):
    """The audit's constructed case: real headroom, 18% of draws invalid.

    The earlier gate dropped those draws and reported a perfect interval.
    """
    half = set(DOCS[: N // 2])
    scores = {
        ("cloze", "target_native"): lambda d: 1.0 if d in half else 0.125,
        ("cloze", "empty_cache"): lambda d: 0.0 if d in half else 0.875,
        ("cloze", "cand"): lambda d: 1.0 if d in half else 0.125,
    }
    rc, d, _ = _run(tmp_path, _rows(tmp_path, "signflip", scores=scores))
    assert rc == 0
    ri = d["tasks"]["cloze"]["recovery_interval"]
    assert ri["undefined"] is True
    assert ri["draws_with_nonpositive_headroom"] > 0
    assert ri["point"] is None and "lo" not in ri
    assert d["verdict"] == "BENCHMARK_UNINFORMATIVE_FOR_RECOVERY_GATE"
    assert d["advances"] is False


def test_rows_outside_the_declared_matrix_are_refused(tmp_path):
    def add(rows):
        return rows + [
            {
                "kind": "retrieval",
                "condition": "surprise",
                "doc": DOCS[0],
                "manifest_version": "v2",
                "primary": 1.0,
            }
        ]

    rc, d, _ = _run(tmp_path, _rows(tmp_path, "extra", add))
    assert rc == 2
    assert "outside the declared matrix" in d["reason"]


def test_mixed_manifest_versions_are_refused(tmp_path):
    def mix(rows):
        rows[0]["manifest_version"] = "v1"
        return rows

    rc, d, _ = _run(tmp_path, _rows(tmp_path, "mixed", mix))
    assert rc == 2
    assert "mixed manifest versions" in d["reason"]


def test_a_nonfinite_score_is_refused(tmp_path):
    def nan(rows):
        for r in rows:
            if r["kind"] == "retrieval" and r["condition"] == "cand":
                r["primary"] = float("nan")
                break
        return rows

    rc, d, _ = _run(tmp_path, _rows(tmp_path, "nan", nan))
    assert rc == 2
    assert "not finite" in d["reason"]


def test_the_screen_stage_uses_its_own_thresholds(tmp_path):
    """R1 is a resource-release screen, not the development gate."""
    m = {
        "retrieval": ["cand", "mapped_unrepaired", "target_native", "empty_cache"],
        "cloze": ["cand", "mapped_unrepaired", "target_native", "empty_cache"],
        "free": ["cand", "mapped_unrepaired", "target_native"],
    }
    (tmp_path / "matrix.json").write_text(json.dumps(m))
    out, rows = tmp_path / "g.json", tmp_path / "r.jsonl"
    recs = []
    for kind in ("retrieval", "cloze"):
        for cond in m[kind]:
            for i, d in enumerate(DOCS):
                if cond == "empty_cache":
                    v = 0.0
                elif cond == "target_native":
                    v = 1.0  # the benchmark must be informative
                elif cond == "mapped_unrepaired":
                    v = 1.0 if i < 8 else 0.0
                else:
                    v = 1.0 if i < 11 else 0.0  # +3 retrieval over unrepaired
                recs.append(
                    {
                        "kind": kind,
                        "condition": cond,
                        "doc": d,
                        "manifest_version": "v2",
                        "primary": v,
                    }
                )
    for cond in m["free"]:
        for d in DOCS:
            recs.append(_free(cond, d))
    rows.write_text("".join(json.dumps(r) + "\n" for r in recs))
    r = subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--rows",
            str(rows),
            "--manifest",
            str(_manifest(tmp_path)),
            "--matrix",
            str(tmp_path / "matrix.json"),
            "--stage",
            "R1",
            "--candidate",
            "cand",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    d = json.loads(out.read_text())
    assert r.returncode == 0, r.stdout + r.stderr
    assert d["screen"]["net_retrieval_over_unrepaired"] == pytest.approx(3.0)
    assert d["screen"]["meets_net_retrieval"] is True
    assert d["verdict"] == "SCREEN_SIGNAL"


def test_one_net_retrieval_answer_does_not_pass_the_screen(tmp_path):
    """The bar is two. One is the boundary that must fail."""
    m = {
        "retrieval": ["cand", "mapped_unrepaired", "target_native", "empty_cache"],
        "cloze": ["cand", "mapped_unrepaired", "target_native", "empty_cache"],
        "free": ["cand", "mapped_unrepaired", "target_native"],
    }
    (tmp_path / "matrix.json").write_text(json.dumps(m))
    out, rows = tmp_path / "g.json", tmp_path / "r.jsonl"
    recs = []
    for kind in ("retrieval", "cloze"):
        for cond in m[kind]:
            for i, d in enumerate(DOCS):
                if cond == "empty_cache":
                    v = 0.0
                elif cond == "target_native":
                    v = 1.0  # the benchmark must be informative
                elif cond == "mapped_unrepaired":
                    v = 1.0 if i < 8 else 0.0
                else:
                    v = 1.0 if i < 9 else 0.0  # +1 only
                recs.append(
                    {
                        "kind": kind,
                        "condition": cond,
                        "doc": d,
                        "manifest_version": "v2",
                        "primary": v,
                    }
                )
    for cond in m["free"]:
        for d in DOCS:
            recs.append(_free(cond, d))
    rows.write_text("".join(json.dumps(r) + "\n" for r in recs))
    subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--rows",
            str(rows),
            "--manifest",
            str(_manifest(tmp_path)),
            "--matrix",
            str(tmp_path / "matrix.json"),
            "--stage",
            "R1",
            "--candidate",
            "cand",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    d = json.loads(out.read_text())
    assert d["screen"]["meets_net_retrieval"] is False
    assert d["advances"] is False
