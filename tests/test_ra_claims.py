"""Tests for the stale-RA-claim checker (scripts/check_ra_claims.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import check_ra_claims as checker


def test_repo_is_clean():
    problems = []
    for path in checker.iter_files():
        problems.extend(checker.scan_file(path))
    assert problems == [], "\n".join(problems)


def _scan_doc(tmp_path, monkeypatch, name, text):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    (tmp_path / "docs").mkdir(exist_ok=True)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return checker.scan_file(path)


def test_forbidden_headline_detected(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/foo.md",
        "RA achieves 82% better perplexity (50.5 vs 282 PPL).\n",
    )
    assert any("82%" in p for p in problems)


def test_invalidation_history_allowed(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/foo.md",
        "An earlier page claimed 82% better PPL; those numbers could\n"
        "not be traced to retained artifacts and were removed.\n",
    )
    assert problems == []


def test_marker_exempts_line(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/foo.md",
        "the banned string 82% better appears here <!-- ra-claims-ok -->\n",
    )
    assert problems == []


def test_scoped_number_requires_scope(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/ra.md",
        "RA reached 68.9 PPL, a great result.\n",
    )
    assert any("scope" in p or "caveat" in p for p in problems)


def test_scoped_number_with_scope_and_caveat_passes(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/ra.md",
        "In a GPT-2 small-scale FineWebEdu run RA reached 68.9 PPL\n"
        "(vs 72.5 baseline); matched 1B runs were neutral, scaling\n"
        "unproven.\n",
    )
    assert problems == []


def test_unscoped_claims_outside_ra_surface_ignore_scoped_rule(tmp_path, monkeypatch):
    # 72.5 in a non-RA file (e.g. a memory table) must not trip the
    # scoped-number rule.
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/resnet.md",
        "| SGD | 12,501.1 MB | +72.5 MB |\n",
    )
    assert problems == []


def test_1b_audit_pending_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    audit = tmp_path / "fim" / "reciprocal_attention" / "LLAMA1B_AUDIT.md"
    audit.parent.mkdir(parents=True)
    audit.write_text("# audit\n\n## Results\n\nPending cloud execution.\n")
    problems = checker.scan_file(audit)
    assert any("Pending" in p for p in problems)


def test_exact_eigmax_outside_history_detected(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/foo.md",
        "We select heads with the exact_eigmax metric.\n",
    )
    assert any("exact_eigmax" in p for p in problems)


def test_exact_eigmax_in_bug_history_allowed(tmp_path, monkeypatch):
    problems = _scan_doc(
        tmp_path,
        monkeypatch,
        "docs/foo.md",
        "The exact_eigmax selector is degenerate: a row-stochastic\n"
        "matrix has spectral radius 1 (Perron-Frobenius).\n",
    )
    assert problems == []
