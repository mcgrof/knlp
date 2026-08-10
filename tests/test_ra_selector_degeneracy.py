"""Regression tests for the degenerate exact_eigmax RA head selector.

The batch-mean post-softmax attention matrix is row-stochastic, so its
spectral radius is exactly 1 for every head (Perron-Frobenius). Ranking
heads by that eigenvalue is random tie-breaking on floating-point
noise. These tests pin the degeneracy itself and the harness guard that
refuses the metric. See docs/ra-evidence.md.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fim.reciprocal_attention.llama150m_matched import (  # noqa: E402
    AttentionStatsCollector,
    _validate_head_score_metric,
)


def _random_row_stochastic(t: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(t, t, generator=gen, dtype=torch.float64)
    return torch.softmax(logits, dim=-1)


def test_row_stochastic_matrix_has_eigmax_one():
    """Any row-stochastic matrix has spectral radius ~1: the metric
    cannot rank heads."""
    for seed in range(5):
        mat = _random_row_stochastic(64, seed)
        eigmax = torch.linalg.eigvals(mat).abs().max().item()
        assert eigmax == pytest.approx(1.0, abs=1e-6)


def test_exact_eigmax_metric_rejected():
    with pytest.raises(ValueError, match="exact_eigmax.*degenerate"):
        _validate_head_score_metric("exact_eigmax")


def test_unknown_metric_rejected():
    with pytest.raises(ValueError, match="unknown head_score_metric"):
        _validate_head_score_metric("fisher_eigmax")


def test_enabled_collector_rejects_exact_eigmax():
    with pytest.raises(ValueError, match="exact_eigmax"):
        AttentionStatsCollector(
            num_layers=2, num_heads=2, enabled=True, score_metric="exact_eigmax"
        )


def test_disabled_collector_construction_still_works():
    # Baseline runs construct a disabled collector with the historical
    # default metric; that must not crash.
    collector = AttentionStatsCollector(num_layers=2, num_heads=2, enabled=False)
    collector.update(0, torch.rand(1, 2, 8, 8))  # no-op when disabled


def test_inbound_mass_var_still_scores():
    collector = AttentionStatsCollector(
        num_layers=1, num_heads=2, enabled=True, score_metric="inbound_mass_var"
    )
    probs = torch.softmax(torch.randn(2, 2, 8, 8), dim=-1)
    collector.update(0, probs)
    assert torch.isfinite(collector.head_score[0]).all()
