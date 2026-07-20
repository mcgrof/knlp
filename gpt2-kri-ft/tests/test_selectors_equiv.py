"""P0 gate: prove the selector equivalences the whole line assumes.

No larger lattice/KRI experiment starts until these pass:
  1. rel_only (top-K cosine) reproduces cosine KRI-Q under the same probe.
  2. leaky_residual(alpha=0) selects IDENTICAL blocks to plain recent-Q top-K.
  3. leaky_residual(alpha=1) is the exact-span orthogonal residual (well-defined,
     deterministic, distinct from alpha=0 whenever centroids are non-orthogonal).
  4. every operator is budget-matched (returns exactly min(K, NB) distinct picks).
  5. position-matched random is present and returns distinct in-range blocks.
"""

import torch

from src.selectors import (
    leaky_residual,
    top_k_relevance,
    top_k_mass,
    position_matched_random,
    select,
)


def _fixture(H=6, NB=40, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q0 = torch.randn(H, D, generator=g)
    kc = torch.randn(H, NB, D, generator=g)
    kc = kc / kc.norm(dim=-1, keepdim=True)
    mass = torch.rand(H, NB, generator=g)
    return q0, kc, mass


def test_rel_only_equals_kri_q_same_probe():
    # rel_only IS cosine KRI-Q; under the same probe they must be identical.
    q0, kc, _ = _fixture()
    a = top_k_relevance(q0, kc, K=8)
    b = select("kri_q", K=8, q0=q0, kc=kc)
    assert torch.equal(a, b)


def test_leaky_alpha0_equals_recentq_topk_exactly():
    q0, kc, _ = _fixture(seed=1)
    for K in (4, 8, 16):
        topk = top_k_relevance(q0, kc, K)  # recent-Q top-K
        leaky0 = leaky_residual(q0, kc, K, alpha=0.0)  # alpha=0 endpoint
        assert torch.equal(topk, leaky0), f"alpha=0 != recent-Q top-K at K={K}"


def test_leaky_alpha1_is_deterministic_and_differs_when_nonorthogonal():
    q0, kc, _ = _fixture(seed=2)
    r1 = leaky_residual(q0, kc, K=12, alpha=1.0)
    r2 = leaky_residual(q0, kc, K=12, alpha=1.0)
    assert torch.equal(r1, r2)  # deterministic
    r0 = leaky_residual(q0, kc, K=12, alpha=0.0)
    # with random (non-orthogonal) centroids the orthogonal residual must diverge
    # from plain relevance at some rank.
    assert not torch.equal(r1, r0)


def test_leaky_alpha1_first_pick_matches_relevance():
    # the first pick has an empty basis -> pure relevance regardless of alpha.
    q0, kc, _ = _fixture(seed=3)
    topk = top_k_relevance(q0, kc, K=1)
    for a in (0.0, 0.25, 0.5, 1.0):
        first = leaky_residual(q0, kc, K=1, alpha=a)
        assert torch.equal(first, topk), f"first pick != top-1 relevance at alpha={a}"


def test_leaky_monotone_endpoints_bracket():
    # alpha in (0,1) should equal alpha=0 up to the first rank and then may
    # diverge; sanity: all alphas agree on rank 0.
    q0, kc, _ = _fixture(seed=4)
    picks = {a: leaky_residual(q0, kc, K=8, alpha=a) for a in (0.0, 0.125, 0.5, 1.0)}
    for a, p in picks.items():
        assert torch.equal(p[:, 0], picks[0.0][:, 0])


def test_budget_matched_distinct():
    q0, kc, mass = _fixture(seed=5)
    K, NB = 8, kc.shape[1]
    for op, kw in [
        ("rel_only", {}),
        ("residual_rel", {"alpha": 1.0}),
        ("h2o", {}),
    ]:
        picks = select(op, K=K, q0=q0, kc=kc, mass=mass, **kw)
        assert picks.shape[-1] == min(K, NB)
        for h in range(picks.shape[0]):
            assert len(set(picks[h].tolist())) == picks.shape[-1], f"{op} has dup picks"


def test_position_matched_random_distinct_in_range():
    H, NB, K = 6, 40, 8
    g = torch.Generator().manual_seed(7)
    picks = position_matched_random(H, NB, K, g)
    assert picks.shape == (H, K)
    for h in range(H):
        s = picks[h].tolist()
        assert len(set(s)) == K
        assert all(0 <= b < NB for b in s)


def test_ordered_selection_logging():
    q0, kc, _ = _fixture(seed=8)
    log = []
    leaky_residual(q0, kc, K=5, alpha=0.5, log=log)
    assert len(log) == 5
    assert log[0]["step"] == 0 and "pick" in log[0]


def test_K_exceeds_NB_clamps():
    q0, kc, _ = _fixture(H=3, NB=5, D=8, seed=9)
    picks = leaky_residual(q0, kc, K=999, alpha=1.0)
    assert picks.shape == (3, 5)
    for h in range(3):
        assert len(set(picks[h].tolist())) == 5
