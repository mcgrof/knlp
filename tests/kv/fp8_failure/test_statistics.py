"""Statistics helpers: deterministic bootstrap CIs that bracket the point estimate, BH-FDR that is
monotone and order-invariant, and LOFO attribution that ranks a hurting factor first. Cheap guards
against noise-mining the atlas's many cells."""

from tools.kv.fp8_failure import statistics as S


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    p1, lo1, hi1 = S.bootstrap_ci(vals, n_boot=500, seed=7)
    p2, lo2, hi2 = S.bootstrap_ci(vals, n_boot=500, seed=7)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)  # seed-deterministic
    assert abs(p1 - 3.0) < 1e-9
    assert lo1 <= p1 <= hi1


def test_paired_delta_sign():
    a = [1.0, 1.1, 0.9, 1.05]  # higher error (fp8)
    b = [0.5, 0.4, 0.6, 0.5]  # lower error (repair)
    d, lo, hi = S.paired_bootstrap_delta(a, b, n_boot=500, seed=3)
    assert d > 0 and lo > 0  # repair strictly helps


def test_bh_fdr_monotone_and_order_invariant():
    pvals = [0.001, 0.04, 0.2, 0.5, 0.9]
    reject, adj = S.benjamini_hochberg(pvals, q=0.05)
    assert reject[0] is True and reject[-1] is False
    # order invariance: permuting input permutes output the same way
    perm = [2, 0, 4, 1, 3]
    rp, ap = S.benjamini_hochberg([pvals[i] for i in perm], q=0.05)
    for new_i, orig_i in enumerate(perm):
        assert rp[new_i] == reject[orig_i]
        assert abs(ap[new_i] - adj[orig_i]) < 1e-12


def test_lofo_ranks_hurting_factor_first():
    # full error 1.0; removing 'fp8_k' drops error to 0.2 (it was hurting most)
    imp = S.lofo_attribution(1.0, {"fp8_k": 0.2, "fp8_v": 0.8, "layout": 0.95})
    assert list(imp.keys())[0] == "fp8_k"
    assert imp["fp8_k"] > imp["fp8_v"] > imp["layout"]
