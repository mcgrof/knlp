# SPDX-License-Identifier: GPL-2.0
"""The estimators, on scores and timings whose answer is known by construction."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from research.kv_translate.sharing import estimate  # noqa: E402


def rows(route, table, mapper=None, health=()):
    """table: {(domain, length, context): [correct, correct]}"""
    out = []
    for (domain, length, cid), marks in table.items():
        for i, ok in enumerate(marks):
            out.append(
                {
                    "context_id": cid,
                    "source_group_id": "g-" + cid,
                    "domain": domain,
                    "length_bucket": length,
                    "query_id": f"{cid}.q{i}",
                    "route": route,
                    "mapper": mapper,
                    "correct": ok,
                    "health_events": ["empty"] if (cid, i) in health else [],
                }
            )
    return out


BASE = {
    ("document", 4096, "a"): [True, True],
    ("document", 4096, "b"): [True, False],
    ("code", 16384, "c"): [True, True],
    ("code", 16384, "d"): [False, False],
}


def test_a_route_that_scores_the_same_has_no_deficit_and_no_spread():
    ctx = estimate.context_scores(rows("restore", BASE) + rows("translate", BASE, "m"))
    got = estimate.paired_deficit(ctx["restore"], ctx["translate:m"], resamples=300)
    assert got["deficit"] == 0.0 and got["deficit_upper_95_one_sided"] == 0.0
    assert got["contexts_equal"] == 4 and got["groups"] == 4


def test_the_deficit_is_signed_and_weighted_by_the_frozen_strata():
    worse = dict(BASE)
    worse[("document", 4096, "a")] = [False, False]  # document/4096 mean falls by 0.5
    better = dict(BASE)
    better[("code", 16384, "d")] = [True, True]  # code/16384 mean rises by 0.5
    ctx = estimate.context_scores(
        rows("restore", BASE)
        + rows("translate", worse, "w")
        + rows("translate", better, "b")
    )
    w = estimate.paired_deficit(ctx["restore"], ctx["translate:w"], resamples=300)
    b = estimate.paired_deficit(ctx["restore"], ctx["translate:b"], resamples=300)
    # weights: document/4096 is 0.5 * 0.25, code/16384 is 0.5 * 0.5
    assert w["deficit"] == pytest.approx(0.5 * 0.125 / 0.375)
    assert b["deficit"] == pytest.approx(-0.5 * 0.25 / 0.375)
    assert w["largest_stratum_deficit"] == pytest.approx(0.5)
    assert w["strata"]["document/4096"]["deficit"] == pytest.approx(0.5)
    assert w["contexts_worse"] == 1 and b["contexts_better"] == 1
    assert w["deficit_upper_95_one_sided"] >= w["deficit"]


def test_a_context_is_scored_by_the_mean_of_its_questions():
    ctx = estimate.context_scores(rows("restore", BASE))
    assert ctx["restore"]["b"]["score"] == 0.5
    assert ctx["restore"]["a"]["queries"] == 2


def test_routes_that_cover_different_contexts_are_refused():
    part = {k: v for k, v in BASE.items() if k[2] != "d"}
    ctx = estimate.context_scores(rows("restore", BASE) + rows("translate", part, "m"))
    with pytest.raises(ValueError, match="same contexts"):
        estimate.paired_deficit(ctx["restore"], ctx["translate:m"], resamples=50)


def test_an_answer_with_no_score_is_never_counted_as_wrong():
    bad = rows("restore", BASE)
    bad[0]["correct"] = None
    with pytest.raises(ValueError, match="no score"):
        estimate.context_scores(bad)


def test_the_exact_limits_match_their_closed_forms():
    assert estimate.binomial_upper(0, 64, 0.025) == pytest.approx(
        1 - 0.025 ** (1 / 64), abs=1e-9
    )
    assert estimate.binomial_upper(0, 512, 0.025) == pytest.approx(
        1 - 0.025 ** (1 / 512), abs=1e-9
    )
    assert estimate.binomial_lower(0, 64, 0.025) == pytest.approx(0.0, abs=1e-12)
    assert estimate.binomial_lower(64, 64, 0.025) == pytest.approx(
        0.025 ** (1 / 64), abs=1e-9
    )
    assert estimate.binomial_upper(64, 64, 0.025) == 1.0
    up = estimate.binomial_upper(3, 40, 0.025)
    assert 3 / 40 < up < 0.25


def test_no_failure_seen_still_leaves_a_bound_above_zero():
    ctx = estimate.context_scores(rows("restore", BASE) + rows("translate", BASE, "m"))
    got = estimate.health_excess(ctx["restore"], ctx["translate:m"])
    assert got["route_unhealthy"] == 0 and got["excess"] == 0.0
    assert got["excess_upper_95_conservative"] == pytest.approx(1 - 0.025 ** (1 / 4))


def test_one_broken_answer_makes_its_context_unhealthy():
    ctx = estimate.context_scores(
        rows("restore", BASE) + rows("translate", BASE, "m", health={("a", 1)})
    )
    got = estimate.health_excess(ctx["restore"], ctx["translate:m"])
    assert got["route_unhealthy"] == 1 and got["excess"] == pytest.approx(0.25)


def timing(route, cid, length, seconds, state):
    return {
        "context_id": cid,
        "source_group_id": "g-" + cid,
        "domain": "document",
        "length_bucket": length,
        "query_id": cid + ".q0",
        "route": route,
        "read_state": "none" if route == "recompute" else state,
        "seconds_total": seconds,
        "seconds_suffix": 0.1,
    }


def test_a_percentile_is_taken_over_requests_under_the_frozen_weights():
    assert (
        estimate.weighted_quantile([(1.0, 1), (2.0, 1), (3.0, 1), (4.0, 1)], 0.5) == 2.0
    )
    assert estimate.weighted_quantile([(1.0, 1), (9.0, 3)], 0.5) == 9.0
    assert estimate.weighted_quantile([(5.0, 1)], 0.95) == 5.0


def test_the_saving_is_measured_against_recomputing():
    t = []
    for cid in "abcd":
        for _ in range(3):
            t.append(timing("recompute", cid, 16384, 4.0, "direct"))
            t.append(timing("translate", cid, 16384, 1.0, "direct"))
            t.append(timing("restore", cid, 16384, 0.5, "direct"))
            t.append(timing("translate", cid, 16384, 0.8, "page_cache"))
    got = estimate.latency(t, "direct", resamples=100)
    assert got["routes"]["recompute"]["p50"] == 4.0
    assert got["routes"]["translate"]["requests"] == 12
    assert got["against_recompute"]["translate"]["p50_saving"] == pytest.approx(0.75)
    assert got["against_recompute"]["restore"]["p50_saving"] == pytest.approx(0.875)
    assert got["against_recompute"]["translate"]["p95_seconds_more"] == pytest.approx(
        -3.0
    )
    warm = estimate.latency(t, "page_cache", resamples=100)
    assert warm["routes"]["translate"]["p50"] == 0.8
    assert "restore" not in warm["routes"]


def test_agreement_counts_identical_outputs_and_verdicts():
    scores = rows("restore", BASE) + rows("translate", BASE, "m")
    gens = []
    for s in scores:
        same = s["context_id"] in ("a", "b")
        toks = [1, 2, 3] if (s["route"] == "restore" or same) else [1, 9]
        gens.append(
            {
                "query_id": s["query_id"],
                "route": s["route"],
                "mapper": s["mapper"],
                "tokens": toks,
            }
        )
    got = estimate.agreement(scores, gens, "restore", "translate:m")
    assert got == {
        "queries": 8,
        "identical_output": 4,
        "identical_first_token": 8,
        "same_verdict": 8,
    }
