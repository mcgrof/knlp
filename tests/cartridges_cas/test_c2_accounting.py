# SPDX-License-Identifier: GPL-2.0
"""The reserve reconciliation, and the rule that an unknown is not a zero.

The failure this guards against is not arithmetic. It is that an allocation
nobody measured reads as free, so every unmeasured hour makes the reserve look
larger than it is, and always in the same direction.
"""

from __future__ import annotations

import datetime as dt

import pytest

from research.cartridges_cas.scripts import c2_accounting as acct
from tests.cartridges_cas.entry_guard import assert_entry_point_is_sound


def test_each_occupancy_is_a_subtraction_between_two_recorded_clocks():
    for a in acct.ALLOCATIONS:
        created = dt.datetime.fromisoformat(a["created_utc"].replace("Z", "+00:00"))
        ended = dt.datetime.fromisoformat(a["terminated_utc"].replace("Z", "+00:00"))
        assert ended > created, a["id"]
        assert acct.occupancy(a) == (ended - created).total_seconds()
        assert a["created_evidence"] and a["terminated_evidence"]


def test_the_two_source_screen_acquisitions_reconcile_to_their_measured_values():
    by_id = {a["id"]: acct.occupancy(a) for a in acct.ALLOCATIONS}
    assert by_id["1fddf8254f334648b26618d8ca8859c2"] == 1046.0
    assert by_id["8a9bd6c4502d44e9bb969430e5692ba0"] == 8691.0


def test_the_wasted_acquisition_is_larger_than_the_work_it_failed_to_do():
    """The finding, pinned so it cannot quietly disappear from the record.

    The second acquisition stayed alive for longer than the screen itself took,
    while the screen ran elsewhere. It was previously carried as an unknown.
    """
    wasted = acct.occupancy(acct.ALLOCATIONS[1])
    screen = acct.LOCAL[0]["seconds"]
    assert wasted > screen
    assert wasted / screen > 2.9


def test_the_residual_is_what_is_left_after_every_measured_debit():
    r = acct.reconcile()
    assert r["rented_seconds"] == 1046.0 + 8691.0
    assert r["local_seconds"] == 2914.6
    assert r["recorded_debit_seconds"] == pytest.approx(12651.6)
    assert r["residual_seconds"] == pytest.approx(1748.4)


def test_the_residual_binds_the_stage_cap_rather_than_the_other_way_round():
    r = acct.reconcile()
    assert r["c2_cap_seconds"] == pytest.approx(1748.4)
    assert r["c2_cap_seconds"] < acct.STAGE_CAP_SECONDS


def test_local_hardware_bills_nothing_and_still_consumes_the_reserve():
    r = acct.reconcile()
    assert all(entry["usd"] == 0.0 for entry in r["local_occupancy"])
    assert r["local_seconds"] > 0
    assert r["recorded_debit_seconds"] > r["rented_seconds"]


def test_unknown_debits_are_named_and_never_entered_as_zero():
    r = acct.reconcile()
    assert r["unresolved_debits"], "an empty list would assert everything is known"
    assert r["residual_is_an_upper_bound"] is True
    assert "may be entered as zero" in r["unresolved_rule"]


def test_the_complete_comparison_does_not_fit_what_is_left():
    """Generation alone, at a rate faster than anything measured here."""
    fastest_plausible_generation = 163840 / 90
    v = acct.verdict_for(fastest_plausible_generation)
    assert v["fits"] is False
    assert v["verdict"] == acct.INSUFFICIENT
    assert v["shortfall_seconds"] > 0


def test_a_small_stage_still_fits():
    v = acct.verdict_for(900.0)
    assert v["fits"] is True and v["verdict"] is None


def test_the_dollar_figures_follow_from_occupancy_and_rate():
    r = acct.reconcile()
    for row in r["rented_allocations"]:
        assert row["usd"] == pytest.approx(
            row["occupied_seconds"] / 3600 * row["usd_per_hour"], abs=0.01
        )
    assert r["rented_usd"] == pytest.approx(3.48, abs=0.02)


def test_the_entry_point_can_actually_be_executed():
    assert_entry_point_is_sound(acct)
