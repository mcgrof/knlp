# SPDX-License-Identifier: GPL-2.0
"""Advancement and deadlines, on the cases the last run got wrong."""

from __future__ import annotations

import json

import pytest

from research.kv_translate.stage_control import (
    Deadline,
    forecast,
    may_advance,
    read_verdict,
)


def gate(tmp_path, name, **body):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(body))
    return str(p)


def test_a_valid_negative_verdict_does_not_advance(tmp_path):
    """The defect this exists for: the gate exits zero for this result."""
    g = gate(tmp_path, "r3", verdict="NO_SIGNAL", advances=False)
    r = may_advance([g])
    assert r["advances"] is False
    assert r["blocked_by"] == ["NO_SIGNAL"]


def test_an_advancing_verdict_advances(tmp_path):
    g = gate(tmp_path, "r1", verdict="SCREEN_SIGNAL", advances=True)
    assert may_advance([g])["advances"] is True


def test_every_named_gate_must_advance(tmp_path):
    a = gate(tmp_path, "a", verdict="SCREEN_SIGNAL", advances=True)
    b = gate(tmp_path, "b", verdict="NO_SIGNAL", advances=False)
    assert may_advance([a, b])["advances"] is False


def test_a_missing_gate_blocks_rather_than_passes(tmp_path):
    r = read_verdict(str(tmp_path / "never_written.json"))
    assert r["advances"] is False and r["verdict"] == "ABSENT"


def test_an_unreadable_gate_blocks(tmp_path):
    p = tmp_path / "torn.json"
    p.write_text('{"verdict": "SCREEN_SIG')
    assert read_verdict(str(p))["advances"] is False


def test_a_gate_with_no_verdict_field_blocks(tmp_path):
    g = gate(tmp_path, "empty", scores={"x": 1})
    assert read_verdict(g)["advances"] is False


def test_a_gate_that_stops_writing_its_flag_fails_closed(tmp_path):
    """Absent the flag, an unknown verdict must not be read as advancing."""
    g = gate(tmp_path, "novel", verdict="SOME_NEW_OUTCOME")
    assert read_verdict(g)["advances"] is False
    g2 = gate(tmp_path, "known", verdict="SCREEN_SIGNAL")
    assert read_verdict(g2)["advances"] is True


def test_the_gate_s_own_flag_outranks_the_verdict_name(tmp_path):
    g = gate(tmp_path, "odd", verdict="SCREEN_SIGNAL", advances=False)
    assert read_verdict(g)["advances"] is False


def test_a_deadline_reserves_time_to_record_what_it_did():
    d = Deadline(100, name="s", reserve=20)
    assert d.remaining() <= 80.0
    assert d.enough_for(10) is True
    assert d.enough_for(500) is False


def test_an_expired_deadline_says_so_instead_of_raising():
    d = Deadline(0.0, name="s")
    assert d.expired() is True
    r = d.record()
    assert r["expired"] is True and r["stage"] == "s"


def test_a_forecast_counts_the_whole_allocation_not_just_the_stages():
    """Summing stage durations understates what the meter charges for."""
    f = forecast({"E1": 3600, "E2": 7200}, setup=1500, export=900, cap=14400)
    assert f["stage_total_s"] == 10800
    assert f["total_occupied_s"] == 13200
    assert f["fits"] is True and f["headroom_s"] == 1200


def test_a_forecast_that_does_not_fit_says_so():
    f = forecast(
        {"E1": 3600, "E2": 7200, "E3": 3600}, setup=1800, export=900, cap=14400
    )
    assert f["total_occupied_s"] == 17100
    assert f["fits"] is False


@pytest.mark.parametrize(
    "v",
    [
        "NO_SIGNAL",
        "INVALID_EVIDENCE",
        "BENCHMARK_UNINFORMATIVE",
        "NO_LOCAL_REPAIR_SIGNAL",
        "ORACLE_RANKING_ONLY",
    ],
)
def test_no_negative_verdict_name_is_treated_as_advancing(v, tmp_path):
    assert read_verdict(gate(tmp_path, v, verdict=v))["advances"] is False


# -- reading a qualification receipt, not just its verdict -------------------


def qual(tmp_path, name, **body):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(body))
    return str(p)


FULL = {
    "passed": True,
    "envelope_checks": {"relative_frobenius": {"measured": 0.01, "passed": True}},
    "exact_checks": {
        "layer0_keys": True,
        "layer0_values": True,
        "ownership_modes_agree": True,
    },
    "native_prefill_repeats_identically": True,
}


def test_a_complete_passing_receipt_advances(tmp_path):
    from research.kv_translate.stage_control import read_qualification

    assert read_qualification(qual(tmp_path, "ok", **FULL))["passed"] is True


def test_a_receipt_missing_its_exact_checks_does_not_advance(tmp_path):
    """An absent check is not a passing one."""
    from research.kv_translate.stage_control import read_qualification

    body = {k: v for k, v in FULL.items() if k != "exact_checks"}
    r = read_qualification(qual(tmp_path, "thin", **body))
    assert r["passed"] is False and "exact_checks" in r["reason"]


def test_a_declared_pass_with_a_failing_exact_check_does_not_advance(tmp_path):
    """The case the ownership gap would have produced."""
    from research.kv_translate.stage_control import read_qualification

    body = dict(FULL)
    body["exact_checks"] = dict(FULL["exact_checks"], ownership_modes_agree=False)
    r = read_qualification(qual(tmp_path, "own", **body))
    assert r["passed"] is False
    assert r["declared_passed"] is True
    assert r["failed_exact_checks"] == ["ownership_modes_agree"]


def test_a_declared_pass_with_a_failing_envelope_check_does_not_advance(tmp_path):
    from research.kv_translate.stage_control import read_qualification

    body = dict(FULL)
    body["envelope_checks"] = {"continuation_KL": {"measured": 1.0, "passed": False}}
    r = read_qualification(qual(tmp_path, "env", **body))
    assert r["passed"] is False and r["failed_envelope_checks"] == ["continuation_KL"]


def test_an_absent_or_unreadable_receipt_does_not_advance(tmp_path):
    from research.kv_translate.stage_control import read_qualification

    assert read_qualification(str(tmp_path / "nope.json"))["passed"] is False
    torn = tmp_path / "torn.json"
    torn.write_text('{"passed": tr')
    assert read_qualification(str(torn))["passed"] is False


def test_the_native_repeat_answer_is_carried_through(tmp_path):
    from research.kv_translate.stage_control import read_qualification

    body = dict(FULL, native_prefill_repeats_identically=False)
    r = read_qualification(qual(tmp_path, "nr", **body))
    assert r["native_prefill_repeats_identically"] is False


# -- forecasting against occupied time, not stage time -----------------------


def test_a_forecast_in_stage_time_understates_a_cap_in_occupied_time():
    """The error the previous accounting made, shown both ways."""
    from research.kv_translate.stage_control import forecast, inclusive_forecast

    plan = {"RQ": 900, "E3": 3600}
    naive = forecast(plan, setup=0, export=0, cap=14400)
    assert naive["total_occupied_s"] == 4500

    real = inclusive_forecast(
        plan,
        setup=600,
        transfers=400,
        diagnosis=300,
        export=200,
        already_used=7031,
        cap=14400,
        occupancy_multiplier=1.0,
    )
    # 4500 of stage work plus 1500 of overheads nobody times
    assert real["forecast_occupied_s"] == 6000
    assert real["remaining_s"] == 14400 - 7031
    assert real["fits"] is True


def test_an_occupancy_multiplier_can_make_a_plan_stop_fitting():
    from research.kv_translate.stage_control import inclusive_forecast

    plan = {"RQ": 900, "E3": 3600}
    r = inclusive_forecast(
        plan,
        setup=600,
        transfers=400,
        diagnosis=300,
        export=200,
        already_used=7031,
        cap=14400,
        occupancy_multiplier=1.9,
    )
    assert r["fits"] is False
    assert r["shortfall_s"] > 0


def test_a_forecast_that_does_not_fit_reports_the_shortfall():
    from research.kv_translate.stage_control import inclusive_forecast

    r = inclusive_forecast(
        {"E3": 3600},
        setup=600,
        transfers=400,
        diagnosis=300,
        export=200,
        already_used=14000,
        cap=14400,
    )
    assert r["fits"] is False and r["shortfall_s"] == pytest.approx(4700.0)
