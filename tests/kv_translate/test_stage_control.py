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
