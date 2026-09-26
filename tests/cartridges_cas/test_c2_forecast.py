# SPDX-License-Identifier: GPL-2.0
"""The occupied-time forecast, and the two guesses it must refuse to make.

The lane's accounting error was structural, not arithmetic: a forecast in stage
time compared against a cap in occupied time. So the tests here are mostly
about what the tool declines to assume.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from research.cartridges_cas.scripts import c2_forecast
from research.kv_translate.stage_control import inclusive_forecast
from tests.cartridges_cas.entry_guard import assert_entry_point_is_sound

# The screen that ran before this one decoded a four-billion-parameter model at
# this rate on an eager path. It is the only measured rate this lane has.
MEASURED_4B_TOKENS_PER_SECOND = 25918 / 2914.6


def test_the_reserve_arithmetic_this_stage_is_judged_against():
    """A flat ten-thousand-eight-hundred-second stage, at three multipliers."""
    base = dict(
        setup=0.0,
        transfers=0.0,
        diagnosis=0.0,
        export=0.0,
        already_used=2914.6,
        cap=14400.0,
    )
    fits = inclusive_forecast({"C2": 10800}, occupancy_multiplier=1.0, **base)
    assert fits["fits"] is True
    assert round(fits["headroom_s"], 1) == 685.4

    measured = inclusive_forecast({"C2": 10800}, occupancy_multiplier=1.39, **base)
    assert measured["fits"] is False
    assert round(measured["shortfall_s"], 1) == 3526.6

    docstring_rate = inclusive_forecast({"C2": 10800}, occupancy_multiplier=1.9, **base)
    assert docstring_rate["fits"] is False
    assert round(docstring_rate["shortfall_s"], 1) == 9034.6


def test_the_evaluation_leg_is_priced_from_a_measured_rate():
    g = c2_forecast.generation_seconds(10.0, 320, 400, 163840)
    assert g["seconds_at_token_ceiling"] == pytest.approx(16384.0)
    assert g["seconds_at_mean_length"] == pytest.approx(12800.0)


def test_the_borrowed_rate_does_not_fit_but_proves_nothing_about_a_device():
    """A sensitivity, not a benchmark.

    Priced at the earlier screen's effective output rate the plan does not fit.
    That rate is NOT a decode rate for this receiver: it came from a model half
    the size, it mixes prompt and prefill work into the token count, and it
    contains no backward pass at all. So this says the plan is expensive at that
    price, and says nothing about whether any particular card can meet it. Only
    a qualification on the actual receiver can say that.
    """
    out = c2_forecast.build(MEASURED_4B_TOKENS_PER_SECOND, 1.0)
    assert out["fits"] is False
    assert out["verdict"] == c2_forecast.BLOCKED
    assert "not a qualification of any device" in out["not_a_device_measurement"]


def test_the_reserve_charges_the_failed_acquisition_too():
    """A failed allocation is occupied time and belongs to this lane."""
    out = c2_forecast.build(60.0, 1.0)
    assert out["recorded_debits_s"] == {
        "c1_local_screen": 2914.6,
        "c1_failed_acquisition": 1020.0,
    }
    assert out["forecast_at_token_ceiling"]["remaining_s"] == pytest.approx(10465.4)


def test_the_stage_cap_binds_when_it_is_smaller_than_the_residual():
    """C2 is limited by min(its own cap, what the reserve actually has left)."""
    out = c2_forecast.build(60.0, 1.0, cap=99999.0, already_used=0.0)
    assert out["binding_limit_s"] == c2_forecast.STAGE_CAP_S
    assert out["binding_limit_is"] == "the stage cap"
    # 163,840 / 10 = 16,384 s: over the 10,800 s stage cap, well under the
    # inflated reserve, so only the cap can be what refuses it.
    huge = c2_forecast.build(10.0, 1.0, cap=99999.0, already_used=0.0)
    assert huge["fits_reserve"] is True and huge["fits_stage_cap"] is False
    assert huge["fits"] is False and huge["verdict"] == c2_forecast.BLOCKED


def test_a_fast_enough_receiver_makes_it_fit():
    out = c2_forecast.build(60.0, 1.4, fit_s=600, train_s=1800)
    assert out["fits"] is True
    assert out["verdict"] is None


def test_fitting_only_at_the_mean_length_is_flagged_rather_than_claimed():
    """A plan that works only if nothing hits the cap is not a plan."""
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if c2_forecast.build(mid, 1.0)["fits"]:
            hi = mid
        else:
            lo = mid
    just_under = c2_forecast.build(lo, 1.0)
    assert just_under["fits"] is False
    assert just_under["fits_only_at_mean_length"] is True


def test_the_remaining_figure_is_labelled_an_upper_bound():
    out = c2_forecast.build(60.0, 1.0)
    assert out["remaining_is_an_upper_bound"] is True
    assert "may be treated as zero" in out["remaining_note"]
    for debt in c2_forecast.UNRESOLVED_DEBITS:
        assert debt in out["remaining_note"]


def run_cli(*args):
    return subprocess.run(
        [sys.executable, c2_forecast.__file__, *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_the_cli_refuses_without_a_measured_occupancy_multiplier():
    """inclusive_forecast treats a missing multiplier as 1.0, silently.

    That is precisely the understatement the function exists to prevent, so the
    multiplier cannot have a default here.
    """
    r = run_cli("--tokens-per-second", "60")
    assert r.returncode != 0
    assert "occupancy-multiplier" in r.stderr


def test_the_cli_refuses_without_a_measured_generation_rate():
    r = run_cli("--occupancy-multiplier", "1.4")
    assert r.returncode != 0
    assert "tokens-per-second" in r.stderr


def test_the_cli_exits_nonzero_when_the_stage_does_not_fit():
    r = run_cli(
        "--tokens-per-second",
        f"{MEASURED_4B_TOKENS_PER_SECOND}",
        "--occupancy-multiplier",
        "1.0",
    )
    assert r.returncode == 1
    assert c2_forecast.BLOCKED in r.stdout


def test_the_cli_exits_zero_when_it_fits():
    r = run_cli("--tokens-per-second", "60", "--occupancy-multiplier", "1.4")
    assert r.returncode == 0
    assert "fits" in r.stdout


def test_the_entry_point_can_actually_be_executed():
    assert_entry_point_is_sound(c2_forecast)
