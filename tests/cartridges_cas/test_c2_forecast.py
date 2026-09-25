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


def test_at_the_only_rate_this_lane_has_measured_the_stage_does_not_fit():
    """Evaluation alone exceeds the remaining reserve on the previous hardware.

    This is the finding, not a fixture: three hundred and twenty generations at
    the earlier screen's decode rate costs more than the whole reserve has left,
    before any training, fitting or setup, and the receiver here is twice the
    size of the model that produced that rate.
    """
    out = c2_forecast.build(MEASURED_4B_TOKENS_PER_SECOND, 1.0)
    assert out["fits"] is False
    assert out["verdict"] == c2_forecast.BLOCKED
    fc = out["forecast_at_token_ceiling"]
    assert fc["remaining_s"] == pytest.approx(11485.4)
    assert fc["shortfall_s"] > 6000


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
    assert "can only shrink" in out["remaining_note"]


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
