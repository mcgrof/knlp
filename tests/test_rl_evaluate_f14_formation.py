"""Tests for the sustained F-14 formation evaluator."""

import argparse

import numpy as np
import pytest

from rl.evaluate_f14_formation import evaluate_turn, parse_turn_rates
from rl.flight.contracts import EnemyPose
from rl.flight.geometry import quaternion_from_euler


class ExactSwarm:
    size = 3
    fighter_contract = type("Contract", (), {"nominal_dt_s": 0.02})()

    @staticmethod
    def _offset(slot):
        return np.asarray((-200.0, 100.0 * (slot - 1), 0.0))

    def update_state(self, player, **frame):
        yaw = frame["sequence"] * frame["dt_s"] * 0.04
        rotation = np.asarray(
            (
                (np.cos(yaw), -np.sin(yaw), 0.0),
                (np.sin(yaw), np.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        return tuple(
            EnemyPose.create(
                slot=slot,
                position_ned_m=player[:3] + rotation @ self._offset(slot),
                quaternion_body_to_ned=quaternion_from_euler(0.0, 0.0, yaw),
            )
            for slot in range(self.size)
        )


def test_parse_turn_rates_rejects_outside_contract():
    assert parse_turn_rates("-0.04,0,0.04") == (-0.04, 0.0, 0.04)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_turn_rates("0.09")


def test_exact_formation_has_zero_error_and_safe_separation():
    result = evaluate_turn(
        ExactSwarm(), 0.04, seconds=0.2, settle_seconds=0.02
    )
    assert result["slot_error_rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["slot_error_p95_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["minimum_pair_separation_m"] == pytest.approx(100.0)
