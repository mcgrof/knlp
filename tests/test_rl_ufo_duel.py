"""Headless actor-versus-enemy UFO duel tests."""

import math
import os

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from rl.ufo_duel import (
    DuelRules,
    pursuit_goal,
    reference_policy_factory,
    relative_geometry,
    run_duel,
)

UFO_ROOT = os.environ.get("XPLANE_UFO_ROOT")


def test_pursuit_goal_stays_inside_the_certified_envelope():
    goal = pursuit_goal(
        10_000.0,
        math.pi,
        np.asarray((10_000.0, 5000.0, -1000.0)),
        100.0,
    )
    assert np.all(goal >= np.asarray((-10.0, -25.0, -12.0, -1.0)))
    assert np.all(goal <= np.asarray((45.0, 25.0, 12.0, 1.0)))


def test_relative_geometry_uses_each_fighter_local_origin():
    state = np.asarray(
        (0.0, 0.0, -100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    distance, bearing, body = relative_geometry(
        state,
        np.asarray((-1500.0, 0.0, 0.0)),
        state,
        np.asarray((1500.0, 0.0, 0.0)),
    )
    assert distance == pytest.approx(3000.0)
    assert bearing == pytest.approx(0.0)
    assert body == pytest.approx((3000.0, 0.0, 0.0))


@pytest.mark.skipif(not UFO_ROOT, reason="XPLANE_UFO_ROOT is not set")
def test_two_reference_pilots_resolve_a_headless_duel():
    result = run_duel(
        reference_policy_factory,
        7000,
        root=UFO_ROOT,
        rules=DuelRules(maximum_seconds=45.0),
    )
    assert result["outcome"] in {"kill", "mutual_kill"}
    assert result["closest_range_m"] <= 900.0
    assert max(result["maximum_tilt_deg"]) <= 20.0


@pytest.mark.skipif(not UFO_ROOT, reason="XPLANE_UFO_ROOT is not set")
def test_time_limit_is_not_reported_as_a_flight_failure():
    def zero_factory(environment):
        return lambda observation: np.zeros(6, dtype=np.float32)

    result = run_duel(
        zero_factory,
        7000,
        root=UFO_ROOT,
        rules=DuelRules(maximum_seconds=0.1),
    )
    assert result["outcome"] == "timeout"
    assert result["winner"] is None


@pytest.mark.skipif(not UFO_ROOT, reason="XPLANE_UFO_ROOT is not set")
def test_actor_and_enemy_policies_load_independently():
    loaded = []

    def factory(role):
        def load(environment):
            loaded.append((role, id(environment)))
            return lambda observation: np.zeros(6, dtype=np.float32)

        return load

    run_duel(
        factory("actor"),
        7000,
        enemy_policy_factory=factory("enemy"),
        root=UFO_ROOT,
        rules=DuelRules(maximum_seconds=0.1),
    )
    assert [role for role, _ in loaded] == ["actor", "enemy"]
    assert loaded[0][1] != loaded[1][1]
