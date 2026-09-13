"""X-Plane bridge tests for a player-led native F-14 formation."""

import math
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from rl.flight.contracts import EnemyPose
from rl.flight.control_f14_formation import (
    OVERRIDE_PLANEPATH,
    f14_player_state,
    formation_pose_writes,
    run_formation,
    write_formation_overrides,
)
from rl.flight.geometry import quaternion_from_euler


def sample(**updates):
    values = {
        "local_x": 1200.0,
        "local_y": 2400.0,
        "local_z": -3600.0,
        "local_vx": 20.0,
        "local_vy": 5.0,
        "local_vz": -180.0,
        "roll_deg": 10.0,
        "pitch_deg": 4.0,
        "heading_deg": 30.0,
        "roll_rate": 0.1,
        "pitch_rate": -0.2,
        "yaw_rate": 0.03,
        "height_agl": 1000.0,
        "on_ground": 0.0,
        "paused": 0.0,
    }
    values.update(updates)
    return values


class FakeClient:
    def __init__(self):
        self.writes = []

    def write_many(self, values):
        self.writes.extend(values)


class FakeLiveClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.subscribed = None

    def subscribe(self, datarefs, frequency_hz):
        self.subscribed = (datarefs, frequency_hz)

    def receive(self, timeout_s):
        assert timeout_s == 0.3
        self.calls += 1
        if self.calls > 1:
            raise KeyboardInterrupt
        return sample()

    def fresh(self, maximum_age_s):
        assert maximum_age_s == 0.25
        return True


class FakeSwarm:
    size = 1
    fighter_contract = SimpleNamespace(nominal_dt_s=0.02)

    def update_state(self, player_state, **frame):
        assert player_state.shape == (13,)
        assert frame["episode_id"].startswith("xplane-f14-formation-")
        return (
            EnemyPose.create(
                slot=0,
                position_ned_m=(100.0, 20.0, -1000.0),
                quaternion_body_to_ned=(1.0, 0.0, 0.0, 0.0),
                native_visual=True,
            ),
        )


def test_player_state_converts_xplane_axes_to_ned():
    state = f14_player_state(sample())
    assert state[:6] == pytest.approx(
        (3600.0, 1200.0, -2400.0, 180.0, 20.0, -5.0)
    )
    assert np.linalg.norm(state[6:10]) == pytest.approx(1.0)
    assert state[10:] == pytest.approx((0.1, -0.2, 0.03))


def test_pose_writes_convert_ned_and_orientation_to_multiplayer_datarefs():
    pose = EnemyPose.create(
        slot=1,
        position_ned_m=(300.0, 400.0, -500.0),
        quaternion_body_to_ned=quaternion_from_euler(
            math.radians(12.0),
            math.radians(-5.0),
            math.radians(275.0),
        ),
        native_visual=True,
    )
    writes = dict(formation_pose_writes((pose,)))
    prefix = "sim/multiplayer/position/plane2"
    assert writes[f"{prefix}_x"] == 400.0
    assert writes[f"{prefix}_y"] == 500.0
    assert writes[f"{prefix}_z"] == -300.0
    assert writes[f"{prefix}_phi"] == pytest.approx(12.0)
    assert writes[f"{prefix}_the"] == pytest.approx(-5.0)
    assert writes[f"{prefix}_psi"] == pytest.approx(275.0)


def test_formation_override_never_touches_player_aircraft():
    client = FakeClient()
    write_formation_overrides(client, 3, True)
    write_formation_overrides(client, 3, False)
    assert client.writes == [
        (f"{OVERRIDE_PLANEPATH}[1]", 1.0),
        (f"{OVERRIDE_PLANEPATH}[2]", 1.0),
        (f"{OVERRIDE_PLANEPATH}[3]", 1.0),
        (f"{OVERRIDE_PLANEPATH}[1]", 0.0),
        (f"{OVERRIDE_PLANEPATH}[2]", 0.0),
        (f"{OVERRIDE_PLANEPATH}[3]", 0.0),
    ]


def test_live_formation_releases_ai_paths_without_touching_player():
    client = FakeLiveClient()
    output = StringIO()
    stats = run_formation(
        client,
        FakeSwarm(),
        output,
        wait_seconds=1.0,
        duration_seconds=None,
    )
    override_writes = [
        write for write in client.writes if write[0].startswith(OVERRIDE_PLANEPATH)
    ]
    assert override_writes == [
        (f"{OVERRIDE_PLANEPATH}[1]", 1.0),
        (f"{OVERRIDE_PLANEPATH}[1]", 0.0),
    ]
    assert all("[0]" not in dataref for dataref, _value in client.writes)
    assert stats.frames == 1
    assert stats.override_releases == 1
    assert stats.interrupted
