"""X-Plane bridge tests for a player-led native F-14 formation."""

import json
import math
from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest

from rl.flight.contracts import EnemyPose
from rl.flight.control_f14_formation import (
    OVERRIDE_PLANEPATH,
    f14_player_state,
    extrapolate_player_values,
    formation_pose_writes,
    formation_datarefs,
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
        "team_status_1": 1.0,
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
        assert 0.0 < timeout_s <= 0.3
        self.calls += 1
        if self.calls > 1:
            raise KeyboardInterrupt
        return sample()

    def fresh(self, maximum_age_s):
        assert maximum_age_s == 0.25
        return True

    def close(self):
        pass


class RetryReleaseClient(FakeLiveClient):
    def __init__(self):
        super().__init__()
        self.release_failures = 0
        self.close_calls = 0

    def write_many(self, values):
        values = tuple(values)
        is_release = values == ((f"{OVERRIDE_PLANEPATH}[1]", 0.0),)
        if is_release and not self.release_failures:
            self.release_failures += 1
            raise ConnectionError("dead formation socket")
        self.writes.extend(values)

    def close(self):
        self.close_calls += 1


class DisconnectClient(FakeLiveClient):
    def receive(self, timeout_s):
        assert 0.0 < timeout_s <= 0.3
        self.calls += 1
        if self.calls > 1:
            raise ConnectionError("X-Plane closed")
        return sample()


class HostileClient(FakeLiveClient):
    def receive(self, timeout_s):
        assert 0.0 < timeout_s <= 0.3
        self.calls += 1
        return sample(team_status_1=2.0)


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


def test_player_sample_is_extrapolated_between_xplane_updates():
    values = sample(
        roll_deg=0.0,
        pitch_deg=0.0,
        heading_deg=30.0,
        roll_rate=0.0,
        pitch_rate=0.0,
        yaw_rate=0.1,
    )
    projected = extrapolate_player_values(values, 0.05)
    assert projected["local_x"] == pytest.approx(1201.0)
    assert projected["local_y"] == pytest.approx(2400.25)
    assert projected["local_z"] == pytest.approx(-3609.0)
    assert projected["heading_deg"] == pytest.approx(
        30.0 + math.degrees(0.005)
    )


def test_formation_subscribes_to_follower_team_status():
    assert formation_datarefs(2)["team_status_2"] == (
        "sim/multiplayer/combat/team_status[2]"
    )


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
    record = json.loads(output.getvalue())
    assert record["schema_version"] == 2
    assert record["team_status"] == [1]
    assert record["player_speed_mps"] == pytest.approx(
        math.sqrt(180.0**2 + 20.0**2 + 5.0**2)
    )
    assert len(record["player_state_ned"]) == 13


def test_live_formation_treats_xplane_disconnect_as_a_clean_stop():
    client = DisconnectClient()
    stats = run_formation(
        client,
        FakeSwarm(),
        StringIO(),
        wait_seconds=1.0,
        duration_seconds=None,
    )
    assert stats.frames == 1
    assert stats.transport_disconnects == 1
    assert stats.override_releases == 1


def test_live_formation_refuses_hostile_ai_slots():
    client = HostileClient()
    stats = run_formation(
        client,
        FakeSwarm(),
        StringIO(),
        wait_seconds=1.0,
        duration_seconds=None,
    )
    assert stats.frames == 0
    assert stats.hostile_samples == 1
    assert not any(
        name.startswith(OVERRIDE_PLANEPATH) for name, _value in client.writes
    )


def test_live_formation_reconnects_to_release_a_dead_socket():
    client = RetryReleaseClient()
    stats = run_formation(
        client,
        FakeSwarm(),
        StringIO(),
        wait_seconds=1.0,
        duration_seconds=None,
    )
    assert client.close_calls == 1
    assert client.release_failures == 1
    assert client.writes[-1] == (f"{OVERRIDE_PLANEPATH}[1]", 0.0)
    assert stats.override_releases == 1
