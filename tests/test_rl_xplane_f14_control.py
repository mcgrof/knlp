"""X-Plane wire and coordinate tests for the learned F-14 player."""

import json
import math
import struct
from io import StringIO
from pathlib import Path

import pytest

from rl.flight.control_f14 import (
    AILERON,
    ELEVATOR,
    OVERRIDE_JOYSTICK,
    OVERRIDE_THROTTLES,
    RUDDER,
    THROTTLE_0,
    THROTTLE_1,
    F14LiveLimiter,
    F14LiveReference,
    f14_observation,
    run_controller,
    safe_handoff,
    sample_rate_hz,
    showcase_goal,
    write_action,
    write_overrides,
)
from rl.flight.geometry import quaternion_body_to_ned
from rl.flight.contracts import FlightContract
from rl.flight.xplane_udp import (
    DREF_HEADER,
    RREF_HEADER,
    RREF_RESPONSE_HEADER,
    pack_dref_write,
    pack_rref_request,
    unpack_rref_response,
)
from rl.flight.xplane_web import XPlaneWeb


class FakeClient:
    def __init__(self):
        self.writes = []

    def write(self, dataref, value):
        self.writes.append((dataref, value))

    def write_many(self, values):
        self.writes.extend(values)


class FakeLiveClient(FakeClient):
    def __init__(self, samples):
        super().__init__()
        self.samples = iter(samples)
        self.subscribed = None

    def subscribe(self, datarefs, frequency_hz):
        self.subscribed = (datarefs, frequency_hz)

    def receive(self, timeout_s):
        del timeout_s
        return next(self.samples)

    def fresh(self, maximum_age_s):
        del maximum_age_s
        return True


class FakePolicy:
    def infer(self, frame):
        del frame
        return [0.7, -0.2, 0.3, -0.4], 123


class FakeReconnectClient(XPlaneWeb):
    def __init__(self, samples):
        self.samples = iter(samples)
        self.writes = []
        self.subscribe_count = 0
        self.close_count = 0

    def subscribe(self, datarefs, frequency_hz):
        self.subscribed = (datarefs, frequency_hz)
        self.subscribe_count += 1

    def receive(self, timeout_s):
        del timeout_s
        value = next(self.samples)
        if isinstance(value, Exception):
            raise value
        return value

    def fresh(self, maximum_age_s):
        del maximum_age_s
        return True

    def write_many(self, values):
        self.writes.extend(values)

    def close(self):
        self.close_count += 1


def sample(**updates):
    values = {
        "local_vx": 0.0,
        "local_vy": 0.0,
        "local_vz": -180.0,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "heading_deg": 0.0,
        "roll_rate": 0.0,
        "pitch_rate": 0.0,
        "yaw_rate": 0.0,
        "height_agl": 1500.0,
        "on_ground": 0.0,
        "paused": 0.0,
    }
    values.update(updates)
    return values


def test_xplane_packets_have_exact_legacy_wire_layout():
    request = pack_rref_request("sim/time/paused", 7, 50)
    assert len(request) == 413
    assert request[:5] == RREF_HEADER
    assert struct.unpack_from("<ii", request, 5) == (50, 7)
    assert request[13:].split(b"\0", 1)[0] == b"sim/time/paused"

    write = pack_dref_write("sim/joystick/yoke_roll_ratio", -0.25)
    assert len(write) == 509
    assert write[:5] == DREF_HEADER
    assert struct.unpack_from("<f", write, 5)[0] == pytest.approx(-0.25)
    assert write[9:].split(b"\0", 1)[0] == b"sim/joystick/yoke_roll_ratio"


def test_rref_response_parser_rejects_invalid_samples():
    packet = RREF_RESPONSE_HEADER + struct.pack("<ifif", 7, 12.5, 8, -3.0)
    assert unpack_rref_response(packet) == {7: 12.5, 8: -3.0}
    with pytest.raises(ValueError, match="not an RREF"):
        unpack_rref_response(b"DATA\0")
    with pytest.raises(ValueError, match="malformed"):
        unpack_rref_response(RREF_RESPONSE_HEADER + b"x")
    with pytest.raises(ValueError, match="invalid sample"):
        unpack_rref_response(
            RREF_RESPONSE_HEADER + struct.pack("<if", 9, math.nan)
        )


def test_xplane_state_converts_to_fighter_ned_contract():
    observation = f14_observation(
        sample(
            local_vx=30.0,
            local_vy=8.0,
            local_vz=-170.0,
            roll_deg=10.0,
            pitch_deg=5.0,
            heading_deg=90.0,
            roll_rate=0.1,
            pitch_rate=-0.2,
            yaw_rate=0.03,
        )
    )
    assert observation[:3] == (170.0, 30.0, -8.0)
    rotation = quaternion_body_to_ned(observation[3:7])
    assert rotation[:, 0] == pytest.approx((0.0, 0.9961947, -0.0871557))
    assert observation[7:] == (0.1, -0.2, 0.03)


def test_xplane_state_converts_to_yaw_invariant_live_contract():
    observation = f14_observation(
        sample(
            local_vx=30.0,
            local_vy=8.0,
            local_vz=-170.0,
            roll_deg=10.0,
            pitch_deg=5.0,
            heading_deg=90.0,
            roll_rate=0.1,
            pitch_rate=-0.2,
            yaw_rate=0.03,
        ),
        revision=2,
    )
    assert observation == pytest.approx(
        (
            math.sqrt(170.0**2 + 30.0**2 + 8.0**2),
            8.0,
            math.radians(10.0),
            math.radians(5.0),
            0.1,
            -0.2,
            0.03,
        )
    )


@pytest.mark.parametrize(
    "updates",
    (
        {"local_vz": -80.0},
        {"height_agl": 29.0},
        {"on_ground": 1.0},
        {"paused": 1.0},
    ),
)
def test_f14_handoff_rejects_unsafe_live_state(updates):
    assert not safe_handoff(sample(**updates))


@pytest.mark.parametrize(
    "updates",
    (
        {"local_vz": -80.0},
        {"local_vz": -510.0},
        {"local_vy": 121.0},
        {"height_agl": 49.0},
        {"height_agl": 58.0, "local_vy": -6.0},
        {"roll_deg": 121.0},
        {"pitch_deg": 71.0},
        {"pitch_rate": 2.01},
    ),
)
def test_f14_initial_handoff_rejects_outside_airborne_envelope(updates):
    assert not safe_handoff(sample(**updates), initial=True)


def test_f14_initial_handoff_accepts_observed_fast_climb():
    assert safe_handoff(
        sample(
            local_vx=270.08,
            local_vy=31.96,
            local_vz=-141.77,
            height_agl=481.5,
            roll_deg=16.38,
            pitch_deg=6.12,
            roll_rate=0.095,
            pitch_rate=0.008,
            yaw_rate=0.009,
        ),
        initial=True,
    )


def test_f14_initial_handoff_accepts_airborne_formation_spawn():
    assert safe_handoff(
        sample(
            local_vx=-80.0,
            local_vy=19.0,
            local_vz=-85.0,
            height_agl=58.0,
            roll_deg=-24.0,
            roll_rate=0.61,
        ),
        initial=True,
    )


def test_live_limiter_blends_bounds_and_slew_limits_actor_actions():
    limiter = F14LiveLimiter()
    assert limiter.apply((0.0, 1.0, -1.0, 1.0), 1_000_000_000) == (
        0.9,
        0.0,
        0.0,
        0.0,
    )
    second = limiter.apply((0.0, 1.0, -1.0, 1.0), 2_000_000_000)
    assert second == pytest.approx((0.8125, 0.2, -0.2, 0.2))
    third = limiter.apply((1.0, 1.0, -1.0, 1.0), 3_000_000_000)
    assert third == pytest.approx((0.9, 0.25, -0.25, 0.25))


def test_live_reference_protects_speed_and_corrects_attitude():
    controller = F14LiveReference()
    action = controller.action(
        sample(
            local_vz=-160.0,
            local_vy=-8.0,
            roll_deg=8.0,
            pitch_deg=-4.0,
            roll_rate=0.1,
            pitch_rate=-0.1,
        ),
        (180.0, 0.0, 0.0),
    )
    assert action[0] == 1.0
    assert action[1] < 0.0
    assert action[2] > 0.0


def test_showcase_builds_altitude_before_aggressive_maneuvers():
    assert showcase_goal((180.0, 9.0, 0.1), 1.9, 200.0) == (
        180.0, 0.0, 0.0
    )
    assert showcase_goal((180.0, 9.0, 0.1), 2.0, 200.0) == (
        180.0, 25.0, 0.025
    )
    assert showcase_goal((180.0, 9.0, 0.1), 2.0, 500.0) == (
        180.0, 15.0, 0.040
    )
    assert showcase_goal((180.0, 9.0, 0.1), 10.0, 500.0) == (
        180.0, 5.0, -0.050
    )


def test_sample_rate_reports_transport_cadence():
    assert sample_rate_hz(tuple(range(0, 1_000_000_000, 100_000_000))) == 10.0
    assert sample_rate_hz((0, 280_000_000, 560_000_000)) == pytest.approx(
        3.57142857
    )


def test_f14_control_owns_and_releases_only_control_axes():
    client = FakeClient()
    write_overrides(client, True)
    write_action(client, (0.7, -0.2, 0.3, -0.4))
    write_overrides(client, False)
    assert client.writes == [
        (OVERRIDE_JOYSTICK, 1),
        (OVERRIDE_THROTTLES, 1),
        (THROTTLE_0, 0.7),
        (THROTTLE_1, 0.7),
        (AILERON, -0.2),
        (ELEVATOR, 0.3),
        (RUDDER, -0.4),
        (OVERRIDE_JOYSTICK, 0),
        (OVERRIDE_THROTTLES, 0),
    ]


def test_live_controller_releases_overrides_on_unsafe_transition():
    contract = FlightContract.from_json(
        Path(__file__).resolve().parents[1]
        / "rl/contracts/fighter-controls-v1.json"
    )
    client = FakeLiveClient(
        [sample() for _ in range(10)] + [sample(height_agl=29.0)]
    )
    output = StringIO()
    stats = run_controller(
        client,
        contract,
        FakePolicy(),
        (180.0, 0.0, 0.0),
        output,
        wait_seconds=1.0,
        duration_seconds=None,
    )
    assert stats.release_reason.startswith("guarded envelope exit:")
    assert client.subscribed[1] == 50
    assert client.writes[:2] == [
        (OVERRIDE_JOYSTICK, 1),
        (OVERRIDE_THROTTLES, 1),
    ]
    assert client.writes[-2:] == [
        (OVERRIDE_JOYSTICK, 0),
        (OVERRIDE_THROTTLES, 0),
    ]
    record = json.loads(output.getvalue())
    assert record["kind"] == "xplane_f14_control"
    assert record["actor_action"] == [0.7, -0.2, 0.3, -0.4]
    assert record["action"] == [0.9, 0.0, 0.0, 0.0]
    assert record["controller"] == "live_reference_actor_shadow"


def test_live_controller_reconnects_before_acquiring_overrides():
    contract = FlightContract.from_json(
        Path(__file__).resolve().parents[1]
        / "rl/contracts/fighter-controls-v2.json"
    )
    client = FakeReconnectClient(
        [ConnectionError("closed")]
        + [sample() for _ in range(10)]
        + [sample(height_agl=29.0)]
    )
    stats = run_controller(
        client,
        contract,
        FakePolicy(),
        (180.0, 0.0, 0.0),
        StringIO(),
        wait_seconds=2.0,
        duration_seconds=None,
    )
    assert stats.release_reason.startswith("guarded envelope exit:")
    assert client.subscribe_count == 2
    assert client.close_count == 1
