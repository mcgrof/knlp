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
    f14_observation,
    run_controller,
    safe_handoff,
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


@pytest.mark.parametrize(
    "updates",
    (
        {"local_vz": -99.0},
        {"height_agl": 149.0},
        {"on_ground": 1.0},
        {"paused": 1.0},
    ),
)
def test_f14_handoff_rejects_unsafe_live_state(updates):
    assert not safe_handoff(sample(**updates))


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
        [sample() for _ in range(10)] + [sample(height_agl=100.0)]
    )
    output = StringIO()
    with pytest.raises(RuntimeError, match="left the live actor"):
        run_controller(
            client,
            contract,
            FakePolicy(),
            (180.0, 0.0, 0.0),
            output,
            wait_seconds=1.0,
            duration_seconds=None,
        )
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
    assert record["action"] == [0.7, -0.2, 0.3, -0.4]
