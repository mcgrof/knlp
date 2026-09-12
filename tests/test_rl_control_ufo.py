"""Fail-closed X-Plane control-sender tests."""

import io
import json
import os
from pathlib import Path

import pytest

from rl.flight.contracts import ControlCommand, FlightContract, TelemetryFrame
from rl.flight.control_ufo import (
    LIVE_ENVELOPES,
    process_stream,
    reference_action,
    validate_live_envelope,
    zero_action,
)
from rl.flight.ufo_reference import UfoReferenceParameters


class CapturingSink:
    def __init__(self):
        self.messages = []

    def sendall(self, data):
        self.messages.append(data)


def load_contract() -> FlightContract:
    return FlightContract.from_json(
        Path(os.environ["XPLANE_UFO_ROOT"]) / "schemas/ufo-wrench-v1.json"
    )


def telemetry(contract, sequence):
    return TelemetryFrame.create(
        contract,
        episode_id="xplane-test",
        sequence=sequence,
        monotonic_ns=1_000_000 + sequence,
        dt_s=0.02,
        observation=(
            0.0,
            0.0,
            -100.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        goal=(0.0, 0.0, 0.0, 0.0),
    )


def test_zero_sender_answers_each_telemetry_frame():
    contract = load_contract()
    frames = [telemetry(contract, sequence) for sequence in (4, 5)]
    source = io.BytesIO(b"".join(frame.to_wire() for frame in frames))
    sink = CapturingSink()
    output = io.StringIO()
    stats = process_stream(
        source,
        sink,
        output,
        contract,
        zero_action(contract),
        validity_ns=100_000_000,
    )
    assert stats.frames == 2
    assert stats.sent == 2
    assert stats.rejected_telemetry == 0
    assert stats.first_sequence == 4
    assert stats.last_sequence == 5
    assert len(sink.messages) == 2
    for frame, wire in zip(frames, sink.messages):
        command = ControlCommand.from_wire(wire, contract)
        command.validate(
            contract,
            frame=frame,
            now_monotonic_ns=command.issued_monotonic_ns,
        )
        assert command.action == (0.0,) * contract.action.width
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert all(record["kind"] == "control_sent" for record in records)


def test_sender_rejects_replayed_telemetry():
    contract = load_contract()
    frame = telemetry(contract, 4)
    source = io.BytesIO(frame.to_wire() * 2)
    sink = CapturingSink()
    stats = process_stream(
        source,
        sink,
        io.StringIO(),
        contract,
        zero_action(contract),
        validity_ns=100_000_000,
    )
    assert stats.frames == 2
    assert stats.sent == 1
    assert stats.rejected_telemetry == 1
    assert len(sink.messages) == 1


def test_reference_sender_matches_hover_feed_forward():
    contract = load_contract()
    frame = telemetry(contract, 0)
    action = reference_action(contract, UfoReferenceParameters())(frame)
    assert tuple(action[:2]) == (0.0, 0.0)
    assert action[2] == pytest.approx(-11767.98)
    assert tuple(action[3:]) == (0.0, 0.0, 0.0)


def test_reference_sender_accepts_a_wider_acceleration_limit():
    contract = load_contract()
    frame = telemetry(contract, 0)
    faster = TelemetryFrame.create(
        contract,
        episode_id=frame.episode_id,
        sequence=frame.sequence,
        monotonic_ns=frame.monotonic_ns,
        dt_s=frame.dt_s,
        observation=frame.observation,
        goal=(90.0, 0.0, 0.0, 0.0),
    )
    ordinary = reference_action(
        contract,
        UfoReferenceParameters(),
        LIVE_ENVELOPES["showcase"],
    )(faster)
    showcase = reference_action(
        contract,
        UfoReferenceParameters(),
        LIVE_ENVELOPES["showcase"],
        max_acceleration_mps2=24.0,
    )(faster)
    assert showcase[0] == pytest.approx(3.0 * ordinary[0])


def test_nonzero_sender_rejects_state_outside_live_envelope():
    contract = load_contract()
    frame = telemetry(contract, 0)
    values = list(frame.observation)
    values[0] = 20.01
    outside = TelemetryFrame.create(
        contract,
        episode_id=frame.episode_id,
        sequence=frame.sequence,
        monotonic_ns=frame.monotonic_ns,
        dt_s=frame.dt_s,
        observation=values,
        goal=frame.goal,
    )
    with pytest.raises(ValueError, match="horizontal"):
        validate_live_envelope(outside)


def test_combat_envelope_allows_pursuit_but_keeps_tilt_guard():
    contract = load_contract()
    frame = telemetry(contract, 0)
    values = list(frame.observation)
    values[0] = 2500.0
    values[3] = 45.0
    pursuit = TelemetryFrame.create(
        contract,
        episode_id=frame.episode_id,
        sequence=frame.sequence,
        monotonic_ns=frame.monotonic_ns,
        dt_s=frame.dt_s,
        observation=values,
        goal=(45.0, 0.0, 0.0, 0.0),
    )
    validate_live_envelope(pursuit, LIVE_ENVELOPES["combat"])
    with pytest.raises(ValueError, match="horizontal"):
        validate_live_envelope(pursuit, LIVE_ENVELOPES["hover"])

    values[6:10] = (0.965925826, 0.258819045, 0.0, 0.0)
    tilted = TelemetryFrame.create(
        contract,
        episode_id=frame.episode_id,
        sequence=frame.sequence,
        monotonic_ns=frame.monotonic_ns,
        dt_s=frame.dt_s,
        observation=values,
        goal=frame.goal,
    )
    with pytest.raises(ValueError, match="tilt"):
        validate_live_envelope(tilted, LIVE_ENVELOPES["combat"])
