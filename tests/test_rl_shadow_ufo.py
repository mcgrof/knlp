"""Read-only X-Plane shadow-policy transport tests."""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl.continuous import SquashedGaussianAgent
from rl.export_ufo_actor import export_checkpoint
from rl.flight.contracts import FlightContract, TelemetryFrame
from rl.flight.shadow_ufo import ShadowPolicy, process_stream


def load_contract() -> FlightContract:
    root = os.environ.get("XPLANE_UFO_ROOT")
    if not root:
        pytest.skip("XPLANE_UFO_ROOT is not set")
    return FlightContract.from_json(Path(root) / "schemas/ufo-wrench-v1.json")


def test_shadow_stream_emits_actions_but_no_control_wire(tmp_path):
    contract = load_contract()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "args.json").write_text(json.dumps({"hidden": 16}))
    agent = SquashedGaussianAgent(
        contract.observation.width + contract.goal.width,
        contract.action.low,
        contract.action.high,
        hidden=16,
    )
    torch.save(
        {"agent": agent.state_dict(), "state": {"action_kind": "continuous"}},
        run_dir / "checkpoint.pt",
    )
    model = run_dir / "actor.npz"
    export_checkpoint(contract, run_dir / "checkpoint.pt", model)
    policy = ShadowPolicy(contract, model)
    frames = []
    for sequence in (10, 11):
        frames.append(
            TelemetryFrame.create(
                contract,
                episode_id="xplane-test",
                sequence=sequence,
                monotonic_ns=sequence * 1_000_000,
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
        )
    source = io.BytesIO(b"".join(frame.to_wire() for frame in frames))
    output = io.StringIO()
    stats = process_stream(source, output, policy)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert stats.frames == 2
    assert stats.rejected == 0
    assert stats.first_sequence == 10
    assert stats.last_sequence == 11
    assert all(record["kind"] == "shadow_action" for record in records)
    assert all("proposed_action" in record for record in records)
    assert all(record["telemetry"]["kind"] == "telemetry" for record in records)
    assert all(record["kind"] != "control" for record in records)
    first_values = np.asarray(
        (*frames[0].observation, *frames[0].goal), dtype=np.float32
    )
    with torch.no_grad():
        expected = agent.act_deterministic(torch.as_tensor(first_values).unsqueeze(0))
    assert np.allclose(records[0]["proposed_action"], expected[0].numpy(), atol=1e-5)


def test_shadow_stream_rejects_replayed_sequence(tmp_path):
    contract = load_contract()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "args.json").write_text(json.dumps({"hidden": 8}))
    agent = SquashedGaussianAgent(
        contract.observation.width + contract.goal.width,
        contract.action.low,
        contract.action.high,
        hidden=8,
    )
    torch.save(
        {"agent": agent.state_dict(), "state": {"action_kind": "continuous"}},
        run_dir / "checkpoint.pt",
    )
    model = run_dir / "actor.npz"
    export_checkpoint(contract, run_dir / "checkpoint.pt", model)
    policy = ShadowPolicy(contract, model)
    frame = TelemetryFrame.create(
        contract,
        episode_id="xplane-test",
        sequence=4,
        monotonic_ns=4_000_000,
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
    output = io.StringIO()
    stats = process_stream(io.BytesIO(frame.to_wire() * 2), output, policy)
    assert stats.frames == 1
    assert stats.rejected == 1
    assert not stats.interrupted


def test_shadow_stream_preserves_stats_on_interrupt(tmp_path):
    class InterruptedSource:
        def readline(self, size):
            raise KeyboardInterrupt

    contract = load_contract()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent = SquashedGaussianAgent(
        contract.observation.width + contract.goal.width,
        contract.action.low,
        contract.action.high,
        hidden=8,
    )
    torch.save(
        {"agent": agent.state_dict(), "state": {"action_kind": "continuous"}},
        run_dir / "checkpoint.pt",
    )
    model = run_dir / "actor.npz"
    export_checkpoint(contract, run_dir / "checkpoint.pt", model)
    policy = ShadowPolicy(contract, model)
    stats = process_stream(InterruptedSource(), io.StringIO(), policy)
    assert stats.frames == 0
    assert stats.interrupted
