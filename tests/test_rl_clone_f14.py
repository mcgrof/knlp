"""Smoke-test the F-14 clone and portable export pipeline."""

import json
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from rl.clone_f14 import main as clone_main  # noqa: E402
from rl.export_f14_actor import main as export_main  # noqa: E402
from rl.flight.contracts import FlightContract  # noqa: E402
from rl.flight.shadow_ufo import ShadowPolicy  # noqa: E402


def test_clone_and_export_f14_actor(tmp_path):
    run_dir = tmp_path / "run"
    assert (
        clone_main(
            [
                "--run-dir",
                str(run_dir),
                "--steps",
                "256",
                "--epochs",
                "1",
                "--batch-size",
                "64",
                "--dagger-iterations",
                "0",
                "--max-seconds",
                "2",
                "--hidden",
                "16",
                "--torch-threads",
                "1",
            ]
        )
        == 0
    )
    actor = run_dir / "actor.npz"
    assert (
        export_main(
            [
                "--checkpoint",
                str(run_dir / "checkpoint.pt"),
                "--output",
                str(actor),
            ]
        )
        == 0
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["vehicle"] == "Laminar Research F-14D integration target"
    assert manifest["dynamics_fidelity"] == "generic fixed-wing training surrogate"
    contract = FlightContract.from_json(
        Path(__file__).parents[1] / "rl/contracts/fighter-controls-v1.json"
    )
    policy = ShadowPolicy(contract, actor)
    assert policy.checkpoint_sha256 == manifest["checkpoint_sha256"]
