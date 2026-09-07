"""Behavioral-cloning warm start for the standalone UFO actor."""

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from rl.clone_ufo import main
from rl.continuous import SquashedGaussianAgent, load_continuous_state_dict

UFO_ROOT = os.environ.get("XPLANE_UFO_ROOT")
pytestmark = pytest.mark.skipif(not UFO_ROOT, reason="XPLANE_UFO_ROOT is not set")


def test_clone_writes_normalized_reloadable_actor(tmp_path):
    run_dir = tmp_path / "clone"
    result = main(
        [
            "--run-dir",
            str(run_dir),
            "--steps",
            "512",
            "--epochs",
            "3",
            "--batch-size",
            "128",
            "--max-seconds",
            "0.5",
            "--hidden",
            "16",
            "--torch-threads",
            "1",
        ]
    )
    assert result == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["method"] == "behavioral_cloning"
    assert manifest["samples"] == 512
    assert manifest["final_validation_loss"] < manifest["initial_validation_loss"]

    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    agent = SquashedGaussianAgent(17, (-1.0,) * 6, (1.0,) * 6, hidden=16)
    load_continuous_state_dict(agent, checkpoint["agent"])
    assert not torch.equal(agent.observation_mean, torch.zeros(17))
    assert torch.all(agent.observation_scale > 0.0)
