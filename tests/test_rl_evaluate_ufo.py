"""Fixed-seed physical evaluation for standalone UFO policies."""

import json
import os

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from rl.controls.ufo import VelocityTargetController, zero_wrench
from rl.envs import make_env
from rl.evaluate_ufo import evaluate, main

UFO_ROOT = os.environ.get("XPLANE_UFO_ROOT")
pytestmark = pytest.mark.skipif(not UFO_ROOT, reason="XPLANE_UFO_ROOT is not set")


def test_reference_beats_zero_wrench_on_fixed_hover():
    env = make_env("ufo:hover", root=UFO_ROOT, max_seconds=2.0, random_start=False)
    controller = VelocityTargetController()
    try:
        results = evaluate(
            env,
            {
                "zero": lambda observation: zero_wrench(env),
                "reference": lambda observation: controller(env),
            },
            [7, 11],
        )
    finally:
        env.close()
    zero = results["zero"]["summary"]
    reference = results["reference"]["summary"]
    assert zero["success_rate"] == 0.0
    assert reference["success_rate"] == 1.0
    assert reference["mean_velocity_rmse_mps"] < zero["mean_velocity_rmse_mps"]
    assert (
        reference["mean_final_velocity_error_mps"]
        < zero["mean_final_velocity_error_mps"]
    )
    assert reference["mean_maximum_tilt_deg"] == pytest.approx(0.0)


def test_cli_writes_strict_json_for_reference_policies(tmp_path, monkeypatch):
    monkeypatch.setenv("XPLANE_UFO_ROOT", UFO_ROOT)
    output = tmp_path / "evaluation.json"
    result = main(
        [
            "--policies",
            "zero",
            "reference",
            "--seeds",
            "3,5",
            "--max-seconds",
            "0.5",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["seeds"] == [3, 5]
    assert set(report["policies"]) == {"zero", "reference"}


def test_maneuver_evaluation_uses_the_goal_applied_to_each_step():
    env = make_env(
        "ufo:maneuver",
        root=UFO_ROOT,
        max_seconds=1.0,
        goal_hold_seconds=0.4,
        random_start=False,
    )
    controller = VelocityTargetController()
    try:
        results = evaluate(
            env,
            {
                "zero": lambda observation: zero_wrench(env),
                "reference": lambda observation: controller(env),
            },
            [13],
        )
    finally:
        env.close()
    reference = results["reference"]["summary"]
    zero = results["zero"]["summary"]
    assert reference["out_of_envelope_rate"] == 0.0
    assert reference["mean_velocity_rmse_mps"] < zero["mean_velocity_rmse_mps"]
