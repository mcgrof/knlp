"""Closed-loop UFO actor certification tests."""

import json

from rl.certify_ufo import build_verdict, main


def evaluation(*, failures=0, tilt=12.0, velocity=12.0, yaw=0.15):
    episodes = [
        {
            "ground_contact": index < failures,
            "out_of_envelope": False,
            "maximum_tilt_deg": tilt,
        }
        for index in range(100)
    ]
    return {
        "environment": "ufo:maneuver",
        "contract_hash": "test-contract",
        "checkpoint": {"sha256": "checkpoint"},
        "policies": {
            "checkpoint": {
                "summary": {
                    "episodes": 100,
                    "mean_velocity_rmse_mps": velocity,
                    "mean_yaw_rate_rmse_radps": yaw,
                },
                "episodes": episodes,
            },
            "reference": {
                "summary": {
                    "episodes": 100,
                    "mean_velocity_rmse_mps": 10.0,
                    "mean_yaw_rate_rmse_radps": 0.1,
                },
                "episodes": [],
            },
        },
    }


def test_certification_accepts_bounded_actor_close_to_reference():
    verdict = build_verdict(evaluation())
    assert verdict["machine_gate_passed"]
    assert all(verdict["gates"].values())
    assert verdict["observed"]["terminal_failures"] == 0


def test_certification_rejects_any_terminal_failure():
    verdict = build_verdict(evaluation(failures=1))
    assert not verdict["machine_gate_passed"]
    assert not verdict["gates"]["no_terminal_failures"]


def test_certification_rejects_tracking_or_tilt_excess():
    verdict = build_verdict(evaluation(tilt=20.1, velocity=12.6, yaw=0.151))
    assert not verdict["gates"]["tilt"]
    assert not verdict["gates"]["velocity_tracking"]
    assert not verdict["gates"]["yaw_tracking"]


def test_cli_returns_nonzero_and_writes_failed_verdict(tmp_path):
    source = tmp_path / "evaluation.json"
    output = tmp_path / "verdict.json"
    source.write_text(json.dumps(evaluation(failures=1)))
    result = main(["--evaluation", str(source), "--output", str(output)])
    assert result == 3
    assert not json.loads(output.read_text())["machine_gate_passed"]
