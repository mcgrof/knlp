"""Gate a UFO actor on closed-loop standalone flight evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path


def _finite_positive(value, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def build_verdict(
    report: dict,
    *,
    minimum_episodes: int = 100,
    maximum_velocity_rmse_ratio: float = 1.25,
    maximum_yaw_rmse_ratio: float = 1.5,
    maximum_tilt_deg: float = 20.0,
) -> dict:
    if minimum_episodes < 1:
        raise ValueError("minimum episodes must be positive")
    maximum_velocity_rmse_ratio = _finite_positive(
        maximum_velocity_rmse_ratio, "maximum velocity RMSE ratio"
    )
    maximum_yaw_rmse_ratio = _finite_positive(
        maximum_yaw_rmse_ratio, "maximum yaw RMSE ratio"
    )
    maximum_tilt_deg = _finite_positive(maximum_tilt_deg, "maximum tilt")
    if report.get("environment") not in {"ufo:maneuver", "ufo:showcase"}:
        raise ValueError(
            "certification requires a maneuver or showcase UFO environment"
        )
    policies = report.get("policies")
    if not isinstance(policies, dict):
        raise TypeError("evaluation report has no policy results")
    try:
        checkpoint = policies["checkpoint"]
        reference = policies["reference"]
        checkpoint_summary = checkpoint["summary"]
        reference_summary = reference["summary"]
        checkpoint_episodes = checkpoint["episodes"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "evaluation must contain checkpoint and reference policies"
        ) from error
    episode_count = int(checkpoint_summary.get("episodes", 0))
    if episode_count != len(checkpoint_episodes):
        raise ValueError("checkpoint episode count does not match its records")

    checkpoint_velocity = _finite_positive(
        checkpoint_summary.get("mean_velocity_rmse_mps"),
        "checkpoint velocity RMSE",
    )
    reference_velocity = _finite_positive(
        reference_summary.get("mean_velocity_rmse_mps"),
        "reference velocity RMSE",
    )
    checkpoint_yaw = _finite_positive(
        checkpoint_summary.get("mean_yaw_rate_rmse_radps"),
        "checkpoint yaw RMSE",
    )
    reference_yaw = _finite_positive(
        reference_summary.get("mean_yaw_rate_rmse_radps"),
        "reference yaw RMSE",
    )
    episode_tilts = [
        float(episode["maximum_tilt_deg"]) for episode in checkpoint_episodes
    ]
    if not episode_tilts or not all(math.isfinite(value) for value in episode_tilts):
        raise ValueError("checkpoint tilt evidence is missing or non-finite")

    velocity_ratio = checkpoint_velocity / reference_velocity
    yaw_ratio = checkpoint_yaw / reference_yaw
    failures = [
        episode
        for episode in checkpoint_episodes
        if episode.get("ground_contact") or episode.get("out_of_envelope")
    ]
    observed_maximum_tilt = max(episode_tilts)
    gates = {
        "coverage": episode_count >= minimum_episodes,
        "no_terminal_failures": not failures,
        "tilt": observed_maximum_tilt <= maximum_tilt_deg,
        "velocity_tracking": velocity_ratio <= maximum_velocity_rmse_ratio,
        "yaw_tracking": yaw_ratio <= maximum_yaw_rmse_ratio,
    }
    return {
        "schema_version": 1,
        "kind": "ufo_closed_loop_certification",
        "knlp_commit": report.get("knlp_commit"),
        "environment_source_commit": report.get(
            "environment_source_commit"
        ),
        "environment": report["environment"],
        "contract_hash": report.get("contract_hash"),
        "dynamics_library": report.get("dynamics_library"),
        "dynamics_library_sha256": report.get(
            "dynamics_library_sha256"
        ),
        "checkpoint": report.get("checkpoint"),
        "thresholds": {
            "minimum_episodes": minimum_episodes,
            "maximum_tilt_deg": maximum_tilt_deg,
            "maximum_velocity_rmse_ratio": maximum_velocity_rmse_ratio,
            "maximum_yaw_rmse_ratio": maximum_yaw_rmse_ratio,
        },
        "observed": {
            "episodes": episode_count,
            "terminal_failures": len(failures),
            "maximum_tilt_deg": observed_maximum_tilt,
            "velocity_rmse_ratio": velocity_ratio,
            "yaw_rmse_ratio": yaw_ratio,
        },
        "gates": gates,
        "machine_gate_passed": all(gates.values()),
        "scope": (
            "This gate covers the standalone dynamics only. It does not arm "
            "or authorize live X-Plane control."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=100)
    parser.add_argument("--maximum-velocity-rmse-ratio", type=float, default=1.25)
    parser.add_argument("--maximum-yaw-rmse-ratio", type=float, default=1.5)
    parser.add_argument("--maximum-tilt-deg", type=float, default=20.0)
    args = parser.parse_args(argv)
    report = json.loads(args.evaluation.read_text())
    verdict = build_verdict(
        report,
        minimum_episodes=args.minimum_episodes,
        maximum_velocity_rmse_ratio=args.maximum_velocity_rmse_ratio,
        maximum_yaw_rmse_ratio=args.maximum_yaw_rmse_ratio,
        maximum_tilt_deg=args.maximum_tilt_deg,
    )
    encoded = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded)
    print(json.dumps(verdict, sort_keys=True), flush=True)
    return 0 if verdict["machine_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
