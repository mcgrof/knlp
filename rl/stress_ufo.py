"""Stress a UFO actor at every corner of the combat goal envelope."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from pathlib import Path

from rl.controls.ufo import VelocityTargetController
from rl.envs.ufo_env import UfoEnv
from rl.evaluate_ufo import (
    checkpoint_policy,
    evaluate_episode,
    parse_seeds,
    summarize,
)
from rl.ppo import _git_head

COMBAT_GOAL_AXES = (
    (-10.0, 45.0),
    (-25.0, 25.0),
    (-12.0, 0.0),
    (-1.0, 1.0),
)


def combat_corner_goals() -> list[tuple[float, ...]]:
    """Return the Cartesian corners of the director goal envelope."""

    return list(itertools.product(*COMBAT_GOAL_AXES))


def _positive(value, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def attach_verdict(
    report: dict,
    *,
    maximum_velocity_rmse_ratio: float = 1.25,
    maximum_yaw_rmse_ratio: float = 1.5,
    maximum_tilt_deg: float = 20.0,
) -> dict:
    """Add fail-closed combat-corner gates to a stress report."""

    maximum_velocity_rmse_ratio = _positive(
        maximum_velocity_rmse_ratio,
        "maximum velocity RMSE ratio",
    )
    maximum_yaw_rmse_ratio = _positive(
        maximum_yaw_rmse_ratio,
        "maximum yaw RMSE ratio",
    )
    maximum_tilt_deg = _positive(maximum_tilt_deg, "maximum tilt")
    try:
        checkpoint = report["policies"]["checkpoint"]
        reference = report["policies"]["reference"]
        checkpoint_summary = checkpoint["summary"]
        reference_summary = reference["summary"]
        checkpoint_episodes = checkpoint["episodes"]
        reference_episodes = reference["episodes"]
        expected_episodes = len(report["goals"]) * len(report["seeds"])
    except (KeyError, TypeError) as error:
        raise ValueError("stress report is missing required evidence") from error

    velocity_ratio = _positive(
        checkpoint_summary.get("mean_velocity_rmse_mps"),
        "checkpoint velocity RMSE",
    ) / _positive(
        reference_summary.get("mean_velocity_rmse_mps"),
        "reference velocity RMSE",
    )
    yaw_ratio = _positive(
        checkpoint_summary.get("mean_yaw_rate_rmse_radps"),
        "checkpoint yaw RMSE",
    ) / _positive(
        reference_summary.get("mean_yaw_rate_rmse_radps"),
        "reference yaw RMSE",
    )
    tilts = [float(episode["maximum_tilt_deg"]) for episode in checkpoint_episodes]
    if not tilts or not all(math.isfinite(value) for value in tilts):
        raise ValueError("checkpoint tilt evidence is missing or non-finite")
    terminal_failures = sum(
        bool(episode.get("ground_contact") or episode.get("out_of_envelope"))
        for episode in checkpoint_episodes
    )
    maximum_tilt = max(tilts)
    gates = {
        "coverage": (
            len(checkpoint_episodes) == expected_episodes
            and len(reference_episodes) == expected_episodes
        ),
        "no_terminal_failures": terminal_failures == 0,
        "tilt": maximum_tilt <= maximum_tilt_deg,
        "velocity_tracking": velocity_ratio <= maximum_velocity_rmse_ratio,
        "yaw_tracking": yaw_ratio <= maximum_yaw_rmse_ratio,
    }
    report.update(
        {
            "thresholds": {
                "maximum_tilt_deg": maximum_tilt_deg,
                "maximum_velocity_rmse_ratio": maximum_velocity_rmse_ratio,
                "maximum_yaw_rmse_ratio": maximum_yaw_rmse_ratio,
            },
            "observed": {
                "episodes": len(checkpoint_episodes),
                "terminal_failures": terminal_failures,
                "maximum_tilt_deg": maximum_tilt,
                "velocity_rmse_ratio": velocity_ratio,
                "yaw_rmse_ratio": yaw_ratio,
            },
            "gates": gates,
            "machine_gate_passed": all(gates.values()),
        }
    )
    return report


def evaluate_corners(
    run_dir: Path,
    seeds: Sequence[int],
    *,
    checkpoint: Path | None = None,
    max_seconds: float = 20.0,
) -> dict:
    knlp_commit = _git_head(Path(__file__).resolve().parents[1])
    goals = combat_corner_goals()
    records = {"checkpoint": [], "reference": []}
    actor = None
    checkpoint_info = None
    contract_hash = None
    for goal in goals:
        env = UfoEnv(
            goal=goal,
            max_seconds=max_seconds,
            random_start=True,
        )
        try:
            if actor is None:
                actor, path, digest, state = checkpoint_policy(
                    env,
                    run_dir,
                    checkpoint,
                )
                checkpoint_info = {
                    "path": str(path),
                    "sha256": digest,
                    **state,
                }
                contract_hash = env.contract.digest
            controller = VelocityTargetController()
            for seed in seeds:
                policies = {
                    "checkpoint": actor,
                    "reference": lambda observation, controller=controller, env=env: (
                        controller(env)
                    ),
                }
                for name, policy in policies.items():
                    episode = evaluate_episode(env, policy, seed)
                    episode["goal"] = list(goal)
                    records[name].append(episode)
        finally:
            env.close()
    report = {
        "schema_version": 1,
        "kind": "ufo_combat_corner_stress",
        "knlp_commit": knlp_commit,
        "environment": "ufo:combat-corners",
        "contract_hash": contract_hash,
        "checkpoint": checkpoint_info,
        "max_seconds": max_seconds,
        "random_start": True,
        "seeds": list(seeds),
        "goals": [list(goal) for goal in goals],
        "policies": {
            name: {
                "summary": summarize(episodes),
                "episodes": episodes,
            }
            for name, episodes in records.items()
        },
        "scope": (
            "This gate covers standalone dynamics at the combat director "
            "goal corners. It does not authorize live X-Plane control."
        ),
    }
    return attach_verdict(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("6000,6001,6002"),
    )
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_corners(
        args.run_dir,
        args.seeds,
        checkpoint=args.checkpoint,
        max_seconds=args.max_seconds,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report["observed"], sort_keys=True), flush=True)
    return 0 if report["machine_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
