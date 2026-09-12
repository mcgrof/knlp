"""Run an actor and an enemy UFO motor policy in a headless duel."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rl.controls.ufo import velocity_target_controller
from rl.envs.ufo_env import UfoEnv
from rl.evaluate_ufo import checkpoint_policy, parse_seeds
from rl.flight.geometry import quaternion_body_to_ned, quaternion_from_euler
from rl.ppo import _git_head, _sha256

MotorPolicy = Callable[[np.ndarray], np.ndarray]
MotorPolicyFactory = Callable[[UfoEnv], MotorPolicy]


@dataclass(frozen=True)
class DuelRules:
    initial_range_m: float = 3000.0
    maximum_seconds: float = 45.0
    firing_range_m: float = 900.0
    firing_bearing_deg: float = 12.0
    firing_height_m: float = 250.0
    firing_dwell_s: float = 0.3
    maximum_tilt_deg: float = 20.0

    def validate(self) -> None:
        values = asdict(self)
        valid = all(
            math.isfinite(value) and value > 0.0
            for value in values.values()
        )
        if not valid:
            raise ValueError("duel rules must be finite and positive")
        if self.firing_range_m >= self.initial_range_m:
            raise ValueError("firing range must be below the initial range")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def relative_geometry(
    ego_state: np.ndarray,
    ego_origin: np.ndarray,
    target_state: np.ndarray,
    target_origin: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return range, body bearing, and body-relative target displacement."""

    delta_ned = target_origin + target_state[:3] - ego_origin - ego_state[:3]
    target_body = quaternion_body_to_ned(ego_state[6:10]).T @ delta_ned
    target_range = float(np.linalg.norm(target_body))
    bearing_rad = math.atan2(float(target_body[1]), float(target_body[0]))
    return target_range, bearing_rad, target_body


def pursuit_goal(
    target_range_m: float,
    target_bearing_rad: float,
    target_body: np.ndarray,
    weave: float,
) -> np.ndarray:
    """Produce a certified-envelope goal for the reusable motor actor."""

    bearing_deg = math.degrees(target_bearing_rad)
    return np.asarray(
        (
            _clamp((target_range_m - 650.0) * 0.03, -10.0, 45.0),
            _clamp(bearing_deg * 0.3 + weave, -25.0, 25.0),
            _clamp(float(target_body[2]) * 0.03, -12.0, 12.0),
            _clamp(target_bearing_rad * 1.5, -1.0, 1.0),
        ),
        dtype=np.float64,
    )


def _maximum_tilt_deg(state: np.ndarray) -> float:
    rotation = quaternion_body_to_ned(state[6:10])
    upright = float(np.clip(rotation[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(upright))


def run_duel(
    actor_policy_factory: MotorPolicyFactory,
    seed: int,
    *,
    enemy_policy_factory: MotorPolicyFactory | None = None,
    root: str | Path | None = None,
    rules: DuelRules | None = None,
) -> dict:
    """Simulate one duel with independently loaded actor and enemy policies."""

    rules = rules or DuelRules()
    rules.validate()
    environments = [
        UfoEnv(
            root=root,
            max_seconds=rules.maximum_seconds,
            random_start=False,
        )
        for _ in range(2)
    ]
    enemy_factory = enemy_policy_factory or actor_policy_factory
    policies = [
        actor_policy_factory(environments[0]),
        enemy_factory(environments[1]),
    ]
    generator = np.random.default_rng(seed)
    origins = [
        np.asarray((-rules.initial_range_m / 2.0, 0.0, 0.0)),
        np.asarray((rules.initial_range_m / 2.0, 0.0, 0.0)),
    ]
    headings = (
        generator.uniform(-math.radians(12.0), math.radians(12.0)),
        math.pi
        + generator.uniform(-math.radians(12.0), math.radians(12.0)),
    )
    fire_steps = [0, 0]
    required_fire_steps = math.ceil(
        rules.firing_dwell_s / environments[0].dt_s
    )
    closest_range = rules.initial_range_m
    maximum_tilt = [0.0, 0.0]
    flight_failures = [False, False]
    outcome = "timeout"
    winner = None
    elapsed_s = 0.0
    try:
        for index, environment in enumerate(environments):
            environment.reset(seed=seed * 2 + index)
            attitude = quaternion_from_euler(0.0, 0.0, headings[index])
            for axis, value in enumerate(attitude):
                environment.dynamics.state.quaternion_body_to_ned[axis] = value

        while elapsed_s < rules.maximum_seconds:
            geometries = []
            for index in range(2):
                other = 1 - index
                geometries.append(
                    relative_geometry(
                        environments[index].dynamics.state_vector(),
                        origins[index],
                        environments[other].dynamics.state_vector(),
                        origins[other],
                    )
                )
            closest_range = min(
                closest_range,
                geometries[0][0],
                geometries[1][0],
            )
            actions = []
            for index, environment in enumerate(environments):
                target_range, target_bearing, target_body = geometries[index]
                phase = elapsed_s * 0.7 + index * math.pi
                weave = 6.0 * math.sin(phase)
                environment.goal = pursuit_goal(
                    target_range,
                    target_bearing,
                    target_body,
                    weave,
                )
                actions.append(policies[index](environment._observation()))

            failed = []
            timed_out = []
            for index, environment in enumerate(environments):
                _, _, terminated, truncated, info = environment.step(
                    actions[index]
                )
                state = np.asarray(info["state"], dtype=np.float64)
                maximum_tilt[index] = max(
                    maximum_tilt[index],
                    _maximum_tilt_deg(state),
                )
                failed.append(bool(terminated))
                timed_out.append(bool(truncated))
            elapsed_s += environments[0].dt_s
            if any(failed):
                flight_failures = failed
                outcome = "flight_failure"
                if failed[0] != failed[1]:
                    winner = 1 if failed[0] else 0
                break
            if any(timed_out):
                break

            hits = []
            for index in range(2):
                other = 1 - index
                target_range, target_bearing, target_body = relative_geometry(
                    environments[index].dynamics.state_vector(),
                    origins[index],
                    environments[other].dynamics.state_vector(),
                    origins[other],
                )
                in_solution = (
                    target_range <= rules.firing_range_m
                    and abs(math.degrees(target_bearing))
                    <= rules.firing_bearing_deg
                    and abs(float(target_body[2])) <= rules.firing_height_m
                )
                fire_steps[index] = fire_steps[index] + 1 if in_solution else 0
                hits.append(fire_steps[index] >= required_fire_steps)
            if any(hits):
                if all(hits):
                    outcome = "mutual_kill"
                else:
                    outcome = "kill"
                    winner = 0 if hits[0] else 1
                break
    finally:
        for environment in environments:
            environment.close()

    return {
        "seed": seed,
        "outcome": outcome,
        "winner": winner,
        "elapsed_s": elapsed_s,
        "closest_range_m": closest_range,
        "maximum_tilt_deg": maximum_tilt,
        "flight_failures": flight_failures,
    }


def summarize_duels(
    episodes: Sequence[dict],
    rules: DuelRules,
) -> tuple[dict, dict, bool]:
    """Summarize outcomes and fail closed on each fighter's safety."""

    resolved = sum(
        episode["outcome"] in {"kill", "mutual_kill"}
        for episode in episodes
    )
    safety = {
        role: {
            "terminal_failures": sum(
                episode["flight_failures"][index]
                for episode in episodes
            ),
            "tilt_violations": sum(
                episode["maximum_tilt_deg"][index]
                > rules.maximum_tilt_deg
                for episode in episodes
            ),
            "maximum_tilt_deg": max(
                episode["maximum_tilt_deg"][index]
                for episode in episodes
            ),
        }
        for index, role in enumerate(("actor", "enemy"))
    }
    fighter_gates = {
        role: not values["terminal_failures"]
        and not values["tilt_violations"]
        for role, values in safety.items()
    }
    summary = {
        "episodes": len(episodes),
        "resolved": resolved,
        "flight_failures": sum(
            episode["outcome"] == "flight_failure"
            for episode in episodes
        ),
        "timeouts": sum(
            episode["outcome"] == "timeout" for episode in episodes
        ),
        "wins": {
            "actor": sum(
                episode["winner"] == 0 for episode in episodes
            ),
            "enemy": sum(
                episode["winner"] == 1 for episode in episodes
            ),
        },
        "safety": safety,
    }
    gates = {
        "all_resolved": resolved == len(episodes),
        "fighters": fighter_gates,
    }
    return summary, gates, gates["all_resolved"] and all(fighter_gates.values())


def reference_policy_factory(environment: UfoEnv) -> MotorPolicy:
    controller = velocity_target_controller(environment)
    return lambda observation: controller(environment)


def checkpoint_policy_factory(
    run_dir: Path,
    checkpoint: Path | None = None,
) -> MotorPolicyFactory:
    def load(environment: UfoEnv) -> MotorPolicy:
        policy, _, _, _ = checkpoint_policy(environment, run_dir, checkpoint)
        return policy

    return load


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor-run-dir",
        "--run-dir",
        dest="actor_run_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--actor-checkpoint",
        "--checkpoint",
        dest="actor_checkpoint",
        type=Path,
    )
    parser.add_argument(
        "--enemy-run-dir",
        type=Path,
        help="enemy motor run; defaults to the actor run",
    )
    parser.add_argument("--enemy-checkpoint", type=Path)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("7000,7001,7002,7003,7004"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rules = DuelRules()
    enemy_run_dir = args.enemy_run_dir or args.actor_run_dir
    actor_checkpoint = (
        args.actor_checkpoint or args.actor_run_dir / "checkpoint.pt"
    )
    enemy_checkpoint = (
        args.enemy_checkpoint or enemy_run_dir / "checkpoint.pt"
    )
    probe = UfoEnv(max_seconds=rules.maximum_seconds, random_start=False)
    try:
        contract_hash = probe.contract.digest
        dynamics_path = probe.dynamics.library_path
        dynamics_hash = _sha256(dynamics_path)
    finally:
        probe.close()
    episodes = [
        run_duel(
            checkpoint_policy_factory(
                args.actor_run_dir,
                args.actor_checkpoint,
            ),
            seed,
            enemy_policy_factory=checkpoint_policy_factory(
                enemy_run_dir,
                args.enemy_checkpoint,
            ),
            rules=rules,
        )
        for seed in args.seeds
    ]
    summary, gates, machine_gate_passed = summarize_duels(episodes, rules)
    report = {
        "schema_version": 2,
        "kind": "ufo_actor_enemy_duel",
        "knlp_commit": _git_head(Path(__file__).resolve().parents[1]),
        "environment_source_commit": os.environ.get(
            "XPLANE_UFO_SOURCE_COMMIT"
        ),
        "contract_hash": contract_hash,
        "dynamics_library": str(dynamics_path),
        "dynamics_library_sha256": dynamics_hash,
        "fighters": {
            "actor": {
                "run_dir": str(args.actor_run_dir),
                "checkpoint": {
                    "path": str(actor_checkpoint),
                    "sha256": _sha256(actor_checkpoint),
                },
            },
            "enemy": {
                "run_dir": str(enemy_run_dir),
                "checkpoint": {
                    "path": str(enemy_checkpoint),
                    "sha256": _sha256(enemy_checkpoint),
                },
            },
        },
        "rules": asdict(rules),
        "seeds": args.seeds,
        "episodes": episodes,
        "summary": summary,
        "gates": gates,
        "machine_gate_passed": machine_gate_passed,
        "scope": (
            "This compares independently loaded actor and enemy motor "
            "policies in shared headless dynamics under deterministic "
            "tactics. It does not yet control an X-Plane AI aircraft."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if machine_gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
