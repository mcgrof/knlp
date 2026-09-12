"""Evaluate standalone UFO policies on fixed physical metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from rl.continuous import (
    SquashedGaussianAgent,
    action_bounds,
    load_continuous_state_dict,
)
from rl.controls.ufo import velocity_target_controller, zero_wrench
from rl.envs import make_env
from rl.flight.geometry import quaternion_body_to_ned
from rl.ppo import _git_head, _sha256

Policy = Callable[[np.ndarray], np.ndarray]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("evaluation seeds must be unique")
    return seeds


def checkpoint_policy(env, run_dir: Path, checkpoint: Path | None = None):
    import torch

    args = json.loads((run_dir / "args.json").read_text())
    low, high = action_bounds(env.action_space)
    agent = SquashedGaussianAgent(
        int(np.prod(env.observation_space.shape)),
        low,
        high,
        args.get("hidden", 128),
    )
    path = checkpoint or run_dir / "checkpoint.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("state", {}) if isinstance(saved, dict) else {}
    if state.get("action_kind", "continuous") != "continuous":
        raise ValueError("UFO evaluation requires a continuous-action checkpoint")
    weights = saved["agent"] if isinstance(saved, dict) and "agent" in saved else saved
    load_continuous_state_dict(agent, weights)
    agent.eval()

    def act(observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(observation).unsqueeze(0)
            return agent.act_deterministic(tensor)[0].numpy()

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return act, path, digest, state


def evaluate_episode(env, policy: Policy, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    action_scale = np.maximum(np.abs(env.action_space.low), env.action_space.high)
    previous_action = np.zeros_like(action_scale, dtype=np.float64)
    total_return = 0.0
    velocity_square_time = 0.0
    yaw_square_time = 0.0
    maximum_tilt = 0.0
    saturated = 0
    action_delta = 0.0
    steps = 0
    done = False
    info = {}
    while not done:
        action = np.asarray(policy(observation), dtype=np.float64)
        if action.shape != env.action_space.shape or not np.isfinite(action).all():
            raise ValueError("policy returned an invalid UFO action")
        observation, reward, terminated, truncated, info = env.step(action)
        state = np.asarray(info["state"], dtype=np.float64)
        applied_goal = np.asarray(info["applied_goal"], dtype=np.float64)
        velocity_error = env.body_velocity(state) - applied_goal[:3]
        yaw_error = state[12] - applied_goal[3]
        rotation = quaternion_body_to_ned(state[6:10])
        tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
        normalized = action / action_scale
        velocity_square_time += float(np.dot(velocity_error, velocity_error)) * env.dt_s
        yaw_square_time += float(yaw_error * yaw_error) * env.dt_s
        maximum_tilt = max(maximum_tilt, tilt)
        saturated += int(np.any(np.abs(normalized) >= 0.999))
        action_delta += float(np.linalg.norm((action - previous_action) / action_scale))
        previous_action = action
        total_return += float(reward)
        steps += 1
        done = bool(terminated or truncated)

    duration = steps * env.dt_s
    stats = info["episode_stats"]
    final_state = np.asarray(info["state"], dtype=np.float64)
    final_goal = np.asarray(info["applied_goal"], dtype=np.float64)
    final_velocity_error = float(
        np.linalg.norm(env.body_velocity(final_state) - final_goal[:3])
    )
    final_yaw_rate_error = float(abs(final_state[12] - final_goal[3]))
    return {
        "seed": seed,
        "steps": steps,
        "simulated_seconds": duration,
        "return": total_return,
        "success": bool(stats["success"]),
        "ground_contact": bool(stats["ground_contact"]),
        "out_of_envelope": bool(stats.get("out_of_envelope", 0.0)),
        "velocity_rmse_mps": math.sqrt(velocity_square_time / (duration * 3.0)),
        "yaw_rate_rmse_radps": math.sqrt(yaw_square_time / duration),
        "final_velocity_error_mps": (
            final_velocity_error if math.isfinite(final_velocity_error) else None
        ),
        "final_yaw_rate_error_radps": (
            final_yaw_rate_error if math.isfinite(final_yaw_rate_error) else None
        ),
        "maximum_tilt_deg": math.degrees(maximum_tilt),
        "saturation_fraction": saturated / steps,
        "mean_normalized_action_delta": action_delta / steps,
    }


def summarize(episodes: Sequence[dict]) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([episode[name] for episode in episodes]))

    def finite_mean(name: str) -> float | None:
        values = [episode[name] for episode in episodes if episode[name] is not None]
        return float(np.mean(values)) if values else None

    return {
        "episodes": len(episodes),
        "success_rate": mean("success"),
        "ground_contact_rate": mean("ground_contact"),
        "out_of_envelope_rate": mean("out_of_envelope"),
        "mean_return": mean("return"),
        "mean_velocity_rmse_mps": mean("velocity_rmse_mps"),
        "mean_yaw_rate_rmse_radps": mean("yaw_rate_rmse_radps"),
        "mean_final_velocity_error_mps": finite_mean(
            "final_velocity_error_mps"
        ),
        "mean_final_yaw_rate_error_radps": finite_mean(
            "final_yaw_rate_error_radps"
        ),
        "mean_maximum_tilt_deg": mean("maximum_tilt_deg"),
        "mean_saturation_fraction": mean("saturation_fraction"),
        "mean_normalized_action_delta": mean("mean_normalized_action_delta"),
    }


def evaluate(
    env,
    policies: dict[str, Policy],
    seeds: Sequence[int],
) -> dict[str, dict]:
    results = {}
    for name, policy in policies.items():
        episodes = [evaluate_episode(env, policy, seed) for seed in seeds]
        results[name] = {"summary": summarize(episodes), "episodes": episodes}
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default="ufo:hover",
        choices=[
            "ufo:hover",
            "ufo:forward",
            "ufo:maneuver",
            "ufo:showcase",
        ],
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("1000,1001,1002,1003,1004"))
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["zero", "reference", "checkpoint"],
        default=["zero", "reference", "checkpoint"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if "checkpoint" in args.policies and args.run_dir is None:
        parser.error("--run-dir is required for checkpoint evaluation")
    env = make_env(
        args.env,
        max_seconds=args.max_seconds,
        random_start=args.random_start,
    )
    policies: dict[str, Policy] = {}
    if "zero" in args.policies:
        policies["zero"] = lambda observation: zero_wrench(env)
    if "reference" in args.policies:
        controller = velocity_target_controller(env)
        policies["reference"] = lambda observation: controller(env)
    checkpoint_metadata = None
    if "checkpoint" in args.policies:
        policy, path, digest, state = checkpoint_policy(
            env, args.run_dir, args.checkpoint
        )
        policies["checkpoint"] = policy
        checkpoint_metadata = {
            "path": str(path),
            "sha256": digest,
            "update": state.get("update"),
            "global_step": state.get("global_step"),
        }
    try:
        policy_results = evaluate(env, policies, args.seeds)
        report = {
            "schema_version": 1,
            "knlp_commit": _git_head(Path(__file__).resolve().parents[1]),
            "environment_source_commit": os.environ.get(
                "XPLANE_UFO_SOURCE_COMMIT"
            ),
            "environment": args.env,
            "contract_hash": env.contract.digest,
            "dynamics_library": str(env.dynamics.library_path),
            "dynamics_library_sha256": _sha256(env.dynamics.library_path),
            "max_seconds": args.max_seconds,
            "random_start": args.random_start,
            "seeds": args.seeds,
            "checkpoint": checkpoint_metadata,
            "policies": policy_results,
        }
    finally:
        env.close()

    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
