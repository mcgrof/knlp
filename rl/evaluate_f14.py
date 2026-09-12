"""Evaluate F-14 motor policies against fixed formation-flight metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl.controls.fighter import fighter_reference_controller
from rl.envs import make_env
from rl.evaluate_ufo import checkpoint_policy, parse_seeds
from rl.ppo import _git_head, _sha256


def evaluate_episode(env, policy, seed: int) -> dict:
    observation, _ = env.reset(seed=seed)
    squared_error_time = np.zeros(3, dtype=np.float64)
    saturated = 0
    steps = 0
    done = False
    info = {}
    while not done:
        action = np.asarray(policy(observation), dtype=np.float64)
        observation, _, terminated, truncated, info = env.step(action)
        error = np.asarray(info["tracking"]) - np.asarray(info["applied_goal"])
        squared_error_time += error * error * env.dt_s
        saturated += int(
            np.any(
                np.isclose(action, env.action_space.low, atol=1e-4)
                | np.isclose(action, env.action_space.high, atol=1e-4)
            )
        )
        steps += 1
        done = bool(terminated or truncated)
    duration = steps * env.dt_s
    rms = np.sqrt(squared_error_time / duration)
    stats = info["episode_stats"]
    return {
        "seed": seed,
        "steps": steps,
        "simulated_seconds": duration,
        "success": bool(stats["success"]),
        "ground_contact": bool(stats["ground_contact"]),
        "out_of_envelope": bool(stats["out_of_envelope"]),
        "airspeed_rmse_mps": float(rms[0]),
        "climb_rate_rmse_mps": float(rms[1]),
        "turn_rate_rmse_radps": float(rms[2]),
        "saturation_fraction": saturated / steps,
    }


def summarize(episodes: list[dict]) -> dict:
    def mean(name: str) -> float:
        return float(np.mean([episode[name] for episode in episodes]))

    return {
        "episodes": len(episodes),
        "success_rate": mean("success"),
        "ground_contact_rate": mean("ground_contact"),
        "out_of_envelope_rate": mean("out_of_envelope"),
        "mean_airspeed_rmse_mps": mean("airspeed_rmse_mps"),
        "mean_climb_rate_rmse_mps": mean("climb_rate_rmse_mps"),
        "mean_turn_rate_rmse_radps": mean("turn_rate_rmse_radps"),
        "mean_saturation_fraction": mean("saturation_fraction"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds(",".join(str(seed) for seed in range(1000, 1100))),
    )
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--knlp-commit")
    args = parser.parse_args(argv)
    env = make_env(
        "f14:formation",
        max_seconds=args.max_seconds,
        random_start=True,
    )
    reference = fighter_reference_controller(env)
    actor, checkpoint, checkpoint_hash, _ = checkpoint_policy(
        env, args.run_dir, args.checkpoint
    )
    try:
        policies = {
            "reference": lambda observation: reference(env),
            "checkpoint": actor,
        }
        results = {
            name: {
                "episodes": [evaluate_episode(env, policy, seed) for seed in args.seeds]
            }
            for name, policy in policies.items()
        }
        for result in results.values():
            result["summary"] = summarize(result["episodes"])
        actor_summary = results["checkpoint"]["summary"]
        reference_summary = results["reference"]["summary"]
        machine_gate = bool(
            actor_summary["ground_contact_rate"] == 0.0
            and actor_summary["out_of_envelope_rate"] == 0.0
            and actor_summary["mean_airspeed_rmse_mps"]
            <= 1.25 * reference_summary["mean_airspeed_rmse_mps"]
            and actor_summary["mean_climb_rate_rmse_mps"]
            <= 1.25 * reference_summary["mean_climb_rate_rmse_mps"]
            and actor_summary["mean_turn_rate_rmse_radps"]
            <= 1.25 * reference_summary["mean_turn_rate_rmse_radps"]
        )
        report = {
            "schema_version": 1,
            "knlp_commit": args.knlp_commit
            or _git_head(Path(__file__).resolve().parents[1]),
            "environment": "f14:formation",
            "dynamics_fidelity": "generic fixed-wing training surrogate",
            "contract_hash": env.contract.digest,
            "dynamics_source_sha256": _sha256(env.dynamics.source_path),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
            },
            "seeds": args.seeds,
            "machine_gate_passed": machine_gate,
            "policies": results,
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
    return 0 if report["machine_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
