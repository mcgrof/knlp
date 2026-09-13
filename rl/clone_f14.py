"""Train an F-14 motor actor by cloning the fixed-wing reference."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rl.clone_ufo import collect_dagger, collect_reference, train_actor
from rl.continuous import SquashedGaussianAgent, action_bounds
from rl.controls.fighter import fighter_reference_controller
from rl.envs import make_env
from rl.envs.fighter_env import DEFAULT_CONTRACT
from rl.ppo import _git_head, _sha256


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dagger-iterations", type=int, default=4)
    parser.add_argument("--dagger-steps", type=int, default=10_000)
    parser.add_argument("--dagger-epochs", type=int, default=5)
    parser.add_argument("--dagger-expert-probability", type=float, default=0.5)
    parser.add_argument("--dagger-expert-decay", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--knlp-commit")
    args = parser.parse_args(argv)
    if args.steps < 10 or args.epochs < 1 or args.batch_size < 1:
        parser.error("steps, epochs, and batch size must be positive")
    if args.dagger_iterations < 0 or args.dagger_steps < 1:
        parser.error("DAgger iterations and steps must be valid")
    if args.dagger_epochs < 1:
        parser.error("DAgger epochs must be positive")
    if not 0.0 <= args.dagger_expert_probability <= 1.0:
        parser.error("DAgger expert probability must be in [0, 1]")
    if not 0.0 <= args.dagger_expert_decay <= 1.0:
        parser.error("DAgger expert decay must be in [0, 1]")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error(f"run directory is not empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    knlp_commit = args.knlp_commit or _git_head(Path(__file__).resolve().parents[1])
    env = make_env(
        "f14:formation",
        contract_path=args.contract,
        max_seconds=args.max_seconds,
        random_start=True,
    )
    try:
        observations, actions, episodes = collect_reference(
            env,
            args.steps,
            args.seed,
            fighter_reference_controller,
        )
        low, high = action_bounds(env.action_space)
        agent = SquashedGaussianAgent(
            int(np.prod(env.observation_space.shape)),
            low,
            high,
            args.hidden,
        )
        rows, initial_loss, mean, scale, action_loss_scale = train_actor(
            agent,
            observations,
            actions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        for row in rows:
            row.update(
                stage="reference",
                iteration=0,
                dataset_samples=len(observations),
                rollout_episodes=episodes,
                rollout_expert_probability=1.0,
                rollout_learner_fraction=0.0,
            )
        dagger_rollouts = []
        for iteration in range(1, args.dagger_iterations + 1):
            expert_probability = args.dagger_expert_probability * (
                args.dagger_expert_decay ** (iteration - 1)
            )
            new_observations, new_actions, rollout_episodes, learner_steps = (
                collect_dagger(
                    env,
                    agent,
                    args.dagger_steps,
                    args.seed + 100_000 * iteration,
                    expert_probability,
                    fighter_reference_controller,
                )
            )
            observations = np.concatenate((observations, new_observations))
            actions = np.concatenate((actions, new_actions))
            dagger_rows, _, mean, scale, action_loss_scale = train_actor(
                agent,
                observations,
                actions,
                epochs=args.dagger_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + iteration,
                update_normalization=False,
            )
            learner_fraction = learner_steps / args.dagger_steps
            for row in dagger_rows:
                row.update(
                    stage="dagger",
                    iteration=iteration,
                    dataset_samples=len(observations),
                    rollout_episodes=rollout_episodes,
                    rollout_expert_probability=expert_probability,
                    rollout_learner_fraction=learner_fraction,
                )
            rows.extend(dagger_rows)
            dagger_rollouts.append(
                {
                    "iteration": iteration,
                    "samples": args.dagger_steps,
                    "episodes": rollout_episodes,
                    "expert_probability": expert_probability,
                    "learner_fraction": learner_fraction,
                }
            )

        checkpoint_path = args.run_dir / "checkpoint.pt"
        torch.save(
            {
                "agent": agent.state_dict(),
                "state": {
                    "action_kind": "continuous",
                    "clone_samples": len(observations),
                    "reference_samples": args.steps,
                    "dagger_iterations": args.dagger_iterations,
                },
            },
            checkpoint_path,
        )
        arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        (args.run_dir / "args.json").write_text(
            json.dumps(arguments, indent=2, sort_keys=True) + "\n"
        )
        with (args.run_dir / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "stage",
                    "iteration",
                    "epoch",
                    "dataset_samples",
                    "rollout_episodes",
                    "rollout_expert_probability",
                    "rollout_learner_fraction",
                    "train_loss",
                    "validation_loss",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        dynamics_source = env.dynamics.source_path
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "knlp_commit": knlp_commit,
            "environment": "f14:formation",
            "vehicle": "Laminar Research F-14D integration target",
            "dynamics_fidelity": "generic fixed-wing training surrogate",
            "contract_hash": env.contract.digest,
            "dynamics_source": str(dynamics_source),
            "dynamics_source_sha256": _sha256(dynamics_source),
            "method": (
                "dagger_behavioral_cloning"
                if args.dagger_iterations
                else "behavioral_cloning"
            ),
            "reference": "FighterReferenceController",
            "samples": len(observations),
            "reference_samples": args.steps,
            "reference_episodes": episodes,
            "dagger_iterations": args.dagger_iterations,
            "dagger_rollouts": dagger_rollouts,
            "model_parameters": sum(
                parameter.numel() for parameter in agent.parameters()
            ),
            "observation_mean": mean.tolist(),
            "observation_scale": scale.tolist(),
            "action_loss_scale": action_loss_scale.tolist(),
            "initial_validation_loss": initial_loss,
            "final_validation_loss": rows[-1]["validation_loss"],
            "checkpoint_sha256": _sha256(checkpoint_path),
            "wall_seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        (args.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    finally:
        env.close()
    print(
        f"[clone-f14] samples={len(observations)} epochs={len(rows)} "
        f"validation={initial_loss:.6g}->{rows[-1]['validation_loss']:.6g} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
