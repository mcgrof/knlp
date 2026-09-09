"""Behavior-clone the deterministic UFO controller into a continuous actor."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rl.continuous import SquashedGaussianAgent, action_bounds
from rl.controls.ufo import VelocityTargetController
from rl.envs import make_env
from rl.ppo import _git_head, _sha256


def collect_reference(env, steps: int, seed: int):
    observations = np.empty((steps, *env.observation_space.shape), dtype=np.float32)
    actions = np.empty((steps, *env.action_space.shape), dtype=np.float32)
    controller = VelocityTargetController()
    episode_seed = seed
    observation, _ = env.reset(seed=episode_seed)
    episodes = 0
    for index in range(steps):
        action = controller(env)
        observations[index] = observation
        actions[index] = action
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            episodes += 1
            episode_seed += 1
            observation, _ = env.reset(seed=episode_seed)
    return observations, actions, episodes


def collect_dagger(
    env,
    agent,
    steps: int,
    seed: int,
    expert_probability: float,
):
    """Collect expert labels on states visited by a learner/expert mixture."""

    observations = np.empty((steps, *env.observation_space.shape), dtype=np.float32)
    actions = np.empty((steps, *env.action_space.shape), dtype=np.float32)
    controller = VelocityTargetController()
    generator = np.random.default_rng(seed)
    episode_seed = seed
    observation, _ = env.reset(seed=episode_seed)
    episodes = 0
    learner_steps = 0
    for index in range(steps):
        expert_action = controller(env)
        observations[index] = observation
        actions[index] = expert_action
        with torch.no_grad():
            learner_action = (
                agent.act_deterministic(torch.as_tensor(observation).unsqueeze(0))[0]
                .cpu()
                .numpy()
            )
        if generator.random() < expert_probability:
            behavior_action = expert_action
        else:
            behavior_action = learner_action
            learner_steps += 1
        observation, _, terminated, truncated, _ = env.step(behavior_action)
        if terminated or truncated:
            episodes += 1
            episode_seed += 1
            observation, _ = env.reset(seed=episode_seed)
    return observations, actions, episodes, learner_steps


def cloning_loss(agent, observations, actions) -> torch.Tensor:
    predicted = agent.act_deterministic(observations)
    normalized_error = (predicted - actions) / agent.action_scale
    return normalized_error.square().mean()


def train_actor(
    agent,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    update_normalization: bool = True,
):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(observations), generator=generator)
    validation_size = max(1, len(observations) // 10)
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    observation_tensor = torch.as_tensor(observations)
    action_tensor = torch.as_tensor(actions)

    if update_normalization:
        training_observations = observation_tensor[training_indices]
        mean = training_observations.mean(dim=0)
        scale = training_observations.std(dim=0).clamp_min(1e-3)
        agent.set_observation_normalization(mean, scale)
    else:
        mean = agent.observation_mean.detach().cpu()
        scale = agent.observation_scale.detach().cpu()
    with torch.no_grad():
        agent.actor_log_std.fill_(-2.0)

    optimizer = torch.optim.Adam(
        agent.actor_mean.parameters(), lr=learning_rate, eps=1e-5
    )
    validation_observations = observation_tensor[validation_indices]
    validation_actions = action_tensor[validation_indices]
    with torch.no_grad():
        initial_validation_loss = float(
            cloning_loss(agent, validation_observations, validation_actions)
        )
    rows = []
    for epoch in range(1, epochs + 1):
        shuffled = training_indices[
            torch.randperm(len(training_indices), generator=generator)
        ]
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            loss = cloning_loss(
                agent, observation_tensor[indices], action_tensor[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        with torch.no_grad():
            validation_loss = float(
                cloning_loss(agent, validation_observations, validation_actions)
            )
        rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / batches,
                "validation_loss": validation_loss,
            }
        )
    return rows, initial_validation_loss, mean.numpy(), scale.numpy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--env",
        default="ufo:hover",
        choices=["ufo:hover", "ufo:forward", "ufo:maneuver"],
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dagger-iterations", type=int, default=0)
    parser.add_argument("--dagger-steps", type=int, default=10_000)
    parser.add_argument("--dagger-epochs", type=int, default=5)
    parser.add_argument("--dagger-expert-probability", type=float, default=0.5)
    parser.add_argument("--dagger-expert-decay", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--environment-source-commit",
        default=os.environ.get("XPLANE_UFO_SOURCE_COMMIT"),
    )
    args = parser.parse_args(argv)
    if args.steps < 10 or args.epochs < 1 or args.batch_size < 1:
        parser.error("steps, epochs, and batch size must be positive")
    if args.dagger_iterations < 0 or args.dagger_steps < 1 or args.dagger_epochs < 1:
        parser.error("DAgger iterations, steps, and epochs must be valid")
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
    env = make_env(args.env, max_seconds=args.max_seconds, random_start=True)
    try:
        observations, actions, episodes = collect_reference(env, args.steps, args.seed)
        low, high = action_bounds(env.action_space)
        agent = SquashedGaussianAgent(
            int(np.prod(env.observation_space.shape)), low, high, args.hidden
        )
        rows, initial_loss, mean, scale = train_actor(
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
                )
            )
            observations = np.concatenate((observations, new_observations))
            actions = np.concatenate((actions, new_actions))
            dagger_rows, _, mean, scale = train_actor(
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
                    "update": 0,
                    "global_step": 0,
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
            )
            writer.writeheader()
            writer.writerows(rows)
        library_path = env.dynamics.library_path
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "knlp_commit": _git_head(Path(__file__).resolve().parents[1]),
            "environment_source_commit": args.environment_source_commit,
            "contract_hash": env.contract.digest,
            "dynamics_library_sha256": _sha256(library_path),
            "environment": args.env,
            "method": (
                "dagger_behavioral_cloning"
                if args.dagger_iterations
                else "behavioral_cloning"
            ),
            "reference": "VelocityTargetController",
            "samples": len(observations),
            "reference_samples": args.steps,
            "reference_episodes": episodes,
            "dagger_iterations": args.dagger_iterations,
            "dagger_rollouts": dagger_rollouts,
            "model_parameters": sum(parameter.numel() for parameter in agent.parameters()),
            "observation_mean": mean.tolist(),
            "observation_scale": scale.tolist(),
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
        f"[clone-ufo] samples={len(observations)} epochs={len(rows)} "
        f"validation={initial_loss:.6g}->{rows[-1]['validation_loss']:.6g} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
