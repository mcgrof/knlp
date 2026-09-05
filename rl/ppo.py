"""Single-file PPO for discrete actions, in the CleanRL style.

Everything that matters for reading the algorithm is in this file:
rollout collection, generalised advantage estimation, the clipped
surrogate with value and entropy terms, minibatch updates, and the
diagnostics (approximate KL, clip fraction, explained variance).
What is added over the reference implementation is operational:
checkpoints that resume exactly, a wall-clock budget, and a yield
protocol so a run stops at an update boundary when the GPU is
reclaimed (see rl/pace) and picks up again later.

    python -m rl.ppo --env sim:sim_trees --total-timesteps 200000 --run-name sim-a
    python -m rl.ppo --env etr:bunny_hill --num-envs 8 --device cuda --run-name bh-seed1
    python -m rl.ppo --run-name bh-seed1 --resume            # continue from the checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from rl.envs import env_factory
from rl.pace.lease import EXIT_YIELD, GpuLease, YieldRequest
from rl.vec import SyncVec


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--env", default="sim:sim_trees", help="<backend>:<course>")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default=os.environ.get("KNLP_RL_RUNS", "runs/rl"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=0)
    # PPO (CleanRL defaults for discrete control)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--hidden", type=int, default=128)
    # environment contract knobs (passed through; defaults are the frozen v0)
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--stuck-seconds", type=float, default=15.0)
    p.add_argument("--action-set", default="v0", choices=["v0", "v1"])
    p.add_argument(
        "--random-start",
        action="store_true",
        help="begin training episodes at a randomly chosen course reset point",
    )
    p.add_argument("--etr-bin", default=None, help="path to the patched etr binary")
    # operations
    p.add_argument(
        "--resume", action="store_true", help="continue from the run's checkpoint"
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=10, help="updates between checkpoints"
    )
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="stop (resumable) after this much wall clock",
    )
    p.add_argument(
        "--no-lease", action="store_true", help="ignore the GPU lease (tests)"
    )
    p.add_argument("--lease-dir", default=None)
    p.add_argument("--log-every", type=int, default=1)
    args = p.parse_args(argv)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_updates = max(1, args.total_timesteps // args.batch_size)
    if args.run_name is None:
        args.run_name = f"{args.env.replace(':', '-')}-s{args.seed}"
    return args


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def act_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x).argmax(dim=-1)


class RunLog:
    """CSV metrics plus a small JSON progress file that survives restarts."""

    FIELDS = [
        "update",
        "global_step",
        "wall_s",
        "sps",
        "lr",
        "value_loss",
        "policy_loss",
        "entropy",
        "approx_kl",
        "clipfrac",
        "explained_var",
        "episodes",
        "ep_return",
        "ep_len",
        "ep_finish",
        "ep_progress",
        "ep_time",
        "ep_herring",
        "ep_collisions",
    ]

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.csv_path = run_dir / "metrics.csv"
        new = not self.csv_path.exists()
        self._f = open(self.csv_path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        if new:
            self._w.writeheader()

    def write(self, row: dict) -> None:
        self._w.writerow(row)
        self._f.flush()
        with open(self.run_dir / "progress.json", "w") as f:
            json.dump(row, f, indent=2, sort_keys=True)

    def close(self) -> None:
        self._f.close()


def pick_device(name: str) -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: Path, agent: Agent, optimizer, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "state": state,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        tmp,
    )
    os.replace(tmp, path)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    lease = None if args.no_lease else GpuLease(args.lease_dir)
    yield_req = YieldRequest(lease) if lease else None
    if lease:
        lease.register(
            f"ppo:{args.run_name}",
            role="worker",
            run_dir=str(run_dir),
            device=str(device),
        )
        yield_req.install_signals()

    env_kwargs = {
        "max_seconds": args.max_seconds,
        "stuck_seconds": args.stuck_seconds,
        "action_set": args.action_set,
        "random_start": args.random_start,
        "binary": args.etr_bin,
    }
    if args.env.startswith("etr"):
        env_kwargs["stderr_path"] = str(run_dir / "etr.stderr.log")
    envs = SyncVec([env_factory(args.env, **env_kwargs) for _ in range(args.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)

    agent = Agent(obs_dim, n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    n_params = sum(p.numel() for p in agent.parameters())

    state = {"update": 0, "global_step": 0, "wall_s": 0.0, "env_seed": args.seed}
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        optimizer.load_state_dict(ck["optimizer"])
        state = ck["state"]
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        print(
            f"[ppo] resumed {args.run_name} at update {state['update']} step {state['global_step']}"
        )
    elif args.resume:
        print(f"[ppo] no checkpoint at {ckpt_path}; starting fresh", flush=True)

    log = RunLog(run_dir)
    print(
        f"[ppo] run={args.run_name} env={args.env} device={device} params={n_params} "
        f"envs={args.num_envs} steps={args.num_steps} batch={args.batch_size} updates={args.num_updates}"
    )

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    act_buf = torch.zeros(
        (args.num_steps, args.num_envs), device=device, dtype=torch.long
    )
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    # Each (re)start seeds the environments from a distinct base so a
    # resumed run does not replay the same episodes.
    env_seed = state["env_seed"] + 1000 * state["update"]
    next_obs = torch.tensor(envs.reset(seed=env_seed), device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    session_start = time.time()
    wall_before = state["wall_s"]
    global_step = state["global_step"]
    update = state["update"]
    exit_code = 0

    def checkpoint() -> None:
        state.update(
            update=update,
            global_step=global_step,
            wall_s=wall_before + time.time() - session_start,
        )
        save_checkpoint(ckpt_path, agent, optimizer, state)
        if lease:
            lease.update(step=global_step, update=update)

    try:
        while update < args.num_updates:
            if yield_req and yield_req.check():
                print(f"[ppo] yielding: {yield_req.reason}", flush=True)
                checkpoint()
                exit_code = EXIT_YIELD
                break
            if (
                args.max_hours is not None
                and (wall_before + time.time() - session_start) > args.max_hours * 3600
            ):
                print("[ppo] wall-clock budget reached; checkpointing", flush=True)
                checkpoint()
                exit_code = EXIT_YIELD
                break

            update += 1
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            episodes: list[dict] = []
            t_roll = time.time()
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs_buf[step] = next_obs
                done_buf[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    val_buf[step] = value.flatten()
                act_buf[step] = action
                logp_buf[step] = logprob
                nobs, reward, done, infos = envs.step(action.cpu().numpy())
                rew_buf[step] = torch.tensor(reward, device=device)
                next_obs = torch.tensor(nobs, device=device)
                next_done = torch.tensor(done, device=device, dtype=torch.float32)
                episodes.extend(infos["episodes"])
            roll_s = time.time() - t_roll

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rew_buf, device=device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = val_buf[t + 1]
                    delta = (
                        rew_buf[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - val_buf[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + val_buf

            b_obs = obs_buf.reshape((-1, obs_dim))
            b_logp = logp_buf.reshape(-1)
            b_act = act_buf.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)
            b_val = val_buf.reshape(-1)

            t_learn = time.time()
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    mb = b_inds[start : start + args.minibatch_size]
                    _, newlogp, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb], b_act[mb]
                    )
                    logratio = newlogp - b_logp[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        )
                    mb_adv = b_adv[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_unclipped = (newvalue - b_ret[mb]) ** 2
                        v_clipped = b_val[mb] + torch.clamp(
                            newvalue - b_val[mb], -args.clip_coef, args.clip_coef
                        )
                        v_loss = (
                            0.5
                            * torch.max(
                                v_unclipped, (v_clipped - b_ret[mb]) ** 2
                            ).mean()
                        )
                    else:
                        v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
            learn_s = time.time() - t_learn

            y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
            var_y = float(np.var(y_true))
            explained_var = (
                float("nan")
                if var_y == 0
                else float(1 - np.var(y_true - y_pred) / var_y)
            )
            wall_s = wall_before + time.time() - session_start
            row = {
                "update": update,
                "global_step": global_step,
                "wall_s": round(wall_s, 1),
                "sps": int(args.batch_size / (roll_s + learn_s)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": float(np.mean(clipfracs)),
                "explained_var": explained_var,
                "episodes": len(episodes),
            }
            if episodes:
                row["ep_return"] = float(np.mean([e["r"] for e in episodes]))
                row["ep_len"] = float(np.mean([e["l"] for e in episodes]))
                for k in ("finish", "progress", "time", "herring", "collisions"):
                    row["ep_" + k] = float(
                        np.mean([e.get(k, np.nan) for e in episodes])
                    )
            log.write(row)
            if update % args.log_every == 0:
                print(
                    f"[ppo] upd {update}/{args.num_updates} step {global_step} sps {row['sps']} "
                    f"ret {row.get('ep_return', float('nan')):.2f} finish {row.get('ep_finish', float('nan')):.2f} "
                    f"prog {row.get('ep_progress', float('nan')):.1f} kl {row['approx_kl']:.4f} "
                    f"ent {row['entropy']:.3f} ev {row['explained_var']:.2f} "
                    f"(roll {roll_s:.1f}s learn {learn_s:.1f}s)"
                )
            if update % args.checkpoint_every == 0 or update == args.num_updates:
                checkpoint()
        else:
            checkpoint()
            torch.save(agent.state_dict(), run_dir / "final_agent.pt")
            print(
                f"[ppo] done: {global_step} steps in {wall_before + time.time() - session_start:.0f}s"
            )
    finally:
        log.close()
        envs.close()
        if lease:
            lease.unregister()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
