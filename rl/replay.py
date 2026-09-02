"""Record, evaluate and compare action traces for the game bridge.

    # deterministic evaluation of a checkpoint, with a trace per episode
    python -m rl.replay trace --run-name bh-s1 --episodes 3 --out-dir traces/

    # a trace from a reference controller instead of a checkpoint
    python -m rl.replay trace --policy centerline --seed 0 --out-dir traces/

    # rendered/headless parity: diff the per-tick logs of the same trace
    python -m rl.replay compare traces/bh-s1-e0.headless.jsonl rendered.jsonl

A trace is what ``etr --rl-replay TRACE LOG`` plays back through the
rendered game: a header line, then one ``turn brake paddle jump`` line
per physics tick. Next to each trace this tool writes the headless
per-tick trajectory of the same actions, so the rendered log can be
compared tick by tick.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from rl.controls.policies import POLICIES
from rl.envs import make_env, parse_env_id
from rl.envs.etr_bridge import DEFAULT_DT, EtrBridge
from rl.envs.etr_env import ACTION_TABLE, FRAME_SKIP


def load_agent(
    run_name: str, runs_dir: str, checkpoint: str | None, obs_dim: int, n_actions: int
):
    import torch

    from rl.ppo import Agent

    run_dir = Path(runs_dir) / run_name
    args = json.load(open(run_dir / "args.json"))
    agent = Agent(obs_dim, n_actions, args.get("hidden", 128))
    path = Path(checkpoint) if checkpoint else run_dir / "checkpoint.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    agent.load_state_dict(ck["agent"] if "agent" in ck else ck)
    agent.eval()
    state = ck.get("state", {})

    def act(obs: np.ndarray, rng) -> int:
        with torch.no_grad():
            return int(agent.act_deterministic(torch.as_tensor(obs).unsqueeze(0))[0])

    return act, state, str(path)


def record_episode(env, policy, seed: int):
    """Run one episode; return (per-tick action rows, episode stats, agent steps)."""
    obs, _ = env.reset(seed=seed)
    rows = []
    done = False
    rng = np.random.default_rng(seed)
    while not done:
        a = policy(obs, rng)
        steer, paddle, brake = ACTION_TABLE[int(a)]
        rows.extend([(steer, int(brake), int(paddle), 0)] * env.frame_skip)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    return rows, info["episode_stats"]


def write_trace(
    path: Path, rows, course: str, group: str, seed: int, autoquit: bool
) -> None:
    with open(path, "w") as f:
        f.write("# etr rl trace v1\n")
        f.write(
            f"course={course} group={group} seed={seed} dt={DEFAULT_DT:.9g} wind=0 light=0 "
            f"mirror=0 autoquit={int(autoquit)}\n"
        )
        for steer, brake, paddle, jump in rows:
            f.write(f"{steer:g} {brake} {paddle} {jump}\n")


def headless_log(
    binary: str | None, path: Path, rows, course: str, group: str, seed: int
) -> dict:
    """Replay the trace tick by tick through the bridge and log each tick."""
    with EtrBridge(binary) as b, open(path, "w") as f:
        last = b.reset(seed=seed, course=course, group=group, dt=DEFAULT_DT)
        for steer, brake, paddle, jump in rows:
            last = b.step(turn=steer, brake=brake, paddle=paddle, jump=jump, ticks=1)
            f.write(
                json.dumps(
                    {
                        "tick": last["tick"],
                        "time": last["time"],
                        "pos": last["pos"],
                        "vel": last["vel"],
                        "herring": last["herring"],
                        "collisions": last["collisions"],
                        "finish": last["finish"],
                    }
                )
                + "\n"
            )
            if last["done"]:
                break
    return last


def cmd_trace(args) -> int:
    backend, course = parse_env_id(args.env)
    if backend != "etr":
        print(
            "trace: only the etr backend produces traces the game can replay",
            file=sys.stderr,
        )
        return 2
    env = make_env(args.env, binary=args.etr_bin)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.policy:
        policy, label, source = POLICIES[args.policy], args.policy, args.policy
    else:
        obs_dim = int(np.prod(env.observation_space.shape))
        policy, state, source = load_agent(
            args.run_name, args.runs_dir, args.checkpoint, obs_dim, env.action_space.n
        )
        label = args.run_name
        print(
            f"[replay] {source}: update {state.get('update')} step {state.get('global_step')}"
        )
    results = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            rows, stats = record_episode(env, policy, seed)
            stem = out_dir / f"{label}-e{ep}"
            write_trace(
                stem.with_suffix(".trace"), rows, course, env.group, seed, args.autoquit
            )
            last = headless_log(
                args.etr_bin,
                stem.with_suffix(".headless.jsonl"),
                rows,
                course,
                env.group,
                seed,
            )
            row = {
                "episode": ep,
                "seed": seed,
                "finish": stats["finish"],
                "time": stats["time"],
                "progress": stats["progress"],
                "herring": stats["herring"],
                "collisions": stats["collisions"],
                "ticks": len(rows),
                "tick_replay_time": last["time"],
                "trace": str(stem.with_suffix(".trace")),
            }
            results.append(row)
            print(
                f"[replay] ep {ep} seed {seed}: finish {stats['finish']:.0f} time {stats['time']:.2f}s "
                f"herring {stats['herring']:.0f} collisions {stats['collisions']:.0f} "
                f"(tick-by-tick replay {last['time']:.2f}s) -> {stem}.trace"
            )
    finally:
        env.close()
    times = [r["time"] for r in results if r["finish"]]
    summary = {
        "source": source,
        "env": args.env,
        "episodes": results,
        "finish_rate": float(np.mean([r["finish"] for r in results])),
        "mean_time": float(np.mean(times)) if times else float("nan"),
    }
    with open(out_dir / f"{label}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[replay] finish rate {summary['finish_rate']:.2f}, mean time {summary['mean_time']:.2f}s"
    )
    return 0


def cmd_compare(args) -> int:
    a = [json.loads(l) for l in open(args.headless)]
    b = [json.loads(l) for l in open(args.rendered)]
    n = min(len(a), len(b))
    if n == 0:
        print("compare: empty log", file=sys.stderr)
        return 2
    worst = 0.0
    worst_tick = 0
    for x, y in zip(a[:n], b[:n]):
        d = math.dist(x["pos"], y["pos"])
        if d > worst:
            worst, worst_tick = d, x["tick"]
    fa, fb = a[n - 1], b[n - 1]
    print(f"ticks compared {n} (headless {len(a)}, rendered {len(b)})")
    print(f"max position error {worst:.6f} m at tick {worst_tick}")
    print(f"final time headless {fa['time']:.4f}s rendered {fb['time']:.4f}s")
    print(
        f"herring {fa['herring']} vs {fb['herring']}; collisions {fa['collisions']} vs {fb['collisions']}"
    )
    ok = (
        worst <= args.tolerance
        and fa["herring"] == fb["herring"]
        and abs(len(a) - len(b)) <= args.tick_slack
    )
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser(
        "trace", help="evaluate a policy deterministically and write traces"
    )
    t.add_argument("--env", default="etr:bunny_hill")
    t.add_argument("--run-name", default=None)
    t.add_argument("--runs-dir", default="runs/rl")
    t.add_argument("--checkpoint", default=None)
    t.add_argument("--policy", default=None, choices=[None, *POLICIES])
    t.add_argument("--episodes", type=int, default=3)
    t.add_argument("--seed", type=int, default=1000)
    t.add_argument("--out-dir", default="traces")
    t.add_argument("--autoquit", action="store_true")
    t.add_argument("--etr-bin", default=None)
    c = sub.add_parser("compare", help="diff a headless and a rendered per-tick log")
    c.add_argument("headless")
    c.add_argument("rendered")
    c.add_argument(
        "--tolerance", type=float, default=1e-3, help="max position error in metres"
    )
    c.add_argument("--tick-slack", type=int, default=2)
    args = p.parse_args(argv)
    if args.cmd == "trace":
        if not args.policy and not args.run_name:
            p.error("trace needs --run-name or --policy")
        return cmd_trace(args)
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
