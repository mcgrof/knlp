"""Serve a trained policy to the game, so a rendered race is driven live.

    python -m rl.serve --run-name bh-gpu-s12 --runs-dir <runs> \
        --env etr:bunny_hill --seed 1000 --port 5077

The game connects with ``etr --rl-live HOST:PORT TRACE_OUT``, says hello,
and receives the race setup as a trace header. From then on it sends the
same observation the headless bridge produces, once per physics tick, and
this server answers with one trace line: four control values, or ``R`` for
the reset key. Decisions are held for the contract's frame skip here, so
the game stays a dumb client and the policy keeps the decision rate it was
trained at.

Because the reply stream is exactly a trace, a live race and a recorded
one can be diffed line for line.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys

import numpy as np

from rl.envs import parse_env_id
from rl.envs.etr_bridge import DEFAULT_DT, EtrBridge
from rl.envs.etr_env import FRAME_SKIP, build_actions, encode_observation
from rl.replay import load_agent, run_action_set


def serve(args) -> int:
    backend, course = parse_env_id(args.env)
    if backend != "etr":
        print("serve: only the etr backend drives the rendered game", file=sys.stderr)
        return 2
    action_set = args.action_set
    if action_set == "auto":
        action_set = run_action_set(args.run_name, args.runs_dir)
    actions, names = build_actions(action_set)
    obs_dim = None
    policy = None

    header = (
        f"course={course} group={args.group} seed={int(args.seed)} dt={DEFAULT_DT:.9g} "
        f"wind=0 light=0 mirror=0 autoquit={int(args.autoquit)}"
    )

    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("", args.port))
    ls.listen(1)
    print(
        f"[serve] {args.run_name} ({action_set}, {len(actions)} actions) on port "
        f"{args.port}; the game should run: etr --rl-live <host>:{args.port} <trace-out>",
        flush=True,
    )

    while True:
        conn, _ = ls.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rwb")
        try:
            hello = f.readline()
            if not hello:
                continue
            f.write((header + "\n").encode())
            f.flush()
            print(
                f"[serve] a client connected; racing {course} seed {args.seed}",
                flush=True,
            )

            rng = np.random.default_rng(0)
            held = None
            ticks_left = 0
            decisions = 0
            while True:
                line = f.readline()
                if not line:
                    break
                raw = json.loads(line)
                if not raw.get("ok", True):
                    break
                if ticks_left == 0:
                    obs = encode_observation(raw)
                    if policy is None:
                        obs_dim = obs.shape[0]
                        policy, state, source = load_agent(
                            args.run_name,
                            args.runs_dir,
                            args.checkpoint,
                            obs_dim,
                            len(actions),
                        )
                        print(
                            f"[serve] {source}: update {state.get('update')} "
                            f"step {state.get('global_step')}",
                            flush=True,
                        )
                    entry = actions[int(policy(obs, rng))]
                    if entry is None:
                        held = "R"
                        ticks_left = 1  # the reset key consumes no physics tick
                    else:
                        steer, paddle, brake, jump = entry
                        held = f"{steer:g} {int(brake)} {int(paddle)} {int(jump)}"
                        ticks_left = args.frame_skip
                    decisions += 1
                f.write((held + "\n").encode())
                f.flush()
                ticks_left -= 1
            print(f"[serve] client finished after {decisions} decisions", flush=True)
        except (ConnectionError, OSError, ValueError) as e:
            print(f"[serve] client dropped: {e}", flush=True)
        finally:
            try:
                f.close()
            finally:
                conn.close()
        if args.once:
            return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default="runs/rl")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--env", default="etr:bunny_hill")
    p.add_argument("--group", default="default")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--port", type=int, default=5077)
    p.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    p.add_argument("--action-set", default="auto", choices=["auto", "v0", "v1"])
    p.add_argument("--autoquit", action="store_true", default=True)
    p.add_argument("--no-autoquit", dest="autoquit", action="store_false")
    p.add_argument("--once", action="store_true", help="serve one race and exit")
    return serve(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
