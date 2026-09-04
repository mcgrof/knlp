"""Run every course: controls, zero-shot transfer of one checkpoint,
PPO, deterministic evaluation; then write one summary.

    python -m rl.campaign --courses all --parallel 4 --total-timesteps 5000000 \
        --runs-dir runs/rl --out-dir results/campaign --transfer-from bh-s2

Each PPO run goes through the pace supervisor, so a hold pauses the
campaign at the next update boundary and it resumes from checkpoints;
re-running the command skips finished stages.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rl.controls.policies import POLICIES, run_episodes, summarize
from rl.envs.etr_bridge import EtrBridge
from rl.evaluate import evaluate

PY = sys.executable


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def course_table(binary) -> list[dict]:
    with EtrBridge(binary) as b:
        info = b.info()
    return sorted(
        (c for c in info["courses"] if c["group"] == "default"),
        key=lambda c: c["play_length"],
    )


def max_seconds(course: dict) -> float:
    return 60.0 + course["play_length"] / 4.0


def stuck_seconds(course: dict) -> float:
    """How long without new maximum progress before an episode is cut off.

    A fixed six seconds cut off legitimate episodes on the long courses,
    where recovering the ground lost to one tree can take longer than
    that; scale the window with the course instead.
    """
    return max(15.0, course["play_length"] / 100.0)


def stage_controls(course, args, out):
    path = out / f"controls-{course['dir']}.json"
    if path.exists():
        return json.load(open(path))
    rows = {}
    for name in ("noop", "centerline"):
        stats, steps, wall = run_episodes(
            f"etr:{course['dir']}",
            POLICIES[name],
            2,
            1000,
            binary=args.etr_bin,
            max_seconds=max_seconds(course),
            stuck_seconds=stuck_seconds(course),
        )
        rows[name] = summarize(name, stats, steps, wall)
    json.dump(rows, open(path, "w"), indent=2)
    return rows


def stage_transfer(course, args, out):
    """One trained checkpoint raced on this course unchanged."""
    path = out / f"transfer-{course['dir']}.json"
    if path.exists():
        return json.load(open(path))
    from rl.envs import make_env
    from rl.replay import load_agent

    env = make_env(
        f"etr:{course['dir']}",
        binary=args.etr_bin,
        max_seconds=max_seconds(course),
        stuck_seconds=stuck_seconds(course),
    )
    res = {}
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        oracle, _, _ = load_agent(
            args.transfer_from,
            args.transfer_runs_dir or args.runs_dir,
            None,
            obs_dim,
            env.action_space.n,
        )
        res["ppo"] = evaluate(env, oracle, [1000, 1001])
    finally:
        env.close()
    json.dump(res, open(path, "w"), indent=2)
    return res


def run_name(course, args) -> str:
    suffix = "" if args.action_set == "v0" else f"-{args.action_set}"
    return f"{course['dir']}-s{args.seed}{suffix}"


def stage_ppo(course, args, out):
    name = run_name(course, args)
    run_dir = Path(args.runs_dir) / name
    if (run_dir / "final_agent.pt").exists():
        return name
    cmd = [
        PY,
        "-m",
        "rl.pace.ctl",
        "run",
        "--name",
        name,
        "--",
        PY,
        "-m",
        "rl.ppo",
        "--env",
        f"etr:{course['dir']}",
        "--run-name",
        name,
        "--runs-dir",
        args.runs_dir,
        "--seed",
        str(args.seed),
        "--device",
        "cpu",
        "--num-envs",
        "8",
        "--torch-threads",
        "2",
        "--total-timesteps",
        str(args.total_timesteps),
        "--max-seconds",
        f"{max_seconds(course):.0f}",
        "--resume",
        "--max-hours",
        "2",
        "--log-every",
        "50",
    ]
    if args.etr_bin:
        cmd += ["--etr-bin", args.etr_bin]
    with open(out / f"ppo-{name}.log", "a") as f:
        rc = subprocess.call(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
        )
    if rc != 0:
        raise RuntimeError(f"ppo {name} exited {rc}")
    return name


def stage_eval(course, name, args, out):
    path = out / f"eval-{name}.json"
    if path.exists():
        return json.load(open(path))
    from rl.envs import make_env
    from rl.replay import load_agent

    env = make_env(
        f"etr:{course['dir']}",
        binary=args.etr_bin,
        max_seconds=max_seconds(course),
        stuck_seconds=stuck_seconds(course),
    )
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        policy, _, _ = load_agent(
            name, args.runs_dir, None, obs_dim, env.action_space.n
        )
        res = evaluate(env, policy, [1000, 1001])
    finally:
        env.close()
    json.dump(res, open(path, "w"), indent=2)
    return res


def run_course(course, args, out):
    d = course["dir"]
    try:
        log(f"{d}: controls")
        stage_controls(course, args, out)
        log(f"{d}: transfer")
        stage_transfer(course, args, out)
        log(f"{d}: ppo")
        name = stage_ppo(course, args, out)
        stage_eval(course, name, args, out)
        log(f"{d}: done")
    except Exception as e:  # keep the campaign going
        log(f"{d}: FAILED {e}")


def summary(courses, args, out):
    lines = [
        "| course | length | no-op | centreline | zero-shot from "
        + args.transfer_from
        + " | PPO (own) |",
        "|---|---|---|---|---|---|",
    ]

    def t(ev):
        if not ev:
            return "-"
        f = ev.get("finish_rate", ev.get("finish"))
        tm = ev.get("mean_time", ev.get("time"))
        return f"{tm:.1f}" if f == 1.0 else f"DNF ({f:.1f})"

    for c in courses:
        d = c["dir"]
        name = run_name(c, args)
        con = (
            json.load(open(out / f"controls-{d}.json"))
            if (out / f"controls-{d}.json").exists()
            else {}
        )
        tr = (
            json.load(open(out / f"transfer-{d}.json"))
            if (out / f"transfer-{d}.json").exists()
            else {}
        )
        ev = (
            json.load(open(out / f"eval-{name}.json"))
            if (out / f"eval-{name}.json").exists()
            else {}
        )
        lines.append(
            f"| {d} | {c['play_length']:.0f} | {t(con.get('noop'))} | {t(con.get('centerline'))} | "
            f"{t(tr.get('ppo'))} | {t(ev)} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--courses", default="all", help="all, or comma-separated course dirs"
    )
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--runs-dir", default=os.environ.get("KNLP_RL_RUNS", "runs/rl"))
    p.add_argument("--out-dir", default="results/campaign")
    p.add_argument("--action-set", default="v0", choices=["v0", "v1"])
    p.add_argument("--transfer-from", default="bh-s2")
    p.add_argument(
        "--transfer-runs-dir", default=None, help="runs dir holding --transfer-from"
    )
    p.add_argument("--etr-bin", default=None)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    courses = course_table(args.etr_bin)
    if args.courses != "all":
        keep = set(args.courses.split(","))
        courses = [c for c in courses if c["dir"] in keep]
    if args.summary_only:
        summary(courses, args, out)
        return 0
    log(
        f"campaign over {len(courses)} courses, {args.parallel} at a time, {args.total_timesteps} steps each"
    )
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(lambda c: run_course(c, args, out), courses))
    summary(courses, args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
