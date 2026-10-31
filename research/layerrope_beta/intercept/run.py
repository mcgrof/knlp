# SPDX-License-Identifier: MIT
"""Run the magnitude-intercept screen and confirmation on one caller-provided GPU."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from research.layerrope_beta.intercept.plan import (
    CONFIRM_SEEDS,
    PROBE_SEED,
    SCREEN_SEED,
    confirm_rows,
    freeze,
    job,
    primary,
    row,
    screen_rows,
    select,
)
from research.mo_variants.stage_c_common import read, sha, write

REPO = Path(__file__).resolve().parents[3]
SOURCES = ("layerrope_beta", "tinystories_depth", "mo_variants")


def prepare(root, prepared, checkpoints):
    root, prepared, checkpoints = (
        Path(x).expanduser().resolve() for x in (root, prepared, checkpoints)
    )
    if root.is_relative_to(REPO) or checkpoints.is_relative_to(REPO):
        raise ValueError("Outputs and checkpoints must be outside the checkout")
    manifest = read(prepared / "DATA.json")
    (root / "data").mkdir(parents=True, exist_ok=False)
    for name in ("train.npy", "curve.npy", "final.npy", "DOCUMENTS.json"):
        if sha(prepared / name) != manifest["files"][name]:
            raise ValueError("Changed prepared input: " + name)
        (root / "data" / name).symlink_to(prepared / name)
    shutil.copyfile(prepared / "DATA.json", root / "data/DATA.json")
    count = (len(np.load(root / "data/train.npy", mmap_mode="r")) - 1) // 256
    for seed in (SCREEN_SEED, PROBE_SEED, *CONFIRM_SEEDS):
        np.save(
            root / f"data/order-{seed}.npy",
            np.random.default_rng(seed).permutation(count).astype(np.int32),
        )
    plan = freeze()
    plan["checkpoint_directory"] = str(checkpoints)
    plan["input_files"] = {
        p.name: sha(p) for p in sorted((root / "data").iterdir()) if p.is_file()
    }
    plan["source_files"] = {
        str(p.relative_to(REPO)): sha(p)
        for directory in SOURCES
        for p in sorted((REPO / "research" / directory).rglob("*.py"))
        if p.name != "modal_app.py"
    }
    for folder in ("jobs", "logs", "language/runs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    write(root / "PLAN.json", plan)
    return plan


def verify(root):
    plan = read(root / "PLAN.json")
    for name, expected in plan["input_files"].items():
        if sha(root / "data" / name) != expected:
            raise ValueError("Changed study input: " + name)
    for name, expected in plan["source_files"].items():
        if sha(REPO / name) != expected:
            raise ValueError("Changed study source: " + name)
    return plan


def wave(root, checkpoints, jobs, workers, max_seconds):
    """Run jobs concurrently; completed results are kept, partial ones restart."""

    def one(j):
        destination = root / "language/runs" / j["id"]
        if destination.exists():
            if read(destination / "RESULT.json").get("status") == "complete":
                return j["id"], "kept"
            shutil.rmtree(destination)
            shutil.rmtree(checkpoints / "language" / j["id"], ignore_errors=True)
        path = root / "jobs" / (j["id"] + ".json")
        write(path, j)
        with (root / "logs" / (j["id"] + ".log")).open("w") as output:
            code = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "research.layerrope_beta.train",
                    "--root",
                    root,
                    "--job",
                    path,
                    "--checkpoints",
                    checkpoints,
                    "--max-seconds",
                    max_seconds,
                ],
                cwd=REPO,
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        return j["id"], f"exit {code}"

    # Slow mHC rows first so the tail of the wave is short.
    jobs = sorted(jobs, key=lambda j: j["arm"].startswith("M"), reverse=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for name, outcome in pool.map(one, jobs):
            print(json.dumps(dict(id=name, outcome=outcome)), flush=True)


def scores(root, rows, stage, seed):
    out = []
    for r in rows:
        path = root / "language/runs" / job(r, stage, seed)["id"] / "RESULT.json"
        out.append((r, primary(read(path)) if path.exists() else None))
    return out


def execute(root, checkpoints, stage, workers, max_seconds):
    root, checkpoints = Path(root).resolve(), Path(checkpoints).resolve()
    verify(root)
    if not torch.cuda.is_available():
        raise RuntimeError("This study trains on one visible GPU")
    began = time.monotonic()
    status = dict(
        status="running",
        stage=stage,
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        workers=workers,
        started=time.time(),
    )
    write(root / "STATUS.json", status)
    try:
        if stage in ("screen", "all"):
            jobs = [job(r, "screen", SCREEN_SEED) for r in screen_rows()]
            wave(root, checkpoints, jobs, workers, max_seconds)
            screened = scores(root, screen_rows(), "screen", SCREEN_SEED)
            chosen = select([(r, v) for r, v in screened if r["arm"] in ("L", "ML")])
            write(
                root / "SELECTED.json",
                dict(
                    chosen=chosen,
                    screen={r["id"]: v for r, v in screened},
                    frozen_before_confirmation=True,
                ),
            )
        if stage in ("confirm", "all"):
            chosen = read(root / "SELECTED.json")["chosen"]
            jobs = [
                job(r, "final", seed)
                for seed in CONFIRM_SEEDS
                for r in confirm_rows(chosen)
            ]
            wave(root, checkpoints, jobs, workers, max_seconds)
        status.update(status="complete")
    except BaseException as error:
        status.update(status="incomplete", error=str(error))
        raise
    finally:
        status["seconds"] = time.monotonic() - began
        write(root / "STATUS.json", status)


def probe(root, arms, steps):
    """Time disposable updates; callers launch several to measure contention."""
    from research.layerrope_beta.model import Model
    from research.tinystories_depth.model import loss
    from research.tinystories_depth.probe import batch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens = np.load(root / "data/train.npy", mmap_mode="r")
    order = np.load(root / f"data/order-{PROBE_SEED}.npy")
    x, y = batch(tokens, order, 0, "cuda")
    for arm in arms:
        r = row(arm, -0.25, -0.2) if arm in ("L", "ML") else row(arm)
        model = Model(
            arm, 32, PROBE_SEED, beta=r["beta"] or -0.5, alpha_out=r["alpha_out"] or 0
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
        times = []
        for step in range(steps):
            torch.cuda.synchronize()
            began = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective = loss(model(x), y)
            objective.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 5:
                times.append(time.monotonic() - began)
        print(
            json.dumps(
                dict(
                    arm=arm,
                    mean_step_seconds=float(np.mean(times)),
                    projected_2000_seconds=float(np.mean(times)) * 2000,
                    pid=os.getpid(),
                    gpu=torch.cuda.get_device_name(0),
                )
            ),
            flush=True,
        )
        del model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "probe"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--stage", choices=("screen", "confirm", "all"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--arms", default="R,M,L,ML")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    checkpoints = args.checkpoints or args.root / "checkpoints"
    if args.action == "prepare":
        prepare(args.root, args.prepared, checkpoints)
    elif args.action == "probe":
        probe(args.root, args.arms.split(","), args.steps)
    else:
        execute(args.root, checkpoints, args.stage, args.workers, args.max_seconds)
