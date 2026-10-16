# SPDX-License-Identifier: MIT
"""Run the paired full-gain beta study on a caller-provided two-GPU allocation."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from research.layerrope_beta.plan import freeze, language_rows, LM_SEEDS, LR_GRID
from research.mo_variants.stage_c_common import read, sha, write
from research.reproduction.data import build
from research.subliminal_learning.recipe import Recipe, validate

REPO = Path(__file__).resolve().parents[2]


def prepare(root, cache, checkpoints):
    root, cache, checkpoints = map(
        lambda x: Path(x).expanduser().resolve(), (root, cache, checkpoints)
    )
    if root.is_relative_to(REPO) or checkpoints.is_relative_to(REPO):
        raise ValueError("Outputs and checkpoints must be outside the checkout")
    root.mkdir(parents=True, exist_ok=False)
    cached = cache / "tinystories-prepared"
    if (cached / "DATA.json").exists():
        manifest = read(cached / "DATA.json")
        (root / "data").mkdir()
        for name in ("train.npy", "curve.npy", "final.npy", "DOCUMENTS.json"):
            if sha(cached / name) != manifest["files"][name]:
                raise ValueError("Changed prepared input: " + name)
            (root / "data" / name).symlink_to(cached / name)
        shutil.copyfile(cached / "DATA.json", root / "data/DATA.json")
    else:
        build(root, cache)
    count = (len(np.load(root / "data/train.npy", mmap_mode="r")) - 1) // 256
    for seed in (77301, 77390, *LM_SEEDS):
        np.save(
            root / f"data/order-{seed}.npy",
            np.random.default_rng(seed).permutation(count).astype(np.int32),
        )
    plan = freeze(root, root / "data", checkpoints)
    from research.layerrope_beta.measure import policies
    from research.tinystories_depth.kv_cache import cases

    plan["floating_interventions"] = policies()
    from research.layerrope_beta.mnist_measure import policies as mnist_policies

    plan["mnist"]["floating_interventions"] = mnist_policies()
    plan["cache_cases"] = cases() + [
        dict(name="fp4_e2m1-K4V4", k_bits=4, v_bits=4, method="fp4_e2m1"),
        dict(name="fp8_e4m3fn-K8V8", k_bits=8, v_bits=8, method="fp8_e4m3fn"),
    ]
    plan["input_files"] = {
        p.name: sha(p) for p in sorted((root / "data").iterdir()) if p.is_file()
    }
    plan["source_files"] = {
        str(p.relative_to(REPO)): sha(p)
        for directory in ("layerrope_beta", "tinystories_depth", "subliminal_learning")
        for p in sorted((REPO / "research" / directory).rglob("*.py"))
    }
    write(root / "PLAN.json", plan)
    for folder in (
        "jobs",
        "logs",
        "language/runs",
        "mnist/runs",
        "mnist/qualification",
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
    return plan


def mnist(root, name):
    from research.subliminal_learning.layerrope_study import qualify
    from research.subliminal_learning.run import cmd_run

    plan = read(root / "PLAN.json")
    recipe = Recipe(**plan["mnist"]["recipes"][name])
    validate(recipe)
    qualification = qualify(recipe)
    write(root / "mnist/qualification" / (name + ".json"), qualification)
    if not qualification["all_pass"]:
        raise RuntimeError("MNIST active controls failed: " + name)
    source = {field: "full-input-output-beta-study" for field in recipe.to_json()}
    result = cmd_run(recipe, source, ["layerrope_beta", "mnist", name])
    if result:
        raise RuntimeError("Incomplete MNIST trajectory: " + name)
    from research.layerrope_beta.mnist_measure import run as measure_mnist

    measure_mnist(root, name)


def execute(root, checkpoints, max_seconds=43200):
    root, checkpoints = Path(root), Path(checkpoints)
    plan = read(root / "PLAN.json")
    for name, expected in plan["input_files"].items():
        if sha(root / "data" / name) != expected:
            raise ValueError("Changed study input: " + name)
    for name, expected in plan["source_files"].items():
        if sha(REPO / name) != expected:
            raise ValueError("Changed study source: " + name)
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Provide exactly two visible GPUs for this frozen study")
    if not all("H100" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("The full study requires the frozen H100 hardware tier")
    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError("The frozen GPU runtime requires torch 2.9.1")
    began = time.monotonic()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(visible) != 2:
        raise RuntimeError("Expected two explicit visible GPU slots")
    active, lock, aborted = {}, threading.Lock(), threading.Event()

    def command(module, args, slot, label):
        elapsed = time.monotonic() - began
        if elapsed >= max_seconds:
            raise TimeoutError("Study elapsed-time limit")
        with (root / "logs" / (label + ".log")).open("w") as output:
            process = subprocess.Popen(
                [sys.executable, "-m", module, *map(str, args)],
                cwd=REPO,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": visible[slot],
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "KNLP_BETA_TEST_DEVICE": "cuda",
                },
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with lock:
                active[process.pid] = process
            try:
                code = process.wait(timeout=max(1, max_seconds - elapsed))
                if code:
                    raise RuntimeError(f"{label} exited {code}; see its log")
            finally:
                if process.poll() is None:
                    stop(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        stop(process, signal.SIGKILL)
                        process.wait()
                with lock:
                    active.pop(process.pid, None)

    def stop(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def wave(jobs):
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)

        def worker(slot):
            while True:
                if aborted.is_set():
                    return
                try:
                    kind, job = pending.get_nowait()
                except queue.Empty:
                    return
                if kind == "mnist":
                    command(
                        "research.layerrope_beta.run",
                        ["mnist", "--root", root, "--row", job],
                        slot,
                        "mnist-" + job,
                    )
                else:
                    path = root / "jobs" / (job["id"] + ".json")
                    write(path, job)
                    command(
                        "research.layerrope_beta.train",
                        [
                            "--root",
                            root,
                            "--job",
                            path,
                            "--checkpoints",
                            checkpoints,
                            "--max-seconds",
                            min(7200, max_seconds),
                        ],
                        slot,
                        job["id"],
                    )
                    result = read(root / "language/runs" / job["id"] / "RESULT.json")
                    if job["stage"] == "final" and result["status"] != "complete":
                        raise RuntimeError(
                            "Incomplete final language-model job: " + job["id"]
                        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            try:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except BaseException:
                aborted.set()
                with lock:
                    for process in active.values():
                        if process.poll() is None:
                            stop(process, signal.SIGTERM)
                raise

    status = dict(status="running", stage="qualification", started=time.time())
    write(root / "STATUS.json", status)
    try:
        for slot in range(2):
            command(
                "pytest",
                [
                    "-q",
                    "research/layerrope_beta/tests",
                    "research/tinystories_depth/tests/test_float_quant.py",
                ],
                slot,
                f"qualification-gpu{slot}",
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            probes = [
                pool.submit(
                    command,
                    "research.layerrope_beta.probe",
                    ["--root", root, "--kind", kind],
                    slot,
                    "probe-" + kind,
                )
                for slot, kind in enumerate(("language", "mnist"))
            ]
            for probe in probes:
                probe.result()
        forecast = (
            read(root / "PROBE-language.json")["projected_train_seconds"]
            + read(root / "PROBE-mnist.json")["projected_train_seconds"]
        )
        write(
            root / "FORECAST.json",
            dict(
                projected_train_gpu_seconds=forecast,
                optimistic_two_gpu_seconds=forecast / 2,
                elapsed_limit_seconds=max_seconds,
                qualification="Projection excludes full quantization, evaluation, scheduling imbalance and checkpoint transfer",
            ),
        )
        if forecast / 2 > max_seconds * 0.85:
            raise RuntimeError(
                "Even the training-only projection exceeds the reserved time; revise resources before training"
            )
        jobs = []
        for row in language_rows():
            for lr in LR_GRID:
                jobs.append(
                    (
                        "language",
                        dict(
                            id=f"pilot-{row['id']}-lr{lr}",
                            row=row["id"],
                            arm=row["arm"],
                            beta=row["beta"],
                            stage="pilot",
                            steps=500,
                            lr=lr,
                            seed=77301,
                        ),
                    )
                )
        jobs += [("mnist", name) for name in plan["mnist"]["recipes"]]
        order = np.random.default_rng(77300).permutation(len(jobs))
        jobs = [jobs[i] for i in order]
        status.update(stage="pilots-and-mnist")
        write(root / "STATUS.json", status)
        wave(jobs)
        selected, finals = {}, []
        for row in language_rows():
            candidates = []
            for lr in LR_GRID:
                result = read(
                    root / "language/runs" / f"pilot-{row['id']}-lr{lr}/RESULT.json"
                )
                if result["status"] == "complete":
                    points = [
                        p["ce"] for p in result["curve"] if p["step"] in (300, 400, 500)
                    ]
                    if len(points) == 3:
                        candidates.append((float(np.mean(points)), lr))
            if not candidates:
                raise RuntimeError("No valid pilot for " + row["id"])
            _, lr = min(candidates)
            selected[row["id"]] = lr
            for seed in LM_SEEDS:
                finals.append(
                    (
                        "language",
                        dict(
                            id=f"final-{row['id']}-seed{seed}",
                            row=row["id"],
                            arm=row["arm"],
                            beta=row["beta"],
                            stage="final",
                            steps=2000,
                            lr=lr,
                            seed=seed,
                        ),
                    )
                )
        write(
            root / "SELECTED.json",
            dict(
                selection_metric=plan["language"]["lr_selection"],
                lrs=selected,
                finals=[j for _, j in finals],
                frozen_before_final_trajectories=True,
            ),
        )
        status.update(stage="final-language-models")
        write(root / "STATUS.json", status)
        wave(finals)
        status.update(
            status="complete", stage="all-trajectories-and-quantization-complete"
        )
    except BaseException as error:
        status.update(status="incomplete", error=str(error))
        raise
    finally:
        status["seconds"] = time.monotonic() - began
        write(root / "STATUS.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "mnist"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--max-seconds", type=int, default=43200)
    parser.add_argument("--row")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(
            args.root, args.data_cache, args.checkpoints or args.root / "checkpoints"
        )
    elif args.action == "mnist":
        mnist(args.root, args.row)
    else:
        execute(
            args.root, args.checkpoints or args.root / "checkpoints", args.max_seconds
        )
