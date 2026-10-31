#!/usr/bin/env python3
"""Fan the magnitude-intercept study out on Modal, one training run per card.

The science is in ``run.py``, ``plan.py`` and the shared trainer; this file
is only the platform. These story models are small and launch-bound, so one
card trains one run no faster than a large card would, and several runs on
one card share its launch queue without gaining throughput. Running every
run on its own container makes a stage take about as long as its slowest
run.

The prepared run directory (inputs, seed orders and the frozen PLAN.json)
lives in the ``knlp-runs`` volume. Each container copies it to local disk,
checks the frozen input and source hashes, trains, and copies its result,
log and checkpoints back. A completed run is kept on restart; a preempted
run restarts from the beginning, which costs at most one run.

Run from the repository root, after uploading the prepared directory to
``knlp-runs/<study>/run``::

    modal run research/layerrope_beta/intercept/modal_app.py --stage probe
    modal run --detach research/layerrope_beta/intercept/modal_app.py \
        --stage screen
    modal run --detach research/layerrope_beta/intercept/modal_app.py \
        --stage confirm
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import modal

STUDY = os.environ.get("KNLP_INTERCEPT_STUDY", "layerrope-intercept-20261030")
# Launch-bound runs gain little from a larger card; pick one with capacity.
GPU = os.environ.get("KNLP_INTERCEPT_GPU", "A10G")
app = modal.App(os.environ.get("KNLP_MODAL_APP", "knlp-layerrope-intercept"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pins follow research/reproduction/requirements.txt, with the NumPy of
    # the measured H100 runtime.
    .uv_pip_install(
        "torch==2.9.1",
        "transformers==5.3.0",
        "tokenizers==0.22.2",
        "numpy==2.4.3",
        "pyarrow==21.0.0",
        "huggingface_hub>=1.0,<2",
        "safetensors>=0.8",
        "scipy>=1.15",
        "einops>=0.8",
        "psutil>=7",
        "packaging>=25",
    ).env({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
)
if modal.is_local():
    research = pathlib.Path(__file__).resolve().parents[2]
    image = image.add_local_dir(
        research,
        "/root/knlp/research",
        # the platform ships this file itself; shipping it twice is refused
        ignore=["**/__pycache__", "**/*.pyc", "**/modal_app.py", "**/.pytest_cache"],
    )

runs = modal.Volume.from_name("knlp-runs", create_if_missing=True)
REPO = "/root/knlp"
VOLUME = pathlib.Path("/runs") / STUDY


def stage_in():
    """Copy the prepared inputs and plan to local disk and check them."""
    sys.path.insert(0, REPO)
    from research.layerrope_beta.intercept.run import verify

    root = pathlib.Path("/tmp/study")
    if not root.exists():
        shutil.copytree(VOLUME / "run/data", root / "data")
        shutil.copyfile(VOLUME / "run/PLAN.json", root / "PLAN.json")
        for folder in ("jobs", "logs", "language/runs"):
            (root / folder).mkdir(parents=True, exist_ok=True)
    verify(root)
    return root


@app.function(
    image=image,
    gpu=GPU,
    cpu=4.0,
    memory=12288,
    timeout=2 * 3600,
    volumes={"/runs": runs},
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
)
def train_one(job: dict) -> dict:
    runs.reload()
    kept = VOLUME / "run/language/runs" / job["id"] / "RESULT.json"
    if kept.exists() and json.loads(kept.read_text()).get("status") == "complete":
        return dict(id=job["id"], outcome="kept", result=json.loads(kept.read_text()))
    root = stage_in()
    checkpoints = pathlib.Path("/tmp/checkpoints")
    destination = root / "language/runs" / job["id"]
    shutil.rmtree(destination, ignore_errors=True)
    shutil.rmtree(checkpoints / "language" / job["id"], ignore_errors=True)
    path = root / "jobs" / (job["id"] + ".json")
    path.write_text(json.dumps(job, indent=2))
    began = time.time()
    with (root / "logs" / (job["id"] + ".log")).open("w") as log:
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "research.layerrope_beta.train",
                "--root",
                str(root),
                "--job",
                str(path),
                "--checkpoints",
                str(checkpoints),
                "--max-seconds",
                "6600",
            ],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    for source, target in (
        (destination, VOLUME / "run/language/runs" / job["id"]),
        (checkpoints / "language" / job["id"], VOLUME / "checkpoints" / job["id"]),
        (path, VOLUME / "run/jobs" / path.name),
        (
            root / "logs" / (job["id"] + ".log"),
            VOLUME / "run/logs" / (job["id"] + ".log"),
        ),
    ):
        if not source.exists():
            continue
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    runs.commit()
    result = destination / "RESULT.json"
    return dict(
        id=job["id"],
        outcome=f"exit {code}",
        wall_seconds=round(time.time() - began, 1),
        result=json.loads(result.read_text()) if result.exists() else None,
    )


@app.function(
    image=image, gpu=GPU, cpu=4.0, memory=12288, timeout=1200, volumes={"/runs": runs}
)
def probe(arms: str, gpu: str) -> str:
    root = stage_in()
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.layerrope_beta.intercept.run",
            "probe",
            "--root",
            str(root),
            "--arms",
            arms,
            "--steps",
            "25",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if out.returncode:
        return f"{gpu}: probe failed\n" + out.stderr[-2000:]
    return f"{gpu}:\n" + out.stdout


@app.function(image=image, cpu=2.0, memory=4096, timeout=900, volumes={"/runs": runs})
def check() -> str:
    """Import every training and measurement module and verify the plan, no card."""
    import importlib

    root = stage_in()
    names = (
        "research.layerrope_beta.train",
        "research.layerrope_beta.measure",
        "research.layerrope_beta.model",
        "research.tinystories_depth.kv_cache",
        "research.tinystories_depth.float_quant",
        "research.tinystories_depth.probe",
        "research.mo_variants.stage_c_common",
    )
    for name in names:
        importlib.import_module(name)
    from research.layerrope_beta.model import Model

    model = Model("ML", 2, 77990, beta=-0.2, vocab=97, alpha_out=-0.25)
    rope = model.lm.model.layerrope
    return json.dumps(
        dict(
            imported=len(names),
            plan_verified=str(root),
            out_alpha=float(rope.out_attn_alpha),
            in_alpha=float(rope.in_attn_alpha),
        )
    )


@app.function(image=image, cpu=1.0, memory=1024, timeout=600, volumes={"/runs": runs})
def put(name: str, text: str):
    runs.reload()
    (VOLUME / "run" / name).write_text(text)
    runs.commit()


@app.function(image=image, cpu=1.0, memory=1024, timeout=600, volumes={"/runs": runs})
def read(name: str) -> str:
    runs.reload()
    return (VOLUME / "run" / name).read_text()


def fan_out(jobs):
    results = []
    for r in train_one.map(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("RUN ERROR", repr(r), flush=True)
            continue
        status = (r["result"] or {}).get("status")
        print("RUN", r["id"], r["outcome"], status, r.get("wall_seconds"), flush=True)
        results.append(r)
    return results


@app.local_entrypoint()
def main(stage: str = "probe", only: str = ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    from research.layerrope_beta.intercept.plan import (
        CONFIRM_SEEDS,
        SCREEN_SEED,
        confirm_rows,
        job,
        primary,
        screen_rows,
        select,
    )

    if stage == "check":
        print("CHECK", check.remote())
        return
    if stage == "probe":
        calls = [
            probe.with_options(gpu=card).spawn("R,M,L,ML", card)
            for card in ("A10G", "L40S")
        ]
        for call in calls:
            print(call.get())
        return
    if stage == "screen":
        jobs = [job(r, "screen", SCREEN_SEED) for r in screen_rows()]
        if only:
            jobs = [j for j in jobs if j["row"] in only.split(",")]
        results = {r["id"]: r["result"] for r in fan_out(jobs)}
        if only:
            return
        scored = []
        for r in screen_rows():
            result = results.get(job(r, "screen", SCREEN_SEED)["id"])
            scored.append((r, primary(result) if result else None))
        missing = [r["id"] for r, v in scored if v is None]
        chosen = select([(r, v) for r, v in scored if r["arm"] in ("L", "ML")])
        selected = dict(
            chosen=chosen,
            screen={r["id"]: v for r, v in scored},
            invalid_or_missing=missing,
            frozen_before_confirmation=True,
        )
        put.remote("SELECTED.json", json.dumps(selected, indent=2))
        print("SELECTED", json.dumps(selected, indent=2))
        return
    if stage == "confirm":
        chosen = json.loads(read.remote("SELECTED.json"))["chosen"]
        jobs = [
            job(r, "final", seed)
            for seed in CONFIRM_SEEDS
            for r in confirm_rows(chosen)
        ]
        fan_out(jobs)
        return
    raise SystemExit("stage must be check, probe, screen or confirm")
