#!/usr/bin/env python3
"""Run a sweep of the CPU-scale cartridge harness on Modal, one arm per
small GPU.

Each arm is one ``smallcart.py run`` invocation on its own L4 container,
so a sweep that takes hours serially on one CPU host finishes in about the
time of its slowest arm. The harness is unchanged; this file is only the
platform: an image with the harness's library versions, a volume holding
the reader's weights, a volume for the per-arm results, and a fan-out.

Arms are short, so a preempted container is simply restarted from the
beginning. Weights are fetched once by a function with no card attached.

The sweeps follow a three-stage tuning order: the learning-rate range at
zero weight decay (``lr``), then weight decay at the winning rate
(``decay``), then half and twice that rate around the winning decay, with
either the decay coefficient or the decay timescale held fixed
(``nearby``). Run from the repository root::

    modal run research/cartridges_small/modal_sweep.py --sweep lr --smoke
    modal run --detach research/cartridges_small/modal_sweep.py --sweep lr
    modal run ... --sweep decay --lr 3e-2
    modal run ... --sweep nearby --lr 3e-2 --tau 100 --to init

``--smoke`` runs only the first arm, to measure time and cost per arm
before fanning out. ``--gpu`` picks the card (default L4); keep one card
type within a sweep, since arithmetic differs between cards. Results land in the ``knlp-runs`` volume under
``cartsmall/<sweep>/<arm>/`` (``result.json`` and ``run.log``).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

import modal

SWEEP_APP = os.environ.get("KNLP_MODAL_APP", "knlp-cartsmall")
app = modal.App(SWEEP_APP)

HERE = pathlib.Path(__file__).parent
MODEL = "Qwen/Qwen3-0.6B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.14.1",
        "transformers==5.21.0",
        "numpy",
        "huggingface_hub",
        "safetensors",
    )
    .env({"HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
)
if modal.is_local():
    image = image.add_local_file(
        HERE / "smallcart.py", "/root/cartsmall/smallcart.py", copy=False
    )

hf_vol = modal.Volume.from_name("knlp-hf", create_if_missing=True)
runs_vol = modal.Volume.from_name("knlp-runs", create_if_missing=True)

# The fixed part of every arm: the calibration recipe of the CPU rung
# (reader, synthetic record, 64 slots, 200 steps, warmup 20, schedule
# horizon 1,000, top-20 distillation, held-out evaluation). The learning
# rate and weight decay come from the sweep.
COMMON = [
    "--model",
    MODEL,
    "--fmt",
    "kv",
    "--entities",
    "8",
    "--slots",
    "64",
    "--steps",
    "200",
    "--warmup",
    "20",
    "--val-every",
    "10",
    "--eval-choice-every",
    "50",
    "--heldout-convs",
    "32",
    "--save-every",
    "0",
    "--sched-steps",
    "1000",
    "--init",
    "first_tokens",
]


SEEDS = (0, 1, 2)


def _seed(seed, lr, *extra):
    return ["--seed", str(seed), "--lr", lr] + list(extra)


def _decay(tau, to):
    return ["--decay-tau", repr(float(tau)), "--decay-to", to]


def _lr_arms(lr="", **_):
    """Stage 1: the learning-rate range at zero weight decay (the native
    Cartridges recipe has none), everything else fixed."""
    return [
        (f"lr{x}_s{seed}", _seed(seed, x))
        for x in ("3e-3", "1e-2", "3e-2", "6e-2", "1e-1")
        for seed in SEEDS
    ]


def _decay_arms(lr="3e-2", **_):
    """Stage 2: weight decay at a fixed learning rate. No decay, decay
    toward the starting prefix and decay toward zero, each at three
    timescales (tau = 1 / (lr * lambda) steps at the peak rate)."""
    arms = []
    for seed in SEEDS:
        arms.append((f"lr{lr}_nodecay_s{seed}", _seed(seed, lr)))
        for to in ("init", "zero"):
            for tau in (25, 100, 400):
                arms.append(
                    (f"lr{lr}_{to}_tau{tau}_s{seed}", _seed(seed, lr, *_decay(tau, to)))
                )
    return arms


def _nearby_arms(lr="3e-2", tau=100.0, to="init", **_):
    """Stage 3: half and twice the winning rate around the winning decay,
    once with the decay coefficient lambda held fixed (tau rescaled) and
    once with the timescale tau held fixed (lambda rescaled)."""
    lam = 1.0 / (float(lr) * float(tau))
    arms = []
    for x in (float(lr) / 2, float(lr) * 2):
        xs = f"{x:.4g}"
        for seed in SEEDS:
            arms.append(
                (
                    f"lr{xs}_{to}_lamfixed_s{seed}",
                    _seed(seed, xs, *_decay(1.0 / (x * lam), to)),
                )
            )
            arms.append(
                (f"lr{xs}_{to}_taufixed_s{seed}", _seed(seed, xs, *_decay(tau, to)))
            )
    return arms


SWEEPS = {"lr": _lr_arms, "decay": _decay_arms, "nearby": _nearby_arms}


@app.function(image=image, volumes={"/hf": hf_vol}, timeout=1800, cpu=2.0)
def fetch_weights():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    hf_vol.commit()
    return "ok"


@app.function(
    image=image,
    gpu="L4",
    cpu=2.0,
    memory=12288,
    timeout=3600,
    volumes={"/hf": hf_vol, "/runs": runs_vol},
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
def run_arm(sweep: str, name: str, extra: list) -> dict:
    out = pathlib.Path("/tmp/out") / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = (
        ["python", "/root/cartsmall/smallcart.py", "run"]
        + COMMON
        + extra
        + ["--device", "cuda", "--out", str(out)]
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    log = proc.stdout + "\n" + proc.stderr
    dest = pathlib.Path("/runs/cartsmall") / sweep / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "run.log").write_text(log)
    res = out / "result.json"
    if res.exists():
        (dest / "result.json").write_text(res.read_text())
    runs_vol.commit()
    done = [ln for ln in log.splitlines() if ln.startswith("RUN_DONE")]
    return {
        "arm": name,
        "rc": proc.returncode,
        "wall_s": round(wall, 1),
        "done": done[-1] if done else log[-2000:],
    }


@app.local_entrypoint()
def main(
    sweep: str = "lr",
    smoke: bool = False,
    lr: str = "3e-2",
    tau: float = 100.0,
    to: str = "init",
    gpu: str = "L4",
):
    arms = SWEEPS[sweep](lr=lr, tau=tau, to=to)
    print(fetch_weights.remote())
    if smoke:
        arms = arms[:1]
    print(f"{sweep}: {len(arms)} arms on {gpu}")
    fn = run_arm.with_options(gpu=gpu)
    for r in fn.starmap([(sweep, n, e) for n, e in arms], order_outputs=False):
        print(f"ARM {r['arm']} rc={r['rc']} wall={r['wall_s']}s {r['done']}")
    print("SWEEP END")
