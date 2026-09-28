#!/usr/bin/env python3
"""Run the paired comparison on Modal, and measure what the platform adds.

The science is in ``run_pair``; this file is only the platform. It is kept
thin on purpose, so the same stages can run on a rented pod from another
provider and the two bills can be compared for identical work.

Three things about this platform shape the design.

Accelerated functions can always be preempted, and cannot be made otherwise.
So the work directory is mirrored to a volume after every stage, and a
restarted function picks up at the first stage without a receipt.

Processor and memory are billed separately from the card, on whichever is
larger of what was requested and what was used. A generous request is
therefore a cost, not a safety margin, and the requests here are sized to the
preset rather than to the largest case.

Weights are fetched by a function with no card attached. On a rented pod the
download happens while the card is being paid for; here it costs processor
time only.

Run it from the repository root::

    modal run --detach research/kv_translate/published/modal_app.py \
        --preset small
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import time

import modal

# one application name per preset keeps their charges apart in the bill
app = modal.App(os.environ.get("KNLP_MODAL_APP", "knlp-published-transfer"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==4.55.0",
        "accelerate",
        "datasets==3.6.0",
        "lm_eval==0.4.9.1",
        "safetensors",
        "numpy",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        }
    )
)
if modal.is_local():
    here = pathlib.Path(__file__).resolve().parent
    image = image.add_local_file(
        here.parent / "__init__.py", "/root/research/kv_translate/__init__.py"
    ).add_local_dir(
        here,
        "/root/research/kv_translate/published",
        # the platform ships this file itself; shipping it twice is refused
        ignore=["__pycache__", "*.pyc", "modal_app.py"],
    )

hf = modal.Volume.from_name("knlp-hf", create_if_missing=True)
runs = modal.Volume.from_name("knlp-runs", create_if_missing=True)

PRESETS = {
    # a rehearsal on a pair with the same key-value geometry and, like the
    # real pair, differing depth and a receiver whose attention is not square
    "small": {
        "source": "Qwen/Qwen3-1.7B",
        "source_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "target": "Qwen/Qwen3-4B",
        "target_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "sequences": 100,
        "eval_examples": 100,
    },
    # the real pair on a fifth of the rows and examples: enough to measure
    # throughput, peak memory and the platform's overheads
    "qualify": {"sequences": 100, "eval_examples": 100},
    "full": {},
}

RESOURCES = {
    "small": {"gpu": "L40S", "cpu": 4.0, "memory": 32768, "timeout": 2 * 3600},
    "qualify": {"gpu": "H200", "cpu": 4.0, "memory": 65536, "timeout": 4 * 3600},
    "full": {"gpu": "H200", "cpu": 4.0, "memory": 32768, "timeout": 12 * 3600},
}


def mirror(src, dst):
    """Copy what is missing or changed. Returns bytes moved and seconds taken."""
    t0, moved = time.time(), 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in files:
            if fn.endswith(".tmp"):
                continue
            a, b = os.path.join(root, fn), os.path.join(dst, rel, fn)
            sa = os.stat(a)
            if os.path.exists(b):
                sb = os.stat(b)
                if sb.st_size == sa.st_size and sb.st_mtime >= sa.st_mtime:
                    continue
            shutil.copyfile(a, b)
            moved += sa.st_size
    return moved, time.time() - t0


@app.function(image=image, volumes={"/hf": hf}, cpu=4.0, memory=8192, timeout=3 * 3600)
def fetch(repos: list):
    """Download model snapshots with no card attached."""
    from huggingface_hub import snapshot_download

    out = []
    for name, rev in repos:
        t0 = time.time()
        p = snapshot_download(name, revision=rev)
        # snapshot entries are links into the blob store; getsize follows them
        size = sum(
            os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(p) for f in fs
        )
        out.append(
            {
                "repo": name,
                "revision": rev,
                "seconds": round(time.time() - t0, 1),
                "bytes": size,
            }
        )
        hf.commit()
    return out


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/hf": hf, "/runs": runs},
    cpu=4.0,
    memory=32768,
    timeout=2 * 3600,
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
)
def pair(cfg: dict, run_name: str, stop_after: str = "", t_submit: float = 0.0):
    import subprocess

    import torch

    from research.kv_translate.published import run_pair

    t_start = time.time()
    local = f"/tmp/work/{run_name}"
    remote = f"/runs/{run_name}"
    os.makedirs(local, exist_ok=True)
    os.makedirs(remote, exist_ok=True)
    restored, restore_s = mirror(remote, local)

    overhead = {"mirror_out_bytes": 0, "mirror_out_seconds": 0.0, "commits": 0}

    def on_stage_done(_name):
        t0 = time.time()
        moved, _secs = mirror(local, remote)
        runs.commit()
        overhead["mirror_out_bytes"] += moved
        overhead["mirror_out_seconds"] += time.time() - t0
        overhead["commits"] += 1

    # starts are appended, so a restart after a preemption shows up as a
    # second line rather than overwriting the first
    with open(os.path.join(local, "STARTS.jsonl"), "a") as f:
        f.write(
            json.dumps(
                {
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seconds_submit_to_running": (
                        round(t_start - t_submit, 1) if t_submit else None
                    ),
                    "restored_bytes": restored,
                    "restore_seconds": round(restore_s, 1),
                }
            )
            + "\n"
        )

    summary = run_pair.run(
        cfg, local, on_stage_done=on_stage_done, stop_after=stop_after or None
    )
    props = torch.cuda.get_device_properties(0)
    with open(os.path.join(local, "STARTS.jsonl")) as f:
        starts = [json.loads(line) for line in f if line.strip()]
    platform = {
        "provider": "modal",
        "card_memory_gib": round(props.total_memory / 2**30, 1),
        "card": torch.cuda.get_device_name(0),
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "processors_visible": os.cpu_count(),
        "starts": starts,
        "seconds_in_this_start": round(time.time() - t_start, 1),
        "durability": {
            k: round(v, 1) if isinstance(v, float) else v for k, v in overhead.items()
        },
    }
    with open(os.path.join(local, "PLATFORM.json"), "w") as f:
        json.dump(platform, f, indent=2, sort_keys=True)
    mirror(local, remote)
    runs.commit()
    return {"summary": summary, "platform": platform}


@app.function(
    image=image,
    gpu="H200",
    volumes={"/hf": hf, "/runs": runs},
    cpu=4.0,
    memory=32768,
    timeout=2 * 3600,
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
def parity(name: str, revision: str, tasks: list, examples: int, dtypes: list):
    """One-pass against from-cache scoring for one receiver."""
    from research.kv_translate.published import parity as par

    out = par.run(name, revision, tasks, examples, tuple(dtypes))
    os.makedirs("/runs/parity", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    with open(f"/runs/parity/{name.split('/')[-1]}-{stamp}.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    runs.commit()
    return out


@app.local_entrypoint()
def check_parity(
    model: str = "target",
    preset: str = "qualify",
    examples: int = 100,
    dtypes: str = "float32",
    gpu: str = "H200",
    memory: int = 32768,
):
    from research.kv_translate.published.run_pair import DEFAULT_TASKS, default_config

    cfg = default_config()
    cfg.update(PRESETS[preset])
    t0 = time.time()
    out = parity.with_options(gpu=gpu, memory=memory).remote(
        cfg[model],
        cfg[f"{model}_revision"],
        list(DEFAULT_TASKS),
        examples,
        dtypes.split(","),
    )
    out["wall_seconds_around_call"] = round(time.time() - t0, 1)
    print("PARITY_RESULT " + json.dumps(out, sort_keys=True), flush=True)


@app.local_entrypoint()
def main(
    preset: str = "small",
    run_name: str = "",
    stop_after: str = "",
    gpu: str = "",
    cpu: float = 0.0,
    memory: int = 0,
    fetch_only: bool = False,
    skip_fetch: bool = False,
):
    from research.kv_translate.published.run_pair import default_config

    cfg = default_config()
    cfg.update(PRESETS[preset])
    res = dict(RESOURCES[preset])
    if gpu:
        res["gpu"] = gpu
    if cpu:
        res["cpu"] = cpu
    if memory:
        res["memory"] = memory
    run_name = run_name or f"{preset}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    print(f"RUN {run_name} preset={preset} resources={res}", flush=True)

    if not skip_fetch:
        t0 = time.time()
        got = fetch.remote(
            [
                (cfg["source"], cfg["source_revision"]),
                (cfg["target"], cfg["target_revision"]),
            ]
        )
        print(
            "FETCH "
            + json.dumps({"wall_seconds": round(time.time() - t0, 1), "repos": got}),
            flush=True,
        )
    if fetch_only:
        return

    t0 = time.time()
    fn = pair.with_options(
        gpu=res["gpu"], cpu=res["cpu"], memory=res["memory"], timeout=res["timeout"]
    )
    out = fn.remote(cfg, run_name, stop_after, t0)
    out["wall_seconds_around_call"] = round(time.time() - t0, 1)
    out["requested"] = res
    out["run_name"] = run_name
    print("RESULT " + json.dumps(out, sort_keys=True), flush=True)
