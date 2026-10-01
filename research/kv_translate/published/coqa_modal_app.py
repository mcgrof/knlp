#!/usr/bin/env python3
"""Modal wrapper for the frozen CoQA cross-model transfer stages."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import time

import modal

app = modal.App(os.environ.get("KNLP_MODAL_APP", "knlp-coqa-transfer"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==4.55.0",
        "accelerate",
        "safetensors",
        "numpy",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
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
        ignore=["__pycache__", "*.pyc", "coqa_modal_app.py"],
    )

hf = modal.Volume.from_name("knlp-hf", create_if_missing=False)
runs = modal.Volume.from_name("knlp-runs", create_if_missing=False)


def mirror(source, destination):
    """Copy changed regular files and return byte/file counts."""
    moved = files = 0
    if not os.path.exists(source):
        return {"bytes": 0, "files": 0}
    for root, _dirs, names in os.walk(source):
        rel = os.path.relpath(root, source)
        target_root = destination if rel == "." else os.path.join(destination, rel)
        os.makedirs(target_root, exist_ok=True)
        for name in names:
            if name.endswith(".tmp"):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(target_root, name)
            stat = os.stat(src)
            if os.path.exists(dst):
                old = os.stat(dst)
                if old.st_size == stat.st_size and old.st_mtime >= stat.st_mtime:
                    continue
            shutil.copyfile(src, dst)
            os.utime(dst, (stat.st_atime, stat.st_mtime))
            moved += stat.st_size
            files += 1
    return {"bytes": moved, "files": files}


@app.function(
    image=image,
    gpu="H200",
    cpu=4.0,
    memory=49152,
    timeout=12 * 3600,
    volumes={"/hf": hf, "/runs": runs},
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
)
def execute(stage: str, asset_dir: str, run_name: str, submitted: float):
    from research.kv_translate.published import coqa_runner

    if stage not in ("smoke", "q1", "q2"):
        raise ValueError(f"unknown stage {stage}")
    started = time.time()
    runs.reload()
    remote_assets = f"/runs/{asset_dir}"
    remote_out = f"/runs/{run_name}"
    local_assets = "/tmp/coqa-assets"
    local_out = "/tmp/coqa-run"
    os.makedirs(local_assets, exist_ok=True)
    os.makedirs(local_out, exist_ok=True)
    restore = {
        "assets": mirror(remote_assets, local_assets),
        "partial_outputs": mirror(remote_out, local_out),
    }
    checkpoint_state = {"last": 0.0, "commits": 0, "bytes": 0, "files": 0}

    def checkpoint():
        now = time.monotonic()
        if now - checkpoint_state["last"] < 5.0:
            return
        copied = mirror(local_out, remote_out)
        runs.commit()
        checkpoint_state["last"] = now
        checkpoint_state["commits"] += 1
        checkpoint_state["bytes"] += copied["bytes"]
        checkpoint_state["files"] += copied["files"]

    coqa_runner.ON_CHECKPOINT = checkpoint
    args = [
        "--stage",
        stage,
        "--frozen-dir",
        os.path.join(local_assets, "q0"),
        "--out-dir",
        local_out,
        "--forward-mapper",
        os.path.join(local_assets, "mapper_forward.safetensors"),
    ]
    if stage in ("smoke", "q2"):
        args += [
            "--reverse-mapper",
            os.path.join(local_assets, "mapper_reverse.safetensors"),
        ]
    try:
        coqa_runner.main(args)
        status = "complete"
    except Exception as error:
        status = "failed"
        with open(os.path.join(local_out, f"FAIL_{stage.upper()}.json"), "w") as stream:
            json.dump(
                {
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
        mirror(local_out, remote_out)
        runs.commit()
        raise
    finally:
        lifecycle = {
            "stage": stage,
            "status": status,
            "submitted_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(submitted)
            ),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "seconds_submit_to_start": started - submitted,
            "seconds_in_container": time.time() - started,
            "restore": restore,
            "durable_checkpoints": checkpoint_state,
        }
        with open(
            os.path.join(local_out, f"MODAL_{stage.upper()}.json"), "w"
        ) as stream:
            json.dump(lifecycle, stream, indent=2, sort_keys=True)
        exported = mirror(local_out, remote_out)
        runs.commit()
    return {"stage": stage, "status": status, "exported": exported, **lifecycle}


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    asset_dir: str = "coqa-quality-20261001/assets",
    run_name: str = "coqa-quality-20261001/run",
    gpu: str = "H200",
    cpu: float = 4.0,
    memory: int = 49152,
    timeout_hours: int = 12,
):
    submitted = time.time()
    print(
        json.dumps(
            {
                "stage": stage,
                "asset_dir": asset_dir,
                "run_name": run_name,
                "gpu": gpu,
                "cpu": cpu,
                "memory": memory,
                "timeout_hours": timeout_hours,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = execute.with_options(
        gpu=gpu, cpu=cpu, memory=memory, timeout=timeout_hours * 3600
    ).remote(stage, asset_dir, run_name, submitted)
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
