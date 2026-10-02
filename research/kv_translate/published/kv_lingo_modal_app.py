#!/usr/bin/env python3
"""Modal mechanics/cost pilot for the KV-Lingo 4B/8B reproduction."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import modal

app = modal.App(os.environ.get("KNLP_MODAL_APP", "knlp-kv-lingo-preflight"))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.14.0",
        "transformers==4.55.0",
        "accelerate",
        "safetensors",
        "numpy",
        "huggingface_hub",
        "pytest",
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
    image = (
        image.add_local_file(
            here.parent / "__init__.py", "/root/research/kv_translate/__init__.py"
        )
        .add_local_file(
            here.parents[2] / "tests/kv_translate/test_published_kv_lingo.py",
            "/root/tests/kv_translate/test_published_kv_lingo.py",
        )
        .add_local_dir(
            here,
            "/root/research/kv_translate/published",
            ignore=["__pycache__", "*.pyc", "kv_lingo_modal_app.py"],
        )
    )

hf = modal.Volume.from_name("knlp-hf", create_if_missing=True)
runs = modal.Volume.from_name("knlp-runs", create_if_missing=True)
PINS = (
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)


@app.function(image=image, cpu=2.0, memory=4096, timeout=10 * 60)
def unit_checks():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "/root/tests/kv_translate/test_published_kv_lingo.py",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@app.function(image=image, volumes={"/hf": hf}, cpu=4.0, memory=8192, timeout=3 * 3600)
def fetch_models():
    from huggingface_hub import snapshot_download

    receipts = []
    for name, revision in PINS:
        started = time.time()
        path = snapshot_download(name, revision=revision)
        size = sum(
            os.path.getsize(os.path.join(root, filename))
            for root, _dirs, files in os.walk(path)
            for filename in files
        )
        hf.commit()
        receipts.append(
            {
                "model": name,
                "revision": revision,
                "bytes": size,
                "seconds": time.time() - started,
            }
        )
    return receipts


@app.function(
    image=image,
    gpu="H100",
    volumes={"/hf": hf, "/runs": runs},
    cpu=4.0,
    memory=65536,
    timeout=2 * 3600,
    retries=0,
)
def preflight(run_name: str, direction: str, cases: str, submitted: float):
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from research.kv_translate.published import kv_lingo_preflight

    started = time.time()
    out_dir = pathlib.Path("/runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = out_dir / f"PREFLIGHT_{direction.upper().replace('-', '_')}.json"
    status = kv_lingo_preflight.main(
        [
            "--direction",
            direction,
            "--cases",
            cases,
            "--out",
            str(receipt),
        ]
    )
    platform = {
        "schema": "kv_lingo_modal_preflight_v1",
        "run_name": run_name,
        "direction": direction,
        "cases": cases,
        "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(submitted)),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "submit_to_start_seconds": started - submitted,
        "container_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(),
        "status": status,
    }
    with (out_dir / f"MODAL_{direction.upper().replace('-', '_')}.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(platform, stream, indent=2, sort_keys=True)
        stream.write("\n")
    runs.commit()
    return platform


@app.local_entrypoint()
def main(
    run_name: str,
    direction: str = "4b-to-8b",
    cases: str = "256x64,2048x256,16384x2048",
    fetch: bool = False,
    check: bool = True,
):
    if check:
        result = unit_checks.remote()
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["returncode"]:
            raise SystemExit("unit checks failed")
    if fetch:
        print(json.dumps(fetch_models.remote(), indent=2, sort_keys=True))
    print(
        json.dumps(
            preflight.remote(run_name, direction, cases, time.time()),
            indent=2,
            sort_keys=True,
        )
    )
