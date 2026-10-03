#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Serve the pinned live-agent model through a temporary Modal endpoint.

Create the named Modal secret before deployment.  Its ``VLLM_API_KEY`` value
must also be exported as ``OPENAI_API_KEY`` for the live client.  The secret is
never written to a run manifest or placed in the endpoint URL.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

import modal

APP_NAME = "knlp-prefixplan-live-server"
API_SECRET_NAME = "knlp-prefixplan-live-api"
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.13.0@"
    "sha256:1ae45c63e0578cf8ea690299c20b6a32a5c3ddf659e92ea15c0ae6b8040ba741"
)
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
SERVED_MODEL = "served-model"

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
    .env(
        {
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )
    .entrypoint([])
)
hf = modal.Volume.from_name("knlp-hf", create_if_missing=False)
api_secret = modal.Secret.from_name(API_SECRET_NAME)


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=2 * 60 * 60,
    volumes={"/hf": hf},
)
def fetch_model() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    started = time.time()
    path = snapshot_download(repo_id=MODEL, revision=MODEL_REVISION)
    hf.commit()
    return {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "path": path,
        "seconds": time.time() - started,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=10.0,
    memory=32768,
    timeout=24 * 60 * 60,
    scaledown_window=15 * 60,
    max_containers=1,
    volumes={"/hf": hf},
    secrets=[api_secret],
)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve() -> None:
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True
    )
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError(f"Modal secret {API_SECRET_NAME} lacks VLLM_API_KEY")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-seqs",
        "1",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--generation-config",
        "vllm",
    ]
    subprocess.Popen(command, stdin=subprocess.DEVNULL)


@app.local_entrypoint()
def main(fetch: bool = False):
    if not fetch:
        raise ValueError("Use --fetch to populate the model volume before deploy")
    print(fetch_model.remote())
