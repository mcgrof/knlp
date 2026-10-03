#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify shared and isolated SGLang cache scopes on a bounded Modal GPU."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import time
import traceback
from typing import Any

import modal

APP_NAME = "knlp-prefixplan-sglang-cache-scope"
SGLANG_REVISION = "cbe070b6bfce18936620f406b9b0d71bdc17452b"
SGLANG_BASE_IMAGE = (
    "lmsysorg/sglang:dev@"
    "sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d"
)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REMOTE_ROOT = "/runs/agent-cache-baselines-20261003/prefixplan-sglang-cache-scope"
ADAPTER_NAME = "sglang_adapter.py"
ADAPTER_REMOTE = f"/opt/knlp/{ADAPTER_NAME}"

app = modal.App(APP_NAME)
image = modal.Image.from_registry(SGLANG_BASE_IMAGE, add_python="3.12").apt_install(
    "git"
)
if modal.is_local():
    adapter_path = pathlib.Path(__file__).with_name(ADAPTER_NAME)
    image = image.add_local_file(adapter_path, ADAPTER_REMOTE, copy=True)
image = image.run_commands(
    "git clone --filter=blob:none https://github.com/sgl-project/sglang "
    "/opt/sglang-src",
    f"git -C /opt/sglang-src checkout {SGLANG_REVISION}",
    "python -m pip install --no-deps --no-build-isolation " "-e /opt/sglang-src/python",
).env(
    {
        "HF_HOME": "/hf",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
)
hf = modal.Volume.from_name("knlp-hf", create_if_missing=False)
runs = modal.Volume.from_name("knlp-runs", create_if_missing=False)


def _write_json(path: str, value: Any) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _copy_tree(source: str, destination: str) -> None:
    os.makedirs(destination, exist_ok=True)
    for entry in os.scandir(source):
        target = os.path.join(destination, entry.name)
        if entry.is_dir():
            _copy_tree(entry.path, target)
        elif not entry.name.endswith((".tmp", ".partial")):
            shutil.copy2(entry.path, target)


def _commit(local_dir: str, run_name: str) -> None:
    _copy_tree(local_dir, os.path.join(REMOTE_ROOT, run_name))
    runs.commit()


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("knlp_commit must be a full lowercase 40-character Git SHA")


def _wait_for_server(process: subprocess.Popen, log_path: str, timeout: int = 600):
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = pathlib.Path(log_path).read_text(errors="replace")[-8000:]
            raise RuntimeError(f"SGLang exited before readiness:\n{tail}")
        try:
            response = requests.get("http://127.0.0.1:30000/health_generate", timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("SGLang did not become ready within 600 seconds")


def _prompt(tokenizer, target_tokens: int = 3000) -> list[int]:
    unit = "PrefixPlan compatible shared cache prefix. "
    text = unit * math.ceil(target_tokens / 6)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    while len(tokens) < target_tokens:
        text += unit * 64
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[:target_tokens]


def _chat(input_ids: list[int], extensions: dict[str, Any]) -> tuple[dict, float]:
    import requests

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Token IDs supplied separately."}],
        "input_ids": input_ids,
        "temperature": 0,
        "seed": 17,
        "max_tokens": 8,
        "stream": False,
        "n": 1,
        **extensions,
    }
    started = time.monotonic()
    response = requests.post(
        "http://127.0.0.1:30000/v1/chat/completions", json=payload, timeout=180
    )
    elapsed = time.monotonic() - started
    response.raise_for_status()
    data = response.json()
    extension = data.get("sglext")
    if not isinstance(extension, dict) or extension.get("input_ids") != input_ids:
        raise ValueError("SGLang response does not preserve the requested input IDs")
    outputs = extension.get("output_ids")
    if not isinstance(outputs, list) or len(outputs) != 1 or not outputs[0]:
        raise ValueError("SGLang response lacks one nonempty output-ID sequence")
    cache = extension.get("cached_tokens_details")
    if cache is not None and not isinstance(cache, dict):
        raise ValueError("SGLang response has invalid cache-source details")
    return data, elapsed


def _response_receipt(response: dict, seconds: float) -> dict[str, Any]:
    extension = response["sglext"]
    return {
        "seconds": seconds,
        # The pinned server omits an all-zero detail object. Missing therefore
        # means a cold request, while a claimed hit must carry positive details.
        "cache": extension.get("cached_tokens_details") or {},
        "output_ids": extension["output_ids"][0],
    }


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800, volumes={"/hf": hf})
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
    cpu=4.0,
    memory=24576,
    timeout=60 * 60,
    volumes={"/hf": hf, "/runs": runs},
)
def qualify(run_name: str, knlp_commit: str) -> dict[str, Any]:
    import sys

    import sglang
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    sys.path.insert(0, "/opt/knlp")
    from sglang_adapter import capability_receipt, request_extensions

    if not run_name or "/" in run_name or run_name.startswith("."):
        raise ValueError("run_name must be one safe path component")
    _validate_commit(knlp_commit)
    source_revision = subprocess.check_output(
        ["git", "-C", "/opt/sglang-src", "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != SGLANG_REVISION:
        raise RuntimeError("SGLang source revision mismatch")

    local_dir = os.path.join("/tmp", run_name)
    remote_result = os.path.join(REMOTE_ROOT, run_name, "result.json")
    os.makedirs(local_dir, exist_ok=True)
    runs.reload()
    if os.path.exists(remote_result):
        prior = json.loads(pathlib.Path(remote_result).read_text())
        if prior.get("status") == "passed":
            return prior
        raise RuntimeError(f"Existing non-passing result refuses overwrite: {run_name}")

    model_path = snapshot_download(
        repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dtype",
        "bfloat16",
        "--context-length",
        "4096",
        "--max-total-tokens",
        "8192",
        "--max-running-requests",
        "1",
        "--chunked-prefill-size",
        "4096",
        "--enable-hierarchical-cache",
        "--hicache-size",
        "1",
        "--hicache-write-policy",
        "write_through",
        "--hicache-io-backend",
        "kernel",
        "--hicache-mem-layout",
        "page_first",
        "--enable-cache-report",
        "--disable-cuda-graph",
    ]
    environment = {
        **os.environ,
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    }
    initial = {
        "status": "running",
        "run_name": run_name,
        "started_at": time.time(),
        "source": {
            "knlp_harness_commit": knlp_commit,
            "harness_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
            ).hexdigest(),
            "adapter_sha256": hashlib.sha256(
                pathlib.Path(ADAPTER_REMOTE).read_bytes()
            ).hexdigest(),
            "sglang_revision": SGLANG_REVISION,
            "sglang_checkout_revision": source_revision,
            "sglang_version": sglang.__version__,
            "base_image": SGLANG_BASE_IMAGE,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "server_command": command,
    }
    _write_json(os.path.join(local_dir, "result.json"), initial)
    _commit(local_dir, run_name)

    log_path = os.path.join(local_dir, "server.log")
    process = None
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=environment,
            )
        _wait_for_server(process, log_path)

        receipt = capability_receipt(
            ["openai_chat", "sglext_token_ids", "cache_source_report"]
        )
        shared_scope = "tenant-a:qwen2.5:no-adapter:campaign-a"
        isolated_scope = "tenant-b:qwen2.5:no-adapter:campaign-a"
        extension_receipts = {
            "first": request_extensions(
                receipt, request_id="task-a:1", cache_scope=shared_scope
            ),
            "shared": request_extensions(
                receipt, request_id="task-b:1", cache_scope=shared_scope
            ),
            "isolated": request_extensions(
                receipt, request_id="task-c:1", cache_scope=isolated_scope
            ),
        }
        prompt = _prompt(tokenizer)
        first_response, first_s = _chat(prompt, extension_receipts["first"])
        shared_response, shared_s = _chat(prompt, extension_receipts["shared"])
        isolated_response, isolated_s = _chat(prompt, extension_receipts["isolated"])
        responses = {
            "first": _response_receipt(first_response, first_s),
            "shared": _response_receipt(shared_response, shared_s),
            "isolated": _response_receipt(isolated_response, isolated_s),
        }
        shared_cache = responses["shared"]["cache"]
        isolated_cache = responses["isolated"]["cache"]
        isolated_cached_tokens = sum(
            isolated_cache.get(tier, 0) for tier in ("device", "host", "storage")
        )
        checks = {
            "adapter_shared_scope_equal": (
                extension_receipts["first"]["extra_key"]
                == extension_receipts["shared"]["extra_key"]
            ),
            "adapter_request_ids_distinct": (
                extension_receipts["first"]["rid"]
                != extension_receipts["shared"]["rid"]
            ),
            "adapter_isolated_scope_distinct": (
                extension_receipts["first"]["extra_key"]
                != extension_receipts["isolated"]["extra_key"]
            ),
            "shared_scope_cache_hit_positive": shared_cache.get("device", 0) > 0,
            "isolated_scope_cache_hit_zero": isolated_cached_tokens == 0,
            "outputs_identical": len(
                {tuple(item["output_ids"]) for item in responses.values()}
            )
            == 1,
        }
        result = {
            **initial,
            "status": "passed" if all(checks.values()) else "failed",
            "finished_at": time.time(),
            "prompt_tokens": len(prompt),
            "extensions": extension_receipts,
            "responses": responses,
            "checks": checks,
            "claim_boundary": (
                "A bounded cache-identity mechanism check. It establishes a live "
                "same-scope hit and different-scope isolation, not agent quality or "
                "host-tier restoration."
            ),
        }
        _write_json(os.path.join(local_dir, "result.json"), result)
        _commit(local_dir, run_name)
        if result["status"] != "passed":
            raise RuntimeError(f"SGLang cache-scope checks failed: {checks}")
        return result
    except BaseException as error:
        failure = {
            **initial,
            "status": "error",
            "finished_at": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(os.path.join(local_dir, "failure.json"), failure)
        _commit(local_dir, run_name)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        if os.path.exists(log_path):
            _commit(local_dir, run_name)


@app.local_entrypoint()
def main(run_name: str, knlp_commit: str):
    fetched = fetch_model.remote()
    print("MODEL_FETCH " + json.dumps(fetched, sort_keys=True), flush=True)
    result = qualify.remote(run_name, knlp_commit)
    print("QUALIFICATION_RESULT " + json.dumps(result, sort_keys=True), flush=True)
