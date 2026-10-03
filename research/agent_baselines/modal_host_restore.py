#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify SGLang HiCache host restoration on a bounded Modal GPU function.

The test forces a small device KV pool, writes prefixes through to host DRAM,
fills the device pool with unrelated prompts, and then reissues the original
prompt.  A pass requires positive engine-reported host retrieval and output
parity with cold, freely generated recomputations.  Repeated cache-flushed
cold controls establish the chosen-token-logprob repeatability floor.

Run from the repository root::

    modal run --detach research/agent_baselines/modal_host_restore.py \
        --run-name prefixplan-20261002-modal-h100-cbe070b \
        --knlp-commit $(git rev-parse HEAD)
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import pathlib
import shutil
import subprocess
import time
import traceback
from typing import Any

import modal

APP_NAME = "knlp-prefixplan-host-restore"
SGLANG_REVISION = "cbe070b6bfce18936620f406b9b0d71bdc17452b"
SGLANG_BASE_IMAGE = (
    "lmsysorg/sglang:dev@"
    "sha256:352f1373e721c37d0849cf10ac9ec3c5b81dce76e148b9f271c1b59085eaba1d"
)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
REMOTE_ROOT = "/runs/agent-cache-baselines-20261002/prefixplan-host-restore"
COLD_REPEATS = 3

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(SGLANG_BASE_IMAGE, add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none https://github.com/sgl-project/sglang "
        "/opt/sglang-src",
        f"git -C /opt/sglang-src checkout {SGLANG_REVISION}",
        "python -m pip install --no-deps --no-build-isolation "
        "-e /opt/sglang-src/python",
    )
    .env(
        {
            "HF_HOME": "/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
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
        elif not entry.name.endswith(".tmp"):
            shutil.copy2(entry.path, target)


def _commit(local_dir: str, run_name: str) -> None:
    remote_dir = os.path.join(REMOTE_ROOT, run_name)
    _copy_tree(local_dir, remote_dir)
    runs.commit()


def _wait_for_server(process: subprocess.Popen, log_path: str, timeout: int = 600):
    import requests

    deadline = time.monotonic() + timeout
    url = "http://127.0.0.1:30000/health_generate"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = pathlib.Path(log_path).read_text(errors="replace")[-8000:]
            raise RuntimeError(f"SGLang exited before readiness:\n{tail}")
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("SGLang did not become ready within 600 seconds")


def _token_logprobs(response: dict) -> list[float]:
    content = response["choices"][0].get("logprobs", {}).get("content") or []
    values = [item.get("logprob") for item in content]
    if not values or any(not isinstance(value, (int, float)) for value in values):
        raise ValueError("Response did not contain chosen-token logprobs")
    return [float(value) for value in values]


def _max_logprob_delta(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Chosen-token logprob sequences have different lengths")
    return max(
        (abs(a - b) for a, b in zip(left, right, strict=True)),
        default=0.0,
    )


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("knlp_commit must be a full lowercase 40-character Git SHA")


def _sglext(response: dict) -> dict:
    extension = response.get("sglext")
    if not isinstance(extension, dict):
        raise ValueError("Response did not contain sglext")
    return extension


def _chat(input_ids: list[int]) -> tuple[dict, float]:
    import requests

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Token IDs supplied separately."}],
        "input_ids": input_ids,
        "temperature": 0,
        "seed": 17,
        "max_tokens": 16,
        "stream": False,
        "n": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "return_input_ids_in_sglext": True,
        "return_output_ids_in_sglext": True,
        "return_cached_tokens_details": True,
    }
    started = time.monotonic()
    response = requests.post(
        "http://127.0.0.1:30000/v1/chat/completions", json=payload, timeout=180
    )
    elapsed = time.monotonic() - started
    response.raise_for_status()
    data = response.json()
    extension = _sglext(data)
    if extension.get("input_ids") != input_ids:
        raise ValueError("SGLang response input IDs do not match the request")
    outputs = extension.get("output_ids")
    if not isinstance(outputs, list) or len(outputs) != 1 or not outputs[0]:
        raise ValueError("SGLang response lacks one nonempty output-ID sequence")
    return data, elapsed


def _prompt(tokenizer, marker: str, target_tokens: int) -> list[int]:
    unit = f"PrefixPlan cache pressure marker {marker}. "
    text = unit * math.ceil(target_tokens / 6)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    while len(tokens) < target_tokens:
        text += unit * 64
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[:target_tokens]


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800, volumes={"/hf": hf})
def fetch_model() -> dict:
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
    timeout=90 * 60,
    volumes={"/hf": hf, "/runs": runs},
    retries=modal.Retries(max_retries=1, initial_delay=5.0),
)
def qualify(run_name: str, knlp_commit: str) -> dict:
    import requests
    import sglang
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if not run_name or "/" in run_name or run_name.startswith("."):
        raise ValueError("run_name must be one safe path component")
    _validate_commit(knlp_commit)
    source_revision = subprocess.check_output(
        ["git", "-C", "/opt/sglang-src", "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != SGLANG_REVISION:
        raise RuntimeError(
            f"Expected SGLang source {SGLANG_REVISION}, got {source_revision}"
        )
    local_dir = os.path.join("/tmp", run_name)
    os.makedirs(local_dir, exist_ok=True)
    report_path = os.path.join(local_dir, "result.json")
    remote_report = os.path.join(REMOTE_ROOT, run_name, "result.json")
    runs.reload()
    if os.path.exists(remote_report):
        with open(remote_report, encoding="utf-8") as stream:
            prior = json.load(stream)
        if prior.get("status") == "passed":
            return prior

    model_path = snapshot_download(
        repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=MODEL_REVISION, local_files_only=True
    )
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
    server_environment = {
        **os.environ,
        # The pinned source checkout does not share the base image's bundled Rust
        # TreeCore extension.  Use SGLang's source-native backend explicitly.
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    }
    initial = {
        "status": "running",
        "run_name": run_name,
        "started_at": time.time(),
        "source": {
            "knlp_harness_commit": knlp_commit,
            "app_source_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()
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
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "server_command": command,
        "server_environment": {
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        },
    }
    _write_json(report_path, initial)
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
                env=server_environment,
            )
        _wait_for_server(process, log_path)

        target = _prompt(tokenizer, "TARGET-A", 3000)
        fillers = [_prompt(tokenizer, f"FILL-{index}", 3000) for index in range(4)]
        first, first_s = _chat(target)
        filler_results = []
        for index, filler in enumerate(fillers):
            response, elapsed = _chat(filler)
            filler_results.append(
                {
                    "index": index,
                    "seconds": elapsed,
                    "cache": _sglext(response).get("cached_tokens_details"),
                }
            )
        restored, restored_s = _chat(target)
        restored_extension = _sglext(restored)
        restored_cache = restored_extension.get("cached_tokens_details") or {}

        flush = requests.post(
            "http://127.0.0.1:30000/flush_cache",
            params={"timeout": 60},
            timeout=70,
        )
        flush.raise_for_status()
        cold_results = []
        for repeat in range(COLD_REPEATS):
            if repeat:
                flush = requests.post(
                    "http://127.0.0.1:30000/flush_cache",
                    params={"timeout": 60},
                    timeout=70,
                )
                flush.raise_for_status()
            cold, cold_s = _chat(target)
            cold_extension = _sglext(cold)
            cold_results.append(
                {
                    "repeat": repeat,
                    "seconds": cold_s,
                    "cache": cold_extension.get("cached_tokens_details") or {},
                    "output_ids": cold_extension["output_ids"][0],
                    "chosen_logprobs": _token_logprobs(cold),
                }
            )

        cold_reference = cold_results[0]
        cold_cache = cold_reference["cache"]

        restored_ids = restored_extension["output_ids"][0]
        cold_ids = cold_reference["output_ids"]
        restored_logprobs = _token_logprobs(restored)
        cold_logprobs = cold_reference["chosen_logprobs"]
        restored_cold_delta = _max_logprob_delta(restored_logprobs, cold_logprobs)
        cold_pair_deltas = []
        for left in range(len(cold_results)):
            for right in range(left + 1, len(cold_results)):
                cold_pair_deltas.append(
                    {
                        "left_repeat": left,
                        "right_repeat": right,
                        "max_abs_chosen_logprob_delta": _max_logprob_delta(
                            cold_results[left]["chosen_logprobs"],
                            cold_results[right]["chosen_logprobs"],
                        ),
                    }
                )
        cold_repeatability_delta = max(
            (item["max_abs_chosen_logprob_delta"] for item in cold_pair_deltas),
            default=0.0,
        )
        required_checks = {
            "positive_host_retrieval": restored_cache.get("host", 0) > 0,
            "restored_output_ids_equal_cold": restored_ids == cold_ids,
            "all_cold_runs_have_no_host_retrieval": all(
                item["cache"].get("host", 0) == 0 for item in cold_results
            ),
        }
        diagnostic_checks = {
            "chosen_logprob_lengths_equal": len(restored_logprobs)
            == len(cold_logprobs),
            "chosen_logprobs_within_1e-5": restored_cold_delta <= 1e-5,
            "chosen_logprobs_within_5e-3": restored_cold_delta <= 5e-3,
            "cold_output_ids_equal": all(
                item["output_ids"] == cold_ids for item in cold_results
            ),
            "restored_delta_within_cold_repeatability": (
                restored_cold_delta <= cold_repeatability_delta
            ),
        }
        result = {
            **initial,
            "status": "passed" if all(required_checks.values()) else "failed",
            "finished_at": time.time(),
            "workload": {
                "device_kv_capacity_tokens": 8192,
                "host_kv_capacity_gib": 1,
                "target_prompt_tokens": len(target),
                "filler_prompt_tokens": [len(item) for item in fillers],
                "generation_tokens": 16,
            },
            "timings_seconds": {
                "first": first_s,
                "restored": restored_s,
                "cold_recompute": cold_reference["seconds"],
                "cold_repeats": [item["seconds"] for item in cold_results],
            },
            "cache": {
                "first": _sglext(first).get("cached_tokens_details"),
                "fillers": filler_results,
                "restored": restored_cache,
                "cold_recompute": cold_cache,
                "cold_repeats": [item["cache"] for item in cold_results],
            },
            "parity": {
                "restored_output_ids": restored_ids,
                "cold_output_ids": cold_ids,
                "restored_chosen_logprobs": restored_logprobs,
                "cold_chosen_logprobs": cold_logprobs,
                "max_abs_chosen_logprob_delta": restored_cold_delta,
                "cold_repeats": cold_results,
                "cold_pair_deltas": cold_pair_deltas,
                "cold_repeatability_max_abs_chosen_logprob_delta": (
                    cold_repeatability_delta
                ),
            },
            "required_checks": required_checks,
            "diagnostic_checks": diagnostic_checks,
        }
        _write_json(report_path, result)
        summary = [
            "# PrefixPlan SGLang host-restore qualification",
            "",
            f"Status: **{result['status'].upper()}**",
            "",
            f"- Engine-reported restored cache: `{json.dumps(restored_cache, sort_keys=True)}`",
            f"- Cold recompute cache: `{json.dumps(cold_cache, sort_keys=True)}`",
            f"- Output token IDs equal: `{required_checks['restored_output_ids_equal_cold']}`",
            f"- Maximum chosen-token logprob delta: `{result['parity']['max_abs_chosen_logprob_delta']}`",
            f"- Maximum cold-vs-cold logprob delta: `{cold_repeatability_delta}`",
            f"- Restore delta within cold repeatability: `{diagnostic_checks['restored_delta_within_cold_repeatability']}`",
            f"- Strict 1e-5 logprob diagnostic: `{diagnostic_checks['chosen_logprobs_within_1e-5']}`",
            f"- Restored latency: `{restored_s:.6f}` seconds",
            f"- First cold recompute latency: `{cold_reference['seconds']:.6f}` seconds",
            "",
            "This is a synthetic forced-eviction micro-workload, not an agent-quality result.",
        ]
        pathlib.Path(local_dir, "REPORT.md").write_text(
            "\n".join(summary) + "\n", encoding="utf-8"
        )
        _commit(local_dir, run_name)
        if result["status"] != "passed":
            raise RuntimeError(f"Qualification checks failed: {required_checks}")
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
