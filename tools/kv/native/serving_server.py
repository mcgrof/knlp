# SPDX-License-Identifier: MIT
"""Prepare and launch the matched Qwen3 HTTP serving arms on loopback."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .current import fixture_root, FLAGS, save, sha
from .october import VARIANTS, check_scales


def prepare(install, output, weight_format, arm, cache_bytes=None):
    variant = "mixed_prefill" if arm == "candidate" else "native"
    definition = VARIANTS[variant]
    definition.verify_sources({name: install / name for name in definition.sources})
    source = Path(__file__).resolve().parents[3]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", "tools/kv"], text=True
    ).strip():
        raise ValueError("use committed, clean measurement code")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    fixture = fixture_root() / weight_format
    protocol = json.loads((fixture / "measured-protocol.json").read_text())
    files = json.loads((fixture / "manifest.json").read_text())
    for name, item in files.items():
        archive = fixture / item["file"]
        if sha(archive) != item["archive_sha256"]:
            raise ValueError("fixture archive changed")
        data = archive.read_bytes()
        (inputs / name).write_bytes(
            data if name.endswith(".gz") else gzip.decompress(data)
        )
        if sha(inputs / name) != item["sha256"]:
            raise ValueError("fixture bytes changed")
    check_scales(json.loads((inputs / "scales.json").read_text()), protocol["geometry"])
    protocol["measured_harness_sha"] = protocol["source_sha"]
    protocol["source_sha"] = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    protocol["runner_files_sha256"] = {
        p.name: sha(p) for p in Path(__file__).parent.glob("*.py")
    }
    protocol["inputs_sha256"] = {name: sha(inputs / name) for name in files}
    protocol["backend_sources"] = dict(definition.sources)
    engine = protocol["engine"]
    engine["worker_cls"] = "tools.kv.native.serving_worker.ServingWorker"
    engine["enable_prefix_caching"] = False
    engine["stream_interval"] = 1
    if cache_bytes is not None:
        engine["kv_cache_memory_bytes"] = cache_bytes
    if arm == "native_fa3":
        engine["attention_config"] = {"backend": "FLASH_ATTN", "flash_attn_version": 3}
    save(inputs / "protocol.json", protocol)
    for name in definition.sources:
        (inputs / name).symlink_to(install / name)
    env = {**os.environ, **FLAGS, **definition.environment({})}
    env.update(
        {
            "VLLM_COMPACT_PREFILL": str(int(variant == "mixed_prefill")),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "KNLP_SERVING_INPUTS": str(inputs),
            "KNLP_SERVING_OUTPUT": str(output),
            "KNLP_SERVING_VARIANT": variant,
            "KNLP_SERVING_ARM": "k16_v8" if arm == "candidate" else "k16_v16",
            "KNLP_NATIVE_ALLOC_TRACE": str(output / "allocation.jsonl"),
            "KNLP_NATIVE_TORCH_LIMIT_BYTES": str(
                protocol["torch_allocator_reserved_cap_bytes"]
            ),
        }
    )
    save(
        output / "environment.json",
        {
            key: value
            for key, value in env.items()
            if key in FLAGS or key.startswith("KNLP_") or key == "VLLM_SERVER_DEV_MODE"
        },
    )
    manifest = json.loads(gzip.decompress((inputs / "qwen.json.gz").read_bytes()))
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "study",
        "--model",
        manifest["model"],
        "--revision",
        manifest["model_revision"],
        "--tokenizer-revision",
        manifest["tokenizer_revision"],
        "--kv-cache-dtype",
        "auto,fp8_e4m3" if arm == "candidate" else "auto",
        "--no-enable-log-requests",
    ]
    for key, value in engine.items():
        flag = key.replace("_", "-")
        if key == "disable_log_stats" and value is False:
            # This legacy flag uses store_true, unlike config booleans.
            continue
        if isinstance(value, bool):
            argv.append("--" + ("" if value else "no-") + flag)
        elif isinstance(value, dict):
            argv.extend(["--" + flag, json.dumps(value)])
        elif value is not None:
            argv.extend(["--" + flag, str(value)])
    save(
        output / "launch.json",
        {
            "argv": argv,
            "arm": arm,
            "format": weight_format,
            "protocol_sha256": sha(inputs / "protocol.json"),
        },
    )
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("awq", "fp16"), required=True)
    parser.add_argument(
        "--arm", choices=("candidate", "native_fi", "native_fa3"), required=True
    )
    parser.add_argument("--cache-bytes", type=int)
    args = parser.parse_args()
    argv, env = prepare(
        args.install.resolve(),
        args.output.resolve(),
        args.format,
        args.arm,
        args.cache_bytes,
    )
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
