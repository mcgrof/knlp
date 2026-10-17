#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare pinned LMCache restore paths inside an existing GPU reservation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from restore_common import (
    LMCACHE_COMMIT,
    VLLM_COMMIT,
    cold_l1,
    create_media,
    http,
    sha256,
    source_identity,
    write_json,
)

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    """Ask the loopback stack for an unused local port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop(process: subprocess.Popen) -> None:
    """Drain only the task's process group before its reservation can end."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def launch(script: str, arguments: list[str], log: Path) -> subprocess.Popen:
    """Start a supervised child with a separate log and process group."""
    with log.open("x") as stream:
        return subprocess.Popen(
            [sys.executable, str(HERE / script), *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def execute(script: str, arguments: list[str], log: Path) -> None:
    """Run one bounded stage and retain failures without continuing to timing."""
    process = launch(script, arguments, log)
    try:
        code = process.wait(timeout=1800)
        if code:
            raise RuntimeError(f"{script} failed with exit {code}: {log}")
    finally:
        stop(process)


def serving(config: dict, destination: Path, samples: int) -> None:
    """Prime one storage owner and use a fresh consumer engine for each arm."""
    destination.mkdir()
    cfg = config | {
        "output": str(destination),
        "media": str(destination / "owned-storage.bin"),
        "http_port": free_port(),
        "mq_port": free_port(),
        "serving_samples": samples,
    }
    create_media(Path(cfg["media"]))
    path = destination / "config.json"
    write_json(path, cfg)
    server = launch("server.py", [str(path)], destination / "server.log")
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"storage server failed: {destination / 'server.log'}"
                )
            try:
                http(cfg["http_port"], "healthcheck")
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise TimeoutError("storage server health check timed out")
        execute("worker.py", [str(path), "producer"], destination / "producer.log")
        execute("worker.py", [str(path), "consumer"], destination / "consumer.log")
        cold_l1(cfg["http_port"])
        write_json(destination / "terminal.json", http(cfg["http_port"], "status"))
    finally:
        stop(server)
        write_json(destination / "server-exit.json", {"returncode": server.returncode})


def main(args: argparse.Namespace) -> None:
    """Run a smoke gate, then balanced repeated direct and staged comparisons."""
    for name in ("output", "model", "vllm_source", "lmcache_source"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir()
    os.environ.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        HF_HUB_OFFLINE="1",
        LMCACHE_TRACK_USAGE="false",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_BATCH_INVARIANT="1",
        PYTHONHASHSEED="0",
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.lmcache_source),
            str(args.vllm_source),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    source = {
        "vllm": source_identity(args.vllm_source, VLLM_COMMIT),
        "lmcache": source_identity(args.lmcache_source, LMCACHE_COMMIT),
    }
    model_files = json.loads((HERE / "model-files.json").read_text())
    for name, expected in model_files.items():
        if sha256(args.model / name) != expected:
            raise RuntimeError(
                f"model artifact differs from the pinned revision: {name}"
            )
    native = {}
    for package in (args.vllm_source / "vllm", args.lmcache_source / "lmcache"):
        for binary in sorted(package.rglob("*.so")):
            native[str(binary)] = sha256(binary)
    raw_block_binary = Path(
        importlib.import_module("lmcache_rust_raw_block_io").__file__
    )
    native[str(raw_block_binary)] = sha256(raw_block_binary)
    write_json(
        args.output / "provenance.json",
        {
            "sources": source,
            "model_files": model_files,
            "harness": {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))},
            "native": native,
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
            "python": sys.version,
        },
    )
    import torch

    write_json(
        args.output / "runtime.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "kernel": os.uname().release,
            "threads": 2,
            "batch_invariant": True,
        },
    )
    config = {"model": str(args.model), "gpu_provider": args.gpu_provider}
    write_json(
        args.output / "plan.json",
        {
            "rounds": args.rounds,
            "adapter_samples": args.adapter_samples,
            "serving_samples": args.serving_samples if args.section == "all" else 0,
            "section": args.section,
            "fanouts": [1, 5],
            "mode_order": [
                list(("staged", "direct") if i % 2 == 0 else ("direct", "staged"))
                for i in range(args.rounds)
            ],
        },
    )
    config_path = args.output / "config.json"
    write_json(config_path, config)
    for mode in ("staged", "direct"):
        log = args.output / ("corruption-" + mode + ".log")
        process = launch(
            "adapter.py",
            [
                str(config_path),
                mode,
                str(args.output / ("corruption-" + mode)),
                "--samples",
                "1",
                "--corrupt",
            ],
            log,
        )
        try:
            code = process.wait(timeout=180)
            if code == 0 or "restored GPU consumer bytes differ" not in log.read_text():
                raise RuntimeError(
                    "corruption control did not fail at the exact-byte gate"
                )
        finally:
            stop(process)
        write_json(
            args.output / ("corruption-" + mode + "-rejected.json"),
            {"returncode": code, "expected_payload_rejection": True},
        )
    sections = ("adapter",) if args.section == "adapter" else ("adapter", "serving")
    for section in sections:
        if section == "serving":
            oracle = args.output / "oracle"
            oracle.mkdir()
            oracle_config = oracle / "config.json"
            write_json(
                oracle_config, config | {"output": str(oracle), "serving_samples": 1}
            )
            execute("worker.py", [str(oracle_config), "oracle"], oracle / "oracle.log")
        for label, adapter_samples, serving_samples, rounds in (
            ("smoke", 2, 1, 1),
            ("measured", args.adapter_samples, args.serving_samples, args.rounds),
        ):
            for repeat in range(rounds):
                modes = (
                    ("staged", "direct") if repeat % 2 == 0 else ("direct", "staged")
                )
                for mode in modes:
                    name = f"{label}-{repeat}-{mode}"
                    print("START", name, section, flush=True)
                    if section == "adapter":
                        execute(
                            "adapter.py",
                            [
                                str(config_path),
                                mode,
                                str(args.output / (name + "-adapter")),
                                "--samples",
                                str(adapter_samples),
                            ],
                            args.output / (name + "-adapter.log"),
                        )
                    else:
                        serving(
                            config | {"mode": mode},
                            args.output / (name + "-serving"),
                            serving_samples,
                        )
                    print("COMPLETE", name, section, flush=True)
            execute(
                "verify.py",
                [str(args.output), "--stage", label, "--section", section],
                args.output / (label + "-" + section + "-verify.log"),
            )
    if args.section == "all":
        execute("verify.py", [str(args.output)], args.output / "measured-verify.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--lmcache-source", type=Path, required=True)
    parser.add_argument(
        "--gpu-provider", choices=("native", "vulkan_rm"), default="native"
    )
    parser.add_argument("--adapter-samples", type=int, default=100)
    parser.add_argument("--serving-samples", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--section", choices=("adapter", "all"), default="all")
    args = parser.parse_args()
    if min(args.adapter_samples, args.serving_samples, args.rounds) <= 0:
        parser.error("sample and round counts must be positive")
    main(args)
