#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the pinned exact-fork allocator experiment without GPU access."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import uuid

from verify import ROOT, verify


def settings(path):
    cfg = {"OUTPUT": "results/kmesh-forks", "DOCKER": "docker"}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_FORK_") and "=" in line:
                key, value = line.split("=", 1)
                cfg[key.removeprefix("CONFIG_KNLP_KMESH_FORK_")] = value.strip('"')
    for key in ("OUTPUT", "DOCKER", "VLLM_SOURCE", "RUNTIME_VOLUME"):
        cfg[key] = os.environ.get("KMESH_FORK_" + key, cfg.get(key, ""))
    return cfg


def identity(path):
    path = Path(path).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": git("status", "--porcelain"),
    }


def validate_source(source, profile):
    if source["dirty"] or any(
        source[key] != value for key, value in profile["vllm"].items()
    ):
        raise ValueError("require the clean pinned fork-accounting vLLM checkout")


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def container_command(cfg, source, output, action, profile, name):
    argv = [
        *shlex.split(cfg["DOCKER"]),
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    for src, dst, mode in (
        (str(ROOT), "/inputs", "ro"),
        (source["path"], "/sources/vllm", "ro"),
        (str(output), "/outputs", "rw"),
        (cfg["RUNTIME_VOLUME"], "/opt/kmesh-target", "ro"),
        ("/etc/passwd", "/etc/passwd", "ro"),
    ):
        argv += ["-v", f"{src}:{dst}:{mode}"]
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "NUMBA_CACHE_DIR": "/tmp/numba",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor",
        "KMESH_FORK_SOURCE": "/sources/vllm",
    }.items():
        argv += ["-e", key + "=" + value]
    return argv + [
        "--entrypoint",
        "/opt/kmesh-target/bin/python",
        profile["image"],
        "/inputs/native.py",
        action,
    ]


def execute(cfg, source, output, action, profile):
    name = "kmesh-forks-" + uuid.uuid4().hex
    argv = container_command(cfg, source, output, action, profile, name)
    write(output / "command.json", argv)
    prefix = shlex.split(cfg["DOCKER"])

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    process = None
    try:
        with (output / "native.log").open("x") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, argv)
    finally:
        subprocess.run(
            [*prefix, "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if process is not None:
            process.wait(timeout=30)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("action", choices=("plan", "run", "contracts", "verify"))
    args = parser.parse_args()
    cfg = settings(args.config)
    profile = json.loads((ROOT / "profile.json").read_text())
    output = Path(cfg["OUTPUT"]).resolve()
    if args.action == "plan":
        print(json.dumps({**profile, "output": str(output), "gpu": False}, indent=2))
        return
    if args.action == "verify":
        report = verify(json.loads((output / "results.json").read_text()))
        report["result_sha256"] = hashlib.sha256(
            (output / "results.json").read_bytes()
        ).hexdigest()
        write(output / "verified.json", report)
        print(json.dumps(report, indent=2))
        return
    if not cfg["VLLM_SOURCE"] or not cfg["RUNTIME_VOLUME"]:
        raise ValueError("set KMESH_FORK_VLLM_SOURCE and KMESH_FORK_RUNTIME_VOLUME")
    source = identity(cfg["VLLM_SOURCE"])
    validate_source(source, profile)
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "manifest.json",
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "profile": profile,
            "action": args.action,
            "gpu": False,
        },
    )
    execute(cfg, source, output, args.action, profile)
    validate_source(identity(cfg["VLLM_SOURCE"]), profile)
    if args.action == "run":
        report = verify(json.loads((output / "results.json").read_text()))
        report["result_sha256"] = hashlib.sha256(
            (output / "results.json").read_bytes()
        ).hexdigest()
        write(output / "verified.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
