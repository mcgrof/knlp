#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resolve the restore defconfig into the same pinned benchmark invocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from restore_common import (
    CHUNK_BYTES,
    CHUNKS,
    LMCACHE_COMMIT,
    SINGLEFLIGHT_COMMIT,
    VLLM_COMMIT,
)

HERE = Path(__file__).resolve().parent
FIELDS = (
    "OUTPUT",
    "MODEL_DIR",
    "VLLM_SOURCE",
    "LMCACHE_SOURCE",
    "GPU_PROVIDER",
    "SECTION",
)
DEFAULTS = {
    "OUTPUT": "results/kmesh-restore",
    "GPU_PROVIDER": "vulkan_rm",
    "SECTION": "adapter",
}


def settings(path: Path, environment: dict) -> dict:
    """Resolve explicit environment values before defconfig paths and defaults."""
    result = DEFAULTS.copy()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_RESTORE_"):
                key, value = line.split("=", 1)
                result[key.removeprefix("CONFIG_KNLP_KMESH_RESTORE_")] = json.loads(
                    value
                )
    for key in FIELDS:
        if "KMESH_RESTORE_" + key in environment:
            result[key] = environment["KMESH_RESTORE_" + key]
    return result


def command(config: dict) -> list[str]:
    """Construct an argv array without interpreting path text through a shell."""
    flags = {
        "OUTPUT": "output",
        "MODEL_DIR": "model",
        "VLLM_SOURCE": "vllm-source",
        "LMCACHE_SOURCE": "lmcache-source",
        "GPU_PROVIDER": "gpu-provider",
        "SECTION": "section",
    }
    result = [sys.executable, str(HERE / "run.py")]
    if config["SECTION"] == "singleflight":
        result = [sys.executable, str(HERE / "singleflight.py")]
        flags.pop("SECTION")
    for key, flag in flags.items():
        result.extend(["--" + flag, config.get(key, "")])
    if config["SECTION"] == "coldburst":
        result.extend(
            ["--rounds", "1", "--serving-samples", "3", "--adapter-samples", "1"]
        )
    return result


def main() -> None:
    """Print the plan, execute the experiment, or verify a capture on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "verify"))
    parser.add_argument("--config", type=Path, default=Path(".config"))
    args = parser.parse_args()
    config = settings(args.config, os.environ)
    missing = [name for name in FIELDS if not config.get(name)]
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "vllm_commit": VLLM_COMMIT,
                    "lmcache_commit": (
                        SINGLEFLIGHT_COMMIT
                        if config["SECTION"] == "singleflight"
                        else LMCACHE_COMMIT
                    ),
                    "ancestor_bytes": CHUNK_BYTES * CHUNKS,
                    "rounds": (
                        1 if config["SECTION"] in ("coldburst", "singleflight") else 4
                    ),
                    "section": config["SECTION"],
                    "adapter_samples_per_round": (
                        0 if config["SECTION"] in ("coldburst", "singleflight") else 100
                    ),
                    "serving_families_per_fanout_per_round": (
                        3
                        if config["SECTION"] == "coldburst"
                        else (20 if config["SECTION"] in ("all", "singleflight") else 0)
                    ),
                    "command": command(config),
                    "missing_inputs": missing,
                    "requires": "prepared native runtime, qualified storage route and an available GPU",
                },
                indent=2,
            )
        )
    elif args.action == "verify" and config["SECTION"] == "singleflight":
        subprocess.run(
            [
                sys.executable,
                str(HERE / "singleflight.py"),
                "--verify",
                "--output",
                config["OUTPUT"],
            ],
            check=True,
        )
    elif args.action == "verify":
        subprocess.run(
            [
                sys.executable,
                str(HERE / "verify.py"),
                config["OUTPUT"],
                "--section",
                config["SECTION"],
            ],
            check=True,
        )
    else:
        if missing:
            parser.error("set KMESH_RESTORE_" + ", KMESH_RESTORE_".join(missing))
        Path(config["OUTPUT"]).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command(config), check=True)


if __name__ == "__main__":
    main()
