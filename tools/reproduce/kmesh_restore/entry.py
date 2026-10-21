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
    FAMILY_PREFETCH_COMMIT,
    LMCACHE_COMMIT,
    RETAINED_DEMAND_COMMIT,
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
PREFETCH_FIELDS = ("PHASE", "QUALIFIED", "PILOT")
DEFAULTS = {
    "OUTPUT": "results/kmesh-restore",
    "GPU_PROVIDER": "vulkan_rm",
    "SECTION": "adapter",
    "PHASE": "qualification",
    "QUALIFIED": "results/kmesh-prefetch/qualification",
    "PILOT": "results/kmesh-prefetch/pilot",
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
    for key in FIELDS + PREFETCH_FIELDS:
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
    elif config["SECTION"] == "prefetch":
        result = [sys.executable, str(HERE / "prefetch_run.py")]
        flags.pop("SECTION")
    for key, flag in flags.items():
        result.extend(["--" + flag, config.get(key, "")])
    if config["SECTION"] == "prefetch":
        result.extend(["--phase", config["PHASE"]])
        if config["PHASE"] != "qualification":
            result.extend(["--qualified", config["QUALIFIED"]])
        if config["PHASE"] == "confirmation":
            result.extend(["--pilot", config["PILOT"]])
    if config["SECTION"] in ("coldburst", "family", "retained"):
        result.extend(
            [
                "--rounds",
                "1",
                "--serving-samples",
                "5" if config["SECTION"] in ("family", "retained") else "3",
                "--adapter-samples",
                "1",
            ]
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
    if config["SECTION"] == "prefetch":
        if config["PHASE"] not in ("qualification", "pilot", "confirmation"):
            parser.error("invalid KMESH_RESTORE_PHASE")
        required = ("QUALIFIED",) if config["PHASE"] != "qualification" else ()
        if config["PHASE"] == "confirmation":
            required += ("PILOT",)
        missing.extend(key for key in required if not config.get(key))
    if args.action == "plan":
        if config["SECTION"] == "prefetch":
            print(
                json.dumps(
                    {
                        "vllm_commit": VLLM_COMMIT,
                        "lmcache_commit": RETAINED_DEMAND_COMMIT,
                        "ancestor_bytes": CHUNK_BYTES * CHUNKS,
                        "phase": config["PHASE"],
                        "blocks_per_path_fanout": {
                            "qualification": 1,
                            "pilot": 2,
                            "confirmation": 10,
                        }[config["PHASE"]],
                        "warmups_per_arm_block": 3,
                        "measured_groups_per_arm_block": {
                            "qualification": 1,
                            "pilot": 5,
                            "confirmation": 20,
                        }[config["PHASE"]],
                        "fanouts": [4, 8] if config["PHASE"] == "confirmation" else [4],
                        "command": command(config),
                        "missing_inputs": missing,
                        "requires": "prepared native runtime, qualified storage route and an available GPU",
                    },
                    indent=2,
                )
            )
            return
        print(
            json.dumps(
                {
                    "vllm_commit": VLLM_COMMIT,
                    "lmcache_commit": (
                        FAMILY_PREFETCH_COMMIT
                        if config["SECTION"] in ("family", "retained")
                        else (
                            SINGLEFLIGHT_COMMIT
                            if config["SECTION"] == "singleflight"
                            else LMCACHE_COMMIT
                        )
                    ),
                    "ancestor_bytes": CHUNK_BYTES * CHUNKS,
                    "rounds": (
                        1
                        if config["SECTION"]
                        in ("coldburst", "singleflight", "family", "retained")
                        else 4
                    ),
                    "section": config["SECTION"],
                    "adapter_samples_per_round": (
                        0
                        if config["SECTION"]
                        in ("coldburst", "singleflight", "family", "retained")
                        else 100
                    ),
                    "serving_families_per_fanout_per_round": (
                        5
                        if config["SECTION"] in ("family", "retained")
                        else (
                            3
                            if config["SECTION"] == "coldburst"
                            else (
                                20
                                if config["SECTION"] in ("all", "singleflight")
                                else 0
                            )
                        )
                    ),
                    "command": command(config),
                    "missing_inputs": missing,
                    "requires": "prepared native runtime, qualified storage route and an available GPU",
                },
                indent=2,
            )
        )
    elif args.action == "verify" and config["SECTION"] == "prefetch":
        subprocess.run(
            [sys.executable, str(HERE / "prefetch_campaign.py"), config["OUTPUT"]],
            check=True,
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
