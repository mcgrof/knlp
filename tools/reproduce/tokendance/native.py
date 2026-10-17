# SPDX-License-Identifier: MIT
"""Run the pinned CUDA baseline in an existing, reserved environment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from contract import (
    ROOT,
    profile,
    source_identity,
    validate_assets,
    validate_model,
    validate_sources,
)
from run import report, settings, trace, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument(
        "action", choices=("run", "preflight", "contracts", "verify", "codec")
    )
    args = parser.parse_args()
    cfg = settings(args.config)
    validate_assets()
    if os.environ.get("KMESH_IMAGE_DIGEST") != profile()["current_cuda"]["image"]:
        raise ValueError(
            "declare the pinned image actually used via KMESH_IMAGE_DIGEST"
        )
    identity = {
        name: source_identity(cfg[name.upper() + "_SOURCE"])
        for name in ("vllm", "lmcache")
    }
    validate_sources(identity, "current")
    model = Path(cfg["MODEL_DIR"]).absolute()
    validate_model(model)
    if not cfg.get("WHEEL_PACKAGE"):
        raise ValueError(
            "set TOKENDANCE_WHEEL_PACKAGE to the matching vLLM wheel package"
        )
    output = Path(cfg["OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "sources.json", identity)
    env = os.environ.copy()
    env.update(
        {
            "TOKENDANCE_PROFILE": "current-cuda",
            "TOKENDANCE_MODEL_DIR": str(model),
            "TOKENDANCE_VLLM_SOURCE": identity["vllm"]["path"],
            "TOKENDANCE_WHEEL_PACKAGE": cfg["WHEEL_PACKAGE"],
            "KMESH_IMAGE_DIGEST": profile()["current_cuda"]["image"],
            "KMESH_TARGET_VOLUME": profile()["current_cuda"]["target_volume"],
            "LMCACHE_CONFIG_FILE": str(ROOT / "lmcache-current.yaml"),
            "PYTHONPATH": identity["lmcache"]["path"]
            + os.pathsep
            + str(ROOT / "harness"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "OMP_NUM_THREADS": "2",
        }
    )

    def step(script, argv, label, gpu=False):
        step_env = env.copy()
        if not gpu:
            step_env.update(
                CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES=""
            )
        command = [
            sys.executable,
            str(ROOT / "harness/bootstrap.py"),
            script,
            *map(str, argv),
        ]
        write(output / (label + "-command.json"), command)
        with (output / (label + ".log")).open("x") as stream:
            subprocess.run(
                command,
                env=step_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    source_args = [
        "--vllm-source",
        identity["vllm"]["path"],
        "--lmcache-source",
        identity["lmcache"]["path"],
        "--wheel-vllm-package",
        cfg["WHEEL_PACKAGE"],
        "--lmcache-config",
        ROOT / "lmcache-current.yaml",
    ]
    try:
        if args.action in ("run", "contracts"):
            step(
                "verify_numerical_warmup_contract.py",
                ["--output", output / "contracts.json"],
                "contracts",
            )
        if args.action in ("run", "preflight"):
            step(
                "preflight_resolved_profile.py",
                [*source_args, "--output", output / "preflight.json"],
                "preflight",
                True,
            )
        if args.action == "run":
            trace(output / "trace")
            for index, arm in enumerate(profile()["arms"]):
                argv = [
                    "--arm",
                    arm,
                    *source_args,
                    "--trace",
                    output / "trace/trace.jsonl",
                    "--manifest",
                    output / "trace/manifest.json",
                    "--output-dir",
                    output / f"arms/{index:02d}-{arm}/artifacts",
                ]
                if arm != "ordinary":
                    argv += [
                        "--baseline-result",
                        output / "arms/00-ordinary/artifacts/result.json",
                    ]
                step("run_online_collective_lockstep_gate.py", argv, "arm-" + arm, True)
        if args.action in ("run", "verify"):
            bundle = output if args.action == "run" else Path(cfg["BUNDLE"]).resolve()
            argv = [
                "--source-pair",
                "current",
                "--output",
                output / "verified.json",
                "--checksums",
                output / "SHA256SUMS",
            ]
            for index, arm in enumerate(profile()["arms"]):
                argv += [
                    "--" + arm.replace("_", "-"),
                    bundle / f"arms/{index:02d}-{arm}/artifacts/result.json",
                ]
            step("verify_online_collective_lockstep_gate.py", argv, "verify")
            report(
                output / "verified.json", output / "report.json", "native_gpu_capture"
            )
        if args.action in ("run", "codec"):
            bundle = output if args.action == "run" else Path(cfg["BUNDLE"]).resolve()
            verification = (
                output / "verified.json"
                if args.action == "run"
                else Path(cfg["VERIFICATION"]).resolve()
            )
            step(
                "measure_family.py",
                [
                    "--bundle",
                    bundle,
                    "--verification",
                    verification,
                    "--lmcache-source",
                    identity["lmcache"]["path"],
                    "--output-dir",
                    output / "codec",
                ],
                "codec",
            )
    except BaseException as error:
        write(
            output / "failure.json",
            {
                "status": "failed",
                "exception": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
