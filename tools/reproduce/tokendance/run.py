#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate, run, and independently verify the shared-update recovery baseline."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import uuid

from contract import (
    ROOT,
    digest,
    profile,
    source_identity,
    validate_assets,
    validate_model,
    validate_sources,
    validate_trace,
)

H = ROOT / "harness"
MODEL_CONTAINER = (
    "/scratch/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    + profile()["model_revision"]
)


def settings(path):
    result = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_TOKENDANCE_") and "=" in line:
                key, value = line.split("=", 1)
                result[key.removeprefix("CONFIG_KNLP_TOKENDANCE_")] = value.strip('"')
    for key in [
        "VLLM_SOURCE",
        "LMCACHE_SOURCE",
        "OUTPUT",
        "RUNTIME_VOLUME",
        "MODEL_DIR",
        "DOCKER",
        "BUNDLE",
        "SOURCE_PAIR",
        "VERIFICATION",
    ]:
        if "TOKENDANCE_" + key in os.environ:
            result[key] = os.environ["TOKENDANCE_" + key]
    result.setdefault("MODE", "cpu")
    result.setdefault("OUTPUT", "results/tokendance")
    result.setdefault("DOCKER", "docker")
    result.setdefault("SOURCE_PAIR", "dated")
    if result["MODE"] not in ("cpu", "gpu"):
        raise ValueError("TOKENDANCE_MODE must be cpu or gpu")
    return result


def write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def command(argv, log=None, env=None):
    if log:
        with Path(log).open("x") as stream:
            subprocess.run(
                argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
    else:
        subprocess.run(argv, env=env, check=True)


def trace(output):
    command(
        [
            sys.executable,
            str(H / "generate_all_gather_trace.py"),
            "--case",
            "same_order_different_offsets",
            "--agents",
            "5",
            "--shared-fraction",
            "0.8",
            "--alignment",
            "unaligned",
            "--block-size",
            "32",
            "--qps",
            "2",
            "--mode",
            "prefill_focus",
            "--seed",
            "20260731",
            "--output-dir",
            str(output),
        ],
        output.parent / "trace.log",
    )
    return validate_trace(output)


def sources(cfg):
    found = {}
    for project in ("vllm", "lmcache"):
        value = cfg.get(project.upper() + "_SOURCE")
        if not value:
            raise ValueError(
                f"set TOKENDANCE_{project.upper()}_SOURCE to the dated checkout"
            )
        found[project] = source_identity(value)
    validate_sources(found)
    return found


def plan(cfg):
    return {
        "schema_version": 1,
        "mode": cfg["MODE"],
        "source_pins": profile()["source_pairs"]["dated"],
        "arms": profile()["arms"],
        "trace_sha256": profile()["trace_sha256"],
        "gpu_execution": cfg["MODE"] == "gpu",
        "gpu_requires": [
            "gpuq reservation",
            "pinned ROCm runtime volume and image",
            "local pinned model snapshot",
            "clean dated source checkouts",
        ],
        "output": cfg["OUTPUT"],
        "performance_claim": False,
        "harness_assets_sha256": digest(ROOT / "assets.json"),
    }


def container_command(cfg, identity, output, script, args, gpu):
    prefix = shlex.split(cfg["DOCKER"])
    env = {
        "KMESH_IMAGE_DIGEST": profile()["image"],
        "KMESH_TARGET_VOLUME": "kmesh-rocm723-target-lmcache-v2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "0" if gpu else "",
        "CUDA_VISIBLE_DEVICES": "0" if gpu else "",
        "ROCR_VISIBLE_DEVICES": "0" if gpu else "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "OMP_NUM_THREADS": "2",
        "NUMBA_CACHE_DIR": "/tmp/numba",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
        "PYTHONPATH": "/sources/lmcache:/inputs/harness",
        "TOKENDANCE_VLLM_SOURCE": "/sources/vllm",
        "TOKENDANCE_WHEEL_PACKAGE": "/opt/kmesh-target/lib/python3.12/site-packages/vllm",
    }
    name = "tokendance-" + uuid.uuid4().hex
    argv = [
        *prefix,
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
    if gpu:
        argv += ["--device", "/dev/kfd", "--device", "/dev/dri"]
        for gid in sorted(set(os.getgroups())):
            argv += ["--group-add", str(gid)]
    mounts = [
        (str(ROOT), "/inputs", "ro"),
        (str(output), "/outputs", "rw"),
        ("/etc/passwd", "/etc/passwd", "ro"),
        (cfg["RUNTIME_VOLUME"], "/opt/kmesh-target", "ro"),
    ]
    for project, value in identity.items():
        mounts += [
            (value["path"], "/sources/" + project, "ro"),
            (value["git_common_dir"], value["git_common_dir"], "ro"),
        ]
        for alternate in value.get("git_alternates", []):
            mounts.append((alternate, alternate, "ro"))
    if gpu:
        for filename in profile()["model_files"]:
            mounts.append(
                (
                    str((Path(cfg["MODEL_DIR"]) / filename).resolve()),
                    MODEL_CONTAINER + "/" + filename,
                    "ro",
                )
            )
    if cfg.get("BUNDLE"):
        mounts.append((str(Path(cfg["BUNDLE"]).resolve()), "/reference", "ro"))
    if cfg.get("VERIFICATION"):
        mounts.append(
            (str(Path(cfg["VERIFICATION"]).resolve()), "/verified.json", "ro")
        )
    for src, dst, permission in mounts:
        argv += ["-v", f"{src}:{dst}:{permission}"]
    for key, value in env.items():
        argv += ["-e", key + "=" + value]
    argv += ["--entrypoint", "/opt/kmesh-target/bin/python", profile()["image"]]
    if script == "verify_numerical_warmup_contract.py":
        argv += ["/inputs/harness/bootstrap.py", script, *args]
    else:
        argv += ["/inputs/harness/" + script, *args]
    return name, argv


def docker_step(cfg, identity, output, script, args, log, gpu):
    name, argv = container_command(cfg, identity, output, script, args, gpu)
    prefix = shlex.split(cfg["DOCKER"])
    old_handlers = {}

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)
    try:
        write(output / (log + ".command.json"), {"argv": argv})
        with (output / (log + ".log")).open("x") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
            try:
                status = process.wait()
                if status:
                    raise subprocess.CalledProcessError(status, argv)
            except BaseException:
                subprocess.run(
                    [*prefix, "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                process.wait(timeout=30)
                raise
    finally:
        subprocess.run(
            [*prefix, "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def verify(bundle, output, pair):
    args = [
        sys.executable,
        str(H / "verify_online_collective_lockstep_gate.py"),
        "--source-pair",
        pair,
        "--output",
        str(output / "verified.json"),
        "--checksums",
        str(output / "SHA256SUMS"),
    ]
    for index, arm in enumerate(profile()["arms"]):
        path = bundle / "arms" / f"{index:02d}-{arm}" / "artifacts/result.json"
        args += ["--" + arm.replace("_", "-"), str(path)]
    command(args, output / "verify.log")
    report(output / "verified.json", output / "report.json")


def report(verified, output=None, kind="verified_capture_replay"):
    value = json.loads(verified.read_text())
    if value.get("status") != "complete" or not value.get("claims", {}).get(
        "full_collective_lockstep_gate_pass"
    ):
        raise ValueError("a complete independent verification is required")
    if value["claims"].get("performance_claim") is not False:
        raise ValueError("the capture experiment cannot establish performance")
    pins = profile()["source_pairs"][value["source_pair"]]
    if value.get("source_pins") != pins:
        raise ValueError("verified source pins do not match the committed profile")
    numerical = value["numerical_comparison"]
    members = numerical["members"]
    result = {
        "schema_version": 1,
        "kind": kind,
        "claims": value["claims"],
        "members": len(members),
        "verified_summary_sha256": digest(verified),
        "serial_collective_kv_bitwise_equal": all(
            m[plane]["serial_collective_raw_bits"]["bitwise_equal"]
            for m in members.values()
            for plane in ("key", "value")
        ),
        "maximum_serial_collective_logit_error": max(
            m["raw_logits"]["serial_collective_delta"]["max_abs"]
            for m in members.values()
        ),
        "maximum_ordinary_serial_logit_error": max(
            m["raw_logits"]["ordinary_serial_delta"]["max_abs"]
            for m in members.values()
        ),
        "source_pins": value["source_pins"],
        "qualification": "Single-device synthetic correctness only; timing is invalid under capture instrumentation.",
    }
    if output is not None:
        write(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument(
        "action",
        choices=["plan", "run", "cpu", "gpu", "contracts", "codec", "verify", "report"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--source-pair", choices=["dated", "reference"])
    args = parser.parse_args()
    cfg = settings(args.config)
    if args.output:
        cfg["OUTPUT"] = str(args.output)
    if args.bundle:
        cfg["BUNDLE"] = str(args.bundle)
    if args.source_pair:
        cfg["SOURCE_PAIR"] = args.source_pair
    action = cfg["MODE"] if args.action == "run" else args.action
    validate_assets()
    if action == "plan":
        print(json.dumps(plan(cfg), indent=2, sort_keys=True))
        return
    if action == "report":
        output = Path(cfg["OUTPUT"]).resolve()
        report(output / "verified.json")
        return
    if action == "verify":
        if not cfg.get("BUNDLE"):
            raise ValueError("set TOKENDANCE_BUNDLE to the eight-arm capture directory")
        bundle = Path(cfg["BUNDLE"]).resolve()
        if not bundle.is_dir():
            raise ValueError("capture directory is absent")
        output = Path(cfg["OUTPUT"]).resolve()
        if output.is_relative_to(bundle):
            raise ValueError(
                "verification output must be outside the immutable input bundle"
            )
        output.mkdir(parents=True, exist_ok=False)
        verify(bundle, output, cfg["SOURCE_PAIR"])
        return
    if action == "gpu" and not os.environ.get("GPUQ_RUN_ID"):
        raise ValueError(
            "GPU execution requires gpuq; submit the make command through gpuq"
        )
    identity = sources(cfg) if action in ("gpu", "contracts", "codec") else None
    if action in ("gpu", "contracts", "codec") and not cfg.get("RUNTIME_VOLUME"):
        raise ValueError(
            "set TOKENDANCE_RUNTIME_VOLUME to the pinned environment volume"
        )
    if action == "gpu":
        if (
            not cfg.get("MODEL_DIR")
            or not (Path(cfg["MODEL_DIR"]) / "config.json").is_file()
        ):
            raise ValueError(
                "set TOKENDANCE_MODEL_DIR to the pinned model snapshot directory"
            )
        validate_model(cfg["MODEL_DIR"])
    output = Path(cfg["OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan(cfg))
    try:
        if action == "cpu":
            manifest = trace(output / "trace")
            write(
                output / "result.json",
                {
                    "status": "complete",
                    "kind": "synthetic_trace_only",
                    "trace_sha256": manifest["trace_sha256"],
                    "gpu_execution": False,
                    "executed_kv_reuse": False,
                    "oracle": manifest["oracle"]["aggregate"],
                },
            )
            print("Synthetic trace validated; no model, GPU or KV recovery executed.")
            return
        write(output / "sources.json", identity)
        if action == "codec":
            if not cfg.get("BUNDLE") or not cfg.get("VERIFICATION"):
                raise ValueError(
                    "codec requires TOKENDANCE_BUNDLE and TOKENDANCE_VERIFICATION"
                )
            docker_step(
                cfg,
                identity,
                output,
                "measure_family.py",
                [
                    "--bundle",
                    "/reference",
                    "--verification",
                    "/verified.json",
                    "--lmcache-source",
                    "/sources/lmcache",
                    "--output-dir",
                    "/outputs",
                ],
                "codec",
                False,
            )
            return
        if action == "contracts":
            docker_step(
                cfg,
                identity,
                output,
                "verify_numerical_warmup_contract.py",
                ["--output", "/outputs/contracts.json"],
                "contracts",
                False,
            )
            return
        trace(output / "trace")
        docker_step(
            cfg,
            identity,
            output,
            "verify_numerical_warmup_contract.py",
            ["--output", "/outputs/contracts.json"],
            "contracts",
            False,
        )
        docker_step(
            cfg,
            identity,
            output,
            "preflight_resolved_profile.py",
            [
                "--vllm-source",
                "/sources/vllm",
                "--lmcache-source",
                "/sources/lmcache",
                "--wheel-vllm-package",
                "/opt/kmesh-target/lib/python3.12/site-packages/vllm",
                "--lmcache-config",
                "/inputs/lmcache.yaml",
                "--output",
                "/outputs/preflight.json",
            ],
            "preflight",
            True,
        )
        for index, arm in enumerate(profile()["arms"]):
            args = [
                "--arm",
                arm,
                "--vllm-source",
                "/sources/vllm",
                "--lmcache-source",
                "/sources/lmcache",
                "--wheel-vllm-package",
                "/opt/kmesh-target/lib/python3.12/site-packages/vllm",
                "--lmcache-config",
                "/inputs/lmcache.yaml",
                "--trace",
                "/outputs/trace/trace.jsonl",
                "--manifest",
                "/outputs/trace/manifest.json",
                "--output-dir",
                f"/outputs/arms/{index:02d}-{arm}/artifacts",
            ]
            if arm != "ordinary":
                args += [
                    "--baseline-result",
                    "/outputs/arms/00-ordinary/artifacts/result.json",
                ]
            docker_step(
                cfg,
                identity,
                output,
                "run_online_collective_lockstep_gate.py",
                args,
                f"arm-{arm}",
                True,
            )
        # Verification runs in the same pinned environment but without GPU access.
        verify_args = [
            "--source-pair",
            "dated",
            "--output",
            "/outputs/verified.json",
            "--checksums",
            "/outputs/SHA256SUMS",
        ]
        for index, arm in enumerate(profile()["arms"]):
            verify_args += [
                "--" + arm.replace("_", "-"),
                f"/outputs/arms/{index:02d}-{arm}/artifacts/result.json",
            ]
        docker_step(
            cfg,
            identity,
            output,
            "verify_online_collective_lockstep_gate.py",
            verify_args,
            "verify",
            False,
        )
        report(
            output / "verified.json", output / "report.json", kind="fresh_gpu_capture"
        )
    except BaseException as error:
        write(
            output / "failure.json",
            {
                "status": "failed",
                "error": str(error),
                "exception": type(error).__name__,
            },
        )
        raise


if __name__ == "__main__":
    main()
