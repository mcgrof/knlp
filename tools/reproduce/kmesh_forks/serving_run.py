# SPDX-License-Identifier: MIT
"""Run each original serving fork case in a fresh GPU process."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from serving_verify import SCENARIOS, verify_bundle

ROOT = Path(__file__).resolve().parent
TOKEN = ROOT.parent / "tokendance"
sys.path.insert(0, str(TOKEN))
from contract import profile, source_identity, validate_model


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def worker():
    source = Path(os.environ["KMESH_SERVING_VLLM_SOURCE"])
    wheel = Path(os.environ["KMESH_SERVING_WHEEL_PACKAGE"])
    spec = importlib.util.spec_from_file_location(
        "vllm",
        source / "vllm/__init__.py",
        submodule_search_locations=[str(source / "vllm"), str(wheel)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm"] = module
    spec.loader.exec_module(module)
    sys.argv = [str(ROOT / "serving.py"), *sys.argv[2:]]
    runpy.run_path(str(ROOT / "serving.py"), run_name="__main__")


def settings(path):
    cfg = {"OUTPUT": "results/kmesh-serving"}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("CONFIG_KNLP_KMESH_SERVING_") and "=" in line:
                key, value = line.split("=", 1)
                cfg[key.removeprefix("CONFIG_KNLP_KMESH_SERVING_")] = value.strip('"')
    for key in ("OUTPUT", "VLLM_SOURCE", "WHEEL_PACKAGE", "MODEL_DIR", "BUNDLE"):
        cfg[key] = os.environ.get("KMESH_SERVING_" + key, cfg.get(key, ""))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("action", choices=("plan", "run", "verify"))
    args = parser.parse_args()
    cfg = settings(args.config)
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "cases": SCENARIOS,
                    "vllm": json.loads((ROOT / "profile-current.json").read_text())[
                        "vllm"
                    ],
                    "GPU": True,
                    "performance_claim": False,
                },
                indent=2,
            )
        )
        return
    output = Path(cfg["OUTPUT"]).resolve()
    if args.action == "verify":
        summary = verify_bundle(cfg["BUNDLE"])
        output.mkdir(parents=True, exist_ok=False)
        write(output / "verified.json", summary)
        return
    if not (os.environ.get("GPUQ_RUN_ID") or os.environ.get("RUNPOD_POD_ID")):
        raise ValueError(
            "serving replay requires a gpuq reservation or an owned RunPod"
        )
    if os.environ.get("KMESH_IMAGE_DIGEST") != profile()["current_cuda"]["image"]:
        raise ValueError("declare the pinned running image with KMESH_IMAGE_DIGEST")
    source = source_identity(cfg["VLLM_SOURCE"])
    pins = json.loads((ROOT / "profile-current.json").read_text())["vllm"]
    if source["status_porcelain"] or any(
        source[key] != value for key, value in pins.items()
    ):
        raise ValueError("serving replay requires the clean pinned current vLLM source")
    validate_model(cfg["MODEL_DIR"])
    if not Path(cfg["WHEEL_PACKAGE"]).is_dir():
        raise ValueError("matching installed vLLM wheel package is absent")
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("serving.py", "serving_run.py", "serving_verify.py")
    }
    write(
        output / "provenance.json",
        {
            "source": source,
            "inputs_sha256": inputs,
            "image": os.environ["KMESH_IMAGE_DIGEST"],
            "model_revision": profile()["model_revision"],
        },
    )
    env = os.environ.copy()
    env.update(
        KMESH_SERVING_VLLM_SOURCE=source["path"],
        KMESH_SERVING_WHEEL_PACKAGE=cfg["WHEEL_PACKAGE"],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    try:
        for name in SCENARIOS:
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--model",
                cfg["MODEL_DIR"],
                "--scenario",
                name,
                "--output",
                str(output / (name + ".json")),
            ]
            write(output / (name + "-command.json"), argv)
            with (output / (name + ".log")).open("x") as stream:
                subprocess.run(
                    argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
            print("completed " + name, flush=True)
        write(output / "verified.json", verify_bundle(output))
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
    if sys.argv[1:2] == ["worker"]:
        worker()
    else:
        main()
