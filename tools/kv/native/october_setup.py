# SPDX-License-Identifier: MIT
"""Install the dated H100 research stack into a caller-owned directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

PINS = {
    "vllm": (
        "20261016-k16v8-compact-prefill",
        "26cd429bac01d07909ed91e353f564d7aaa8d38f",
    ),
    "flashinfer": (
        "20261016-asym-k16v8-serving",
        "800a550f8b736e49f8cfa136b82bfaf6d9d32711",
    ),
}
BASE = "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c"
WHEEL = (
    "https://wheels.vllm.ai/"
    + BASE
    + "/vllm-0.23.1rc1.dev734%2Bge24d1b24f.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA = "06a3ba0d94e210e46be6638a325322ad8142e1cfc6b30538c1b0816c839ca0ef"
COMPILED = [
    "csrc",
    "CMakeLists.txt",
    "cmake",
    "setup.py",
    "pyproject.toml",
    "third_party",
    "rust",
    "tools/build_rust.py",
]


def recipe_pins(recipe):
    """Keep the historical install explicit and select current serving."""
    if recipe == "historical-compact":
        return dict(PINS)
    if recipe == "current":
        return {
            **PINS,
            "vllm": (
                "2026-10-17-feature-mixed-fresh-prefill",
                "276305b29e0d11f95ada503ba76323ab24992e59",
            ),
        }
    raise ValueError("unknown installation recipe: " + recipe)


EVENTS = []


def run(*args, **kwargs):
    started = time.perf_counter()
    try:
        subprocess.run(list(map(str, args)), check=True, **kwargs)
    finally:
        EVENTS.append(
            {"argv": list(map(str, args)), "seconds": time.perf_counter() - started}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--recipe",
        choices=("historical-compact", "current"),
        default="historical-compact",
    )
    args = parser.parse_args()
    pins = recipe_pins(args.recipe)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = root / "env"
    if env.exists():
        raise ValueError("use a fresh installation directory")
    constraints = Path(__file__).with_name("october-constraints.txt")
    install_constraints = root / "vllm-install-constraints.txt"
    install_constraints.write_text(
        "\n".join(
            line
            for line in constraints.read_text().splitlines()
            if not line.startswith(("numpy==", "apache-tvm-ffi=="))
        )
        + "\n"
    )
    run("uv", "venv", "--python", "3.12.11", "--seed", env)
    python = env / "bin/python"

    def pip(*packages, constraint=constraints):
        run(
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--constraint",
            constraint,
            *packages,
        )

    pip(
        "--index",
        "https://download.pytorch.org/whl/cu129",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
    )
    pip(
        "setuptools>=77,<81",
        "setuptools_scm",
        "setuptools-rust==1.12.0",
        "wheel",
        "cmake",
        "ninja",
        "packaging",
        "apache-tvm-ffi==0.1.14.post1",
        "nvidia-cutlass-dsl==4.5.2",
        "cuda-python==12.9.4",
        "pytest",
        "scipy",
        "numpy==1.26.4",
    )
    for name, (branch, revision) in pins.items():
        checkout = root / name
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--branch",
            branch,
            "https://github.com/mcgrof/" + name + ".git",
            checkout,
        )
        # A branch is a discovery hint; an immutable revision is the recipe.
        run("git", "-C", checkout, "checkout", "--detach", revision)
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != revision:
            raise RuntimeError("source revision differs: " + name)
    run(
        "git",
        "-C",
        root / "vllm",
        "diff",
        "--exit-code",
        BASE,
        pins["vllm"][1],
        "--",
        *COMPILED,
    )
    wheel = root / "vllm-base.whl"
    started = time.perf_counter()
    urllib.request.urlretrieve(WHEEL, wheel)
    EVENTS.append(
        {"download": "upstream vLLM wheel", "seconds": time.perf_counter() - started}
    )
    if hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest() != WHEEL_SHA:
        raise RuntimeError("upstream binary digest differs")
    build_env = {
        **os.environ,
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_PRECOMPILED_WHEEL_LOCATION": str(wheel),
        "VLLM_PRECOMPILED_WHEEL_COMMIT": BASE,
    }
    run(
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "--constraint",
        install_constraints,
        "-e",
        root / "vllm",
        "--no-build-isolation",
        env=build_env,
    )
    pip(
        "--index",
        "https://download.pytorch.org/whl/cu129",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "torchaudio==2.11.0",
    )
    pip(
        "transformers==5.5.3",
        "apache-tvm-ffi==0.1.14.post1",
        "nvidia-cutlass-dsl==4.5.2",
        "cuda-python==12.9.4",
        "numpy==1.26.4",
        "pluggy==1.6.0",
    )
    run(python, "-c", "import cutlass")
    run(
        "git", "-C", root / "flashinfer", "submodule", "update", "--init", "--recursive"
    )
    # FlashInfer's metadata bootstrap invokes uv itself. pip avoids holding
    # uv's environment lock during that nested dependency installation.
    run(
        python,
        "-m",
        "pip",
        "install",
        "-e",
        root / "flashinfer",
        "--no-deps",
        "--no-build-isolation",
    )
    run(
        "uv",
        "pip",
        "uninstall",
        "--python",
        python,
        "flashinfer-cubin",
        "flashinfer-jit-cache",
    )
    run(
        python,
        "-c",
        "import torch, vllm, flashinfer; assert torch.__version__ == '2.11.0+cu129'; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() == (9, 0); print(vllm.__file__, flashinfer.__file__)",
    )
    with (root / "requirements.lock").open("w") as stream:
        run("uv", "pip", "freeze", "--python", python, stdout=stream)
    (root / "setup.json").write_text(
        json.dumps(
            {
                "recipe": args.recipe,
                "sources": pins,
                "compiled_base": BASE,
                "wheel_sha256": WHEEL_SHA,
                "steps": EVENTS,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
