# SPDX-License-Identifier: MIT
"""Run CPU ownership controls with GPUs hidden in a prepared environment."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.reproduce.kmesh_contracts.contract import (
    CHECKOUT,
    HERE,
    digest,
    helper_hashes,
    profile,
)
from tools.reproduce.kmesh_native.cell import Capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if any(output == tree or tree in output.parents for tree in (source, CHECKOUT)):
        raise RuntimeError("test captures must remain outside both source checkouts")
    if any(
        (output / name).exists()
        for name in ("receipt.json", "junit.xml", "verified.json")
    ):
        raise RuntimeError("test capture already exists")
    output.mkdir(parents=True, exist_ok=True)
    cfg = profile()
    profile_hash = digest(HERE / "profile.json")
    helpers = helper_hashes()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    hashes = {name: digest(source / name) for name in cfg["source_files_sha256"]}
    if commit != cfg["lmcache_commit"] or dirty or hashes != cfg["source_files_sha256"]:
        raise RuntimeError("CPU qualification requires the pinned clean source")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU controls require CUDA devices hidden before import")

    import lmcache
    import lmcache.lmcache_native as native
    import pytest
    import torch

    if Path(lmcache.__file__).resolve().parent != source / "lmcache":
        raise RuntimeError("another LMCache Python package was loaded")
    if lmcache.torch_device_type != "cpu" or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU ownership controls require the CPU fallback")
    native_hash = digest(Path(native.__file__))
    capture = Capture()
    os.chdir(source)
    sys.path.insert(0, str(source))
    code = int(
        pytest.main(
            [
                "-q",
                "-ra",
                "--rootdir",
                str(source),
                "-c",
                "/dev/null",
                "--noconftest",
                "-p",
                "no:cacheprovider",
                "--junitxml",
                str(output / "junit.xml"),
                *cfg["selectors"],
            ],
            plugins=[capture],
        )
    )
    if hashes != {name: digest(source / name) for name in hashes}:
        raise RuntimeError("source files changed during execution")
    if native_hash != digest(Path(native.__file__)):
        raise RuntimeError("native extension changed during execution")
    if profile_hash != digest(HERE / "profile.json") or helpers != helper_hashes():
        raise RuntimeError("qualification program changed during execution")
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        or subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != commit
    ):
        raise RuntimeError("source checkout changed during execution")
    receipt = {
        "profile_sha256": profile_hash,
        "harness_files_sha256": helpers,
        "source": {"commit": commit, "clean": not dirty},
        "source_files_sha256": hashes,
        "native_sha256": {
            "lmcache_native": native_hash,
        },
        "loaded_modules": {
            "lmcache": lmcache.__file__,
            "lmcache_native": native.__file__,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            **{
                name: importlib.metadata.version(name)
                for name in ("pytest", "grpcio", "protobuf")
            },
        },
        "execution": {
            "backend": lmcache.torch_device_type,
            "visible_cuda_devices": torch.cuda.device_count(),
            "cuda_hidden": True,
        },
        "selectors": cfg["selectors"],
        "collected": capture.collected,
        "reports": capture.reports,
        "exit_code": code,
        "junit_sha256": digest(output / "junit.xml"),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
