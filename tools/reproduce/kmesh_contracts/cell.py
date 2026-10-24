# SPDX-License-Identifier: MIT
"""Run CPU ownership controls with GPUs hidden in a prepared environment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.reproduce.kmesh_contracts.contract import (
    CHECKOUT,
    digest,
    helper_hashes,
    profile,
    profile_path,
)
from tools.reproduce.kmesh_native.cell import Capture


def source_snapshot(root: Path, commit: str, files: dict) -> dict:
    """Require a clean pinned checkout and record its selected source files."""
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    hashes = {name: digest(root / name) for name in files}
    if actual != commit or dirty or hashes != files:
        raise RuntimeError("CPU qualification requires the pinned clean source")
    return {"commit": actual, "clean": True, "source_files_sha256": hashes}


def protobuf_hashes(cfg: dict) -> dict:
    """Hash the loaded wire schemas, including generated optional-field presence."""
    return {
        name: hashlib.sha256(
            importlib.import_module(name).DESCRIPTOR.serialized_pb
        ).hexdigest()
        for name in cfg.get("protobuf_descriptors_sha256", {})
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="contexts")
    parser.add_argument("--vllm-source", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    additional_paths = {"vllm": args.vllm_source.resolve()} if args.vllm_source else {}
    if any(
        output == tree or tree in output.parents
        for tree in (source, CHECKOUT, *additional_paths.values())
    ):
        raise RuntimeError("test captures must remain outside all source checkouts")
    if any(
        (output / name).exists()
        for name in ("receipt.json", "junit.xml", "verified.json")
    ):
        raise RuntimeError("test capture already exists")
    output.mkdir(parents=True, exist_ok=True)
    cfg = profile(args.profile)
    profile_hash = digest(profile_path(args.profile))
    helpers = helper_hashes()
    main_source = source_snapshot(
        source, cfg["lmcache_commit"], cfg["source_files_sha256"]
    )
    pins = cfg.get("additional_sources", {})
    if set(pins) != set(additional_paths):
        raise RuntimeError("additional source paths do not match this profile")
    additional = {
        name: source_snapshot(
            additional_paths[name], pin["commit"], pin["source_files_sha256"]
        )
        for name, pin in pins.items()
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU controls require CUDA devices hidden before import")

    import lmcache
    import lmcache.lmcache_native as native
    import pytest
    import torch

    loaded_additional = {}
    for name, root in additional_paths.items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != root / name:
            raise RuntimeError("another additional-source Python package was loaded")
        loaded_additional[name] = module.__file__

    if Path(lmcache.__file__).resolve().parent != source / "lmcache":
        raise RuntimeError("another LMCache Python package was loaded")
    if lmcache.torch_device_type != "cpu" or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU ownership controls require the CPU fallback")
    native_hash = digest(Path(native.__file__))
    descriptors = protobuf_hashes(cfg)
    if descriptors != cfg.get("protobuf_descriptors_sha256", {}):
        raise RuntimeError("generated protobuf descriptors differ from the profile")
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
    if main_source != source_snapshot(
        source, cfg["lmcache_commit"], cfg["source_files_sha256"]
    ) or additional != {
        name: source_snapshot(
            additional_paths[name], pin["commit"], pin["source_files_sha256"]
        )
        for name, pin in pins.items()
    }:
        raise RuntimeError("source files changed during execution")
    if native_hash != digest(Path(native.__file__)):
        raise RuntimeError("native extension changed during execution")
    if profile_hash != digest(profile_path(args.profile)) or helpers != helper_hashes():
        raise RuntimeError("qualification program changed during execution")
    if descriptors != protobuf_hashes(cfg):
        raise RuntimeError("loaded wire descriptors changed during execution")
    receipt = {
        "profile_name": args.profile,
        "profile_sha256": profile_hash,
        "harness_files_sha256": helpers,
        "source": {"commit": main_source["commit"], "clean": True},
        "source_files_sha256": main_source["source_files_sha256"],
        "additional_sources": additional,
        "protobuf_descriptors_sha256": descriptors,
        "native_sha256": {
            "lmcache_native": native_hash,
        },
        "loaded_modules": {
            "lmcache": lmcache.__file__,
            "lmcache_native": native.__file__,
            **loaded_additional,
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
