# SPDX-License-Identifier: MIT
"""Execute pinned native-layout or registered-copy controls on CUDA."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.reproduce.kmesh_native.contract import (
    HARNESS_FILES,
    digest,
    profile,
    profile_path,
    selectors,
)


class Capture:
    """Retain collected cases and all test phases, including setup skips."""

    def __init__(self) -> None:
        self.collected: list[str] = []
        self.reports: list[dict] = []

    def pytest_collection_finish(self, session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report) -> None:
        self.reports.append(
            {
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "wasxfail": getattr(report, "wasxfail", None),
            }
        )


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
    parser.add_argument("--profile", default="paged-cache")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if any(output == tree or tree in output.parents for tree in (source, checkout)):
        raise RuntimeError("test captures must remain outside both source checkouts")
    if any(
        (output / name).exists()
        for name in ("receipt.json", "junit.xml", "verified.json")
    ):
        raise RuntimeError("test capture already exists")
    output.mkdir(parents=True, exist_ok=True)
    cfg = profile(args.profile)
    profile_hash = digest(profile_path(args.profile))
    helper_hashes = {
        name: digest(Path(__file__).with_name(name)) for name in HARNESS_FILES
    }
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty or commit != cfg["lmcache_commit"]:
        raise RuntimeError(
            "native qualification requires the pinned clean LMCache source"
        )
    source_hashes = {name: digest(source / name) for name in cfg["source_files_sha256"]}
    if source_hashes != cfg["source_files_sha256"]:
        raise RuntimeError("pinned source file hashes differ")

    import lmcache
    import lmcache.lmcache_native as native
    import pytest
    import torch
    from lmcache import cuda_ops

    if Path(lmcache.__file__).resolve().parent != source / "lmcache":
        raise RuntimeError("another LMCache Python package was loaded")
    descriptors = protobuf_hashes(cfg)
    if descriptors != cfg.get("protobuf_descriptors_sha256", {}):
        raise RuntimeError("generated protobuf descriptors differ from the profile")
    native_hashes = {
        "cuda_ops": digest(Path(cuda_ops.__file__)),
        "lmcache_native": digest(Path(native.__file__)),
    }
    if native_hashes != cfg["native_sha256"]:
        raise RuntimeError("another native build was loaded")
    for name in cfg["required_symbols"]:
        if not hasattr(cuda_ops, name):
            raise RuntimeError("required CUDA binding missing: " + name)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select one available CUDA device")
    capture = Capture()
    os.chdir(source)
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
                *selectors(args.profile),
            ],
            plugins=[capture],
        )
    )
    torch.cuda.synchronize()
    if descriptors != protobuf_hashes(cfg):
        raise RuntimeError("loaded wire descriptors changed during execution")
    if source_hashes != {
        name: digest(source / name) for name in cfg["source_files_sha256"]
    }:
        raise RuntimeError("source files changed during execution")
    if native_hashes != {
        "cuda_ops": digest(Path(cuda_ops.__file__)),
        "lmcache_native": digest(Path(native.__file__)),
    }:
        raise RuntimeError("native extensions changed during execution")
    if (
        profile_hash != digest(profile_path(args.profile))
        or helper_hashes
        != {name: digest(Path(__file__).with_name(name)) for name in HARNESS_FILES}
        or subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != commit
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
    ):
        raise RuntimeError("qualification program or checkout changed during execution")
    receipt = {
        "profile_name": args.profile,
        "profile_sha256": profile_hash,
        "harness_files_sha256": helper_hashes,
        "source": {"commit": commit, "clean": not dirty},
        "source_files_sha256": source_hashes,
        "protobuf_descriptors_sha256": descriptors,
        "native_sha256": native_hashes,
        "loaded_modules": {
            "lmcache": lmcache.__file__,
            "cuda_ops": cuda_ops.__file__,
            "lmcache_native": native.__file__,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pytest": importlib.metadata.version("pytest"),
        },
        "device": {
            "backend": "cuda",
            "count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
            "capacity_bytes": torch.cuda.get_device_properties(0).total_memory,
            "name": torch.cuda.get_device_properties(0).name,
        },
        "selectors": selectors(args.profile),
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
