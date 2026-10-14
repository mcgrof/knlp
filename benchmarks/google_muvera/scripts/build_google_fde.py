#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build Google's pinned, unmodified FDE source and the knlp I/O adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream"
DEFAULT_BUILD_DIR = Path(__file__).resolve().parents[3] / ".cache/google-muvera"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources():
    lock = json.loads((UPSTREAM / "source-lock.json").read_text())
    for relative, expected in lock["files"].items():
        path = UPSTREAM / "google/graph-mining" / relative
        if sha256(path) != expected:
            raise RuntimeError(f"official Google source checksum mismatch: {path}")
    return lock


def prepare_dependency(name, spec, build_dir):
    downloads = build_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"{name}.tar.gz"
    if not archive.exists():
        print(f"Downloading {name}: {spec['url']}", flush=True)
        temporary = archive.with_suffix(".tmp")
        request = urllib.request.Request(
            spec["url"], headers={"User-Agent": "knlp-google-muvera"}
        )
        with urllib.request.urlopen(request, timeout=180) as source:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        if sha256(temporary) != spec["sha256"]:
            temporary.unlink()
            raise RuntimeError(f"download checksum mismatch for {name}")
        temporary.replace(archive)
    if sha256(archive) != spec["sha256"]:
        raise RuntimeError(f"cached archive checksum mismatch: {archive}")
    destination = build_dir / "deps" / name
    marker = destination / ".knlp-archive-sha256"
    if destination.exists():
        if not marker.exists() or marker.read_text().strip() != spec["sha256"]:
            raise RuntimeError(f"stale dependency directory; remove it: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive, "r:gz") as source:
            # The archive is pinned. Also reject traversal and special files.
            for member in source.getmembers():
                target = (temporary / member.name).resolve()
                if not target.is_relative_to(temporary.resolve()):
                    raise RuntimeError("unsafe archive path")
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError(f"unsupported archive entry: {member.name}")
            if hasattr(tarfile, "data_filter"):
                source.extractall(temporary, filter="data")
            else:
                # Older Python releases lack filters; all members above have
                # already been restricted to contained regular files/dirs.
                source.extractall(temporary)
        roots = list(temporary.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError(f"unexpected archive structure for {name}")
        (roots[0] / marker.name).write_text(spec["sha256"] + "\n")
        roots[0].replace(destination)


def build(build_dir=DEFAULT_BUILD_DIR, jobs=None):
    build_dir = Path(build_dir).expanduser().resolve()
    source_lock = verify_sources()
    dependency_lock = json.loads((UPSTREAM / "dependencies-lock.json").read_text())
    cmake = shutil.which("cmake")
    if not cmake:
        # Also support `python -m pip install cmake` without PATH edits.
        try:
            from cmake import CMAKE_BIN_DIR

            cmake = str(Path(CMAKE_BIN_DIR) / "cmake")
        except ImportError as error:
            raise RuntimeError(
                "CMake >=3.24 is required; install cmake first"
            ) from error
    if not (shutil.which(os.environ.get("CXX", "c++"))):
        raise RuntimeError("A C++20 compiler is required (set CXX if necessary)")
    if not shutil.which("make") and not shutil.which("ninja"):
        raise RuntimeError("A build tool is required: make or ninja")
    if jobs is None:
        jobs = min(4, os.cpu_count() or 1)
    if jobs < 1:
        raise ValueError("jobs must be positive")
    build_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in dependency_lock.items():
        prepare_dependency(name, spec, build_dir)
    cmake_build = build_dir / "cmake-build"
    configure = [
        cmake,
        "-S",
        str(UPSTREAM),
        "-B",
        str(cmake_build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DDEPS_DIR={build_dir / 'deps'}",
        f"-DOUTPUT_DIR={build_dir}",
    ]
    if not (cmake_build / "CMakeCache.txt").exists() and shutil.which("ninja"):
        configure.extend(["-G", "Ninja"])
    subprocess.run(configure, check=True)
    subprocess.run(
        [cmake, "--build", str(cmake_build), "--target", "google_fde", "-j", str(jobs)],
        check=True,
    )
    binary = build_dir / "google_fde"
    compiler = os.environ.get("CXX", "c++")
    cmake_settings = {}
    for line in (cmake_build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:FILEPATH="):
            compiler = line.split("=", 1)[1]
        if line.startswith(
            ("CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE:", "CMAKE_GENERATOR:")
        ):
            key, value = line.split("=", 1)
            cmake_settings[key] = value
    provenance = {
        "upstream": source_lock,
        "dependencies": dependency_lock,
        "binary_sha256": sha256(binary),
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "cmake": subprocess.check_output([cmake, "--version"], text=True),
        "adapter_sha256": sha256(UPSTREAM / "google_fde_cli.cc"),
        "cmake_recipe_sha256": sha256(UPSTREAM / "CMakeLists.txt"),
        "build_script_sha256": sha256(Path(__file__)),
        "build_type": "Release",
        "cmake_settings": cmake_settings,
    }
    (build_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Built {binary}", flush=True)
    return binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    build(args.build_dir, args.jobs)
