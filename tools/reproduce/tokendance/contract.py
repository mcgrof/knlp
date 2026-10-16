# SPDX-License-Identifier: MIT
"""Validate committed source and artifact identities without importing a GPU stack."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def profile():
    return json.loads((ROOT / "profile.json").read_text())


def validate_sources(source, pair="dated"):
    expected = profile()["source_pairs"][pair]
    for project, pins in expected.items():
        observed = source[project]
        if any(observed.get(key) != value for key, value in pins.items()):
            raise ValueError(f"{project} does not match the {pair} source pins")
        if observed.get("status_porcelain", ""):
            raise ValueError(f"{project} source is dirty")


def source_identity(path):
    path = Path(path).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    common = Path(git("rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = path / common
    common = common.resolve()
    alternates = set()

    def collect(objects):
        listing = objects / "info/alternates"
        if not listing.is_file():
            return
        for line in listing.read_text().splitlines():
            alternate = Path(line)
            if not alternate.is_absolute():
                alternate = objects / alternate
            alternate = alternate.resolve()
            if not alternate.is_dir():
                raise ValueError(
                    f"Git alternate object directory is absent: {alternate}"
                )
            if alternate not in alternates:
                alternates.add(alternate)
                collect(alternate)

    collect(common / "objects")
    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "status_porcelain": git("status", "--porcelain"),
        "git_common_dir": str(common),
        "git_alternates": sorted(map(str, alternates)),
    }


def validate_assets():
    expected = json.loads((ROOT / "assets.json").read_text())
    for name, value in expected.items():
        if digest(ROOT / name) != value:
            raise ValueError(f"reproducer input changed: {name}")
    return expected


def validate_trace(directory):
    pins = profile()
    for name, key in [
        ("trace.jsonl", "trace_sha256"),
        ("manifest.json", "manifest_sha256"),
    ]:
        if digest(Path(directory) / name) != pins[key]:
            raise ValueError(f"{name} is not the pinned synthetic workload")
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest["trace_sha256"] != pins["trace_sha256"]:
        raise ValueError("manifest is bound to a different trace")
    return manifest


def validate_model(directory):
    for name, expected in profile()["model_files"].items():
        path = Path(directory) / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(
                f"model snapshot does not match the pinned revision: {name}"
            )
