# SPDX-License-Identifier: MIT
"""Verify installed package code against the pinned public source checkout."""

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path


def verify_package(module, checkout, revision):
    """Accept a normal wheel or editable install only when its code matches.

    Generated version metadata and compiled extensions are recorded separately.
    They are not substituted for checking the package's Python implementation.
    """
    checkout = Path(checkout).resolve()
    observed = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != revision:
        raise ValueError("package source revision mismatch")
    subprocess.run(
        ["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source = checkout / module.__name__
    installed = Path(module.__file__).resolve().parent
    files = {}
    for path in sorted(source.rglob("*.py")):
        if path.name == "_version.py":
            continue
        relative = path.relative_to(source)
        target = installed / relative
        wanted = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != wanted
        ):
            raise ValueError(
                f"installed package differs from pinned source: {relative}"
            )
        files[str(relative)] = wanted
    if not files:
        raise ValueError("empty package source inventory")
    unexpected = {
        str(p.relative_to(installed))
        for p in installed.rglob("*.py")
        if p.name != "_version.py"
    } - files.keys()
    if unexpected:
        raise ValueError(
            f"installed package has unexpected Python files: {sorted(unexpected)}"
        )
    return {
        "revision": revision,
        "installed_path": str(installed),
        "source_path": str(source),
        "installation": "editable" if installed == source else "wheel",
        "python_files": len(files),
        "python_tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "extensions": {
            str(p.relative_to(installed)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(installed.rglob("*.so"))
        },
    }


def dependency_problems():
    """Check every installed distribution's active base requirements."""
    from packaging.requirements import Requirement

    problems = []
    for dist in metadata.distributions():
        for specification in dist.requires or []:
            req = Requirement(specification)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            try:
                version = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                problems.append(f"{dist.name} requires missing {req}")
                continue
            if req.specifier and version not in req.specifier:
                problems.append(f"{dist.name} requires {req}; installed {version}")
    return sorted(problems)
