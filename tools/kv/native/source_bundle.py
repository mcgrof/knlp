# SPDX-License-Identifier: MIT
"""Import one hashed dependency change without publishing it first."""

import hashlib
from pathlib import Path
import re
import subprocess


def commit_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("require a full lowercase SHA-1 commit identity")
    return value


def git(checkout, *args):
    return subprocess.check_output(
        ["git", "-C", str(checkout), *map(str, args)], text=True
    ).strip()


def bundle_identity(path, expected_sha256, base, target):
    """Accept one v2 incremental bundle with exactly one base and head."""
    commit_id(base)
    commit_id(target)
    if base == target:
        raise ValueError("bundle target must differ from its base")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("require a full lowercase bundle SHA-256")
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        # This bootstrap runs before the installer creates its Python 3.12 env.
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("bundle SHA-256 differs")
        stream.seek(0)
        if stream.readline(256) != b"# v2 git bundle\n":
            raise ValueError("require a version 2 Git bundle")
        lines = []
        while True:
            line = stream.readline(4097)
            if not line or len(line) > 4096 or len(lines) > 2:
                raise ValueError("invalid or excessive bundle header")
            if line == b"\n":
                break
            lines.append(line.decode("utf-8").rstrip("\n"))
    if len(lines) != 2 or not lines[0].startswith("-" + base + " "):
        raise ValueError("bundle must have exactly the expected prerequisite")
    parts = lines[1].split(" ", 1)
    if len(parts) != 2 or parts[0] != target or not parts[1].startswith("refs/heads/"):
        raise ValueError("bundle must have exactly the expected target branch")
    subprocess.run(["git", "check-ref-format", parts[1]], check=True)
    return {
        "base": base,
        "target": target,
        "bundle_sha256": expected_sha256,
        "head_ref": parts[1],
    }


def require_descendant(checkout, base, target):
    commit_id(base)
    commit_id(target)
    for revision in (base, target):
        if git(checkout, "rev-parse", revision + "^{commit}") != revision:
            raise ValueError("source is not the expected commit")
    subprocess.run(
        ["git", "-C", str(checkout), "merge-base", "--is-ancestor", base, target],
        check=True,
    )


def import_bundle(checkout, path, expected_sha256, base, target):
    """Verify and import objects; leave the worktree and its HEAD unchanged."""
    path = Path(path).resolve(strict=True)
    receipt = bundle_identity(path, expected_sha256, base, target)
    if git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("bundle import requires a clean checkout")
    if git(checkout, "rev-parse", base + "^{commit}") != base:
        raise ValueError("bundle base is not a commit")
    before = git(checkout, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "-C", str(checkout), "bundle", "verify", str(path)], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "fetch",
            "--no-tags",
            str(path),
            receipt["head_ref"],
        ],
        check=True,
    )
    if git(checkout, "rev-parse", "FETCH_HEAD") != target:
        raise ValueError("imported bundle commit differs")
    require_descendant(checkout, base, target)
    if bundle_identity(path, expected_sha256, base, target) != receipt:
        raise ValueError("bundle changed during import")
    if git(checkout, "rev-parse", "HEAD") != before or git(
        checkout, "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("bundle import changed the checkout")
    return receipt
