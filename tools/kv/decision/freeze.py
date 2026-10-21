# SPDX-License-Identifier: MIT
"""Create a byte-bound source manifest and portable reference archive."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    paths = sorted(
        [str(p) for p in Path("tools/kv/decision").glob("*.py")]
        + [
            "tools/kv/native/gsm8k.py",
            "tools/kv/prequantized/fixtures.py",
            "tools/kv/prequantized/contract.py",
            "tools/kv/fp8_failure/statistics.py",
            "tests/test_decision_cache.py",
            "tests/test_decision_statistics.py",
        ]
    )
    for prefix in [
        "tools",
        "tools/kv",
        "tools/kv/native",
        "tools/kv/prequantized",
        "tools/kv/fp8_failure",
    ]:
        init = Path(prefix) / "__init__.py"
        if init.exists():
            paths.append(str(init))
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], text=True
    )
    if dirty.strip():
        raise ValueError("Commit source changes before freezing")
    source = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        files={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
    )
    encoded = (json.dumps(source, indent=2) + "\n").encode()
    with tarfile.open(a.out, "w:gz") as out:
        for path in paths:
            out.add(path, arcname=path)
        info = tarfile.TarInfo("SOURCE.json")
        info.size = len(encoded)
        info.mtime = 0
        out.addfile(info, io.BytesIO(encoded))
    # Prove the shipped archive imports without the checkout on sys.path.
    with tempfile.TemporaryDirectory(prefix="kv-decision-archive-") as tmp:
        with tarfile.open(a.out) as archive:
            archive.extractall(tmp, filter="data")
        env = dict(os.environ, PYTHONPATH=tmp)
        for module in ["run", "oracle", "analyze"]:
            subprocess.run(
                [sys.executable, "-m", "tools.kv.decision." + module, "--help"],
                cwd=tmp,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_decision_cache.py",
                "tests/test_decision_statistics.py",
            ],
            cwd=tmp,
            env=env,
            check=True,
        )
    print(
        json.dumps(
            dict(
                source_commit=source["commit"],
                source_archive_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
