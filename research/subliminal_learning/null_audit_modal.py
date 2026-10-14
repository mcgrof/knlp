#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the archived target-zero numerical audit on a Modal L4.

Inputs are uploaded to the ``knlp-runs`` volume before this app starts. The
GPU function runs the complete subliminal test suite, executes the audit, and
commits either a success or failure receipt to that volume before returning.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import traceback

import modal

APP_NAME = "knlp-subliminal-null-audit"
VOL = "/vol"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.14.0",
    "torchvision",
    "numpy",
    "pytest",
    "pandas",
    "tqdm",
    "matplotlib",
)
if modal.is_local():
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent.parent
    image = image.add_local_dir(
        here,
        "/root/research/subliminal_learning",
        ignore=["**/__pycache__", "**/*.pyc", "null_audit_modal.py"],
    ).add_local_dir(
        root / "defconfigs",
        "/root/defconfigs",
        ignore=lambda p: not pathlib.Path(p).name.startswith("subliminal-mnist-"),
    )

runs = modal.Volume.from_name("knlp-runs", create_if_missing=False)


def _write_json(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _capture_hardware(dest):
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    commands = {
        "00_uname.txt": ["uname", "-a"],
        "01_nvidia.txt": ["nvidia-smi"],
        "02_cpu.txt": ["lscpu"],
        "03_memory.txt": ["bash", "-lc", "head -8 /proc/meminfo"],
        "04_df.txt": ["df", "-hT"],
        "05_mount.txt": ["mount"],
        "06_python.txt": ["python3", "-VV"],
    }
    for name, command in commands.items():
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        (dest / name).write_text(proc.stdout + proc.stderr)


def _publish(local_dir, remote_dir):
    remote = pathlib.Path(VOL) / remote_dir.lstrip("/")
    shutil.rmtree(remote, ignore_errors=True)
    shutil.copytree(local_dir, remote)
    runs.commit()


@app.function(
    image=image,
    gpu="L4",
    cpu=2.0,
    memory=8192,
    # Three total attempts at this timeout remain below the confirmed $1.29
    # cap at the current L4 + CPU + memory rates.
    timeout=20 * 60,
    volumes={VOL: runs},
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
)
def run_null_audit(input_prefix: str, output_prefix: str, source: dict) -> dict:
    import sys

    import torch

    os.environ.update(source)
    sys.path.insert(0, "/root")
    from research.subliminal_learning import null_audit

    runs.reload()
    input_dir = pathlib.Path(VOL) / input_prefix.lstrip("/")
    run_dir = input_dir / "residual-target-zero"
    teacher = input_dir / "pack00_teacher.pt"
    remote_status = pathlib.Path(VOL) / output_prefix.lstrip("/") / "status.json"
    if remote_status.exists() and json.loads(remote_status.read_text()).get("complete"):
        return {"complete": True, "skipped": True, "output": output_prefix}

    local = pathlib.Path("/tmp/null-audit-result")
    shutil.rmtree(local, ignore_errors=True)
    local.mkdir(parents=True)
    started = time.time()
    status = {
        "complete": False,
        "started_unix": started,
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "source": source,
    }
    _capture_hardware(local / "hwinfo")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Modal L4 function has no visible CUDA device")
        status["gpu"] = torch.cuda.get_device_name(0)
        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "/root/research/subliminal_learning/tests",
            ],
            cwd="/root",
            capture_output=True,
            text=True,
            check=False,
        )
        (local / "tests.log").write_text(tests.stdout + tests.stderr)
        status["tests_returncode"] = tests.returncode
        if tests.returncode:
            raise RuntimeError("subliminal test suite failed; see tests.log")

        result = null_audit.run_audit(str(run_dir), str(teacher), steps=4)
        _write_json(local / "null-audit.json", result)
        digest = hashlib.sha256((local / "null-audit.json").read_bytes()).hexdigest()
        (local / "null-audit.json.sha256").write_text(f"{digest}  null-audit.json\n")
        status.update(
            complete=True,
            elapsed_s=time.time() - started,
            result_sha256=digest,
            result_bytes=(local / "null-audit.json").stat().st_size,
        )
        _write_json(local / "status.json", status)
        _publish(local, output_prefix)
        return status
    except BaseException as exc:
        status.update(
            complete=False,
            elapsed_s=time.time() - started,
            error=repr(exc),
        )
        (local / "FAIL.txt").write_text(traceback.format_exc())
        _write_json(local / "status.json", status)
        _publish(local, output_prefix)
        raise


def tree_sha256(package_root, defconfig_root):
    """Hash the exact package and defconfig bytes shipped by this app."""
    h = hashlib.sha256()
    package_root = pathlib.Path(package_root)
    defconfig_root = pathlib.Path(defconfig_root)
    package_files = sorted(
        p
        for p in package_root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name != "null_audit_modal.py"
    )
    defconfigs = sorted(defconfig_root.glob("subliminal-mnist-*"))
    for path in package_files:
        h.update(f"package/{path.relative_to(package_root)}".encode())
        h.update(path.read_bytes())
    for path in defconfigs:
        h.update(f"defconfigs/{path.name}".encode())
        h.update(path.read_bytes())
    return h.hexdigest()


@app.local_entrypoint()
def main(
    input_prefix: str,
    output_prefix: str,
    commit: str,
    dirty: str = "0",
    branch: str = "main",
):
    here = pathlib.Path(__file__).resolve().parent
    defconfigs = here.parent.parent / "defconfigs"
    source = {
        "KNLP_SOURCE_COMMIT": commit,
        "KNLP_SOURCE_BRANCH": branch,
        "KNLP_SOURCE_DIRTY": dirty,
        "KNLP_SOURCE_TREE_SHA256": tree_sha256(here, defconfigs),
    }
    print(json.dumps(source, sort_keys=True), flush=True)
    result = run_null_audit.remote(input_prefix, output_prefix, source)
    print(json.dumps(result, sort_keys=True), flush=True)
