#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download the course dataset straight into a Modal Volume, without a GPU.

The course README expects a local download followed by a 20 GB upload.
Running the course's own download_data module inside a small Modal
container writes the files directly to the data Volume at the layout the
course launcher mounts, and costs cents rather than an upload.
"""

import os
from pathlib import Path

import modal

ASSIGNMENTS_DIR = Path(os.environ.get("CS312_ASSIGNMENTS_DIR", ".")).resolve()
VOLUME_NAME = os.environ.get("CS312_MODAL_DATA_VOLUME", "hard-dl-dclm-v1")
DATASET_DIR_NAME = "dclm_9p6m_ctx1024"
REMOTE_REPO = "/root/assignments"
MOUNT = "/vol"

app = modal.App("cs312-seed-data")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def _ignore(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {".git", ".venv", "__pycache__", "data", "wandb", "ckpts"})


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "torch==2.6.0",
    index_url="https://download.pytorch.org/whl/cpu",
)
if modal.is_local():
    image = image.add_local_dir(
        ASSIGNMENTS_DIR, remote_path=REMOTE_REPO, ignore=_ignore
    )


@app.function(
    image=image,
    volumes={MOUNT: volume},
    timeout=4 * 60 * 60,
    cpu=2.0,
    memory=4096,
)
def seed(force: bool = False):
    import subprocess
    import sys

    target = f"{MOUNT}/datasets/{DATASET_DIR_NAME}"
    command = [sys.executable, "-m", "download_data", "--output-dir", target]
    if force:
        command.append("--force")
    subprocess.run(command, cwd=REMOTE_REPO, check=True)
    volume.commit()
    files = []
    for path in sorted(Path(f"{MOUNT}/datasets").rglob("*")):
        if path.is_file():
            files.append(f"{path.relative_to(MOUNT)} {path.stat().st_size}")
    return files


@app.local_entrypoint()
def main(force: bool = False):
    for line in seed.remote(force=force):
        print(line)
