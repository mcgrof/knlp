#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run subliminal-learning cells on Modal, one small GPU container per cell.

The science is in run.py; this file only ships the package, fans a list of
cells out over containers, and copies each finished run to a volume. A
cell is a defconfig name plus recipe overrides, exactly what
`make defconfig-<base> SUBLIMINAL_SET="..."` would run locally, so the same
cell can be repeated on another accelerator.

GPU functions can be preempted, so a cell writes nothing to the volume
until it has finished; a restarted container reruns the cell from the
start and a cell whose status.json is already complete is skipped. MNIST
comes from the normalized tensor cache placed on the volume beforehand,
so the image needs no torchvision.

Run it from the repository root, with the cells in a JSON file::

    modal run --detach research/subliminal_learning/modal_app.py \\
        --cells cells.json --commit <knlp revision>

cells.json: {"base": "<defconfig>", "common": {field: value},
             "cells": [{"group": g, "name": n, "overrides": {...}}]}
"""

import hashlib
import json
import os
import pathlib
import shutil
import time

import modal

app = modal.App(os.environ.get("KNLP_MODAL_APP", "knlp-subliminal"))

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch==2.14.0", "numpy"
)
if modal.is_local():
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent.parent
    image = image.add_local_dir(
        here,
        "/root/research/subliminal_learning",
        # the platform ships this file itself; shipping it twice is refused
        ignore=[
            "**/__pycache__",
            "**/*.pyc",
            "modal_app.py",
            "tests",
            "tests/**",
            "upstream",
            "upstream/**",
        ],
    ).add_local_dir(
        root / "defconfigs",
        "/root/defconfigs",
        ignore=lambda p: not pathlib.Path(p).name.startswith("subliminal-mnist-"),
    )

runs = modal.Volume.from_name("knlp-runs", create_if_missing=True)
VOL = "/vol"


@app.function(
    image=image,
    gpu="L4",
    cpu=2.0,
    memory=8192,
    timeout=4 * 3600,
    volumes={VOL: runs},
    retries=modal.Retries(max_retries=3, initial_delay=5.0),
)
def run_cell(cell: dict, base: str, common: dict, source: dict) -> dict:
    import sys

    import torch

    os.environ.update(source)
    sys.path.insert(0, "/root")
    from research.subliminal_learning import recipe as rmod
    from research.subliminal_learning import run as rrun

    group, name = cell["group"], cell["name"]
    dest = f"{VOL}/subliminal/{group}/{name}"
    runs.reload()
    status = os.path.join(dest, "status.json")
    if os.path.exists(status) and json.load(open(status)).get("complete"):
        return {"name": name, "skipped": True}
    fields = {**common, **cell["overrides"]}
    overrides = [f"{k}={v}" for k, v in fields.items()] + [
        f"run_name={name}",
        "output_dir=/tmp/out",
        "checkpoint_dir=/tmp/ckpt",
        f"data_dir={VOL}/subliminal/mnist",
        "max_seconds=0",
    ]
    shutil.rmtree("/tmp/out", ignore_errors=True)
    shutil.rmtree("/tmp/ckpt", ignore_errors=True)
    recipe, where = rmod.resolve(f"/root/defconfigs/{base}", overrides)
    t0 = time.time()
    rc = rrun.cmd_run(
        recipe, where, ["modal", "run_cell", f"defconfig={base}", *overrides]
    )
    elapsed = time.time() - t0
    out = f"/tmp/out/{name}"
    ckpt_dest = f"{VOL}/subliminal-ckpt/{group}/{name}"
    manifest_path = os.path.join(out, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        for fn, entry in manifest.items():
            entry["path"] = f"modal:knlp-runs/subliminal-ckpt/{group}/{name}/{fn}"
        json.dump(manifest, open(manifest_path, "w"), indent=2)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(out, dest)
    if os.path.isdir(f"/tmp/ckpt/{name}"):
        shutil.copytree(f"/tmp/ckpt/{name}", ckpt_dest, dirs_exist_ok=True)
    runs.commit()
    return {
        "name": name,
        "group": group,
        "rc": rc,
        "elapsed_s": elapsed,
        "gpu": torch.cuda.get_device_name(0),
    }


def tree_sha256(root):
    """SHA-256 over the shipped package and defconfig files, path-sorted."""
    h = hashlib.sha256()
    here = pathlib.Path(root)
    files = sorted(
        p
        for p in here.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and "tests" not in p.parts
        and "upstream" not in p.parts
        and p.name != "modal_app.py"
    )
    for p in files:
        h.update(str(p.relative_to(here)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@app.local_entrypoint()
def main(cells: str, commit: str, dirty: str = "0", branch: str = "main"):
    spec = json.load(open(cells))
    here = pathlib.Path(__file__).resolve().parent
    source = {
        "KNLP_SOURCE_COMMIT": commit,
        "KNLP_SOURCE_BRANCH": branch,
        "KNLP_SOURCE_DIRTY": dirty,
        "KNLP_SOURCE_TREE_SHA256": tree_sha256(here),
    }
    print(json.dumps(source), flush=True)
    results = []
    for r in run_cell.map(
        spec["cells"],
        kwargs={
            "base": spec["base"],
            "common": spec.get("common", {}),
            "source": source,
        },
        return_exceptions=True,
        order_outputs=False,
    ):
        results.append(r if isinstance(r, dict) else {"error": repr(r)})
        print(json.dumps(results[-1]), flush=True)
    out = pathlib.Path(cells).with_suffix(".results.json")
    out.write_text(json.dumps(results, indent=2))
