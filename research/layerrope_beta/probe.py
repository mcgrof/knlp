# SPDX-License-Identifier: MIT
"""Measure disposable costs before starting the full frozen study."""

import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from research.layerrope_beta.model import Model
from research.layerrope_beta.plan import language_rows
from research.mo_variants.stage_c_common import read, write
from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.run import cmd_probe
from research.tinystories_depth.model import loss
from research.tinystories_depth.probe import batch


def run(root, kind):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    if kind == "mnist":
        for name, values in read(root / "PLAN.json")["mnist"]["recipes"].items():
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cmd_probe(Recipe(**values), steps=10)
            result = json.loads(output.getvalue())
            result["id"] = name
            rows.append(result)
        projected = sum(r["projected_train_seconds"] for r in rows)
    else:
        tokens = np.load(root / "data/train.npy", mmap_mode="r")
        order = np.load(root / "data/order-77390.npy")
        x, y = batch(tokens, order, 0, "cuda")
        for row in language_rows():
            model = Model(
                row["arm"],
                32,
                77390,
                beta=row["beta"] if row["beta"] is not None else -0.5,
            ).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0,
            )
            times = []
            for step in range(15):
                torch.cuda.synchronize()
                began = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    objective = loss(model(x), y)
                objective.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if not torch.isfinite(objective):
                    raise FloatingPointError("Nonfinite disposable probe: " + row["id"])
                if step >= 5:
                    times.append(time.monotonic() - began)
            rows.append(
                dict(
                    id=row["id"],
                    mean_step_seconds=float(np.mean(times)),
                    steps=len(times),
                )
            )
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        projected = sum(r["mean_step_seconds"] * (4 * 500 + 3 * 2000) for r in rows)
    result = dict(
        kind=kind,
        rows=rows,
        projected_train_seconds=projected,
        gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
    )
    write(root / f"PROBE-{kind}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("language", "mnist"), required=True)
    args = parser.parse_args()
    run(args.root, args.kind)
