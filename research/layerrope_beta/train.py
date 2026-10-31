# SPDX-License-Identifier: MIT
"""Train paired story models with saved optimizer/RNG states and fixed evaluations."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from research.layerrope_beta.model import Model
from research.layerrope_beta.measure import distributions, quantization
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.kv_cache import evaluate
from research.tinystories_depth.model import digest, loss
from research.tinystories_depth.probe import batch
from research.tinystories_depth.train import state_hash


def save(model, optimizer, step, job, folder):
    path = folder / f"step{step}.pt"
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        rng_cpu=torch.get_rng_state(),
        rng_gpu=torch.cuda.get_rng_state(),
        metadata=dict(job=job),
    )
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    temp.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if state_hash(loaded["model"]) != state_hash(state["model"]):
        raise RuntimeError("Checkpoint reload mismatch")
    return dict(
        path=str(path),
        sha256=sha(path),
        bytes=path.stat().st_size,
        parameter_sha256=state_hash(loaded["model"]),
        optimizer_sha256=state_hash(loaded["optimizer"]),
        rng_cpu_sha256=digest(loaded["rng_cpu"]),
        rng_gpu_sha256=digest(loaded["rng_gpu"]),
        model_finite=all(
            torch.isfinite(v).all().item() for v in loaded["model"].values()
        ),
        step=step,
    )


def train(root, job, checkpoints, max_seconds):
    destination = root / "language/runs" / job["id"]
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = checkpoints / "language" / job["id"]
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    arrays = {
        n: np.load(data / f"{n}.npy", mmap_mode="r")
        for n in ("train", "curve", "final")
    }
    order = np.load(data / f"order-{job['seed']}.npy")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(job["seed"])
    model = Model(
        job["arm"],
        32,
        job["seed"],
        beta=job["beta"] if job["beta"] is not None else -0.5,
        alpha_out=job.get("alpha_out") or 0.0,
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=job["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0
    )
    began = time.monotonic()
    result = dict(
        status="running",
        job=job,
        checkpoints=[],
        curve=[],
        steps=[],
        common_initial_weights=model.common_hashes,
        order_sha256=sha(data / f"order-{job['seed']}.npy"),
        model_parameters=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
    )
    write(destination / "RESULT.json", result)
    curve_case = dict(name="clean", k_bits=16, v_bits=16, method="none")

    def validation(step):
        model.eval()
        value = evaluate(model, arrays["curve"], curve_case, "cuda", batch_size=32)
        result["curve"].append(
            dict(
                step=step,
                tokens=step * 2048,
                ce=value["ce"],
                seconds=time.monotonic() - began,
            )
        )
        if not math.isfinite(value["ce"]):
            raise FloatingPointError("Nonfinite clean validation loss")
        model.train()

    completed = 0
    # A screen keeps final checkpoints but defers the quantization grid.
    full = job["stage"] in ("final", "screen")
    measure = job["stage"] == "final"
    try:
        if measure:
            result["checkpoints"].append(save(model, optimizer, 0, job, checkpoint_dir))
            write(
                destination / "statistics-step0.json",
                distributions(model, arrays["final"], "cuda"),
            )
        validation(0)
        for step in range(1, job["steps"] + 1):
            if time.monotonic() - began > max_seconds:
                raise TimeoutError("Language job elapsed-time limit")
            started = time.monotonic()
            x, y = batch(arrays["train"], order, step - 1, "cuda")
            for group in optimizer.param_groups:
                group["lr"] = job["lr"] * min(step / 100, 1.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective = loss(model(x), y)
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite training loss")
            objective.backward()
            if step % 25 == 0 or step == 1:
                if not all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            completed = step
            result["steps"].append(
                dict(step=step, ce=objective.item(), seconds=time.monotonic() - started)
            )
            if step % 100 == 0:
                validation(step)
                write(destination / "RESULT.json", result)
            if full and step in (1000, 2000):
                result["checkpoints"].append(
                    save(model, optimizer, step, job, checkpoint_dir)
                )
                if measure:
                    write(
                        destination / f"statistics-step{step}.json",
                        distributions(model, arrays["final"], "cuda"),
                    )
        if job["stage"] == "pilot":
            result["checkpoints"].append(
                save(model, optimizer, completed, job, checkpoint_dir)
            )
        elif measure:
            model.eval()
            write(
                destination / "QUANTIZATION.json",
                quantization(model, arrays["final"], "cuda"),
            )
        else:
            model.eval()
            result["final_clean"] = evaluate(
                model, arrays["final"], curve_case, "cuda", batch_size=32
            )
        if not all(c["model_finite"] for c in result["checkpoints"]):
            raise FloatingPointError("Nonfinite saved model state")
        result["status"] = "complete"
    except BaseException as error:
        result.update(
            status=(
                "invalid_numerical"
                if isinstance(error, FloatingPointError)
                else "incomplete"
            ),
            error=str(error),
        )
        if completed and not any(c["step"] == completed for c in result["checkpoints"]):
            result["checkpoints"].append(
                save(model, optimizer, completed, job, checkpoint_dir)
            )
        if not isinstance(error, FloatingPointError):
            raise
    finally:
        result.update(
            completed_steps=completed, elapsed_seconds=time.monotonic() - began
        )
        write(destination / "RESULT.json", result)
        print(
            json.dumps(
                dict(
                    id=job["id"],
                    status=result["status"],
                    steps=completed,
                    seconds=result["elapsed_seconds"],
                )
            ),
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    train(args.root, read(args.job), args.checkpoints, args.max_seconds)
