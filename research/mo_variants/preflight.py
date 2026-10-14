"""Qualify inherited operators and measure cost before outcome jobs."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import Model, digest, loss
from research.tinystories_depth.probe import batch, evaluate, finite
from research.tinystories_depth.qualify import run as fidelity


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def clients():
    from research.tinystories_depth.probe import clients as inspect_clients

    return inspect_clients()


def run(root, data):
    root, data = Path(root), Path(data)
    root.mkdir(parents=True, exist_ok=True)
    assert os.environ.get("GPUQ_RUN_ID"), "invoke through the study runner"
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    record = {
        "study": "depth-code-discovery",
        "status": "running",
        "scope": "inherited finite operators, disposable timing; no search outcomes",
        "gpuq_run_id": os.environ["GPUQ_RUN_ID"],
        "gpuq_run_dir": os.environ["GPUQ_RUN_DIR"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "tokenizers", "pyarrow")
        },
        "kernel": os.uname().release,
        "hip": torch.version.hip,
        "seed": 76390,
        "warmup_steps": 5,
        "timed_steps": 20,
        "cells": [],
        "owners": [clients()],
    }
    output = root / "PREFLIGHT.json"
    write(output, record)
    try:
        assert torch.cuda.is_available()
        record["device"] = torch.cuda.get_device_name(0)
        train = np.load(data / "train.npy", mmap_mode="r")
        curve = np.load(data / "curve.npy", mmap_mode="r")
        manifest = json.loads((data / "DATA.json").read_text())
        record["verified_data"] = {}
        for name in ("train.npy", "curve.npy", "DOCUMENTS.json"):
            actual = sha(data / name)
            assert actual == manifest["files"][name], name
            record["verified_data"][name] = actual
        order = np.random.default_rng(76390).permutation((len(train) - 1) // 256)
        # Pristine finite-function tests are inherited, not novel-family gates.
        fidelity(root, "cpu")
        fidelity(root, "cuda")
        record["inherited_fidelity"] = "CPU and HIP passed"
        x, y = batch(train, order, 0, "cuda")
        checkpoints = Path(os.environ["GPUQ_RUN_DIR"]) / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        for depth, arm in ((32, "M"), (32, "M-O"), (32, "ML-O"), (64, "M-O")):
            record["owners"].append(clients())
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(76390)
            before = time.perf_counter()
            model = Model(arm, depth, 76390).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0,
            )
            torch.cuda.synchronize()
            setup = time.perf_counter() - before
            times = []
            for step in range(25):
                torch.cuda.synchronize()
                before = time.perf_counter()
                xb, yb = batch(train, order, step, "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    value = loss(logits, yb)
                assert torch.isfinite(value).item() and torch.isfinite(logits).all()
                value.backward()
                finite(model)
                optimizer.step()
                finite(model, optimizer)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                if step >= 5:
                    times.append(elapsed)
            before = time.perf_counter()
            disposable_ce = evaluate(model, curve, "cuda")
            eval_seconds = time.perf_counter() - before
            model.eval()
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                saved_logits = model(x).detach().cpu()
            before = time.perf_counter()
            checkpoint = checkpoints / f"cost-{arm}-D{depth}-seed76390.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_gpu": torch.cuda.get_rng_state(),
                    "step": 25,
                    "seed": 76390,
                    "scope": "disposable cost probe",
                },
                checkpoint,
            )
            checkpoint_seconds = time.perf_counter() - before
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            expected = {k: digest(v) for k, v in payload["model"].items()}
            model.load_state_dict(payload["model"], strict=True)
            assert {k: digest(v) for k, v in model.state_dict().items()} == expected
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                replay = model(x).detach().cpu()
            assert torch.equal(replay, saved_logits), "same-device checkpoint replay"
            cell = {
                "arm": arm,
                "depth": depth,
                "parameters": sum(p.numel() for p in model.parameters()),
                "setup_seconds": setup,
                "step_times_seconds": times,
                "step_mean_seconds": statistics.mean(times),
                "step_median_seconds": statistics.median(times),
                "curve_32768_tokens_seconds": eval_seconds,
                "disposable_curve_ce": disposable_ce,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "save_seconds": checkpoint_seconds,
                    "exact_replay": True,
                },
                "owners": clients(),
            }
            record["cells"].append(cell)
            write(output, record)
            print(json.dumps(cell), flush=True)
            del model, optimizer, payload
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write(output, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    run(args.root, args.data)
