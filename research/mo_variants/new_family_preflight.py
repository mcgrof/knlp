"""Qualified-family HIP fixtures, parent checkpoint replay, and cost probes."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import Model as Baseline, digest, loss
from research.tinystories_depth.probe import batch, evaluate, finite
from research.tinystories_depth.train import state_hash
from .preflight import clients, write
from .model import Model
from .qualify import run as qualify


def replay(root, device):
    root = Path(root)
    if device == "cuda":
        assert os.environ.get("GPUQ_RUN_ID"), "use gpuq"
        clients()
    torch.set_num_threads(2 if device == "cuda" else 8)
    curve = np.load(root / "data/curve.npy", mmap_mode="r")
    x, y = batch(curve, np.arange((len(curve) - 1) // 256), 0, device)
    record = {
        "device": device,
        "array_sha256": sha(root / "data/curve.npy"),
        "gpuq_run_id": os.environ.get("GPUQ_RUN_ID"),
        "scope": "old checkpoints as provenance, not fresh trials",
        "cells": [],
    }
    for v, arm in (("V00", "M"), ("V01", "M-O")):
        path = root / f"inputs/final-{arm}-D32-seed75211.pt"
        expected = json.loads((path.parent / (path.stem + ".RESULT.json")).read_text())
        assert sha(path) == expected["checkpoint"]["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert (
            state_hash(payload["model"]) == expected["checkpoint"]["parameter_sha256"]
        )
        assert (
            state_hash(payload["optimizer"])
            == expected["checkpoint"]["optimizer_state_sha256"]
        )
        assert digest(payload["rng_cpu"]) == expected["checkpoint"]["rng_cpu_sha256"]
        assert digest(payload["rng_gpu"]) == expected["checkpoint"]["rng_gpu_sha256"]
        inherited = Baseline(arm, 32, 75211).to(device)
        model = Model(v, 32, 75211).to(device)
        inherited.load_state_dict(payload["model"], strict=True)
        model.load_state_dict(payload["model"], strict=True)
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            a, b = model(x), inherited(x)
            ca, cb = loss(a, y), loss(b, y)
        assert torch.equal(a, b) and torch.equal(ca, cb)
        ce = evaluate(model, curve, device)
        prior = next(p["ce"] for p in expected["validation"] if p["step"] == 2000)
        record["cells"].append(
            {
                "variant": v,
                "arm": arm,
                "checkpoint_sha256": sha(path),
                "parameter_hash_verified": True,
                "optimizer_hash_verified": True,
                "rng_hashes_verified": True,
                "same_device_inherited_logits_bit_equal": True,
                "same_device_inherited_loss_bit_equal": True,
                "first_2048_tokens_ce": ca.item(),
                "curve_ce": ce,
                "prior_RTX_curve_ce": prior,
                "curve_minus_prior_RTX": ce - prior,
                "note": "Cross-hardware loss difference is descriptive, not a same-device replay failure",
            }
        )
        del model, inherited, payload
        gc.collect()
    write(root / f"records/PARENT-REPLAY-{device}.json", record)
    print(json.dumps(record), flush=True)


def run(root):
    root = Path(root)
    assert os.environ.get("GPUQ_RUN_ID")

    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    record = {
        "gpuq_run_id": os.environ["GPUQ_RUN_ID"],
        "device": torch.cuda.get_device_name(0),
        "status": "running",
        "scope": "disposable representative cost probes; no pilot or discovery outcomes",
        "cells": [],
        "source_files": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "owners": [clients()],
    }
    output = root / "records/NEW-FAMILY-PREFLIGHT.json"
    write(output, record)
    try:
        qualify(root / "records/variant-fixtures-attempt3", "cuda")
        record["new_family_fidelity"] = "HIP passed all 32 rows"
        replay(root, "cuda")
        train = np.load(root / "data/train.npy", mmap_mode="r")
        curve = np.load(root / "data/curve.npy", mmap_mode="r")
        order = np.load(root / "data/order-76390.npy")
        checkpoints = Path(os.environ["GPUQ_RUN_DIR"]) / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        for v in ("V20", "V24", "V25", "V26", "V27", "V28", "V29", "V30"):
            clients()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            model = Model(v, 32, 76390).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.001,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0,
            )
            setup = time.perf_counter() - before
            times = []
            for step in range(25):
                torch.cuda.synchronize()
                before = time.perf_counter()
                model.step = step + 1
                x, y = batch(train, order, step, "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    value = loss(logits, y)
                assert torch.isfinite(value) and torch.isfinite(logits).all()
                value.backward()
                finite(model)
                optimizer.step()
                finite(model, optimizer)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                if step >= 5:
                    times.append(elapsed)
            before = time.perf_counter()
            ce = evaluate(model, curve, "cuda")
            eval_time = time.perf_counter() - before
            checkpoint = checkpoints / f"cost-{v}-D32-seed76390.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_gpu": torch.cuda.get_rng_state(),
                    "step": 25,
                    "seed": 76390,
                    "variant": v,
                    "scope": "disposable cost probe",
                },
                checkpoint,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert state_hash(payload["model"]) == state_hash(model.state_dict())
            record["cells"].append(
                {
                    "variant": v,
                    "depth": 32,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "setup_seconds": setup,
                    "step_times_seconds": times,
                    "step_mean_seconds": statistics.mean(times),
                    "curve_32768_tokens_seconds": eval_time,
                    "disposable_curve_ce": ce,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": sha(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                        "state_hash_reload_match": True,
                    },
                    "owners": clients(),
                }
            )
            write(output, record)
            print(json.dumps(record["cells"][-1]), flush=True)
            del model, optimizer, payload
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write(output, record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--cpu-replay", action="store_true")
    args = p.parse_args()
    replay(args.root, "cpu") if args.cpu_replay else run(args.root)
