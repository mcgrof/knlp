"""Disposable throughput probes and complete-tier cost forecasts."""

import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .data import sha
from .model import Model, digest, loss


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


class ForeignClientError(RuntimeError):
    pass


def clients(strict=True):
    """Inspect device owners only when exclusive-device checking is requested."""
    if os.environ.get("KNLP_CHECK_GPU_OWNERS") != "1":
        return {
            "checked": False,
            "reason": "device allocation is managed by the caller",
        }
    if torch.version.hip:
        result = subprocess.run(
            ["fuser", "/dev/kfd", "/dev/dri/renderD128"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError("Device owner query failed: " + result.stderr)
        owners = [int(p) for p in result.stdout.split()]
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        owners = [int(p.strip()) for p in result.stdout.splitlines() if p.strip()]
    foreign = [p for p in owners if p != os.getpid()]
    if strict and foreign:
        raise RuntimeError(f"Other device owners are active: {foreign}")
    return {"checked": True, "owners": owners, "self": os.getpid(), "foreign": foreign}


def finite(model, optimizer=None):
    checks = [torch.isfinite(p).all() for p in model.parameters()]
    checks += [
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    ]
    if optimizer:
        for state in optimizer.state.values():
            for value in state.values():
                if not torch.is_tensor(value):
                    continue
                if value.device.type == "cpu":
                    assert torch.isfinite(value).all().item()
                else:
                    checks.append(torch.isfinite(value).all())
    assert (
        torch.stack(checks).all().item()
    ), "nonfinite parameter, gradient, or optimizer state"
    if model.invariants is not None:
        v = model.invariants.cpu().tolist()
        assert all(x <= 1e-7 for x in v[2:]), v


def batch(tokens, order, step, device):
    ix = order[step * 8 : step * 8 + 8].astype(np.int64) * 256
    source = np.stack([tokens[i : i + 257] for i in ix]).astype(np.int64)
    t = torch.from_numpy(source).to(device)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, tokens, device):
    model.eval()
    ce = torch.zeros((), device=device)
    count = (len(tokens) - 1) // 256
    for start in range(0, count, 8):
        ids, target = batch(tokens, np.arange(count), start // 8, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(ids)
            value = loss(logits, target)
        assert torch.isfinite(value).item()
        assert torch.isfinite(logits).all().item()
        finite(model)
        ce += value * target.numel()
    model.train()
    return (ce / (count * 256)).item()


def forecast(cells, pilot=250, final=1000, include16=False):
    # Missing timings contribute zero: this is a lower bound, never a
    # claim that the unmeasured cells will be free or faster than D=4.
    evals = (pilot // (50 if pilot == 250 else 100) + 1) * 4 + (final // 100 + 1) * 3
    totals = {
        "train_seconds": 0.0,
        "scheduled_eval_seconds": 0.0,
        "final_eval_seconds": 0.0,
        "setup_io_seconds": 0.0,
        "qualification_seconds": 0.0,
        "diagnostics_seconds": 0.0,
    }
    for c in cells:
        totals["train_seconds"] += (pilot * 4 + final * 3) * c["step_mean_seconds"]
        totals["scheduled_eval_seconds"] += evals * c["validation_seconds"]
        totals["final_eval_seconds"] += 12 * c["validation_seconds"]
        totals["setup_io_seconds"] += 7 * (c["setup_seconds"] + c["checkpoint_seconds"])
        totals["qualification_seconds"] += (
            50 * c["step_mean_seconds"] if c["depth"] in (4, 32) else 0
        )
        totals["diagnostics_seconds"] += 7 * 4 * c["diagnostic_seconds"]
    return {
        "tier": "SCREEN" if pilot == 250 else "CORE",
        "pilot_steps": pilot,
        "final_steps": final,
        "measured_cells": len(cells),
        "missing_cells": (
            [f"{arm}-D16" for arm in ("R", "L", "M", "ML")] if not include16 else []
        ),
        "components": totals,
        "contingency": 1.25,
        "forecast_seconds": 1.25 * sum(totals.values()),
        "is_lower_bound": not include16,
        "final_eval_assumption": "131072-token endpoint evaluated at same tokens/sec as 32768-token curve pass",
    }


def run(root, data, checkpoints, extra=False):
    root, data, checkpoints = Path(root), Path(data), Path(checkpoints)
    checkpoints.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    result = {
        "status": "running",
        "device": torch.cuda.get_device_name(0),
        "dtype": "BF16 block GEMM; FP32 master/state/normalization/routing/Sinkhorn/CE",
        "seed": 75290,
        "warmup": 5,
        "measured": 10,
        "effective_predicted_tokens": 2048,
        "microbatch_sequences": 8,
        "repeat": 20,
        "cells": [],
        "clients": [clients()],
    }
    if extra:
        result = json.loads((root / "PREFLIGHT.json").read_text())
        assert result["status"] == "needs_D16_and_channel_probe"
        result["status"] = "running_extra"
        result["clients"].append(clients())
    train = np.load(data / "train.npy", mmap_mode="r")
    curve = np.load(data / "curve.npy", mmap_mode="r")
    order = np.load(data / "order-75290.npy")
    result["data_manifest_sha256"] = sha(data / "DATA.json")
    result["source"] = json.loads(os.environ["KNLP_SUPERVISED_SOURCE_JSON"])
    sequence = [(arm, d) for d in (4, 32) for arm in ("R", "L", "M", "ML")]
    if extra:
        sequence = [(arm, 16) for arm in ("R", "L", "M", "ML")] + [("ML-O", 32)]
    for arm, depth in sequence:
        result["clients"].append(clients())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = Model(arm, depth, 75290).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0
        )
        torch.cuda.synchronize()
        setup = time.perf_counter() - started
        identities = model.identities()
        x, y = batch(train, order, 0, device)
        started = time.perf_counter()
        model.capture = True
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            init = model(x)
        init_samples = model.samples
        model.capture = False
        torch.cuda.synchronize()
        diagnostic = time.perf_counter() - started
        step_times, values, invariants = [], [], []
        for i in range(15):
            x, y = batch(train, order, i, device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(x)
                ce = loss(logits, y)
            assert torch.isfinite(ce).item() and torch.isfinite(logits).all().item()
            ce.backward()
            finite(model)
            optimizer.step()
            finite(model, optimizer)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
            values.append(ce.item())
            invariants.append(
                model.invariants.cpu().tolist()
                if model.invariants is not None
                else None
            )
            if i in (4, 9, 14):
                result["clients"].append(clients())
        torch.cuda.synchronize()
        started = time.perf_counter()
        validation_ce = evaluate(model, curve, device)
        torch.cuda.synchronize()
        validation_time = time.perf_counter() - started
        result["clients"].append(clients())
        checkpoint = checkpoints / f"probe-{arm}-D{depth}.pt"
        started = time.perf_counter()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "initialization_samples": init_samples,
                "source": result["source"],
                "rng_cpu": torch.get_rng_state(),
                "rng_gpu": torch.cuda.get_rng_state(),
                "data_manifest_sha256": result["data_manifest_sha256"],
                "identities": identities,
                "kind": "disposable_probe",
            },
            checkpoint,
        )
        checksum = sha(checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert all(torch.isfinite(t).all() for t in loaded["model"].values())
        checkpoint_time = time.perf_counter() - started
        cell = {
            "arm": arm,
            "depth": depth,
            "identities": identities,
            "setup_seconds": setup,
            "diagnostic_seconds": diagnostic,
            "step_times_seconds": step_times,
            "step_mean_seconds": float(np.mean(step_times[5:])),
            "step_min_seconds": min(step_times[5:]),
            "step_max_seconds": max(step_times[5:]),
            "disposable_losses": values,
            "validation_ce_disposable": validation_ce,
            "validation_seconds": validation_time,
            "invariants": invariants,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint_seconds": checkpoint_time,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checksum,
                "bytes": checkpoint.stat().st_size,
                "reload_finite": True,
            },
        }
        result["cells"].append(cell)
        write(root / "PREFLIGHT.json", result)
        print(
            json.dumps(
                {
                    "arm": arm,
                    "depth": depth,
                    "step": cell["step_mean_seconds"],
                    "validation": validation_time,
                    "memory": cell["peak_allocated_bytes"],
                }
            ),
            flush=True,
        )
        del loaded, model, optimizer, init_samples, init, logits, ce
    core = [c for c in result["cells"] if c["arm"] in ("R", "L", "M", "ML")]
    result["forecasts"] = [
        forecast(core, 250, 1000, extra),
        forecast(core, 500, 2000, extra),
    ]
    if extra:
        coded = next(c for c in result["cells"] if c["arm"] == "ML-O")
        proxies = []
        for arm in ("ML-O", "ML-S", "ML-C", "M-O"):
            proxy = dict(coded)
            proxy["arm"] = arm
            proxies.append(proxy)
        channel = forecast(proxies, 500, 2000, True)
        result["forecasts"].append(
            {
                "tier": "CORE_CHANNELS",
                "forecast_seconds": result["forecasts"][1]["forecast_seconds"]
                + channel["forecast_seconds"],
                "channel_components": channel,
                "cost_proxy": "All four channel arms use measured ML-O D32 rates; M-O proxy is conservative relative to M/ML ordering.",
            }
        )
    elapsed = time.monotonic() - float(os.environ["KNLP_GPU_COMMAND_STARTED_MONOTONIC"])
    remaining = float(os.environ["KNLP_GPU_REMAINING_SECONDS"]) - elapsed
    result["remaining_seconds_at_forecast"] = remaining
    if result["forecasts"][0]["forecast_seconds"] > remaining:
        result["status"] = "blocked_runtime_even_without_D16"
    else:
        result["status"] = "cost_tier_ready" if extra else "needs_D16_and_channel_probe"
    if extra and result["status"] == "cost_tier_ready":
        result["selected_tier"] = next(
            (
                f["tier"]
                for f in reversed(result["forecasts"])
                if f["forecast_seconds"] <= remaining
            ),
            None,
        )
    result["clients"].append(clients())
    write(root / "PREFLIGHT.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "forecasts": result["forecasts"],
                "remaining": remaining,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--extra", action="store_true")
    a = p.parse_args()
    run(a.root, a.data, a.checkpoints, a.extra)
