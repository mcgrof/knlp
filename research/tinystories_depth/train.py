"""Frozen paired TinyStories grid with numerical checks and budget stops."""

import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import sha
from .model import Model, digest, loss
from .probe import ForeignClientError, batch, clients, evaluate, finite, write

OPTIMIZER = {
    "type": "AdamW",
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "weight_decay": 0,
    "clipping": None,
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def remaining():
    return float(os.environ["KNLP_GPU_REMAINING_SECONDS"]) - (
        time.monotonic() - float(os.environ["KNLP_GPU_COMMAND_STARTED_MONOTONIC"])
    )


def state_hash(state):
    h = hashlib.sha256()

    def walk(value, path=""):
        if torch.is_tensor(value):
            h.update(path.encode())
            h.update(digest(value).encode())
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, path + "/" + str(k))
        elif isinstance(value, (tuple, list)):
            for k, v in enumerate(value):
                walk(v, path + "/" + str(k))
        else:
            h.update(path.encode())
            h.update(repr(value).encode())

    walk(state)
    return h.hexdigest()


@torch.no_grad()
def snapshot(model, x, step, run_id, metadata, directory):
    model.capture = True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(x)
    model.capture = False
    schedule = {}
    if model.lm.model.layerrope:
        rope = model.lm.model.layerrope
        schedule = {k: v.cpu().tolist() for k, v in rope.state_dict().items()}
        schedule["site_gains_by_block"] = {
            site: [
                rope(site, torch.tensor(np.log(l + 1), device="cuda")).cpu().tolist()
                for l in range(model.depth)
            ]
            for site in rope.sites
        }
    elif model.angular:
        schedule = {k: v.cpu().tolist() for k, v in model.angular.state_dict().items()}
    path = directory / f"diagnostic-{run_id}-step{step}.pt"
    torch.save(
        {
            "samples": model.samples,
            "schedule": schedule,
            "input_ids": x.cpu(),
            "step": step,
            "run_id": run_id,
            "code_assignment": model.assignment,
            "metadata": metadata,
        },
        path,
    )
    return {
        "path": str(path),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "step": step,
    }


def flops(depth, mhc_arm):
    # Dense matrix convention: multiply+add=2 FLOPs, training=3*forward.
    # Excludes elementwise norms, gains, CE, Sinkhorn, optimizer and I/O.
    tokens = 2048
    dense = 2 * tokens * (depth * (4 * 128**2 + 3 * 128 * 352) + 128 * 50257)
    attention = 4 * 8 * 256**2 * 128 * depth
    routing = 2 * tokens * 24 * 512 * (2 * depth) if mhc_arm else 0
    return 3 * (dense + attention + routing)


def trajectory(root, data, checkpoints, job, plan, meta):
    run_id = job["id"]
    destination = root / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(job["seed"])
    model = Model(job["arm"], job["depth"], job["seed"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=job["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0
    )
    order = np.load(data / f"order-{job['seed']}.npy")
    train = np.load(data / "train.npy", mmap_mode="r")
    curve = np.load(data / "curve.npy", mmap_mode="r")
    x, _ = batch(train, order, 0, "cuda")
    metadata = {
        **meta,
        "job": job,
        "order_sha256": sha(data / f"order-{job['seed']}.npy"),
        "optimizer_config_sha256": json_hash({**OPTIMIZER, "lr": job["lr"]}),
        "initial_rng_cpu": digest(torch.get_rng_state()),
        "initial_rng_gpu": digest(torch.cuda.get_rng_state()),
    }
    result = {
        "metadata": metadata,
        "status": "running",
        "identities": model.identities(),
        "training": [],
        "validation": [],
        "diagnostics": [],
        "clients": [clients(strict=False)],
        "optimizer_step_flops_estimate": flops(
            job["depth"], job["arm"].startswith("M")
        ),
        "step_times_seconds": [],
        "training_seconds": 0.0,
        "scheduled_eval_seconds": 0.0,
    }
    write(destination / "RESULT.json", result)
    intervals = 50 if plan["tier"] == "SCREEN" and job["stage"] == "pilot" else 100
    qualification = job["stage"] == "qualification"
    try:
        result["clients"].append(clients())
        result["diagnostics"].append(
            snapshot(model, x, 0, run_id, metadata, checkpoints)
        )
        if not qualification:
            started = time.perf_counter()
            v = evaluate(model, curve, "cuda")
            elapsed = time.perf_counter() - started
            result["scheduled_eval_seconds"] += elapsed
            result["validation"].append(
                {
                    "step": 0,
                    "tokens": 0,
                    "ce": v,
                    "training_seconds": 0.0,
                    "end_to_end_seconds": time.perf_counter() - t0,
                    "eval_seconds": elapsed,
                }
            )
        with (destination / "TRAINING.jsonl").open("x") as log:
            log.write(json.dumps({"metadata": metadata}) + "\n")
            for step in range(1, job["steps"] + 1):
                if remaining() < 30:
                    raise TimeoutError("cumulative GPU-command cap")
                torch.cuda.synchronize()
                started = time.perf_counter()
                x, y = batch(train, order, step - 1, "cuda")
                lr = job["lr"] if qualification else job["lr"] * min(step / 100, 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    value = loss(logits, y)
                result["last_attempt"] = {
                    "step": step,
                    "lr": lr,
                    "ce": (
                        value.item()
                        if torch.isfinite(value).item()
                        else str(value.item())
                    ),
                }
                assert (
                    torch.isfinite(value).item() and torch.isfinite(logits).all().item()
                ), "nonfinite loss or output"
                value.backward()
                finite(model)
                optimizer.step()
                finite(model, optimizer)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                result["training_seconds"] += elapsed
                result["step_times_seconds"].append(elapsed)
                record = {
                    "step": step,
                    "tokens": step * 2048,
                    "ce": value.item(),
                    "lr": lr,
                    "step_seconds": elapsed,
                    "end_to_end_seconds": time.perf_counter() - t0,
                    "invariants": (
                        model.invariants.cpu().tolist()
                        if model.invariants is not None
                        else None
                    ),
                }
                log.write(json.dumps(record) + "\n")
                if step in (50, 500, job["steps"]):
                    fixed, _ = batch(train, order, 0, "cuda")
                    result["diagnostics"].append(
                        snapshot(model, fixed, step, run_id, metadata, checkpoints)
                    )
                if not qualification and step % intervals == 0:
                    started = time.perf_counter()
                    v = evaluate(model, curve, "cuda")
                    elapsed = time.perf_counter() - started
                    result["scheduled_eval_seconds"] += elapsed
                    result["validation"].append(
                        {
                            "step": step,
                            "tokens": step * 2048,
                            "ce": v,
                            "training_seconds": result["training_seconds"],
                            "end_to_end_seconds": time.perf_counter() - t0,
                            "eval_seconds": elapsed,
                        }
                    )
                    result["clients"].append(clients())
                    write(destination / "RESULT.json", result)
        if job["stage"] == "final":
            final = np.load(data / "final.npy", mmap_mode="r")
            started = time.perf_counter()
            result["final_report_ce"] = evaluate(model, final, "cuda")
            result["final_report_seconds"] = time.perf_counter() - started
        result["status"] = "complete"
    except TimeoutError as e:
        result["status"] = "incomplete_budget"
        result["reason"] = str(e)
    except ForeignClientError as e:
        result["status"] = "incomplete_contention"
        result["reason"] = str(e)
    except AssertionError as e:
        result["status"] = "invalid_numerical"
        result["reason"] = str(e)
    finally:
        result["completed_steps"] = len(result["step_times_seconds"])
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        checkpoint = checkpoints / f"{run_id}.pt"
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_gpu": torch.cuda.get_rng_state(),
            "metadata": metadata,
            "status": result["status"],
            "step": result["completed_steps"],
        }
        started = time.perf_counter()
        torch.save(state, checkpoint)
        result["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "bytes": checkpoint.stat().st_size,
        }
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result["checkpoint"]["parameter_sha256"] = state_hash(loaded["model"])
        result["checkpoint"]["optimizer_state_sha256"] = state_hash(loaded["optimizer"])
        result["checkpoint"]["rng_cpu_sha256"] = digest(loaded["rng_cpu"])
        result["checkpoint"]["rng_gpu_sha256"] = digest(loaded["rng_gpu"])
        result["checkpoint"]["reload_finite"] = all(
            torch.isfinite(v).all().item() for v in loaded["model"].values()
        )
        result["checkpoint_seconds"] = time.perf_counter() - started
        result["clients"].append(clients(strict=False))
        result["end_to_end_seconds"] = time.perf_counter() - t0
        result["training_log_sha256"] = (
            sha(destination / "TRAINING.jsonl")
            if (destination / "TRAINING.jsonl").exists()
            else None
        )
        result["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(destination / "RESULT.json", result)
        print(
            json.dumps(
                {
                    "id": run_id,
                    "status": result["status"],
                    "steps": result["completed_steps"],
                    "seconds": result["end_to_end_seconds"],
                    "remaining": remaining(),
                }
            ),
            flush=True,
        )
    return result


def run(root, data, checkpoints, qualify):
    root, data, checkpoints = Path(root), Path(data), Path(checkpoints)
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((root / "PLAN.json").read_text())
    meta = {
        "source": json.loads(os.environ["KNLP_SUPERVISED_SOURCE_JSON"]),
        "plan_sha256": sha(root / "PLAN.json"),
        "data_sha256": sha(data / "DATA.json"),
    }
    assert plan["frozen_before_outcomes"]
    assert plan["tier"] in ("SCREEN", "CORE", "CORE_CHANNELS")
    if qualify:
        results = []
        for job in plan["qualification_jobs"]:
            results.append(trajectory(root, data, checkpoints, job, plan, meta))
            if results[-1]["status"].startswith("incomplete"):
                break
        q = json.loads((root / "QUALIFICATION.json").read_text())
        q["training"] = [
            {
                "id": r["metadata"]["job"]["id"],
                "status": r["status"],
                "completed_steps": r["completed_steps"],
            }
            for r in results
        ]
        q["status"] = (
            "qualified"
            if len(results) == len(plan["qualification_jobs"])
            and all(r["status"] == "complete" for r in results)
            else "blocked_training_qualification"
        )
        write(root / "QUALIFICATION.json", q)
        return
    assert (
        json.loads((root / "QUALIFICATION.json").read_text())["status"] == "qualified"
    )
    status = {
        "status": "running",
        "tier": plan["tier"],
        "metadata": meta,
        "started": [],
        "completed": [],
        "invalid": [],
        "missing": [],
        "selected_lrs": {},
    }
    write(root / "STATUS.json", status)
    for cell in plan["cells"]:
        candidates = []
        for job in cell["pilots"]:
            status["started"].append(job["id"])
            write(root / "STATUS.json", status)
            result = trajectory(root, data, checkpoints, job, plan, meta)
            if result["status"].startswith("incomplete"):
                status["status"] = result["status"]
                write(root / "STATUS.json", status)
                return
            status["completed" if result["status"] == "complete" else "invalid"].append(
                job["id"]
            )
            if result["status"] == "complete":
                points = (
                    (150, 200, 250) if plan["tier"] == "SCREEN" else (300, 400, 500)
                )
                score = np.mean(
                    [r["ce"] for r in result["validation"] if r["step"] in points]
                )
                assert np.isfinite(score)
                candidates.append((float(score), job["lr"]))
        if not candidates:
            status["missing"] += [
                {"id": j["id"], "reason": "all_LRs_invalid_trainability_failure"}
                for j in cell["finals"]
            ]
            write(root / "STATUS.json", status)
            continue
        chosen = min(candidates)[1]
        status["selected_lrs"][cell["id"]] = chosen
        write(root / "STATUS.json", status)
        for template in cell["finals"]:
            job = {**template, "lr": chosen}
            status["started"].append(job["id"])
            write(root / "STATUS.json", status)
            result = trajectory(root, data, checkpoints, job, plan, meta)
            if result["status"].startswith("incomplete"):
                status["status"] = result["status"]
                write(root / "STATUS.json", status)
                return
            status["completed" if result["status"] == "complete" else "invalid"].append(
                job["id"]
            )
        write(root / "STATUS.json", status)
    status["status"] = "complete"
    write(root / "STATUS.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()
    run(a.root, a.data, a.checkpoints, a.qualify)
