"""One fixed H100 confirmation trajectory with retained recovery states."""

import argparse
import datetime as dt
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from research.tinystories_depth.model import digest, loss
from research.tinystories_depth.probe import batch, evaluate, finite
from research.tinystories_depth.train import flops, state_hash
from .job import checkpoint, snapshot
from .model import Model
from .stage_c_common import plan_guard, read, sha, write


def run(root, jobid, resume=False):
    root = Path(root)
    plan, auth = plan_guard(root)
    job = next(j for j in plan["jobs"] if j["id"] == jobid)
    run_id = root.name
    assert torch.__version__ == "2.9.1+cu128" and torch.version.hip is None
    assert "H100" in torch.cuda.get_device_name(0)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = root / "records/runs" / jobid
    cp = root / "checkpoints" / jobid
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
        cp.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(cp).free >= 50 * 2**30
    for name, expected in (
        ("train.npy", plan["data"]["verified_inherited_files"]["train.npy"]),
        (f"order-{job['seed']}.npy", job["order_sha256"]),
    ):
        assert sha(root / "data" / name) == expected
    curve_path = root / "data/confirmation/curve.npy"
    final_path = root / "data/confirmation/final.npy"
    assert sha(curve_path) == plan["data"]["confirmation_files"]["curve.npy"]
    assert sha(final_path) == plan["data"]["confirmation_files"]["final.npy"]
    train = np.load(root / "data/train.npy", mmap_mode="r")
    curve = np.load(curve_path, mmap_mode="r")
    final = np.load(final_path, mmap_mode="r")
    order = np.load(root / "data" / f"order-{job['seed']}.npy")
    assert len(order) >= 4000 * 8
    torch.manual_seed(job["seed"])
    model = Model(job["variant"], 32, job["seed"], mapping=job["mapping"]).cuda()
    assert (
        model.code_map == job["code_map"]
        and model.mapping_lookup == job["mapping_lookup"]
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0
    )
    metadata = {
        "job": job,
        "run_id": run_id,
        "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(0),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "plan_sha256": sha(root / "records/STAGE-C-PLAN.json"),
        "source_sha256": sha(root / "records/STAGE-C-SOURCE.json"),
        "curve_sha256": sha(curve_path),
        "final_sha256": sha(final_path),
        "selected_lr": 0.003,
        "data_order_sha256": job["order_sha256"],
    }
    output = out / "RESULT.json"
    start, offset = 0, 0.0
    result = {
        "status": "running",
        "metadata": metadata,
        "identities": model.identities(),
        "validation": [],
        "diagnostics": [],
        "checkpoints": [],
        "training_seconds": 0.0,
        "scheduled_eval_seconds": 0.0,
        "completed_step": 0,
        "optimizer_step_flops_estimate": flops(32, True),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempts": [],
    }
    if resume:
        previous = read(output)
        assert previous["status"] == "running" and not previous.get("error")
        assert len(previous["attempts"]) < 1
        receipt = previous["checkpoint"]
        assert sha(receipt["path"]) == receipt["sha256"]
        saved = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        assert saved["metadata"] == metadata
        assert state_hash(saved["model"]) == receipt["parameter_sha256"]
        assert state_hash(saved["optimizer"]) == receipt["optimizer_sha256"]
        assert digest(saved["rng_cpu"]) == receipt["rng_cpu_sha256"]
        assert digest(saved["rng_gpu"]) == receipt["rng_gpu_sha256"]
        model.load_state_dict(saved["model"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_gpu"])
        start = saved["step"]
        result = previous
        write(out / "INTERRUPTED-ATTEMPT-0.json", previous)
        (out / "TRAINING.jsonl").rename(out / "INTERRUPTED-ATTEMPT-0.jsonl")
        result["attempts"].append(
            {
                "kind": "infrastructure exact resume",
                "checkpoint": receipt,
                "prior_completed_step": previous["completed_step"],
            }
        )
        result["validation"] = [x for x in result["validation"] if x["step"] <= start]
        result["diagnostics"] = [x for x in result["diagnostics"] if x["step"] <= start]
        offset = result.get("end_to_end_seconds", 0)
        del saved
    model.step = start
    fixed, _ = batch(train, order, 0, "cuda")
    began = time.perf_counter()
    write(output, result)
    try:
        if not resume:
            result["diagnostics"].append(
                snapshot(model, fixed, 0, cp / "routing-0.pt", metadata)
            )
        with (out / "TRAINING.jsonl").open("x") as log:
            for step in range(start + 1, 4001):
                torch.cuda.synchronize()
                before = time.perf_counter()
                model.step = step
                x, y = batch(train, order, step - 1, "cuda")
                for group in opt.param_groups:
                    group["lr"] = 0.003 * min(step / 100, 1.0)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    value = loss(logits, y)
                assert (
                    torch.isfinite(value) and torch.isfinite(logits).all()
                ), "nonfinite output/loss"
                value.backward()
                finite(model)
                opt.step()
                finite(model, opt)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                result["training_seconds"] += elapsed
                result["completed_step"] = step
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "tokens": step * 2048,
                            "ce": value.item(),
                            "lr": opt.param_groups[0]["lr"],
                            "step_seconds": elapsed,
                            "invariants": model.invariants.cpu().tolist(),
                        }
                    )
                    + "\n"
                )
                if step in plan["numerical_steps"]:
                    result["diagnostics"].append(
                        snapshot(
                            model, fixed, step, cp / f"routing-{step}.pt", metadata
                        )
                    )
                if step % 100 == 0:
                    before = time.perf_counter()
                    ce = evaluate(model, curve, "cuda")
                    seconds = time.perf_counter() - before
                    result["scheduled_eval_seconds"] += seconds
                    result["validation"].append(
                        {
                            "step": step,
                            "tokens": step * 2048,
                            "ce": ce,
                            "training_seconds": result["training_seconds"],
                            "end_to_end_seconds": offset + time.perf_counter() - began,
                            "eval_seconds": seconds,
                        }
                    )
                    log.flush()
                    os.fsync(log.fileno())
                    assert shutil.disk_usage(cp).free >= 50 * 2**30
                    result["end_to_end_seconds"] = offset + time.perf_counter() - began
                    write(output, result)
                if step in plan["checkpoint_steps"]:
                    receipt = checkpoint(
                        model, opt, step, metadata, cp / f"step{step}.pt"
                    )
                    result["checkpoints"].append(receipt)
                    result["checkpoint"] = receipt
                    write(output, result)
            before = time.perf_counter()
            result["final_gpu_clean_ce"] = evaluate(model, final, "cuda")
            result["final_gpu_eval_seconds"] = time.perf_counter() - before
            result["status"] = "complete"
    except BaseException as exc:
        result["status"], result["error"] = "failed", repr(exc)
        try:
            result["partial_checkpoint"] = checkpoint(
                model, opt, result["completed_step"], metadata, cp / "FAILED-partial.pt"
            )
            result["partial_checkpoint"]["eligible_for_resume"] = False
        except BaseException as save_error:
            result["partial_checkpoint_error"] = repr(save_error)
        raise
    finally:
        result["end_to_end_seconds"] = offset + time.perf_counter() - began
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(output, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    run(a.root, a.job_id, a.resume)
