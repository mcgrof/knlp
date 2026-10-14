"""One manifest-listed trajectory under gpuq, with exact continuation states."""

import argparse
import datetime as dt
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import digest, loss, mhc
from research.tinystories_depth.probe import batch, evaluate, finite
from research.tinystories_depth.train import flops, state_hash
from .preflight import clients, write
from .model import Model


from .job import checkpoint, snapshot


def run(root, jobid):
    root = Path(root)
    from .stage_b_prepare import verify_source, verify_checkpoint
    from .coordinator import read

    assert os.environ.get("GPUQ_RUN_ID"), "production gpuq required"
    verify_source(root)
    authorization = read(root / "records/STAGE-B-RESOURCE-DECISION.json")
    assert authorization["status"] == "approved" and authorization["scope"] == "B"
    selection = read(root / "records/STAGE-B-SELECTION.json")
    job = next(j for j in selection["jobs"] if j["id"] == jobid)
    assert job["stage"] == "B" and job["resume_step"] == 1000 and job["steps"] == 2000

    assert shutil.disk_usage(Path(os.environ["GPUQ_RUN_DIR"])).free >= 50 * 2**30
    clients()
    metadata = {
        "job": job,
        "source": read(root / "records/STAGE-B-SOURCE.json"),
        "resource_decision_sha256": sha(
            root / "records/STAGE-B-RESOURCE-DECISION.json"
        ),
        "selection_sha256": sha(root / "records/STAGE-B-SELECTION.json"),
        "gpuq_run_id": os.environ["GPUQ_RUN_ID"],
        "gpuq_run_dir": os.environ["GPUQ_RUN_DIR"],
        "device": torch.cuda.get_device_name(0),
    }
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = root / "records/runs" / jobid
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = Path(os.environ["GPUQ_RUN_DIR"]) / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    data = root / "data"
    expected = json.loads((root / "records/DATA.json").read_text())
    for name in ("train.npy", f"order-{job['seed']}.npy"):
        assert sha(data / name) == expected["training_and_order_files"][name]
    confirmation = job["stage"] in ("C",)
    curve_path = data / "confirmation/curve.npy" if confirmation else data / "curve.npy"
    assert (
        sha(curve_path) == expected["confirmation_files"]["curve.npy"]
        if confirmation
        else sha(curve_path) == expected["verified_inherited_files"]["curve.npy"]
    )
    train = np.load(data / "train.npy", mmap_mode="r")
    curve = np.load(curve_path, mmap_mode="r")
    order = np.load(data / f"order-{job['seed']}.npy")
    metadata["data_order_sha256"] = sha(data / f"order-{job['seed']}.npy")
    metadata["curve_sha256"] = sha(curve_path)
    torch.manual_seed(job["seed"])
    model = Model(
        job["variant"],
        job["depth"],
        job["seed"],
        vocab=job.get("vocab", 50257),
        mapping=job.get("mapping", "ordered"),
    ).cuda()
    lr = job["lr"]
    if not isinstance(lr, (int, float)):
        lr = json.loads((root / "records/SELECTED-LRS.json").read_text())[
            job["variant"]
        ]["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0
    )
    start = 0
    if job.get("resume_from"):
        previous = json.loads(
            (root / "records/runs" / job["resume_from"] / "RESULT.json").read_text()
        )
        receipt = previous["checkpoint"]
        assert (
            previous["status"] == "complete"
            and sha(receipt["path"]) == receipt["sha256"]
        )
        saved = torch.load(receipt["path"], map_location="cpu", weights_only=False)
        prior = saved["metadata"]["job"]
        assert all(prior[k] == job[k] for k in ("variant", "depth", "seed"))
        assert saved["metadata"]["selected_lr"] == lr
        assert saved["metadata"]["data_order_sha256"] == metadata["data_order_sha256"]
        assert saved["metadata"]["curve_sha256"] == metadata["curve_sha256"]
        verify_checkpoint(root, job, previous, saved)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_gpu"])
        assert state_hash(model.state_dict()) == receipt["parameter_sha256"]
        assert state_hash(optimizer.state_dict()) == receipt["optimizer_sha256"]
        assert digest(torch.get_rng_state()) == receipt["rng_cpu_sha256"]
        assert digest(torch.cuda.get_rng_state()) == receipt["rng_gpu_sha256"]
        start = saved["step"]
        assert start == job["resume_step"]
    assert start == 1000 and len(order) >= 2000 * 8
    metadata["selected_lr"] = lr
    if not isinstance(job["lr"], (int, float)):
        metadata["selected_lrs_sha256"] = sha(root / "records/SELECTED-LRS.json")
    result = {
        "metadata": metadata,
        "status": "running",
        "identities": model.identities(),
        "validation": [],
        "diagnostics": [],
        "checkpoints": [],
        "completed_step": start,
        "training_seconds": 0.0,
        "scheduled_eval_seconds": 0.0,
        "optimizer_step_flops_estimate": flops(job["depth"], True),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if job.get("resume_from"):
        result["prior_validation"] = previous["validation"]
        result["resume_receipt"] = {
            "checkpoint": receipt,
            "verified_loaded_hashes": True,
            "global_data_cursor_windows": start * 8,
            "scheduler": "global step; warmup not repeated",
        }
        result["prior_diagnostics_reused"] = previous["diagnostics"]
        result["continuation_of"] = job["resume_from"]
    model.step = start
    fixed, _ = batch(train, order, 0, "cuda")
    began = time.perf_counter()
    output = out / "RESULT.json"
    write(output, result)
    try:
        with (out / "TRAINING.jsonl").open("x") as log:
            for step in range(start + 1, job["steps"] + 1):
                torch.cuda.synchronize()
                before = time.perf_counter()
                model.step = step
                x, y = batch(train, order, step - 1, "cuda")
                for group in optimizer.param_groups:
                    group["lr"] = lr * min(step / 100, 1.0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    value = loss(logits, y)
                assert (
                    torch.isfinite(value) and torch.isfinite(logits).all()
                ), "nonfinite loss/output"
                value.backward()
                finite(model)
                optimizer.step()
                finite(model, optimizer)
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
                            "lr": optimizer.param_groups[0]["lr"],
                            "step_seconds": elapsed,
                            "end_to_end_seconds": time.perf_counter() - began,
                            "invariants": model.invariants.cpu().tolist(),
                        }
                    )
                    + "\n"
                )
                if step % 100 == 0 or step == job["steps"]:
                    log.flush()
                    os.fsync(log.fileno())
                if step == job["steps"]:
                    result["diagnostics"].append(
                        snapshot(
                            model,
                            fixed,
                            step,
                            checkpoints / f"routing-{jobid}-{step}.pt",
                            metadata,
                        )
                    )
                if step % 100 == 0 and job["stage"] != "A-qualification":
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
                            "end_to_end_seconds": time.perf_counter() - began,
                            "eval_seconds": seconds,
                        }
                    )
                    clients()
                    assert (
                        shutil.disk_usage(checkpoints).free >= 50 * 2**30
                    ), "disk watermark"
                    write(output, result)
                if step in (1000, 2000, 4000) or step == job["steps"]:
                    receipt = checkpoint(
                        model,
                        optimizer,
                        step,
                        metadata,
                        checkpoints / f"{jobid}-step{step}.pt",
                    )
                    result["checkpoints"].append(receipt)
                    result["checkpoint"] = receipt
                    write(output, result)
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        # Retain the failed state explicitly. It is never an endpoint or an
        # eligible continuation, including when its tensors are nonfinite.
        try:
            result["partial_checkpoint"] = checkpoint(
                model,
                optimizer,
                result["completed_step"],
                metadata,
                checkpoints / f"{jobid}-FAILED-partial.pt",
            )
            result["partial_checkpoint"]["eligible_for_resume"] = False
        except BaseException as save_error:
            result["partial_checkpoint_error"] = repr(save_error)
        raise
    finally:
        result["end_to_end_seconds"] = time.perf_counter() - began
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(output, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    args = p.parse_args()
    run(args.root, args.job_id)
