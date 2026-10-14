"""Run one manifest-listed trajectory with exact continuation states."""

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


def checkpoint(model, optimizer, step, metadata, path):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metadata": metadata,
        "rng_cpu": torch.get_rng_state(),
        "rng_gpu": torch.cuda.get_rng_state(),
    }
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert state_hash(loaded["model"]) == state_hash(model.state_dict())
    assert state_hash(loaded["optimizer"]) == state_hash(optimizer.state_dict())
    return {
        "path": str(path),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "step": step,
        "parameter_sha256": state_hash(loaded["model"]),
        "optimizer_sha256": state_hash(loaded["optimizer"]),
        "rng_cpu_sha256": digest(loaded["rng_cpu"]),
        "rng_gpu_sha256": digest(loaded["rng_gpu"]),
        "reload_hashes_verified": True,
    }


@torch.no_grad()
def snapshot(model, x, step, path, metadata):
    model.capture = True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(x)
    model.capture = False
    # Every sublayer retains 16 fixed positions and the worst row/column
    # residual, following the inherited bounded sample rule.
    for sample in model.samples:
        z = sample["logits"].cuda().detach().requires_grad_()
        g = torch.Generator().manual_seed(76390 + step * 1000 + sample["sublayer"])
        cot = torch.randn(z.shape, generator=g).cuda()
        with torch.enable_grad():
            native = mhc.sinkhorn_normalize_ref(z, repeat=20, eps=1e-6)
            vjp = torch.autograd.grad((native * cot).sum(), z)[0]
        assert torch.equal(native.cpu(), sample["carry"])
        sample["finite_cotangent"] = cot.cpu()
        sample["execution_device_finite_vjp"] = vjp.cpu()
    schedule = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if n.startswith("angular.")
        or n.startswith("lm.model.layerrope.")
        or n
        in (
            "free_angles",
            "strength_logits",
            "normalized_frequency_logits",
            "normalized_offsets",
        )
    }
    torch.save(
        {
            "samples": model.samples,
            "step": step,
            "schedule": schedule,
            "mapping": model.code_map,
            "metadata": metadata,
            "scope": "actual execution-device finite output/VJP; independent CPU reference is a later check",
        },
        path,
    )
    return {
        "path": str(path),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "step": step,
    }


def guard(root, job):
    plan = json.loads((root / "records/PLAN.json").read_text())
    authorization = json.loads((root / "records/RESOURCE-DECISION.json").read_text())
    assert authorization["status"] == "approved" and authorization["user_message"]
    allowed = {
        "A": {"A-qualification", "A-pilot", "A-discovery"},
        "AB": {"A-qualification", "A-pilot", "A-discovery", "B"},
        "full": {
            "A-qualification",
            "A-pilot",
            "A-discovery",
            "B",
            "C",
            "D-pilot",
            "D",
        },
    }
    assert job["stage"] in allowed[authorization["scope"]]
    declared = next(j for j in plan["potential_jobs_A_B"] if j["id"] == job["id"])
    assert all(job[k] == v for k, v in declared.items()), "job differs from frozen plan"
    if job["stage"] == "B":
        shortlist = json.loads((root / "records/SHORTLIST.json").read_text())
        assert (
            len(shortlist["variants"]) <= 12 and job["variant"] in shortlist["variants"]
        )
    assert plan["frozen_before_outcomes"]
    source = json.loads((root / "records/SOURCE.json").read_text())
    for relative, expected in source["implemented_files"].items():
        assert sha(root / "code" / relative) == expected, relative
    for device in ("cpu", "cuda"):
        fixture = json.loads(
            (
                root / f"records/variant-fixtures-final/VARIANT-FIDELITY-{device}.json"
            ).read_text()
        )
        assert fixture["status"] == "complete" and len(fixture["full_network"]) == 32

    assert shutil.disk_usage(root).free >= 50 * 2**30
    clients()
    return {
        "job": job,
        "source": source,
        "plan_sha256": sha(root / "records/PLAN.json"),
        "resource_decision_sha256": sha(root / "records/RESOURCE-DECISION.json"),
        "device": torch.cuda.get_device_name(0),
    }


def run(root, jobid, disposable_fixture=False):
    root = Path(root)
    registry = json.loads((root / "records/JOBS.json").read_text())
    assert registry["frozen"]
    jobs = {j["id"]: j for j in registry["jobs"]}
    assert len(jobs) == len(registry["jobs"]), "duplicate manifest IDs"
    job = jobs[jobid]
    if disposable_fixture:
        assert job["id"].startswith("runner-fixture-") and job["steps"] == 3
        assert (
            job["depth"] == 4
            and job["vocab"] == 128
            and job["stage"] == "A-qualification"
        )
        metadata = {
            "job": job,
            "scope": "bounded disposable run-loop fixture",
        }
    else:
        metadata = guard(root, job)
        assert job["stage"] in (
            "A-qualification",
            "A-pilot",
            "A-discovery",
            "B",
        ), "C/D endpoints need their frozen selection-specific launcher"
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = root / "records/runs" / jobid
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = root / "checkpoints"
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
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_gpu"])
        start = saved["step"]
        assert start == job["resume_step"]
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
        result["continuation_of"] = job["resume_from"]
    model.step = start
    fixed, _ = batch(train, order, 0, "cuda")
    began = time.perf_counter()
    output = out / "RESULT.json"
    write(output, result)
    try:
        result["diagnostics"].append(
            snapshot(
                model,
                fixed,
                start,
                checkpoints / f"routing-{jobid}-{start}.pt",
                metadata,
            )
        )
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
                if step in (50, 500, job["steps"]):
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


def resume_fixture(root, device):
    """Disposable exact-resume proof: three steps plus reload versus five."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    record = {
        "device": device,
        "scope": "tiny-vocabulary disposable runner fixture; no study outcomes",
        "rows": [],
    }
    for v in ("V01", "V10", "V24"):
        gen = torch.Generator().manual_seed(76390)
        x = torch.randint(0, 128, (2, 8), generator=gen).to(device)
        y = torch.randint(0, 128, (2, 8), generator=gen).to(device)
        reference = Model(v, 4, 76390, vocab=128).to(device)
        partial = Model(v, 4, 76390, vocab=128).to(device)

        def opt(m):
            return torch.optim.AdamW(
                m.parameters(),
                lr=0.001,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0,
            )

        a, b = opt(reference), opt(partial)

        def steps(m, o, first, last):
            for step in range(first, last + 1):
                m.step = step
                o.zero_grad(set_to_none=True)
                for group in o.param_groups:
                    group["lr"] = 0.001 * min(step / 100, 1.0)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    value = loss(m(x), y)
                value.backward()
                o.step()

        steps(reference, a, 1, 5)
        steps(partial, b, 1, 3)
        path = root / f"resume-fixture-{v}-{device}.pt"
        torch.save(
            {
                "model": partial.state_dict(),
                "optimizer": b.state_dict(),
                "step": 3,
            },
            path,
        )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        resumed = Model(v, 4, 76390, vocab=128).to(device)
        c = opt(resumed)
        resumed.load_state_dict(saved["model"], strict=True)
        c.load_state_dict(saved["optimizer"])
        steps(resumed, c, 4, 5)
        assert state_hash(reference.state_dict()) == state_hash(resumed.state_dict())
        assert state_hash(a.state_dict()) == state_hash(c.state_dict())
        record["rows"].append(
            {
                "id": v,
                "parameter_state_bit_identical": True,
                "optimizer_state_bit_identical": True,
                "checkpoint_sha256": sha(path),
            }
        )
    record["status"] = "complete"
    write(root / f"RESUME-FIXTURE-{device}.json", record)


def run_loop_fixture(root):
    root = Path(root) / "runner-preflight"
    (root / "records").mkdir(parents=True, exist_ok=False)
    (root / "data").mkdir()
    gen = np.random.default_rng(76390)
    np.save(root / "data/train.npy", gen.integers(0, 128, 10241, dtype=np.uint16))
    np.save(root / "data/curve.npy", gen.integers(0, 128, 32769, dtype=np.uint16))
    np.save(root / "data/order-76390.npy", np.arange(40, dtype=np.int32))
    write(
        root / "records/DATA.json",
        {
            "training_and_order_files": {
                n: sha(root / "data" / n) for n in ("train.npy", "order-76390.npy")
            },
            "verified_inherited_files": {"curve.npy": sha(root / "data/curve.npy")},
        },
    )
    jobs = [
        {
            "id": f"runner-fixture-{v}",
            "variant": v,
            "seed": 76390,
            "depth": 4,
            "vocab": 128,
            "steps": 3,
            "lr": 0.001,
            "stage": "A-qualification",
        }
        for v in ("V01", "V10", "V27")
    ]
    write(root / "records/JOBS.json", {"frozen": True, "jobs": jobs})
    for job in jobs:
        run(root, job["id"], disposable_fixture=True)
    summary = {
        "scope": "three-step toy-vocabulary run-loop fixtures; no study outcomes",
        "status": "complete",
        "results": [
            {
                "job": j["id"],
                "path": str(root / "records/runs" / j["id"] / "RESULT.json"),
                "sha256": sha(root / "records/runs" / j["id"] / "RESULT.json"),
            }
            for j in jobs
        ],
    }
    write(root / "RUN-LOOP-FIXTURE.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id")
    p.add_argument("--resume-fixture", choices=("cpu", "cuda"))
    p.add_argument("--run-loop-fixture", action="store_true")
    args = p.parse_args()
    if args.run_loop_fixture:
        run_loop_fixture(args.root)
    elif args.resume_fixture:
        resume_fixture(args.root, args.resume_fixture)
    else:
        run(args.root, args.job_id)
