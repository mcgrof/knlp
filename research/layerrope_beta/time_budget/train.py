# SPDX-License-Identifier: MIT
"""Train complete updates under a synchronized budget; never load final holdouts."""

import argparse
import math
import random
from pathlib import Path
import signal
import time
import numpy as np
import torch
from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.routing import RoutingObservation
from research.layerrope_beta.time_budget.state import (
    TrainingClock,
    DataCursor,
    json_hash,
    save,
    restore,
    synchronize,
    runtime,
)
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.kv_cache import evaluate
from research.tinystories_depth.model import loss
from research.tinystories_depth.train import state_hash


def source_identity():
    base = Path(__file__).resolve().parents[3]
    paths = set()
    for directory in ("layerrope_beta", "tinystories_depth"):
        paths.update((base / "research" / directory).rglob("*.py"))
    paths.update(
        base / name
        for name in (
            "research/mo_variants/stage_c_common.py",
            "research/mo_variants/model.py",
            "research/reproduction/run.py",
            "research/reproduction/Kconfig",
            "Makefile.residual_reproduction",
            "defconfigs/layerrope-mhc-time-budget",
            "defconfigs/layerrope-mhc-time-budget-smoke",
        )
    )
    return {str(p.relative_to(base)): sha(p) for p in sorted(paths) if p.is_file()}


def finite(model, optimizer):
    values = [torch.isfinite(p).all() for p in model.parameters()]
    values += [
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    ]
    if not torch.stack(values).all().item():
        raise FloatingPointError("Nonfinite model/gradient")
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v) and not torch.isfinite(v).all().item():
                raise FloatingPointError("Nonfinite optimizer state")


def train(
    data,
    job,
    out,
    *,
    device="cuda",
    resume=None,
    resume_sha=None,
    stop_after_updates=None,
    smoke=False
):
    protocol = freeze()
    data = Path(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # Training capability consists of these two arrays only; final is never
    # loaded, distributed or inspected by this module.
    manifest = read(data / "TRAINING.json")
    if set(manifest["files"]) != {"train.npy", "validation.npy"}:
        raise ValueError("Training capability must contain exactly two arrays")
    for name, digest in manifest["files"].items():
        if name not in ("train.npy", "validation.npy"):
            raise ValueError("Unexpected training capability")
        if sha(data / name) != digest:
            raise ValueError("Changed training data")
    tokens = np.load(data / "train.npy", mmap_mode="r")
    validation = np.load(data / "validation.npy", mmap_mode="r")
    if not smoke and job.get("depth", 32) != 32:
        raise ValueError("Scientific depth is frozen")
    if job["arm"] not in protocol["rows"] or job["stage"] not in (
        "pilot",
        "final",
        "qualification",
    ):
        raise ValueError("Unknown row or trajectory stage")
    if not smoke and (
        job.get("window", 256) != 256
        or job.get("batch_size", 8) != 8
        or job.get("vocab", 50257) != 50257
    ):
        raise ValueError("Scientific model/data geometry is frozen")
    torch.set_num_threads(2 if smoke else 4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    model = Model(
        job["arm"],
        job.get("depth", 32),
        job["seed"],
        beta=0.0,
        vocab=job.get("vocab", 50257),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=job["lr"], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    stream = DataCursor(
        tokens, job["seed"], job.get("window", 256), job.get("batch_size", 8)
    )
    identity = dict(
        job=job,
        protocol_sha256=json_hash(protocol),
        data_sha256=sha(data / "TRAINING.json"),
        source=source_identity(),
    )
    clock = TrainingClock(device)
    step = 0
    history = dict(updates=[], landmarks=[], routing=[], attempts=[])
    if resume:
        state = restore(resume, model, optimizer, stream, identity, device, resume_sha)
        step = state["step"]
        clock.seconds = state["training_seconds"]
        history = state["history"]
        # A landmark snapshot cannot contain its own file hash. Rebuild its
        # receipt from the verified resume input rather than losing that point.
        for record in history["landmarks"]:
            if "checkpoint" not in record:
                if (
                    record["step"] != step
                    or record["training_seconds"] != clock.seconds
                ):
                    raise ValueError("Missing earlier landmark checkpoint receipt")
                record["checkpoint"] = dict(
                    path=str(resume),
                    sha256=resume_sha,
                    step=step,
                    training_seconds=clock.seconds,
                    model_sha256=state_hash(state["model"]),
                    optimizer_sha256=state_hash(state["optimizer"]),
                    cursor=state["cursor"],
                    bytes=Path(resume).stat().st_size,
                )
    started = time.monotonic()
    history["attempts"].append(
        dict(
            resumed_from=str(resume) if resume else None,
            restored_training_seconds=clock.seconds,
            runtime=runtime(device),
        )
    )
    result = dict(
        status="running",
        identity=identity,
        common_initial_weights=model.common_hashes,
        history=history,
        scientific_results=False if smoke or job["stage"] == "qualification" else None,
    )
    stop = [False]

    def request_stop(*_):
        stop[0] = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    targets = [0.0, 1e12] if smoke else protocol["landmarks"]
    done = {v["target_seconds"] for v in history["landmarks"]}
    clean = protocol["cache_cases"][0]

    def checkpoint(label):
        history["attempts"][-1]["process_seconds"] = time.monotonic() - started
        before = time.monotonic()
        receipt = save(
            out / (label + ".pt"),
            model,
            optimizer,
            clock,
            stream,
            step,
            identity,
            device,
            history,
        )
        receipt["complete_checkpoint_seconds"] = time.monotonic() - before
        write(out / (label + "-receipt.json"), receipt)
        return receipt

    def landmark(target, preceding):
        from research.layerrope_beta.time_budget.state import landmark_valid

        valid = landmark_valid(
            target, clock.seconds, preceding, clock.last_update_seconds
        )
        if not valid:
            raise RuntimeError("Training-clock landmark overshoot violation")
        synchronize(device)
        model.eval()
        offline = time.monotonic()
        finite(model, optimizer)
        value = evaluate(
            model, validation, clean, device, window=stream.window, batch_size=32
        )
        if not math.isfinite(value["ce"]):
            raise FloatingPointError("Nonfinite validation")
        observer = RoutingObservation()
        model.routing_observation = observer
        ids = torch.as_tensor(
            np.array(validation[: 32 * stream.window], dtype=np.int64).reshape(
                32, stream.window
            ),
            device=device,
        )
        try:
            with torch.no_grad(), torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ):
                model(ids)
        finally:
            model.routing_observation = None
        route = observer.report(2 * model.depth if model.routes is not None else 0)
        synchronize(device)
        record = dict(
            target_seconds=target,
            training_seconds=clock.seconds,
            preceding_training_seconds=preceding,
            last_update_seconds=clock.last_update_seconds,
            timing_valid=valid,
            step=step,
            predicted_tokens=step * stream.batch_size * stream.window,
            validation=value,
            offline_validation_seconds=time.monotonic() - offline,
            routing=route,
        )
        history["landmarks"].append(record)
        done.add(target)
        model.train()
        record["checkpoint"] = checkpoint("time-" + str(target))
        write(out / "RESULT.json", result)

    try:
        if 0.0 not in done:
            landmark(0.0, 0.0)
        while clock.seconds < targets[-1]:
            if time.monotonic() - started > job.get("max_process_seconds", 7200):
                raise TimeoutError("Attempt elapsed-time cap")
            preceding = clock.seconds
            next_step = step + 1
            with clock.update():
                x, y = stream.batch(device)
                for group in optimizer.param_groups:
                    group["lr"] = job["lr"] * min(next_step / 100, 1.0)
                optimizer.zero_grad(set_to_none=True)
                observed = (
                    next_step == 1 or next_step % protocol["routing_cadence"] == 0
                )
                route = RoutingObservation() if observed else None
                model.routing_observation = route
                try:
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        objective = loss(model(x), y)
                finally:
                    model.routing_observation = None
                if not torch.isfinite(objective).item():
                    raise FloatingPointError("Nonfinite objective")
                objective.backward()
                optimizer.step()
                if next_step == 1 or next_step % protocol["finite_cadence"] == 0:
                    finite(model, optimizer)
                if route is not None:
                    observed_route = route.report(
                        2 * model.depth if model.routes is not None else 0
                    )
                    history["routing"].append(dict(step=next_step, **observed_route))
                    if not observed_route["finite"]:
                        raise FloatingPointError("Nonfinite trained routing")
                ce = objective.item()
                step = next_step
                stream.advance()
            history["updates"].append(
                dict(
                    step=step,
                    training_seconds=clock.seconds,
                    update_seconds=clock.last_update_seconds,
                    ce=ce,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
            for target in targets:
                if target not in done and clock.seconds >= target:
                    landmark(target, preceding)
            if stop[0] or stop_after_updates is not None and step >= stop_after_updates:
                result["status"] = "stopped"
                break
        else:
            result["status"] = "complete"
        result["last_checkpoint"] = checkpoint("resume")
    except BaseException as error:
        # No checkpoint of a partially completed update. Previously fsynced
        # states remain valid and failed attempt process time remains visible.
        result.update(
            status=(
                "invalid_numerical"
                if isinstance(error, FloatingPointError)
                else "incomplete"
            ),
            error=repr(error),
        )
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        result.update(
            completed_steps=step,
            training_seconds=clock.seconds,
            cursor=stream.identity(),
            attempt_process_seconds=time.monotonic() - started,
        )
        history["attempts"][-1]["process_seconds"] = result["attempt_process_seconds"]
        write(out / "RESULT.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha")
    a = p.parse_args()
    train(
        a.data,
        read(a.job),
        a.out,
        device=a.device,
        resume=a.resume,
        resume_sha=a.resume_sha,
    )
