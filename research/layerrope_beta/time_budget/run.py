# SPDX-License-Identifier: MIT
"""Frozen preparation, bounded hardware qualification and two-device execution.

This module never rents resources. A separate, explicit resource decision is
required for qualification and again for the scientific matrix.
"""

import argparse
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.analysis import analyze
from research.layerrope_beta.time_budget.evaluation import endpoint
from research.layerrope_beta.time_budget.protocol import (
    freeze,
    choose_lrs,
    edge_extensions,
)
from research.layerrope_beta.time_budget.state import (
    json_hash,
    runtime,
    synchronize,
    landmark_valid,
)
from research.layerrope_beta.time_budget.train import train, source_identity
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.train import state_hash


def seeds_in(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if (k == "seed" or k.endswith("_seed")) and isinstance(v, int):
                yield v
            elif k.endswith("seeds") and isinstance(v, list):
                yield from (s for s in v if isinstance(s, int))
            else:
                yield from seeds_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from seeds_in(v)


def prepare(data, audit_inputs, out):
    data = Path(data).resolve()
    p = freeze()
    manifest = read(data / "PREPARED.json")
    audit = read(audit_inputs)
    used = set()
    receipts = []
    if not audit:
        raise ValueError("Prior seed manifests are required")
    for receipt in audit:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Seed provenance mismatch")
        observed = set(seeds_in(read(receipt["path"])))
        if not observed:
            raise ValueError("Seed manifest contains no seeds")
        used.update(observed)
        receipts.append(dict(**receipt, observed_seeds=sorted(observed)))
    proposed = {p["pilot_seed"], p["probe_seed"], *p["final_seeds"]}
    if used & proposed:
        raise ValueError("Proposed seed has prior use")
    for sub, name, digest in [
        ("training", "TRAINING.json", manifest["training_manifest_sha256"]),
        ("sealed", "SEALED.json", manifest["sealed_manifest_sha256"]),
    ]:
        if sha(data / sub / name) != digest:
            raise ValueError("Prepared manifest changed")
    card = dict(
        protocol=p,
        protocol_sha256=json_hash(p),
        source=source_identity(),
        data=str(data),
        prepared_sha256=sha(data / "PREPARED.json"),
        training_sha256=manifest["training_manifest_sha256"],
        sealed_sha256=manifest["sealed_manifest_sha256"],
        seed_audit=dict(
            receipts=receipts,
            proposed=sorted(proposed),
            collisions=[],
            scope="Declared predecessor language/MNIST job and plan manifests",
        ),
        hardware_requirement="Two matching NVIDIA H100 GPUs; one process per visible GPU",
        execution="No automatic allocation, retry, seed replacement or final evaluation before frozen selection and complete final training",
    )
    if Path(out).exists():
        raise FileExistsError(out)
    write(out, card)
    return card


def load_card(path):
    card = read(path)
    if card["protocol"] != freeze() or card["protocol_sha256"] != json_hash(freeze()):
        raise ValueError("Protocol changed after freeze")
    if card["source"] != source_identity():
        raise ValueError("Source changed after freeze")
    data = Path(card["data"])
    for relative, digest in [
        ("PREPARED.json", card["prepared_sha256"]),
        ("training/TRAINING.json", card["training_sha256"]),
        ("sealed/SEALED.json", card["sealed_sha256"]),
    ]:
        if sha(data / relative) != digest:
            raise ValueError("Data capability changed")
    return card


def resource_gate(card_path, resource_path, action):
    card = load_card(card_path)
    decision = read(resource_path)
    if decision.get("authorized") is not True or decision.get("action") != action:
        raise ValueError("An explicit resource decision for this action is required")
    if decision.get("card_sha256") != sha(card_path):
        raise ValueError("Resource decision is for another card")
    if (
        decision.get("gpu_count") != 2
        or not 0 < decision.get("max_seconds", 0) <= 86400
    ):
        raise ValueError("Two GPUs and a finite process limit are required")
    if not 0 < decision.get("max_aggregate_gpu_hours", 0) <= 48:
        raise ValueError("Finite aggregate allocation cap required")
    if not 0 < decision.get("max_output_bytes", 0) <= 10**12:
        raise ValueError("Finite storage cap required")
    if action == "run":
        q = decision.get("qualification", {})
        if sha(q["path"]) != q["sha256"]:
            raise ValueError("Qualification receipt changed")
        evidence = read(q["path"])
        if evidence.get("status") != "qualified" or evidence["card_sha256"] != sha(
            card_path
        ):
            raise ValueError("Hardware qualification is incomplete or for another card")
        measurements = evidence.get("measurements", [])
        operators = evidence.get("operator_checks", [])
        if len(operators) != 2 or not all(
            v.get("status") == "passed" and v.get("runtime") == evidence["runtime"]
            for v in operators
        ):
            raise ValueError("Numerical operators must qualify on both GPUs")
        if (
            len(measurements) != 4
            or {m["row"] for m in measurements} != set(card["protocol"]["rows"])
            or not all(m.get("clock_landmarks_qualified") for m in measurements)
        ):
            raise ValueError("Full training-clock qualification is required")
        if evidence["runtime"] != runtime("cuda"):
            raise ValueError("Qualification runtime/hardware differs")
    return card, decision


def job(row, seed, lr, stage):
    return dict(
        arm=row,
        seed=seed,
        lr=lr,
        stage=stage,
        depth=32,
        vocab=50257,
        window=256,
        batch_size=8,
        max_process_seconds=7200,
    )


def checkpoint_for(result, target):
    match = [v for v in result["history"]["landmarks"] if v["target_seconds"] == target]
    if len(match) != 1:
        raise ValueError("Missing/duplicate checkpoint landmark")
    receipt = match[0]["checkpoint"]
    if sha(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Endpoint checkpoint changed")
    return receipt


def evaluate_checkpoint(receipt, row, seed, tokens, supporting=False, identity=None):
    began = time.monotonic()
    if sha(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Evaluation checkpoint changed")
    model = Model(row, 32, seed, beta=0.0).to("cuda")
    state = torch.load(receipt["path"], map_location="cuda", weights_only=False)
    if identity is not None and state["identity"] != identity:
        raise ValueError("Checkpoint source/data/recipe differs from training receipt")
    if (
        state["step"] != receipt["step"]
        or state["training_seconds"] != receipt["training_seconds"]
    ):
        raise ValueError("Checkpoint step/clock differs from receipt")
    if state_hash(state["model"]) != receipt["model_sha256"]:
        raise ValueError("Checkpoint parameter digest differs from receipt")
    model.load_state_dict(state["model"])
    model.eval()
    synchronize("cuda")
    before = time.monotonic()
    value = endpoint(model, tokens, "cuda", supporting=supporting)
    synchronize("cuda")
    value["evaluation_seconds"] = time.monotonic() - before
    value["complete_process_seconds"] = time.monotonic() - began
    return value


def final_training_barrier(card, selection, receipts):
    """Validate every final recipe and clock before granting holdout access."""
    p = card["protocol"]
    expected = {(r, s) for r in p["rows"] for s in p["final_seeds"]}
    if set(selection["selected_lrs"]) != set(p["rows"]) or len(receipts) != len(
        expected
    ):
        raise ValueError("Final matrix barrier not met")
    results, common = {}, {}
    for receipt in receipts:
        path = str(Path(receipt["path"]).resolve())
        if path in results or sha(path) != receipt["sha256"]:
            raise ValueError("Duplicate or changed final training receipt")
        result = read(path)
        identity = result["identity"]
        j = identity["job"]
        key = j["arm"], j["seed"]
        if key not in expected or j != job(
            j["arm"], j["seed"], selection["selected_lrs"][j["arm"]], "final"
        ):
            raise ValueError("Final training is outside frozen recipe/selection")
        if (
            result["status"] != "complete"
            or identity["source"] != card["source"]
            or identity["protocol_sha256"] != card["protocol_sha256"]
            or identity["data_sha256"] != card["training_sha256"]
        ):
            raise ValueError("Final training source/data/protocol barrier not met")
        frames = result["history"]["landmarks"]
        if len(frames) != len(p["landmarks"]) or {
            v["target_seconds"] for v in frames
        } != set(p["landmarks"]):
            raise ValueError("Final training landmark coverage mismatch")
        for frame in frames:
            if not frame["timing_valid"] or not landmark_valid(
                frame["target_seconds"],
                frame["training_seconds"],
                frame["preceding_training_seconds"],
                frame["last_update_seconds"],
                p["overshoot_fraction"],
            ):
                raise ValueError("Invalid final training clock landmark")
            if not math.isfinite(frame["validation"]["ce"]):
                raise ValueError("Nonfinite final training validation")
        last = next(v for v in frames if v["target_seconds"] == p["primary_seconds"])
        if (
            last["training_seconds"] != result["training_seconds"]
            or last["step"] != result["completed_steps"]
            or result["cursor"]["cursor"]
            != result["completed_steps"] * p["train_batch"]
        ):
            raise ValueError("Final training clock/step/cursor mismatch")
        weights = result["common_initial_weights"]
        if not weights or j["seed"] in common and common[j["seed"]] != weights:
            raise ValueError("Final common initial matrices are not paired")
        common[j["seed"]] = weights
        results[path] = result
    if {
        (r["identity"]["job"]["arm"], r["identity"]["job"]["seed"])
        for r in results.values()
    } != expected:
        raise ValueError("Final matrix barrier not met")
    return results


def worker(card_path, task, out):
    card = load_card(card_path)
    p = card["protocol"]
    data = Path(card["data"])
    out = Path(out)
    if task["kind"] == "operators":
        out.mkdir(parents=True, exist_ok=False)
        with (out / "pytest.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "research/layerrope_beta/tests/test_models.py",
                    "research/tinystories_depth/tests/test_float_quant.py",
                ],
                env={**os.environ, "KNLP_BETA_TEST_DEVICE": "cuda"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        write(
            out / "OPERATORS.json",
            dict(
                status="passed",
                runtime=runtime("cuda"),
                log_sha256=sha(out / "pytest.log"),
            ),
        )
    elif task["kind"] == "train":
        try:
            train(data / "training", task["job"], out, device="cuda")
        except FloatingPointError:
            if task["job"]["stage"] != "pilot":
                raise
            # Preserve invalid numerical pilots as trials; they cannot win.
            if read(out / "RESULT.json")["status"] != "invalid_numerical":
                raise
    elif task["kind"] == "qualify":
        out.mkdir(parents=True, exist_ok=False)
        j = task["job"]
        whole = train(data / "training", j, out / "whole", stop_after_updates=8)
        part = train(data / "training", j, out / "partial", stop_after_updates=4)
        resumed = train(
            data / "training",
            j,
            out / "resumed",
            resume=part["last_checkpoint"]["path"],
            resume_sha=part["last_checkpoint"]["sha256"],
            stop_after_updates=8,
        )
        a = torch.load(
            whole["last_checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        b = torch.load(
            resumed["last_checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        for field in ("model", "optimizer", "rng"):
            if state_hash(a[field]) != state_hash(b[field]):
                raise RuntimeError("Hardware resume mismatch: " + field)
        if a["cursor"] != b["cursor"]:
            raise RuntimeError("Hardware cursor mismatch")
        timed = train(data / "training", j, out / "timed")
        if timed["status"] != "complete" or {
            v["target_seconds"] for v in timed["history"]["landmarks"]
        } != set(p["landmarks"]):
            raise RuntimeError("Hardware training-clock qualification incomplete")
        # Representative full-length timing uses repeated validation tokens,
        # never final text/targets. It is not a new quality measurement.
        validation = np.load(data / "training/validation.npy", mmap_mode="r")
        timing_tokens = np.resize(validation, p["final_predicted_tokens"] + 1)
        primary = evaluate_checkpoint(
            checkpoint_for(timed, 787.0),
            j["arm"],
            j["seed"],
            timing_tokens,
            identity=timed["identity"],
        )
        supporting = evaluate_checkpoint(
            checkpoint_for(timed, 615.0),
            j["arm"],
            j["seed"],
            timing_tokens,
            True,
            identity=timed["identity"],
        )
        write(
            out / "QUALIFICATION.json",
            dict(
                row=j["arm"],
                resume_equal=True,
                runtime=runtime("cuda"),
                training_updates=whole["history"]["updates"],
                timed_training_seconds=timed["training_seconds"],
                timed_steps=timed["completed_steps"],
                clock_landmarks=timed["history"]["landmarks"],
                clock_landmarks_qualified=True,
                checkpoint_bytes=whole["last_checkpoint"]["bytes"],
                process_seconds=whole["attempt_process_seconds"]
                + part["attempt_process_seconds"]
                + resumed["attempt_process_seconds"]
                + timed["attempt_process_seconds"],
                primary=primary,
                supporting=supporting,
                scope="Qualification on repeated validation; no final holdout access",
            ),
        )
    elif task["kind"] == "evaluate":
        selection = read(task["selection"]["path"])
        if sha(task["selection"]["path"]) != task["selection"]["sha256"] or selection[
            "card_sha256"
        ] != sha(card_path):
            raise ValueError("Frozen selection mismatch")
        # This barrier is checked in every evaluation worker, before np.load.
        results = final_training_barrier(
            card, selection, task["completed_final_training"]
        )
        path = str(Path(task["result"]).resolve())
        if path not in results:
            raise ValueError("Evaluation target is outside completed final matrix")
        result = results[path]
        j = result["identity"]["job"]
        seal = read(data / "sealed/SEALED.json")
        if sha(data / "sealed/final.npy") != seal["sha256"]:
            raise ValueError("Final array changed")
        tokens = np.load(data / "sealed/final.npy", mmap_mode="r")
        values = {}
        for target in (615.0, 787.0):
            values[str(target)] = evaluate_checkpoint(
                checkpoint_for(result, target),
                j["arm"],
                j["seed"],
                tokens,
                target == 615.0,
                identity=result["identity"],
            )
        routes = [v for v in result["history"]["routing"]] + [
            v["routing"] for v in result["history"]["landmarks"]
        ]
        routes += [v["parity"]["routing"] for v in values.values()]
        write(
            out / "EVALUATION.json",
            dict(
                row=j["arm"],
                seed=j["seed"],
                endpoints=values,
                routing_constraint_pass=all(v["constraint_pass"] for v in routes),
                training_receipt=task["result"],
                training_seconds=result["training_seconds"],
                completed_steps=result["completed_steps"],
            ),
        )
    else:
        raise ValueError("Unknown worker task")


class Supervisor:
    def __init__(self, card, decision, root):
        self.card = Path(card).resolve()
        self.decision = decision
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.ledger = []
        write(self.root / "RESOURCE.json", decision)
        write(self.root / "CARD.json", read(card))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
        if len(visible) != 2:
            raise ValueError("Expose exactly the reserved two GPUs")
        self.slots = visible

    def check(self):
        elapsed = time.monotonic() - self.started
        if elapsed >= min(
            self.decision["max_seconds"],
            self.decision["max_aggregate_gpu_hours"] * 1800,
        ):
            raise TimeoutError("Aggregate allocation/process cap")
        size = output_bytes(self.root)
        if size > self.decision["max_output_bytes"]:
            raise RuntimeError("Output storage cap")
        return elapsed

    def batch(self, tasks):
        paths = []
        active = []
        for start in range(0, len(tasks), 2):
            self.check()
            try:
                for slot, task in zip(self.slots, tasks[start : start + 2]):
                    n = len(self.ledger)
                    out = self.root / f"task-{n:03}"
                    task_path = self.root / f"task-{n:03}.json"
                    write(task_path, task)
                    log = (self.root / f"task-{n:03}.log").open("w")
                    command = [
                        sys.executable,
                        "-m",
                        "research.layerrope_beta.time_budget.run",
                        "worker",
                        "--card",
                        str(self.card),
                        "--task",
                        str(task_path),
                        "--root",
                        str(out),
                    ]
                    entry = dict(
                        task=task,
                        slot=slot,
                        output=str(out),
                        status="running",
                        started_seconds=time.monotonic() - self.started,
                    )
                    self.ledger.append(entry)
                    try:
                        process = subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": slot},
                            start_new_session=True,
                        )
                    except BaseException as error:
                        log.close()
                        entry.update(
                            status="launch_failed",
                            error=repr(error),
                            ended_seconds=time.monotonic() - self.started,
                        )
                        write(self.root / "OPERATIONS.json", self.ledger)
                        raise
                    active.append((process, log, entry))
                    paths.append(out)
                write(self.root / "OPERATIONS.json", self.ledger)
                while any(p.poll() is None for p, _, _ in active):
                    self.check()
                    if any(p.poll() not in (None, 0) for p, _, _ in active):
                        raise RuntimeError("Worker failed; no automatic retry")
                    time.sleep(0.2)
                if any(p.returncode for p, _, _ in active):
                    raise RuntimeError("Worker failed; no automatic retry")
            finally:
                for process, log, entry in active:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                    entry.update(
                        returncode=process.returncode,
                        status="complete" if process.returncode == 0 else "failed",
                        ended_seconds=time.monotonic() - self.started,
                    )
                    log.close()
                write(self.root / "OPERATIONS.json", self.ledger)
                active = []
        return paths


def output_bytes(root):
    total = 0
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # A worker can atomically rename its temporary checkpoint here.
            continue
    return total


def execute(action, card_path, resource_path, root):
    card, decision = resource_gate(card_path, resource_path, action)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two allocated CUDA GPUs required")
    devices = [torch.cuda.get_device_name(i) for i in range(2)]
    if devices[0] != devices[1] or "H100" not in devices[0]:
        raise RuntimeError("Frozen two-H100 hardware regime required")
    if any(
        torch.cuda.get_device_properties(i).total_memory < 75_000_000_000
        for i in range(2)
    ):
        raise RuntimeError("The frozen hardware tier needs two 80GB GPUs")
    supervisor = Supervisor(card_path, decision, root)
    root = supervisor.root
    p = card["protocol"]
    status = dict(
        status="incomplete",
        action=action,
        card_sha256=sha(card_path),
        runtime=runtime("cuda"),
        devices=devices,
    )
    write(root / "STATUS.json", status)

    def interrupted(*_):
        raise InterruptedError("Supervisor interrupted; retain completed checkpoints")

    previous_handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        if action == "qualify":
            operator_paths = supervisor.batch(
                [dict(kind="operators"), dict(kind="operators")]
            )
            tasks = [
                dict(
                    kind="qualify", job=job(r, p["probe_seed"], 0.001, "qualification")
                )
                for r in p["rows"]
            ]
            paths = supervisor.batch(tasks)
            evidence = [read(path / "QUALIFICATION.json") for path in paths]
            # Measurements include the complete menu, restoration, parity and checkpoints.
            measured_eval = (
                sum(
                    v["primary"]["complete_process_seconds"]
                    + v["supporting"]["complete_process_seconds"]
                    for v in evidence
                )
                / 4
            )
            checkpoint = max(v["checkpoint_bytes"] for v in evidence)
            supervisor.check()
            status.update(
                status="qualified",
                operator_checks=[
                    read(path / "OPERATORS.json") for path in operator_paths
                ],
                measurements=evidence,
                forecast=dict(
                    base_training_gpu_hours=52 * 787 / 3600,
                    max_training_gpu_hours=60 * 787 / 3600,
                    endpoint_gpu_hours=32 * measured_eval / 3600,
                    retained_checkpoint_bytes_upper_bound=60 * 10 * checkpoint,
                    qualification_process_seconds=sum(
                        v["process_seconds"] for v in evidence
                    ),
                    landmark_offline_seconds=sum(
                        sum(
                            f["offline_validation_seconds"]
                            + f["checkpoint"]["complete_checkpoint_seconds"]
                            for f in v["clock_landmarks"]
                        )
                        for v in evidence
                    ),
                    warning="Add measured landmark validation, import/setup, export and idle time; resource decision must include explicit contingency",
                ),
            )
        elif action == "run":
            tasks = [
                dict(kind="train", job=job(r, p["pilot_seed"], lr, "pilot"))
                for r in p["rows"]
                for lr in p["lr_grid"]
            ]
            # A deterministic shuffle interleaves architectures and rates across workers.
            random.Random(p["probe_seed"]).shuffle(tasks)
            paths = supervisor.batch(tasks)
            trials = []

            def collect(paths):
                for path in paths:
                    r = read(path / "RESULT.json")
                    j = r["identity"]["job"]
                    ce = next(
                        (
                            v["validation"]["ce"]
                            for v in r["history"]["landmarks"]
                            if v["target_seconds"] == 787.0
                        ),
                        None,
                    )
                    if r["status"] == "complete" and ce is None:
                        raise ValueError("Pilot missing final validation")
                    trials.append(
                        dict(
                            row=j["arm"],
                            lr=j["lr"],
                            status=r["status"],
                            validation_ce=ce,
                            result=str(path / "RESULT.json"),
                        )
                    )

            collect(paths)
            selected = choose_lrs(trials)
            extensions = edge_extensions(selected)
            extra = [
                dict(kind="train", job=job(r, p["pilot_seed"], lr, "pilot"))
                for lr in extensions
                for r in p["rows"]
            ]
            if extra:
                collect(supervisor.batch(extra))
            selected = choose_lrs(trials, grid=p["lr_grid"] + extensions)
            selection = dict(
                card_sha256=sha(card_path),
                selected_lrs=selected,
                trials=trials,
                extensions=extensions,
                unbracketed={
                    r: lr
                    in (min(p["lr_grid"] + extensions), max(p["lr_grid"] + extensions))
                    for r, lr in selected.items()
                },
            )
            write(root / "SELECTION.json", selection)
            selection_receipt = dict(
                path=str(root / "SELECTION.json"), sha256=sha(root / "SELECTION.json")
            )
            finals = [
                dict(kind="train", job=job(r, s, selected[r], "final"))
                for s in p["final_seeds"]
                for r in p["rows"]
            ]
            random.Random(p["probe_seed"] + 1).shuffle(finals)
            paths = supervisor.batch(finals)
            barrier = [
                dict(path=str(path / "RESULT.json"), sha256=sha(path / "RESULT.json"))
                for path in paths
            ]
            evals = [
                dict(
                    kind="evaluate",
                    result=x["path"],
                    selection=selection_receipt,
                    completed_final_training=barrier,
                )
                for x in barrier
            ]
            evaluated = supervisor.batch(evals)
            result = analyze([read(path / "EVALUATION.json") for path in evaluated])
            write(root / "ANALYSIS.json", result)
            status.update(status="complete", selection=selection_receipt)
        else:
            raise ValueError("Unknown action")
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        status.update(
            process_seconds=time.monotonic() - supervisor.started,
            aggregate_reserved_gpu_hours=2
            * (time.monotonic() - supervisor.started)
            / 3600,
            output_bytes=output_bytes(root),
        )
        write(root / "STATUS.json", status)
    if action == "qualify" and status["status"] == "qualified":
        write(root / "QUALIFIED.json", status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "qualify", "run", "worker"])
    p.add_argument("--card", type=Path)
    p.add_argument("--resource", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--seed-audit", type=Path)
    p.add_argument("--task", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.data, a.seed_audit, a.root)
    elif a.action == "worker":
        worker(a.card, read(a.task), a.root)
    else:
        execute(a.action, a.card, a.resource, a.root)


if __name__ == "__main__":
    main()
