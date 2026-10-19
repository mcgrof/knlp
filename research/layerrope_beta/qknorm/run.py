# SPDX-License-Identifier: MIT
"""Prepare, qualify, train and independently resume a bounded QKNorm study."""

import argparse
import contextlib
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.run import seeds_in
from research.layerrope_beta.time_budget.state import json_hash, runtime, synchronize
from research.layerrope_beta.time_budget.train import source_identity, train
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.kv_cache import evaluate, parity
from research.tinystories_depth.train import state_hash
from .diagnostics import measure
from .protocol import freeze


def prepare(data, audit, out):
    p = freeze()
    used, receipts = set(), []
    for receipt in read(audit):
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Seed provenance changed")
        values = set(seeds_in(read(receipt["path"])))
        if not values:
            raise ValueError("Empty predecessor seed receipt")
        used.update(values)
        receipts.append(dict(**receipt, observed_seeds=sorted(values)))
    proposed = {*p["final_seeds"], p["probe_seed"]}
    if not receipts or proposed & used:
        raise ValueError("Fresh seed audit failed")
    data = Path(data).resolve()
    prepared = read(data / "PREPARED.json")
    card = dict(
        protocol=p,
        protocol_sha256=json_hash(p),
        source=source_identity(),
        data=str(data),
        prepared_sha256=sha(data / "PREPARED.json"),
        training_sha256=prepared["training_manifest_sha256"],
        sealed_sha256=prepared["sealed_manifest_sha256"],
        seed_audit=dict(receipts=receipts, proposed=sorted(proposed), collisions=[]),
        execution="One GPU, one worker; no cloud allocation by this program",
    )
    if Path(out).exists():
        raise FileExistsError(out)
    write(out, card)
    return card


def load_card(path):
    c = read(path)
    if c["protocol"] != freeze() or c["protocol_sha256"] != json_hash(freeze()):
        raise ValueError("Frozen QKNorm protocol changed")
    if c["source"] != source_identity():
        raise ValueError("Frozen source changed")
    for name, digest in (
        ("PREPARED.json", c["prepared_sha256"]),
        ("training/TRAINING.json", c["training_sha256"]),
        ("sealed/SEALED.json", c["sealed_sha256"]),
    ):
        if sha(Path(c["data"]) / name) != digest:
            raise ValueError("Frozen data capability changed")
    return c


def resource_gate(card, resource, action):
    c, r = load_card(card), read(resource)
    if (
        r.get("authorized") is not True
        or r.get("action") != action
        or r.get("card_sha256") != sha(card)
    ):
        raise ValueError("Matching explicit resource decision is required")
    if r.get("gpu_count") != 1 or not 0 < r.get("max_aggregate_gpu_hours", 0) <= 12:
        raise ValueError("One GPU and at most 12 allocated GPU-hours")
    if (
        not 0 < r.get("max_seconds", 0) <= 43200
        or not 0 < r.get("max_total_usd", 0) <= 60
    ):
        raise ValueError("Finite process and dollar caps are required")
    if not 0 < r.get("max_output_bytes", 0) <= 100_000_000_000:
        raise ValueError("Finite output cap is required")
    if action in ("run", "cache"):
        q = r.get("qualification", {})
        if sha(q["path"]) != q["sha256"]:
            raise ValueError("Qualification changed")
        evidence = read(q["path"])
        if evidence.get("status") != "qualified" or evidence["card_sha256"] != sha(
            card
        ):
            raise ValueError("Matching qualification is required")
        if evidence["runtime"] != runtime("cuda"):
            raise ValueError("Qualification runtime differs")
    return c, r


def jobs(card):
    p = card["protocol"]
    result = []
    for i, seed in enumerate(p["final_seeds"]):
        arms = p["arms"][i * 2 :] + p["arms"][: i * 2]
        if i % 2:
            arms = list(reversed(arms))
        for arm in arms:
            result.append(
                dict(
                    **arm,
                    seed=seed,
                    stage="final",
                    depth=32,
                    vocab=50257,
                    window=256,
                    batch_size=8,
                    max_process_seconds=3600,
                )
            )
    return result


def name(job):
    return f"{job['name']}-seed{job['seed']}"


def training_barrier(card, root):
    results = {}
    for job in jobs(card):
        path = Path(root) / "training" / name(job) / "RESULT.json"
        r = read(path)
        identity = r["identity"]
        if identity != dict(
            job=job,
            protocol_sha256=card["protocol_sha256"],
            data_sha256=card["training_sha256"],
            source=card["source"],
        ):
            raise ValueError("Training identity differs from frozen card")
        if r["status"] not in ("complete", "invalid_numerical"):
            raise ValueError("All run statuses must close before final holdout access")
        if r["status"] == "complete":
            endpoints = [
                x for x in r["history"]["landmarks"] if x["target_seconds"] == 787.0
            ]
            if len(endpoints) != 1 or not endpoints[0]["timing_valid"]:
                raise ValueError("Missing valid time endpoint")
            if r["cursor"]["cursor"] != r["completed_steps"] * 8:
                raise ValueError("Training cursor mismatch")
            for x in endpoints + r["history"]["update_landmarks"]:
                if sha(x["checkpoint"]["path"]) != x["checkpoint"]["sha256"]:
                    raise ValueError("Endpoint checkpoint changed")
            updates = r["history"]["update_landmarks"]
            if len(updates) != int(r["completed_steps"] >= 1000) or any(
                x["step"] != 1000 for x in updates
            ):
                raise ValueError("Equal-token endpoint coverage differs")
        results[name(job)] = dict(path=str(path), sha256=sha(path), result=r)
    for seed in card["protocol"]["final_seeds"]:
        common = [
            results[f"{r}-off-seed{seed}"]["result"]["common_initial_weights"]
            for r in ("R", "L", "M", "ML")
        ]
        if not all(x == common[0] for x in common):
            raise ValueError("Shared architecture weights are not paired")
        for arm in ("R", "L", "M", "ML"):
            off, on = [
                results[f"{arm}-{q}-seed{seed}"]["result"]["initial_parameter_hashes"]
                for q in ("off", "on")
            ]
            if off != {n: v for n, v in on.items() if not n.endswith("qknorm_gain")}:
                raise ValueError("QKNorm pair has different initial weights")
    return results


class Supervisor:
    def __init__(self, card, resource, root):
        self.card, self.resource, self.root = Path(card), resource, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lease = (self.root / ".supervisor.lock").open("a")
        fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.started = time.monotonic()
        self.deadline = min(
            time.time() + resource["max_seconds"],
            resource.get("deadline_epoch", float("inf")),
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.slots = (
            visible.split(",")
            if visible is not None
            else [str(i) for i in range(torch.cuda.device_count())]
        )
        if len(self.slots) < resource["gpu_count"]:
            raise ValueError("Insufficient allocated GPUs")
        self.slots = self.slots[: resource["gpu_count"]]
        operations_path = self.root / "OPERATIONS.json"
        self.operations = read(operations_path) if operations_path.exists() else []

    def batch(self, tasks):
        pending, active = list(tasks), {}
        try:
            while pending or active:
                if (
                    time.time() >= self.deadline
                    or (time.monotonic() - self.started) * len(self.slots) / 3600
                    >= self.resource["max_aggregate_gpu_hours"]
                ):
                    raise TimeoutError("Bounded resource decision expired")
                size = sum(
                    p.stat().st_size for p in self.root.rglob("*") if p.is_file()
                )
                if size > self.resource["max_output_bytes"]:
                    raise RuntimeError("Output byte cap reached")
                for slot in self.slots:
                    if slot in active or not pending:
                        continue
                    task = pending.pop(0)
                    folder = Path(task["out"])
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    spec = folder.parent / (folder.name + "-TASK.json")
                    write(spec, task)
                    log = (folder.parent / (folder.name + ".log")).open("a")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "research.layerrope_beta.qknorm.run",
                            "worker",
                            "--card",
                            str(self.card),
                            "--task",
                            str(spec),
                        ],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": slot},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    op = dict(
                        task=task,
                        slot=slot,
                        pid=process.pid,
                        status="running",
                        started_epoch=time.time(),
                    )
                    self.operations.append(op)
                    active[slot] = (process, log, op)
                for slot, (process, log, op) in list(active.items()):
                    rc = process.poll()
                    if rc is not None:
                        log.close()
                        del active[slot]
                        op.update(
                            returncode=rc,
                            status="complete" if rc == 0 else "failed",
                            ended_epoch=time.time(),
                        )
                        if op["task"]["kind"] == "train":
                            result_path = Path(op["task"]["out"]) / "RESULT.json"
                            if result_path.exists():
                                value = read(result_path)
                                value["full_process_seconds_upper_bound"] = (
                                    op["ended_epoch"] - op["started_epoch"]
                                )
                                value["exit_poll_interval_seconds"] = 2.0
                                write(result_path, value)
                        if rc:
                            raise RuntimeError("Worker failed: " + str(op["task"]))
                write(self.root / "OPERATIONS.json", self.operations)
                if active:
                    time.sleep(2)
        finally:
            for process, log, op in active.values():
                process.terminate()
            for process, log, op in active.values():
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            write(self.root / "OPERATIONS.json", self.operations)


def worker(card_path, task):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    card = load_card(card_path)
    device = "cuda"
    if task["kind"] == "train":
        try:
            train(
                Path(card["data"]) / "training",
                task["job"],
                task["out"],
                contract=card["protocol"],
                diagnostic=measure,
            )
        except FloatingPointError:
            if read(Path(task["out"]) / "RESULT.json")["status"] != "invalid_numerical":
                raise
        return
    results = training_barrier(card, task["run_root"])
    row = results[task["run_name"]]
    r = row["result"]
    p = card["protocol"]
    sealed = read(Path(card["data"]) / "sealed/SEALED.json")
    final = Path(card["data"]) / "sealed/final.npy"
    if sha(final) != sealed["sha256"]:
        raise ValueError("Sealed final tokens changed")
    tokens = np.load(final, mmap_mode="r")
    if len(tokens) != p["final_predicted_tokens"] + 1:
        raise ValueError("Final token coverage differs")
    j = r["identity"]["job"]
    model = (
        Model(j["arm"], 32, j["seed"], beta=0.0, qknorm=j["qknorm"]).to(device).eval()
    )
    out = Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    for endpoint, frames in (
        (
            "time",
            [f for f in r["history"]["landmarks"] if f["target_seconds"] == 787.0],
        ),
        ("tokens", r["history"]["update_landmarks"]),
    ):
        if not frames:
            write(
                out / (endpoint + "-MISSING.json"),
                dict(reason="Endpoint missing/censored", training_status=r["status"]),
            )
            continue
        frame = frames[0]
        checkpoint = frame["checkpoint"]
        if sha(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        state = torch.load(checkpoint["path"], map_location=device, weights_only=False)
        if (
            state["identity"] != r["identity"]
            or state_hash(state["model"]) != checkpoint["model_sha256"]
        ):
            raise ValueError("Checkpoint identity/parameters changed")
        model.load_state_dict(state["model"])
        for policy in p["cache_cases"]:
            path = out / (endpoint + "-" + policy["name"] + ".json")
            if path.exists():
                old = read(path)
                if (
                    old["checkpoint_sha256"] != checkpoint["sha256"]
                    or old["card_sha256"] != sha(card_path)
                    or old["policy"] != policy
                ):
                    raise ValueError("Existing evaluation belongs to different inputs")
                continue
            began = time.monotonic()
            value = evaluate(model, tokens, policy, device)
            write(
                path,
                dict(
                    card_sha256=sha(card_path),
                    checkpoint_sha256=checkpoint["sha256"],
                    job=j,
                    endpoint=endpoint,
                    policy=policy,
                    step=state["step"],
                    training_seconds=state["training_seconds"],
                    result=value,
                    cache_geometry=dict(
                        depth=model.depth,
                        heads=model.lm.model.layers[0].self_attn.num_heads,
                        head_dim=model.lm.model.layers[0].self_attn.head_dim,
                    ),
                    qknorm_model_parameter_bytes=sum(
                        p.numel() * p.element_size()
                        for n, p in model.named_parameters()
                        if n.endswith("qknorm_gain")
                    ),
                    process_seconds=time.monotonic() - began,
                ),
            )


def execute(card_path, resource, root, cache_only=False):
    card, decision = resource_gate(
        card_path, resource, "cache" if cache_only else "run"
    )
    supervisor = Supervisor(card_path, decision, root)
    if not cache_only:
        tasks = []
        for j in jobs(card):
            out = Path(root) / "training" / name(j)
            if (out / "RESULT.json").exists():
                if read(out / "RESULT.json")["status"] not in (
                    "complete",
                    "invalid_numerical",
                ):
                    raise ValueError(
                        "Interrupted training requires explicit state/resume action"
                    )
                continue
            tasks.append(dict(kind="train", job=j, out=str(out)))
        supervisor.batch(tasks)
    results = training_barrier(card, root)
    write(
        Path(root) / "TRAINING-CLOSED.json",
        dict(
            card_sha256=sha(card_path),
            results={
                k: {n: r[n] for n in ("path", "sha256")} for k, r in results.items()
            },
            opened_final_after_barrier=True,
        ),
    )
    supervisor.batch(
        [
            dict(
                kind="evaluate",
                run_name=k,
                run_root=str(root),
                out=str(Path(root) / "evaluation" / k),
            )
            for k in results
        ]
    )
    from .analysis import analyze

    return analyze(card, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "qualify", "run", "cache", "worker", "smoke")
    )
    for n in ("card", "resource", "root", "data", "audit", "out", "task"):
        parser.add_argument("--" + n, type=Path)
    a = parser.parse_args()
    if a.action == "prepare":
        prepare(a.data, a.audit, a.out)
    elif a.action == "worker":
        worker(a.card, read(a.task))
    elif a.action == "smoke":
        from .qualification import smoke

        smoke(a.root)
    elif a.action == "qualify":
        from .qualification import qualify

        qualify(a.card, a.resource, a.root)
    else:
        execute(a.card, a.resource, a.root, cache_only=a.action == "cache")


if __name__ == "__main__":
    main()
