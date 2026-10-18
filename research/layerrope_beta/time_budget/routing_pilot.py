# SPDX-License-Identifier: MIT
"""A fixed-rate paired pilot, admitted only by an independent routing replay."""

import argparse
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from research.layerrope_beta.time_budget import train as training
from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.routing_audit import (
    DetailedObservation,
    solver,
    REPEATS,
)
from research.layerrope_beta.time_budget.run import seeds_in
from research.layerrope_beta.time_budget.state import synchronize
from research.mo_variants.stage_c_common import read, write, sha
from research.tinystories_depth.kv_cache import evaluate

ARMS = (
    ("R", "R", 0.001, False),
    ("M-original", "M", 0.003, False),
    ("M-candidate", "M", 0.003, True),
    ("ML-original", "ML", 0.01, False),
    ("ML-candidate", "ML", 0.01, True),
)
COMMON_UPDATES = (25, 100)


def common_updates(validation, device, updates=COMMON_UPDATES):
    def observe(model, optimizer, stream, clock, step, history, checkpoint):
        if step not in updates:
            return
        began = time.monotonic()
        model.eval()
        gradients = [p.grad.float() for p in model.parameters() if p.grad is not None]
        grad_norm = (
            torch.stack([g.square().sum() for g in gradients]).sum().sqrt().item()
        )
        if not math.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite common-update gradient norm")
        value = evaluate(
            model,
            validation,
            dict(name="clean"),
            device,
            window=stream.window,
            batch_size=32,
        )
        observer = DetailedObservation()
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
            step=step,
            training_seconds=clock.seconds,
            validation=value,
            gradient_norm=grad_norm,
            routing=route,
            offline_seconds=time.monotonic() - began,
        )
        history.setdefault("common_update_landmarks", []).append(record)
        model.train()
        record["checkpoint"] = checkpoint("update-" + str(step))

    return observe


def pilot_card(proposal, audit_path):
    audit = read(audit_path)
    if audit["status"] != "candidate_qualified":
        raise ValueError("Routing replay did not qualify a candidate")
    if audit["candidate_repeats"] not in REPEATS[1:]:
        raise ValueError("Candidate is outside the declared finite replay set")
    if sha(proposal["audit_inputs"]["path"]) != proposal["audit_inputs"]["sha256"]:
        raise ValueError("Audit inputs changed")
    if audit["input_sha256"] != proposal["audit_inputs"]["sha256"]:
        raise ValueError("Candidate is from another audit")
    if proposal["source"] != training.source_identity():
        raise ValueError("Pilot source changed")
    if not proposal["seed_audit"]:
        raise ValueError("Prior seed provenance is required")
    used = set()
    for receipt in proposal["seed_audit"]:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Seed ledger changed")
        observed = set(seeds_in(read(receipt["path"])))
        if not observed:
            raise ValueError("Empty seed provenance")
        used.update(observed)
    seeds = proposal["seeds"]
    if len(seeds) != 2 or len(set(seeds)) != 2 or used.intersection(seeds):
        raise ValueError("Two fresh paired pilot seeds required")
    return dict(
        schema="routing-fixed-rate-pilot.v1",
        source=proposal["source"],
        training=proposal["training"],
        seeds=seeds,
        audit=dict(path=str(audit_path), sha256=sha(audit_path)),
        candidate_repeats=audit["candidate_repeats"],
        arms=ARMS,
        common_updates=COMMON_UPDATES,
        parent_clock=freeze(),
        continuation="For M or ML, candidate must pass every observed routing bound and beat its matched original 787-second validation CE by at least 0.01 nat in both seed pairs; otherwise stop. No significance claim or seed expansion.",
    )


def run(proposal_path, audit_path, root, device="cuda"):
    proposal = read(proposal_path)
    card = pilot_card(proposal, audit_path)
    card.update(
        proposal_sha256=sha(proposal_path),
        order_seed=proposal["order_seed"],
        max_process_seconds=proposal["max_process_seconds"],
    )
    if not 0 < card["max_process_seconds"] <= 18000:
        raise ValueError("A finite bounded pilot lifetime is required")
    if (
        device != "cuda"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("Reserve one CUDA GPU for the bounded pilot")
    props = torch.cuda.get_device_properties(0)
    if "H100" not in props.name or props.total_memory < 75_000_000_000:
        raise RuntimeError("The declared pilot requires the H100 80GB tier")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write(root / "PILOT-CARD.json", card)
    training_dir = Path(card["training"]["path"])
    if sha(training_dir / "TRAINING.json") != card["training"]["sha256"]:
        raise ValueError("Training capability changed")
    validation = np.load(training_dir / "validation.npy", mmap_mode="r")
    tasks = [(seed, *arm) for seed in card["seeds"] for arm in ARMS]
    random.Random(card["order_seed"]).shuffle(tasks)
    results = []
    started = time.monotonic()
    paired = {}
    for seed, label, arm, lr, changed in tasks:
        if time.monotonic() - started > card["max_process_seconds"]:
            raise TimeoutError("Bounded pilot elapsed-time cap")
        job = dict(
            arm=arm,
            seed=seed,
            lr=lr,
            stage="pilot",
            depth=32,
            vocab=50257,
            window=256,
            batch_size=8,
            max_process_seconds=1800,
            solver_iterations=card["candidate_repeats"] if changed else 20,
            common_update_landmarks=list(COMMON_UPDATES),
            routing_pilot_card_sha256=sha(root / "PILOT-CARD.json"),
        )
        out = root / (str(seed) + "-" + label)
        try:
            with solver(job["solver_iterations"]), patch.object(
                training, "RoutingObservation", DetailedObservation
            ):
                result = training.train(
                    training_dir,
                    job,
                    out,
                    device=device,
                    on_update=common_updates(validation, device),
                )
        except FloatingPointError:
            result = read(out / "RESULT.json")
            if result["status"] != "invalid_numerical":
                raise
        weights = result["common_initial_weights"]
        if seed in paired and paired[seed] != weights:
            raise ValueError("Common initialization is not paired")
        paired[seed] = weights
        results.append(
            dict(
                label=label,
                seed=seed,
                result=result,
                receipt=dict(
                    path=str(out / "RESULT.json"), sha256=sha(out / "RESULT.json")
                ),
            )
        )
        write(
            root / "PILOT-PROGRESS.json",
            dict(
                completed=len(results),
                total=10,
                records=[
                    dict(
                        label=x["label"],
                        seed=x["seed"],
                        status=x["result"]["status"],
                        receipt=x["receipt"],
                    )
                    for x in results
                ],
            ),
        )
    result = analyze(results, card)
    write(root / "PILOT.json", result)
    return result


def analyze(results, card):
    expected = {(seed, arm[0]) for seed in card["seeds"] for arm in ARMS}
    if len(results) != 10 or {(r["seed"], r["label"]) for r in results} != expected:
        raise ValueError("All ten distinct pilot trajectories are required")
    comparisons = []
    for family in ("M", "ML"):
        pairs = []
        for seed in card["seeds"]:
            records = [
                next(
                    r
                    for r in results
                    if r["label"] == family + suffix and r["seed"] == seed
                )["result"]
                for suffix in ("-original", "-candidate")
            ]
            original, candidate = records
            checks = (
                candidate["history"]["routing"]
                + [f["routing"] for f in candidate["history"]["landmarks"]]
                + [
                    f["routing"]
                    for f in candidate["history"].get("common_update_landmarks", [])
                ]
            )
            complete = all(
                r["status"] == "complete"
                and {f["step"] for f in r["history"].get("common_update_landmarks", [])}
                == set(COMMON_UPDATES)
                for r in records
            )
            delta = (
                (
                    candidate["history"]["landmarks"][-1]["validation"]["ce"]
                    - original["history"]["landmarks"][-1]["validation"]["ce"]
                )
                if complete
                else None
            )
            common = {
                r["identity"]["job"]["solver_iterations"]: {
                    f["step"]: f["validation"]["ce"]
                    for f in r["history"].get("common_update_landmarks", [])
                }
                for r in records
            }
            pairs.append(
                dict(
                    seed=seed,
                    complete=complete,
                    delta_ce=delta,
                    candidate_constraints_pass=all(
                        c["constraint_pass"] for c in checks
                    ),
                    common_update_ce=common,
                    pass_screen=complete
                    and all(c["constraint_pass"] for c in checks)
                    and delta <= -0.01,
                )
            )
        comparisons.append(
            dict(
                family=family,
                pairs=pairs,
                pass_screen=all(p["pass_screen"] for p in pairs),
            )
        )
    result = dict(
        status="complete",
        continuation=any(x["pass_screen"] for x in comparisons),
        comparisons=comparisons,
        records=[
            dict(label=x["label"], seed=x["seed"], receipt=x["receipt"])
            for x in results
        ],
        scope="Fixed-rate two-seed descriptive pilot; no automatic confirmation, final holdout or quantization evaluation",
    )
    return result


def smoke(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data = root / "training"
    data.mkdir()
    tokens = np.random.default_rng(7).integers(0, 97, 1025, dtype=np.uint16)
    for name in ("train", "validation"):
        np.save(data / (name + ".npy"), tokens)
    write(
        data / "TRAINING.json",
        dict(files={n: sha(data / n) for n in ("train.npy", "validation.npy")}),
    )
    outputs = []
    for n in (20, 80):
        job = dict(
            arm="M",
            seed=7,
            lr=0.003,
            stage="qualification",
            depth=2,
            vocab=97,
            window=8,
            batch_size=2,
            solver_iterations=n,
        )
        with solver(n), patch.object(
            training, "RoutingObservation", DetailedObservation
        ):
            r = training.train(
                data,
                job,
                root / str(n),
                device="cpu",
                smoke=True,
                stop_after_updates=2,
                on_update=common_updates(tokens, "cpu", (1, 2)),
            )
        assert [f["step"] for f in r["history"]["common_update_landmarks"]] == [1, 2]
        outputs.append(dict(repeats=n, status=r["status"], common_updates=2))
    write(root / "SMOKE.json", dict(records=outputs, scientific_results=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proposal", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        smoke(a.out)
    else:
        run(a.proposal, a.audit, a.out)
