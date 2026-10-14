# SPDX-License-Identifier: MIT
"""Run the corrected residual-scale student comparison on archived teachers.

This is a deliberately narrow continuation of the post-scale experiment.  It
reconstructs each arm from its own archived receipt, uses its own frozen
teacher, and trains only aux_shared and full_shared students.  Ordinary targets
come directly from teacher(batch); they never pass through the algebraically
equivalent but numerically different W0 + 1 * (teacher - W0) expression.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import epoch_order, load_mnist, make_noise, tensor_digest
from .models import build, init_model
from .null_audit import _per_model_stats, _recipe_from_receipt
from .numerical_kl import EXACT_CANCEL_KL
from .repair_bridge import (
    _digit_scoring,
    _file_receipt,
    _manifest,
    _pack,
    _parameter_displacement,
    _parameter_values,
    _write_json,
)
from .repair_qualification import (
    _condition_trace,
    _fp64_kl_mean,
    _moment_stats,
    _optimizer_digest,
)
from .run import (
    environment,
    git_identity,
    logits_on,
    make_optimizer,
    output_index,
    pick_device,
    rows_unchanged,
    state_digest,
    sync,
    train_student_epoch,
)

MILESTONES = (0, 5, 20, 80)
KINDS = ("aux_shared", "full_shared")
TARGET_MODE = "direct_teacher_v1"
SCHEMA = "subliminal-residual-scale-repair.v1"


def _verified_file(entry, label):
    path = entry["path"]
    actual = _file_receipt(path)
    expected = {key: entry[key] for key in ("path", "bytes", "sha256")}
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected}, got {actual}")
    return actual


def _compare_per_model(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(
            f"comparison shape/dtype mismatch: {left.shape}/{left.dtype} "
            f"!= {right.shape}/{right.dtype}"
        )
    delta = (left.detach() - right.detach()).reshape(left.shape[0], -1).double()
    equal = (left.detach() == right.detach()).reshape(left.shape[0], -1).all(1)
    return {
        "bitwise_equal": equal.cpu().tolist(),
        "different_elements": (delta != 0).sum(1).cpu().tolist(),
        "max_abs": delta.abs().amax(1).cpu().tolist(),
        "l2": torch.linalg.vector_norm(delta, dim=1).cpu().tolist(),
    }


def _direct_diagnostic(w0_logits, teacher_logits, recipe):
    reconstructed = w0_logits + (teacher_logits - w0_logits)
    out = {}
    for name in ("aux_shared", "full_shared"):
        idx = output_index(name, recipe)
        direct = teacher_logits[..., idx]
        rebuilt = reconstructed[..., idx]
        out[name] = {
            "logits": _compare_per_model(direct, rebuilt),
            "probabilities": _compare_per_model(
                F.softmax(direct, -1), F.softmax(rebuilt, -1)
            ),
        }
    return out


def _schedule_receipt(keys, examples, epochs):
    per_epoch = []
    aggregate = hashlib.sha256()
    for epoch in range(epochs):
        digest = hashlib.sha256()
        for key in keys:
            value = tensor_digest(epoch_order(key, "student_order", epoch, examples))
            digest.update(key.encode())
            digest.update(value.encode())
        value = digest.hexdigest()
        per_epoch.append(value)
        aggregate.update(str(epoch).encode())
        aggregate.update(value.encode())
    return {"per_epoch_sha256": per_epoch, "aggregate_sha256": aggregate.hexdigest()}


def _compatible(parent, candidate):
    excluded = {
        "residual_scale",
        "output_dir",
        "checkpoint_dir",
        "run_name",
        "max_seconds",
        "student_loss_mode",
    }
    left, right = parent.to_json(), candidate.to_json()
    differences = {
        key: [left.get(key), right.get(key)]
        for key in sorted((set(left) | set(right)) - excluded)
        if left.get(key) != right.get(key)
    }
    if differences:
        raise ValueError(f"archived arm recipes differ unexpectedly: {differences}")


def _first_batch(noise, keys, batch_size):
    rows = torch.arange(len(keys), device=noise.device)[:, None]
    order = torch.stack(
        [epoch_order(key, "student_order", 0, noise.shape[1]) for key in keys]
    ).to(noise.device)
    return noise[rows, order[:, :batch_size]]


def _common_evaluation(
    model,
    initial,
    initial_parameters,
    optimizer,
    heldout,
    w0_heldout,
    teacher_heldout,
    recipe,
    kind,
):
    idx = output_index(kind, recipe)
    with torch.no_grad():
        student_heldout = logits_on(model, heldout)
    return {
        "target_fit_kl_fp64": _fp64_kl_mean(
            teacher_heldout[..., idx], student_heldout[..., idx]
        ),
        "logit_displacement_from_w0": _per_model_stats(
            student_heldout[..., idx] - w0_heldout[..., idx]
        ),
        "parameter_displacement_from_w0": _parameter_displacement(
            model, initial_parameters
        ),
        "model_digest": [
            state_digest(model, row) for row in range(next(model.parameters()).shape[0])
        ],
        "optimizer_digest": _optimizer_digest(model, optimizer),
        "optimizer_moments": (
            _moment_stats(model, optimizer) if optimizer.state else None
        ),
        "digit_head_rows_unchanged": (
            rows_unchanged(model, initial, list(range(10))).tolist()
            if kind.startswith("aux")
            else None
        ),
    }


def _save_checkpoint(path, arm, kind, epoch, model, optimizer, source):
    torch.save(
        {
            "schema": "subliminal-residual-scale-repair-checkpoint.v1",
            "arm": arm,
            "kind": kind,
            "epoch": epoch,
            "target_mode": TARGET_MODE,
            "student_loss_mode": EXACT_CANCEL_KL,
            "source": source,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _nonzero_parameter_elements(point):
    return [
        sum(group["summary"]["nonzero"][row] for group in point.values())
        for row in range(5)
    ]


def _short_qualification(prepared, steps=4):
    recipe = prepared["recipe"]
    w0 = prepared["w0"]
    teacher = prepared["teacher"]
    noise = prepared["noise"]
    keys = recipe.model_keys
    zero = torch.zeros(len(keys), device=noise.device)
    conditions = [
        _condition_trace(
            "scaled_shared_zero", w0, teacher, w0, noise, keys, recipe, zero, steps
        ),
        _condition_trace(
            "scaled_shared_direct_teacher",
            w0,
            teacher,
            w0,
            noise,
            keys,
            recipe,
            None,
            steps,
        ),
    ]
    null, active = conditions
    for condition in conditions:
        if condition["calls"] != {
            "loss": steps,
            "backward": steps,
            "optimizer_step": steps,
        }:
            raise AssertionError(
                f"short qualification call mismatch: {condition['calls']}"
            )
        for point in condition["trace"]:
            if not all(point["digit_head_rows_unchanged"]):
                raise AssertionError("auxiliary training changed digit head rows")
            if max(point["logit_gradient_reference"]["error"]["max_abs"]) >= 2e-7:
                raise AssertionError(
                    "repaired gradient missed FP64 reference tolerance"
                )
    for point in null["trace"]:
        if _nonzero_parameter_elements(point["parameter_gradient"]) != [0] * 5:
            raise AssertionError("scaled exact null produced parameter gradients")
        if point["model_digest"] != null["initial_model_digest"]:
            raise AssertionError("scaled exact null moved model parameters")
        if point["optimizer_moments"]["exp_avg"]["nonzero"] != [0] * 5:
            raise AssertionError("scaled exact null produced Adam first moments")
        if point["optimizer_moments"]["exp_avg_sq"]["nonzero"] != [0] * 5:
            raise AssertionError("scaled exact null produced Adam second moments")
    for point in active["trace"]:
        if not all(
            value > 0
            for value in _nonzero_parameter_elements(point["parameter_gradient"])
        ):
            raise AssertionError(
                "direct teacher target did not produce active gradients"
            )
        if not all(
            after != before
            for before, after in zip(
                active["initial_model_digest"], point["model_digest"]
            )
        ):
            raise AssertionError(
                "direct teacher target did not move every packed model"
            )
    return {
        "steps": steps,
        "conditions": conditions,
        "all_gates_pass": True,
        "scaled_exact_null_stationary": True,
        "scaled_direct_teacher_active": True,
        "aux_digit_rows_unchanged": True,
    }


def _trajectory(
    arm,
    kind,
    prepared,
    test_x,
    test_y,
    checkpoint_dir,
    progress_path,
    source,
    deadline,
):
    recipe = prepared["recipe"]
    w0 = prepared["w0"]
    teacher = prepared["teacher"]
    heldout = prepared["heldout"]
    w0_heldout = prepared["w0_heldout"]
    teacher_heldout = prepared["teacher_heldout"]
    student = copy.deepcopy(w0)
    optimizer = make_optimizer(student, recipe)
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in student.named_parameters()
    }
    points, curve, epochs = {}, {}, []
    totals = {
        "training_seconds": 0.0,
        "digit_evaluation_seconds": 0.0,
        "common_evaluation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }
    cumulative_steps = 0
    started = time.time()
    for epoch in range(recipe.student_epochs + 1):
        if time.time() > deadline:
            raise TimeoutError("two-hour end-to-end cap reached")
        digit = None
        if kind == "full_shared" or epoch in MILESTONES:
            sync(test_x.device)
            t0 = time.time()
            digit = _digit_scoring(student, test_x, test_y)
            sync(test_x.device)
            seconds = time.time() - t0
            totals["digit_evaluation_seconds"] += seconds
            curve[str(epoch)] = {
                "correct_counts": digit["correct_counts"],
                "accuracy": digit["accuracy"],
                "seconds": seconds,
            }
        if epoch in MILESTONES:
            sync(test_x.device)
            t0 = time.time()
            common = _common_evaluation(
                student,
                w0,
                initial_parameters,
                optimizer,
                heldout,
                w0_heldout,
                teacher_heldout,
                recipe,
                kind,
            )
            sync(test_x.device)
            common_seconds = time.time() - t0
            totals["common_evaluation_seconds"] += common_seconds

            name = f"{arm}-{kind}-e{epoch}.pt"
            path = os.path.join(checkpoint_dir, name)
            sync(test_x.device)
            t0 = time.time()
            _save_checkpoint(path, arm, kind, epoch, student, optimizer, source)
            sync(test_x.device)
            checkpoint_seconds = time.time() - t0
            totals["checkpoint_seconds"] += checkpoint_seconds
            points[str(epoch)] = {
                "digit_scoring": digit,
                **common,
                "training_steps": cumulative_steps,
                "checkpoint": _file_receipt(path),
                "timing": {
                    "common_evaluation_seconds": common_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                },
            }
        if epoch == recipe.student_epochs:
            break

        sync(test_x.device)
        t0 = time.time()
        kl_mean, optimized_loss, steps = train_student_epoch(
            student,
            teacher,
            optimizer,
            kind,
            recipe.model_keys,
            epoch,
            prepared["noise"],
            recipe,
            None,
            None,
        )
        sync(test_x.device)
        seconds = time.time() - t0
        totals["training_seconds"] += seconds
        cumulative_steps += steps
        row = {
            "arm": arm,
            "kind": kind,
            "epoch": epoch + 1,
            "steps": steps,
            "cumulative_steps": cumulative_steps,
            "examples_per_model": recipe.noise_samples,
            "packed_examples": recipe.noise_samples * len(recipe.model_keys),
            "kl_mean": kl_mean.tolist(),
            "optimized_loss": optimized_loss.tolist(),
            "training_seconds": seconds,
        }
        epochs.append(row)
        with open(progress_path, "a") as progress:
            progress.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    totals["end_to_end_seconds"] = time.time() - started
    totals["overhead_seconds"] = (
        totals["end_to_end_seconds"] - totals["training_seconds"]
    )
    expected_steps = (
        math.ceil(recipe.noise_samples / recipe.batch_size) * recipe.student_epochs
    )
    if cumulative_steps != expected_steps:
        raise AssertionError(f"trajectory steps {cumulative_steps} != {expected_steps}")
    if kind.startswith("aux"):
        for point in points.values():
            if not all(point["digit_head_rows_unchanged"]):
                raise AssertionError(f"{arm}/{kind} changed frozen digit rows")
    return {
        "arm": arm,
        "kind": kind,
        "target_mode": TARGET_MODE,
        "student_loss_mode": recipe.student_loss_mode,
        "points": points,
        "accuracy_curve": curve,
        "epochs": epochs,
        "steps": cumulative_steps,
        "examples_per_model": recipe.noise_samples * recipe.student_epochs,
        "packed_examples": recipe.noise_samples
        * recipe.student_epochs
        * len(recipe.model_keys),
        "timing": totals,
    }


def _boot(values, level=0.95, resamples=100000, seed=20260928):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(values.mean()), float(lo), float(hi)]


def _first_crossing(curve, threshold=0.95):
    per_model = []
    epochs = sorted(int(epoch) for epoch in curve)
    count = len(curve[str(epochs[0])]["accuracy"])
    for row in range(count):
        crossing = next(
            (
                epoch
                for epoch in epochs
                if curve[str(epoch)]["accuracy"][row] >= threshold
            ),
            None,
        )
        per_model.append(crossing)
    return per_model


def _archived_teacher_curve(run_dir, keys):
    key_of = {index: key for index, key in enumerate(keys)}
    curve = {key: {} for key in keys}
    with open(os.path.join(run_dir, "metrics.jsonl")) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("stage") != "teacher" or "test_acc" not in row:
                continue
            for index, accuracy in enumerate(row["test_acc"]):
                curve[key_of[index]][str(row["epoch"])] = float(accuracy)
    return curve


def _speed_screen(parent_epochs, candidate_epochs):
    rows, logs = [], []
    status = "ok"
    for parent, candidate in zip(parent_epochs, candidate_epochs):
        rows.append({"parent": parent, "candidate": candidate})
        if parent is None:
            status = "unresolved"
        elif candidate is None and status != "unresolved":
            status = "fail"
        elif candidate is not None:
            logs.append(math.log(candidate / parent))
    out = {"per_seed": rows, "status": status, "pass": False}
    if status == "ok" and logs:
        interval = _boot(logs)
        out.update(
            {
                "mean_log_ratio": interval,
                "geometric_epoch_ratio": math.exp(interval[0]),
                "pass": interval[2] <= math.log(1.10),
            }
        )
    return out


def _analyze(result, prepared):
    parent_name, candidate_name = [arm["name"] for arm in result["arms"]]
    parent, candidate = (
        result["trajectories"][parent_name],
        result["trajectories"][candidate_name],
    )
    parent_initial = parent["aux_shared"]["points"]["0"]["digit_scoring"]["accuracy"]
    candidate_initial = candidate["aux_shared"]["points"]["0"]["digit_scoring"][
        "accuracy"
    ]
    parent_aux = parent["aux_shared"]["points"]["80"]["digit_scoring"]["accuracy"]
    candidate_aux = candidate["aux_shared"]["points"]["80"]["digit_scoring"]["accuracy"]
    transfer = [
        100 * ((ca - ci) - (pa - pi))
        for pa, pi, ca, ci in zip(
            parent_aux, parent_initial, candidate_aux, candidate_initial
        )
    ]
    parent_teacher = result["arm_provenance"][parent_name]["teacher_digit_scoring"][
        "accuracy"
    ]
    candidate_teacher = result["arm_provenance"][candidate_name][
        "teacher_digit_scoring"
    ]["accuracy"]
    teacher_utility = [
        100 * (candidate_value - parent_value)
        for parent_value, candidate_value in zip(parent_teacher, candidate_teacher)
    ]
    parent_full = parent["full_shared"]["points"]["80"]["digit_scoring"]["accuracy"]
    candidate_full = candidate["full_shared"]["points"]["80"]["digit_scoring"][
        "accuracy"
    ]
    full_utility = [
        100 * (candidate_value - parent_value)
        for parent_value, candidate_value in zip(parent_full, candidate_full)
    ]
    student_speed = _speed_screen(
        _first_crossing(parent["full_shared"]["accuracy_curve"]),
        _first_crossing(candidate["full_shared"]["accuracy_curve"]),
    )
    archived = {}
    for name in (parent_name, candidate_name):
        keys = prepared[name]["recipe"].model_keys
        curve = _archived_teacher_curve(prepared[name]["run_dir"], keys)
        archived[name] = [
            next(
                (
                    int(epoch)
                    for epoch in sorted(curve[key], key=int)
                    if curve[key][epoch] >= 0.95
                ),
                None,
            )
            for key in keys
        ]
    teacher_speed = _speed_screen(archived[parent_name], archived[candidate_name])
    intervals = {
        "transfer_aux_gain_difference_pp": _boot(transfer),
        "teacher_accuracy_difference_pp": _boot(teacher_utility),
        "full_shared_e80_accuracy_difference_pp": _boot(full_utility),
    }
    screens = {
        "transfer": intervals["transfer_aux_gain_difference_pp"][0] <= -5
        and intervals["transfer_aux_gain_difference_pp"][2] < 0,
        "utility_teacher": intervals["teacher_accuracy_difference_pp"][1] >= -2,
        "utility_full_shared_e80": intervals["full_shared_e80_accuracy_difference_pp"][
            1
        ]
        >= -2,
        "speed_full_shared": student_speed["pass"],
        "speed_teacher": teacher_speed["pass"],
    }
    return {
        "schema": "subliminal-residual-scale-repair-analysis.v1",
        "parent": parent_name,
        "candidate": candidate_name,
        "level": 0.95,
        "bootstrap_resamples": 100000,
        "accuracy_threshold": 0.95,
        "paired_values": {
            "transfer_aux_gain_difference_pp": transfer,
            "teacher_accuracy_difference_pp": teacher_utility,
            "full_shared_e80_accuracy_difference_pp": full_utility,
        },
        "intervals_mean_lo_hi": intervals,
        "speed": {"full_shared": student_speed, "teacher_archived": teacher_speed},
        "screens": screens,
        "promote": all(screens.values()),
        "screen_definitions": {
            "transfer": "mean <= -5 pp and 95% paired-bootstrap upper bound < 0",
            "utility": "95% paired-bootstrap lower bound >= -2 pp",
            "speed": "95% upper bound on mean log candidate/parent epoch ratio <= log(1.10)",
            "candidate_censored": "fail when parent crosses; parent censored is unresolved",
        },
    }


def run(input_manifest, out_dir, checkpoint_dir):
    overall_started = time.time()
    manifest_receipt = _file_receipt(input_manifest)
    frozen = json.load(open(input_manifest))
    if frozen.get("schema") != "subliminal-residual-scale-repair-inputs.v1":
        raise ValueError("unexpected input manifest schema")
    source = git_identity()
    if source.get("commit") != frozen["source"]["commit"]:
        raise ValueError(f"source commit mismatch: {source}")
    if source.get("workload_dirty") is not False:
        raise ValueError(f"source is not frozen clean: {source}")
    for name, entry in frozen["source"]["files"].items():
        _verified_file(entry, f"source file {name}")
    _verified_file(frozen["plan"], "frozen analysis plan")
    _verified_file(frozen["mnist_cache"], "MNIST tensor cache")
    if os.path.exists(checkpoint_dir):
        raise FileExistsError(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "progress.jsonl")
    if os.path.exists(progress_path):
        raise FileExistsError(progress_path)

    arms = frozen["arms"]
    if [arm["name"] for arm in arms] != ["residual", "residual-post025"]:
        raise ValueError("input manifest must contain only the frozen residual pair")
    for arm in arms:
        for name, entry in arm["receipts"].items():
            _verified_file(entry, f"{arm['name']} {name}")
        _verified_file(arm["teacher_checkpoint"], f"{arm['name']} teacher")

    recipes = {}
    for arm in arms:
        recipe, _receipt = _recipe_from_receipt(arm["run_dir"])
        if recipe.residual_scale != arm["expected_residual_scale"]:
            raise ValueError(f"{arm['name']} residual scale changed")
        if recipe.student_epochs != 80 or recipe.student_eval_epochs != [5, 20, 80]:
            raise ValueError("repair requires the archived 0/5/20/80 horizon")
        if recipe.weight_decay != 0 or recipe.protocol != "per_example_v1":
            raise ValueError("repair requires zero-decay per-example recipes")
        recipe.student_loss_mode = EXACT_CANCEL_KL
        recipes[arm["name"]] = recipe
    _compatible(recipes["residual"], recipes["residual-post025"])

    device = pick_device(recipes["residual"])
    torch.backends.cudnn.benchmark = False
    _train_x, _train_y, test_x, test_y = load_mnist(
        os.path.expanduser(recipes["residual"].data_dir)
    )
    test_x, test_y = test_x.to(device), test_y.to(device)
    prepared = {}
    provenance = {}
    schedule = None
    for arm in arms:
        name, recipe = arm["name"], recipes[arm["name"]]
        expected = _pack(arm["run_dir"])["results"]
        keys = recipe.model_keys
        w0 = build(recipe, len(keys)).to(device)
        init_model(w0, keys, "init")
        w0.eval()
        got_w0 = [state_digest(w0, row) for row in range(len(keys))]
        if got_w0 != expected["w0_digest"]:
            raise ValueError(f"{name} W0 differs from archived receipt")
        teacher = build(recipe, len(keys)).to(device)
        teacher.load_state_dict(
            torch.load(
                arm["teacher_checkpoint"]["path"],
                map_location=device,
                weights_only=True,
            )
        )
        teacher.eval().requires_grad_(False)
        got_teacher = [state_digest(teacher, row) for row in range(len(keys))]
        if got_teacher != expected["teacher_digest"]:
            raise ValueError(f"{name} teacher differs from archived receipt")
        archived_manifest = _manifest(arm["run_dir"])["pack00_teacher.pt"]
        if (
            archived_manifest["sha256"] != arm["teacher_checkpoint"]["sha256"]
            or archived_manifest["bytes"] != arm["teacher_checkpoint"]["bytes"]
        ):
            raise ValueError(f"{name} teacher manifest cross-receipt mismatch")

        noise_cpu = torch.stack([make_noise(key, recipe.noise_samples) for key in keys])
        noise_digest = [tensor_digest(noise_cpu[row]) for row in range(len(keys))]
        if noise_digest != expected["noise_digest"]:
            raise ValueError(f"{name} noise differs from archived receipt")
        noise = noise_cpu.to(device)
        del noise_cpu
        heldout = torch.stack(
            [
                make_noise(key, recipe.heldout_noise, stream="heldout_noise")
                for key in keys
            ]
        ).to(device)
        heldout_digest = [tensor_digest(heldout[row].cpu()) for row in range(len(keys))]
        first = _first_batch(noise, keys, recipe.batch_size)
        with torch.no_grad():
            w0_first = w0(first)
            teacher_first = teacher(first)
            w0_heldout = logits_on(w0, heldout)
            teacher_heldout = logits_on(teacher, heldout)
        arm_schedule = _schedule_receipt(
            keys, recipe.noise_samples, recipe.student_epochs
        )
        if schedule is None:
            schedule = arm_schedule
        elif schedule != arm_schedule:
            raise ValueError("student epoch schedule differs across arms")
        teacher_score = _digit_scoring(teacher, test_x, test_y)
        archived_teacher_counts = [
            round(value * teacher_score["examples"])
            for value in expected["acc_teacher"]
        ]
        if teacher_score["correct_counts"] != archived_teacher_counts:
            raise ValueError(f"{name} teacher accuracy differs from archived result")
        prepared[name] = {
            "run_dir": arm["run_dir"],
            "recipe": recipe,
            "w0": w0,
            "teacher": teacher,
            "noise": noise,
            "heldout": heldout,
            "w0_heldout": w0_heldout,
            "teacher_heldout": teacher_heldout,
            "w0_first": w0_first,
        }
        provenance[name] = {
            "recipe": recipe.to_json(),
            "keys": keys,
            "w0_state_digest": got_w0,
            "teacher_state_digest_before": got_teacher,
            "noise_digest": noise_digest,
            "heldout_noise_digest": heldout_digest,
            "first_batch_digest": [
                tensor_digest(first[row].cpu()) for row in range(len(keys))
            ],
            "first_batch_w0_logits_digest": [
                tensor_digest(w0_first[row].cpu()) for row in range(len(keys))
            ],
            "teacher_digit_scoring": teacher_score,
            "direct_vs_reconstructed": {
                "first_archived_batch": _direct_diagnostic(
                    w0_first, teacher_first, recipe
                ),
                "heldout_noise": _direct_diagnostic(
                    w0_heldout, teacher_heldout, recipe
                ),
            },
        }

    parent = prepared["residual"]
    candidate = prepared["residual-post025"]
    if (
        provenance["residual"]["w0_state_digest"]
        != provenance["residual-post025"]["w0_state_digest"]
    ):
        raise ValueError("residual pair no longer shares initial parameter tensors")
    functional_difference = _compare_per_model(
        parent["w0_first"], candidate["w0_first"]
    )
    if all(functional_difference["bitwise_equal"]):
        raise ValueError("residual scales unexpectedly implement the same W0 function")
    if (
        provenance["residual"]["noise_digest"]
        != provenance["residual-post025"]["noise_digest"]
    ):
        raise ValueError("residual pair no longer shares the fixed noise inputs")

    preflight_seconds = time.time() - overall_started
    deadline = overall_started + frozen.get("max_end_to_end_seconds", 7200)
    qualification = _short_qualification(candidate)
    trajectories = {}
    for arm in arms:
        name = arm["name"]
        trajectories[name] = {}
        for kind in KINDS:
            trajectories[name][kind] = _trajectory(
                name,
                kind,
                prepared[name],
                test_x,
                test_y,
                checkpoint_dir,
                progress_path,
                source,
                deadline,
            )
            after = [
                state_digest(prepared[name]["teacher"], row)
                for row in range(len(recipes[name].model_keys))
            ]
            if after != provenance[name]["teacher_state_digest_before"]:
                raise AssertionError(f"{name} teacher mutated during student training")
        provenance[name]["teacher_state_digest_after"] = after
        _verified_file(arm["teacher_checkpoint"], f"{name} teacher after run")

    result = {
        "schema": SCHEMA,
        "source": source,
        "environment": environment(device),
        "input_manifest": manifest_receipt,
        "arms": arms,
        "milestones": list(MILESTONES),
        "target_mode": TARGET_MODE,
        "student_loss_mode": EXACT_CANCEL_KL,
        "student_schedule": schedule,
        "cross_arm_checks": {
            "same_keys": recipes["residual"].model_keys
            == recipes["residual-post025"].model_keys,
            "same_w0_parameter_digests": True,
            "same_noise_digests": True,
            "same_epoch_schedule": True,
            "different_w0_function": functional_difference,
        },
        "arm_provenance": provenance,
        "scaled_short_qualification": qualification,
        "initial_parameter_values": {
            name: _parameter_values(prepared[name]["w0"]) for name in prepared
        },
        "trajectories": trajectories,
        "timing": {
            "end_to_end_seconds": time.time() - overall_started,
            "preflight_seconds": preflight_seconds,
            "two_hour_cap_seconds": frozen.get("max_end_to_end_seconds", 7200),
            "trajectory_training_seconds": sum(
                trajectories[name][kind]["timing"]["training_seconds"]
                for name in trajectories
                for kind in KINDS
            ),
            "trajectory_digit_evaluation_seconds": sum(
                trajectories[name][kind]["timing"]["digit_evaluation_seconds"]
                for name in trajectories
                for kind in KINDS
            ),
            "trajectory_common_evaluation_seconds": sum(
                trajectories[name][kind]["timing"]["common_evaluation_seconds"]
                for name in trajectories
                for kind in KINDS
            ),
            "trajectory_checkpoint_seconds": sum(
                trajectories[name][kind]["timing"]["checkpoint_seconds"]
                for name in trajectories
                for kind in KINDS
            ),
        },
        "validation": {
            "all_gates_pass": True,
            "frozen_inputs_verified": True,
            "own_w0_and_teacher_per_arm": True,
            "direct_teacher_targets": True,
            "teacher_immutable": True,
            "full_accuracy_recorded_every_epoch": all(
                len(trajectories[name]["full_shared"]["accuracy_curve"]) == 81
                for name in trajectories
            ),
            "milestone_checkpoints": sum(
                len(trajectories[name][kind]["points"])
                for name in trajectories
                for kind in KINDS
            ),
        },
    }
    if result["validation"]["milestone_checkpoints"] != 16:
        raise AssertionError("expected exactly 16 milestone checkpoints")
    analysis = _analyze(result, prepared)
    result["analysis"] = analysis
    _write_json(os.path.join(out_dir, "run.json"), result)
    _write_json(os.path.join(out_dir, "analysis.json"), analysis)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args(argv)
    result = run(
        os.path.abspath(args.input_manifest),
        os.path.abspath(args.out_dir),
        os.path.abspath(args.checkpoint_dir),
    )
    print(
        json.dumps(
            {
                "complete": True,
                "elapsed_s": result["timing"]["end_to_end_seconds"],
                "analysis": result["analysis"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
