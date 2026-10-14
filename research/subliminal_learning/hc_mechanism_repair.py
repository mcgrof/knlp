# SPDX-License-Identifier: MIT
"""Run repaired HC/post-scale students with carry/write mechanism probes."""

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback

import torch

from .data import epoch_order, load_mnist, make_noise, tensor_digest
from .hc_diagnostics import (
    capture_step_before,
    collect_hc_probe,
    optimizer_step_diagnostics,
    parameter_partition,
)
from .models import build, init_model
from .null_audit import _recipe_from_receipt
from .numerical_kl import EXACT_CANCEL_KL
from .repair_bridge import (
    _digit_scoring,
    _file_receipt,
    _manifest,
    _pack,
    _parameter_values,
    _write_json,
)
from .repair_qualification import (
    _condition_trace,
    _optimizer_digest,
)
from .residual_scale_repair import (
    _archived_teacher_curve,
    _boot,
    _common_evaluation,
    _compare_per_model,
    _direct_diagnostic,
    _schedule_receipt,
    _speed_screen,
    _verified_file,
)
from .run import (
    environment,
    git_identity,
    logits_on,
    make_optimizer,
    output_index,
    pick_device,
    state_digest,
    student_loss,
    sync,
)

MILESTONES = (0, 5, 20, 80)
KINDS = ("aux_shared", "full_shared")
STEP_DIAGNOSTIC_EPOCHS = (1, 5, 20, 80)
TARGET_MODE = "direct_teacher_v1"
SCHEMA = "subliminal-hc-mechanism-repair.v1"


def _compatible(parent, candidate):
    excluded = {
        "post_scale",
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
        raise ValueError(f"archived HC recipes differ unexpectedly: {differences}")


def _first_batch(noise, keys, batch_size):
    rows = torch.arange(len(keys), device=noise.device)[:, None]
    order = torch.stack(
        [epoch_order(key, "student_order", 0, noise.shape[1]) for key in keys]
    ).to(noise.device)
    return noise[rows, order[:, :batch_size]]


def _save_checkpoint(path, arm, kind, epoch, model, optimizer, source):
    torch.save(
        {
            "schema": "subliminal-hc-mechanism-repair-checkpoint.v1",
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


def _load_archived_model(arm, checkpoint_name, recipe, device):
    entry = arm["checkpoints"][checkpoint_name]
    _verified_file(entry, f"{arm['name']} {checkpoint_name}")
    archived = _manifest(arm["run_dir"])[checkpoint_name]
    if archived["sha256"] != entry["sha256"] or archived["bytes"] != entry["bytes"]:
        raise ValueError(f"{arm['name']} {checkpoint_name} cross-receipt mismatch")
    model = build(recipe, len(recipe.model_keys)).to(device)
    model.load_state_dict(
        torch.load(entry["path"], map_location=device, weights_only=True)
    )
    model.eval()
    return model


def _probe_pair(model, noise_probe, mnist_probe):
    return {
        "heldout_noise": collect_hc_probe(model, noise_probe, "heldout_noise"),
        "mnist_images": collect_hc_probe(model, mnist_probe, "mnist_images"),
    }


def _capture_forward_parity(model, probe):
    rows = next(model.parameters()).shape[0]
    before = [state_digest(model, row) for row in range(rows)]
    with torch.no_grad():
        plain = model(probe)
        model.set_hc_capture(True)
        try:
            captured = model(probe)
        finally:
            model.set_hc_capture(False)
    after = [state_digest(model, row) for row in range(rows)]
    comparison = _compare_per_model(plain, captured)
    if not all(comparison["bitwise_equal"]) or before != after:
        raise AssertionError("enabling HC capture changed forward output or state")
    return {
        "output_comparison": comparison,
        "state_digest_before": before,
        "state_digest_after": after,
        "pass": True,
    }


def _captured_optimizer_parity(prepared):
    recipe = prepared["recipe"]
    keys = recipe.model_keys
    batch = _first_batch(prepared["noise"], keys, recipe.batch_size)
    idx = output_index("aux_shared", recipe)
    with torch.no_grad():
        targets = prepared["teacher"](batch)[..., idx]
    models = [copy.deepcopy(prepared["w0"]), copy.deepcopy(prepared["w0"])]
    optimizers = [make_optimizer(model, recipe) for model in models]
    outputs, losses, gradients = [], [], []
    for index, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.set_hc_capture(index == 1)
        logits = model(batch)[..., idx]
        loss, _kl_sum, _kl_mean = student_loss(
            logits, targets, recipe.protocol, recipe.cohort, EXACT_CANCEL_KL
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        outputs.append(logits.detach())
        losses.append(loss.detach())
        gradients.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
            }
        )
        optimizer.step()
        model.set_hc_capture(False)
    output_comparison = _compare_per_model(outputs[0], outputs[1])
    if not all(output_comparison["bitwise_equal"]):
        raise AssertionError("HC capture changed sampled training logits")
    if not torch.equal(losses[0], losses[1]):
        raise AssertionError("HC capture changed sampled loss")
    for name in gradients[0]:
        if not torch.equal(gradients[0][name], gradients[1][name]):
            raise AssertionError(f"HC capture changed gradient {name}")
    model_digests = [
        [state_digest(model, row) for row in range(len(keys))] for model in models
    ]
    optimizer_digests = [
        _optimizer_digest(model, optimizer)
        for model, optimizer in zip(models, optimizers)
    ]
    if (
        model_digests[0] != model_digests[1]
        or optimizer_digests[0] != optimizer_digests[1]
    ):
        raise AssertionError("HC capture changed sampled Adam update")
    return {
        "target_mode": TARGET_MODE,
        "student_loss_mode": EXACT_CANCEL_KL,
        "output_comparison": output_comparison,
        "loss_bitwise_equal": True,
        "all_parameter_gradients_bitwise_equal": True,
        "model_digest": model_digests[0],
        "optimizer_digest": optimizer_digests[0],
        "pass": True,
    }


def _nonzero_parameter_elements(point):
    return [
        sum(group["summary"]["nonzero"][row] for group in point.values())
        for row in range(5)
    ]


def _short_qualification(prepared, steps=4):
    recipe = prepared["recipe"]
    w0, teacher, noise = prepared["w0"], prepared["teacher"], prepared["noise"]
    keys = recipe.model_keys
    zero = torch.zeros(len(keys), device=noise.device)
    conditions = [
        _condition_trace(
            "hc_scaled_shared_zero", w0, teacher, w0, noise, keys, recipe, zero, steps
        ),
        _condition_trace(
            "hc_scaled_shared_direct_teacher",
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
            raise AssertionError("HC short qualification call mismatch")
        for point in condition["trace"]:
            if not all(point["digit_head_rows_unchanged"]):
                raise AssertionError("HC auxiliary qualification changed digit rows")
            if max(point["logit_gradient_reference"]["error"]["max_abs"]) >= 2e-7:
                raise AssertionError("HC repaired gradient missed FP64 tolerance")
    for point in null["trace"]:
        if _nonzero_parameter_elements(point["parameter_gradient"]) != [0] * 5:
            raise AssertionError("HC exact null produced parameter gradients")
        if point["model_digest"] != null["initial_model_digest"]:
            raise AssertionError("HC exact null moved parameters")
        if point["optimizer_moments"]["exp_avg"]["nonzero"] != [0] * 5:
            raise AssertionError("HC exact null produced Adam first moments")
        if point["optimizer_moments"]["exp_avg_sq"]["nonzero"] != [0] * 5:
            raise AssertionError("HC exact null produced Adam second moments")
    for point in active["trace"]:
        if not all(
            value > 0
            for value in _nonzero_parameter_elements(point["parameter_gradient"])
        ):
            raise AssertionError("HC direct target did not activate every packed model")
        if not all(
            after != before
            for before, after in zip(
                active["initial_model_digest"], point["model_digest"]
            )
        ):
            raise AssertionError("HC direct target did not move every packed model")
    return {
        "steps": steps,
        "conditions": conditions,
        "all_gates_pass": True,
        "scaled_exact_null_stationary": True,
        "scaled_direct_teacher_active": True,
        "aux_digit_rows_unchanged": True,
    }


def _train_epoch(
    student,
    teacher,
    optimizer,
    kind,
    keys,
    epoch,
    noise,
    recipe,
    parts,
    initial_parameters,
):
    count, examples = noise.shape[:2]
    rows = torch.arange(count, device=noise.device)[:, None]
    order = torch.stack(
        [epoch_order(key, "student_order", epoch, examples) for key in keys]
    ).to(noise.device)
    idx = output_index(kind, recipe)
    kl_total = torch.zeros(count, device=noise.device)
    loss_total = torch.zeros(count, device=noise.device)
    steps = 0
    selected_epoch = epoch + 1 in STEP_DIAGNOSTIC_EPOCHS
    step_record = None
    diagnostic_seconds = 0.0
    for start in range(0, examples, recipe.batch_size):
        batch = noise[rows, order[:, start : start + recipe.batch_size]]
        with torch.no_grad():
            targets = teacher(batch)[..., idx]
        logits = student(batch)[..., idx]
        loss, kl_sum, kl_mean = student_loss(
            logits, targets, recipe.protocol, recipe.cohort, EXACT_CANCEL_KL
        )
        optimizer.zero_grad(set_to_none=True)
        before = None
        if selected_epoch and steps == 0:
            sync(noise.device)
            diagnostic_started = time.time()
            before = capture_step_before(parts)
            sync(noise.device)
            diagnostic_seconds += time.time() - diagnostic_started
        loss.backward()
        optimizer.step()
        if before is not None:
            sync(noise.device)
            diagnostic_started = time.time()
            step_record = {
                "epoch": epoch + 1,
                "step_in_epoch": 1,
                "global_step": epoch * math.ceil(examples / recipe.batch_size) + 1,
                "batch_digest": [
                    tensor_digest(batch[row].cpu()) for row in range(count)
                ],
                "target_logits_digest": [
                    tensor_digest(targets[row].cpu()) for row in range(count)
                ],
                "student_logits_digest": [
                    tensor_digest(logits[row].detach().cpu()) for row in range(count)
                ],
                "parameter_groups": optimizer_step_diagnostics(
                    student, parts, before, initial_parameters
                ),
            }
            sync(noise.device)
            diagnostic_seconds += time.time() - diagnostic_started
        kl_total += kl_mean.detach()
        per_model = kl_sum if recipe.protocol == "upstream_batched_v1" else kl_mean
        loss_total += per_model.detach() / (
            recipe.cohort if recipe.protocol == "upstream_batched_v1" else 1
        )
        steps += 1
    if selected_epoch and step_record is None:
        raise AssertionError(
            f"missing selected optimizer diagnostic at epoch {epoch + 1}"
        )
    return (
        (kl_total / steps).cpu(),
        (loss_total / steps).cpu(),
        steps,
        step_record,
        diagnostic_seconds,
    )


def _trajectory(
    arm,
    kind,
    prepared,
    test_x,
    test_y,
    probes,
    checkpoint_dir,
    progress_path,
    source,
    deadline,
):
    recipe = prepared["recipe"]
    w0, teacher = prepared["w0"], prepared["teacher"]
    student = copy.deepcopy(w0)
    optimizer = make_optimizer(student, recipe)
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in student.named_parameters()
    }
    parts = parameter_partition(student)
    points, curve, epochs, step_records = {}, {}, [], []
    totals = {
        "training_seconds": 0.0,
        "digit_evaluation_seconds": 0.0,
        "common_evaluation_seconds": 0.0,
        "diagnostic_probe_seconds": 0.0,
        "optimizer_diagnostic_seconds": 0.0,
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
                prepared["heldout"],
                prepared["w0_heldout"],
                prepared["teacher_heldout"],
                recipe,
                kind,
            )
            sync(test_x.device)
            common_seconds = time.time() - t0
            totals["common_evaluation_seconds"] += common_seconds

            sync(test_x.device)
            t0 = time.time()
            mechanism = _probe_pair(student, probes["noise"], probes["mnist"])
            sync(test_x.device)
            probe_seconds = time.time() - t0
            totals["diagnostic_probe_seconds"] += probe_seconds

            checkpoint_name = f"{arm}-{kind}-e{epoch}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            sync(test_x.device)
            t0 = time.time()
            _save_checkpoint(
                checkpoint_path, arm, kind, epoch, student, optimizer, source
            )
            sync(test_x.device)
            checkpoint_seconds = time.time() - t0
            totals["checkpoint_seconds"] += checkpoint_seconds
            points[str(epoch)] = {
                "digit_scoring": digit,
                **common,
                "mechanism_probes": mechanism,
                "training_steps": cumulative_steps,
                "checkpoint": _file_receipt(checkpoint_path),
                "timing": {
                    "common_evaluation_seconds": common_seconds,
                    "diagnostic_probe_seconds": probe_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                },
            }
        if epoch == recipe.student_epochs:
            break

        sync(test_x.device)
        t0 = time.time()
        kl_mean, optimized_loss, steps, step_record, diagnostic_seconds = _train_epoch(
            student,
            teacher,
            optimizer,
            kind,
            recipe.model_keys,
            epoch,
            prepared["noise"],
            recipe,
            parts,
            initial_parameters,
        )
        sync(test_x.device)
        measured_seconds = time.time() - t0
        training_seconds = measured_seconds - diagnostic_seconds
        totals["training_seconds"] += training_seconds
        totals["optimizer_diagnostic_seconds"] += diagnostic_seconds
        cumulative_steps += steps
        if step_record is not None:
            step_records.append(step_record)
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
            "training_seconds": training_seconds,
            "optimizer_diagnostic_seconds": diagnostic_seconds,
            "measured_epoch_seconds": measured_seconds,
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
    if cumulative_steps != expected_steps or len(step_records) != 4:
        raise AssertionError(
            "HC trajectory schedule or sparse step diagnostics incomplete"
        )
    if kind.startswith("aux"):
        for point in points.values():
            if not all(point["digit_head_rows_unchanged"]):
                raise AssertionError(f"{arm}/{kind} changed digit rows")
    return {
        "arm": arm,
        "kind": kind,
        "target_mode": TARGET_MODE,
        "student_loss_mode": EXACT_CANCEL_KL,
        "points": points,
        "accuracy_curve": curve,
        "epochs": epochs,
        "optimizer_step_diagnostics": step_records,
        "parameter_partition": {
            "groups": sorted({part["group"] for part in parts}),
            "covered_elements": sum(
                (
                    part["parameter"].numel()
                    if part["selector"] == "all"
                    else (
                        part["parameter"][..., :1].numel()
                        if part["selector"] == "first_column"
                        else part["parameter"][..., 1:].numel()
                    )
                )
                for part in parts
            ),
            "total_elements": sum(
                parameter.numel() for parameter in student.parameters()
            ),
        },
        "steps": cumulative_steps,
        "examples_per_model": recipe.noise_samples * recipe.student_epochs,
        "packed_examples": recipe.noise_samples
        * recipe.student_epochs
        * len(recipe.model_keys),
        "timing": totals,
    }


def _first_crossing(curve, threshold=0.95):
    epochs = sorted(int(epoch) for epoch in curve)
    count = len(curve[str(epochs[0])]["accuracy"])
    return [
        next(
            (
                epoch
                for epoch in epochs
                if curve[str(epoch)]["accuracy"][row] >= threshold
            ),
            None,
        )
        for row in range(count)
    ]


def _analyze(result, prepared):
    parent_name, candidate_name = [arm["name"] for arm in result["arms"]]
    parent = result["trajectories"][parent_name]
    candidate = result["trajectories"][candidate_name]
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
        "schema": "subliminal-hc-mechanism-repair-analysis.v1",
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


def _mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def _mechanism_summary(result):
    summary = {}
    for arm, trajectories in result["trajectories"].items():
        summary[arm] = {}
        for kind, trajectory in trajectories.items():
            summary[arm][kind] = {}
            for epoch in map(str, MILESTONES):
                summary[arm][kind][epoch] = {}
                for probe_name, probe in trajectory["points"][epoch][
                    "mechanism_probes"
                ].items():
                    layers = []
                    for layer_index, layer in enumerate(probe["layers"]):
                        layers.append(
                            {
                                "layer": layer_index,
                                "carry_l2_relative_to_h_seed_mean": _mean(
                                    layer["state_terms"]["carry"]["l2_relative_to_h"]
                                ),
                                "write_l2_relative_to_h_seed_mean": _mean(
                                    layer["state_terms"]["write"]["l2_relative_to_h"]
                                ),
                                "combined_l2_relative_to_h_seed_mean": _mean(
                                    layer["state_terms"]["carry_plus_write"][
                                        "l2_relative_to_h"
                                    ]
                                ),
                                "stream_mean_carry_relative_seed_mean": _mean(
                                    layer["stream_mean_terms"]["carry"][
                                        "l2_relative_to_h_mean"
                                    ]
                                ),
                                "stream_mean_write_relative_seed_mean": _mean(
                                    layer["stream_mean_terms"]["write"][
                                        "l2_relative_to_h_mean"
                                    ]
                                ),
                                "stream_mean_alignment_seed_mean": _mean(
                                    layer["stream_mean_terms"]["carry_write_alignment"][
                                        "cosine"
                                    ]
                                ),
                                "block_output_rms_seed_mean": _mean(
                                    layer["block_output"]["rms"]
                                ),
                                "beta_unscaled_median_seed_mean": _mean(
                                    [
                                        values[3]
                                        for values in layer["beta"]["unscaled"][
                                            "values"
                                        ]
                                    ]
                                ),
                                "beta_scaled_median_seed_mean": _mean(
                                    [
                                        values[3]
                                        for values in layer["beta"]["scaled"]["values"]
                                    ]
                                ),
                                "p_minus_identity_frobenius_seed_mean": _mean(
                                    layer["carry_matrix"][
                                        "p_minus_identity_frobenius_mean"
                                    ]
                                ),
                                "stream_variance_after_seed_mean": _mean(
                                    layer["stream_structure"]["after"][
                                        "mean_variance_across_streams"
                                    ]
                                ),
                            }
                        )
                    summary[arm][kind][epoch][probe_name] = layers
    return summary


def run(input_manifest, out_dir, checkpoint_dir):
    overall_started = time.time()
    manifest_receipt = _file_receipt(input_manifest)
    frozen = json.load(open(input_manifest))
    if frozen.get("schema") != "subliminal-hc-mechanism-inputs.v1":
        raise ValueError("unexpected HC input manifest schema")
    source = git_identity()
    if source.get("commit") != frozen["source"]["commit"]:
        raise ValueError(f"source commit mismatch: {source}")
    if source.get("workload_dirty") is not False:
        raise ValueError(f"source is not frozen clean: {source}")
    for name, entry in frozen["source"]["files"].items():
        _verified_file(entry, f"source file {name}")
    _verified_file(frozen["plan"], "frozen HC plan")
    _verified_file(frozen["mnist_cache"], "MNIST tensor cache")
    if os.path.exists(checkpoint_dir):
        raise FileExistsError(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "progress.jsonl")
    if os.path.exists(progress_path):
        raise FileExistsError(progress_path)

    arms = frozen["arms"]
    if [arm["name"] for arm in arms] != ["hc", "hc-post025"]:
        raise ValueError("input manifest must contain only the frozen HC pair")
    for arm in arms:
        for name, entry in arm["receipts"].items():
            _verified_file(entry, f"{arm['name']} {name}")
        for name, entry in arm["checkpoints"].items():
            _verified_file(entry, f"{arm['name']} {name}")

    recipes = {}
    for arm in arms:
        recipe, _receipt = _recipe_from_receipt(arm["run_dir"])
        if recipe.arch != "hc" or recipe.post_scale != arm["expected_post_scale"]:
            raise ValueError(f"{arm['name']} architecture or post scale changed")
        if recipe.student_epochs != 80 or recipe.student_eval_epochs != [5, 20, 80]:
            raise ValueError("HC repair requires archived 0/5/20/80 horizon")
        if recipe.weight_decay != 0 or recipe.protocol != "per_example_v1":
            raise ValueError("HC repair requires zero-decay per-example recipes")
        recipe.student_loss_mode = EXACT_CANCEL_KL
        recipes[arm["name"]] = recipe
    _compatible(recipes["hc"], recipes["hc-post025"])

    device = pick_device(recipes["hc"])
    torch.backends.cudnn.benchmark = False
    _train_x, _train_y, test_x, test_y = load_mnist(
        os.path.expanduser(recipes["hc"].data_dir)
    )
    test_x, test_y = test_x.to(device), test_y.to(device)
    mnist_probe = test_x[:256]
    mnist_probe_labels = test_y[:256]
    prepared, provenance = {}, {}
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
        archived_w0 = _load_archived_model(arm, "pack00_w0.pt", recipe, device)
        archived_w0_digest = [
            state_digest(archived_w0, row) for row in range(len(keys))
        ]
        if archived_w0_digest != got_w0:
            raise ValueError(f"{name} archived W0 checkpoint differs")
        teacher = _load_archived_model(arm, "pack00_teacher.pt", recipe, device)
        teacher.requires_grad_(False)
        got_teacher = [state_digest(teacher, row) for row in range(len(keys))]
        if got_teacher != expected["teacher_digest"]:
            raise ValueError(f"{name} teacher differs from archived receipt")

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
        noise_probe = heldout[:, :256]
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
            raise ValueError("student epoch schedule differs across HC arms")
        teacher_score = _digit_scoring(teacher, test_x, test_y)
        archived_teacher_counts = [
            round(value * teacher_score["examples"])
            for value in expected["acc_teacher"]
        ]
        if teacher_score["correct_counts"] != archived_teacher_counts:
            raise ValueError(f"{name} teacher accuracy differs from archived result")

        legacy_models = {
            kind: _load_archived_model(arm, f"pack00_{kind}.pt", recipe, device)
            for kind in KINDS
        }
        legacy_digit_scoring = {
            kind: _digit_scoring(model, test_x, test_y)
            for kind, model in legacy_models.items()
        }
        for kind, scoring in legacy_digit_scoring.items():
            archived_counts = [
                round(value * scoring["examples"]) for value in expected[f"acc_{kind}"]
            ]
            if scoring["correct_counts"] != archived_counts:
                raise ValueError(
                    f"{name} legacy {kind} accuracy differs from archived result"
                )
        preflight_start = time.time()
        preflight_probes = {
            "w0": _probe_pair(w0, noise_probe, mnist_probe),
            "teacher": _probe_pair(teacher, noise_probe, mnist_probe),
            "legacy_aux_shared_e80": _probe_pair(
                legacy_models["aux_shared"], noise_probe, mnist_probe
            ),
            "legacy_full_shared_e80": _probe_pair(
                legacy_models["full_shared"], noise_probe, mnist_probe
            ),
        }
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
            "noise_probe": noise_probe,
        }
        provenance[name] = {
            "recipe": recipe.to_json(),
            "keys": keys,
            "w0_state_digest": got_w0,
            "teacher_state_digest_before": got_teacher,
            "noise_digest": noise_digest,
            "heldout_noise_digest": [
                tensor_digest(heldout[row].cpu()) for row in range(len(keys))
            ],
            "first_batch_digest": [
                tensor_digest(first[row].cpu()) for row in range(len(keys))
            ],
            "teacher_digit_scoring": teacher_score,
            "legacy_digit_scoring": legacy_digit_scoring,
            "direct_vs_reconstructed": {
                "first_archived_batch": _direct_diagnostic(
                    w0_first, teacher_first, recipe
                ),
                "heldout_noise": _direct_diagnostic(
                    w0_heldout, teacher_heldout, recipe
                ),
            },
            "preflight_mechanism_probes": preflight_probes,
            "preflight_probe_seconds": time.time() - preflight_start,
            "capture_forward_parity": _capture_forward_parity(w0, noise_probe),
        }
        del archived_w0, legacy_models

    parent, candidate = prepared["hc"], prepared["hc-post025"]
    if (
        provenance["hc"]["w0_state_digest"]
        != provenance["hc-post025"]["w0_state_digest"]
    ):
        raise ValueError("HC pair no longer shares W0 parameter tensors")
    functional_difference = _compare_per_model(
        parent["w0_first"], candidate["w0_first"]
    )
    if all(functional_difference["bitwise_equal"]):
        raise ValueError("HC post scales unexpectedly implement the same W0 function")
    if provenance["hc"]["noise_digest"] != provenance["hc-post025"]["noise_digest"]:
        raise ValueError("HC pair no longer shares fixed noise")
    if (
        provenance["hc"]["heldout_noise_digest"]
        != provenance["hc-post025"]["heldout_noise_digest"]
    ):
        raise ValueError("HC pair no longer shares held-out noise")
    if (
        provenance["hc"]["first_batch_digest"]
        != provenance["hc-post025"]["first_batch_digest"]
    ):
        raise ValueError("HC pair no longer shares first training batch")

    probes = {
        "noise": candidate["noise_probe"],
        "mnist": mnist_probe,
        "receipts": {
            "heldout_noise": {
                "per_seed_digest": [
                    tensor_digest(candidate["noise_probe"][row].cpu())
                    for row in range(len(recipes["hc"].model_keys))
                ],
                "examples_per_seed": 256,
            },
            "mnist_images": {
                "image_digest": tensor_digest(mnist_probe.cpu()),
                "label_digest": tensor_digest(mnist_probe_labels.cpu()),
                "examples": 256,
                "labels_not_used_for_optimization_or_checkpoint_selection": True,
            },
        },
    }
    parity = _captured_optimizer_parity(candidate)
    qualification = _short_qualification(candidate)
    preflight_seconds = time.time() - overall_started
    deadline = overall_started + frozen.get("max_end_to_end_seconds", 7200)
    trajectories = {}
    for arm in arms:
        name = arm["name"]
        trajectories[name] = {}
        arm_probes = {"noise": prepared[name]["noise_probe"], "mnist": mnist_probe}
        for kind in KINDS:
            trajectories[name][kind] = _trajectory(
                name,
                kind,
                prepared[name],
                test_x,
                test_y,
                arm_probes,
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
        for checkpoint_name, entry in arm["checkpoints"].items():
            _verified_file(entry, f"{name} {checkpoint_name} after run")

    result = {
        "schema": SCHEMA,
        "source": source,
        "environment": environment(device),
        "input_manifest": manifest_receipt,
        "arms": arms,
        "milestones": list(MILESTONES),
        "optimizer_step_diagnostic_epochs": list(STEP_DIAGNOSTIC_EPOCHS),
        "target_mode": TARGET_MODE,
        "student_loss_mode": EXACT_CANCEL_KL,
        "student_schedule": schedule,
        "probe_receipts": probes["receipts"],
        "capture_optimizer_step_parity": parity,
        "cross_arm_checks": {
            "same_keys": recipes["hc"].model_keys == recipes["hc-post025"].model_keys,
            "same_w0_parameter_digests": True,
            "same_noise_digests": True,
            "same_heldout_noise_digests": True,
            "same_first_batch_digests": True,
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
            "trajectory_diagnostic_probe_seconds": sum(
                trajectories[name][kind]["timing"]["diagnostic_probe_seconds"]
                for name in trajectories
                for kind in KINDS
            ),
            "trajectory_optimizer_diagnostic_seconds": sum(
                trajectories[name][kind]["timing"]["optimizer_diagnostic_seconds"]
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
            "capture_forward_bitwise_invariant": True,
            "capture_optimizer_step_bitwise_invariant": True,
            "full_accuracy_recorded_every_epoch": all(
                len(trajectories[name]["full_shared"]["accuracy_curve"]) == 81
                for name in trajectories
            ),
            "sparse_optimizer_steps_per_trajectory": all(
                len(trajectories[name][kind]["optimizer_step_diagnostics"]) == 4
                for name in trajectories
                for kind in KINDS
            ),
            "milestone_checkpoints": sum(
                len(trajectories[name][kind]["points"])
                for name in trajectories
                for kind in KINDS
            ),
        },
    }
    if result["validation"]["milestone_checkpoints"] != 16:
        raise AssertionError("expected exactly 16 HC milestone checkpoints")
    analysis = _analyze(result, prepared)
    result["analysis"] = analysis
    result["mechanism_summary"] = _mechanism_summary(result)
    result["timing"]["end_to_end_seconds"] = time.time() - overall_started
    _write_json(os.path.join(out_dir, "run.json"), result)
    _write_json(os.path.join(out_dir, "analysis.json"), analysis)
    _write_json(
        os.path.join(out_dir, "mechanism_summary.json"), result["mechanism_summary"]
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(
            os.path.abspath(args.input_manifest),
            os.path.abspath(args.out_dir),
            os.path.abspath(args.checkpoint_dir),
        )
    except Exception as error:
        os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
        _write_json(
            os.path.join(os.path.abspath(args.out_dir), "failure.json"),
            {
                "schema": "subliminal-hc-mechanism-failure.v1",
                "complete": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed_at_unix_seconds": time.time(),
            },
        )
        raise
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
