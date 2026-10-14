# SPDX-License-Identifier: MIT
"""Run the bounded legacy-versus-repaired residual distillation bridge."""

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import torch

from .data import load_mnist, make_noise, tensor_digest
from .models import build, init_model
from .null_audit import (
    _displacement_stats,
    _group_tensor_stats,
    _per_model_stats,
    _recipe_from_receipt,
    _sha256,
)
from .numerical_kl import EXACT_CANCEL_KL
from .repair_qualification import _fp64_kl_mean, _moment_stats, _optimizer_digest
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


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def _write_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _manifest(run_dir):
    value = _load_json(os.path.join(run_dir, "manifest.json"))
    return value.get("files", value)


def _pack(run_dir):
    value = _load_json(os.path.join(run_dir, "results.json"))
    if sorted(value) != ["0"]:
        raise ValueError(f"{run_dir} does not contain exactly one pack")
    return value["0"]


def _file_receipt(path):
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "sha256": _sha256(path),
    }


@torch.no_grad()
def _digit_scoring(model, test_x, test_y, chunk=2000):
    predictions = []
    for start in range(0, test_x.shape[0], chunk):
        predictions.append(model(test_x[start : start + chunk])[..., :10].argmax(-1))
    predicted = torch.cat(predictions, dim=1)
    correct = [
        int((predicted[row] == test_y).sum()) for row in range(predicted.shape[0])
    ]
    return {
        "examples": int(test_y.numel()),
        "correct_counts": correct,
        "accuracy": [value / test_y.numel() for value in correct],
        "predicted_digit_counts": [
            torch.bincount(predicted[row], minlength=10).cpu().tolist()
            for row in range(predicted.shape[0])
        ],
    }


def _group_summaries(groups):
    return {name: value["summary"] for name, value in groups.items()}


def _parameter_values(model):
    return _group_summaries(
        _group_tensor_stats(model, lambda _name, parameter: parameter.detach())
    )


def _parameter_displacement(model, initial):
    return _group_summaries(_displacement_stats(model, initial))


def _checkpoint(path, condition, kind, epoch, model, optimizer, source):
    torch.save(
        {
            "schema": "subliminal-repair-bridge-checkpoint.v1",
            "condition": condition,
            "kind": kind,
            "epoch": epoch,
            "source": source,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _evaluate(
    model,
    initial,
    initial_parameters,
    optimizer,
    test_x,
    test_y,
    heldout,
    w0_heldout,
    target_heldout,
    recipe,
    kind,
):
    idx = output_index(kind, recipe)
    with torch.no_grad():
        student_heldout = logits_on(model, heldout)
    return {
        "digit_scoring": _digit_scoring(model, test_x, test_y),
        "target_fit_kl_fp64": _fp64_kl_mean(
            target_heldout[..., idx], student_heldout[..., idx]
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


def _load_legacy_model(run_dir, kind, recipe, device):
    manifest = _manifest(run_dir)
    name = f"pack00_{kind}.pt"
    entry = manifest[name]
    path = entry["path"]
    if _sha256(path) != entry["sha256"] or os.path.getsize(path) != entry["bytes"]:
        raise ValueError(f"legacy checkpoint differs from manifest: {path}")
    model = build(recipe, len(recipe.model_keys)).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model, {"name": name, **entry}


def _legacy_accuracy(run_dir, kind):
    pack = _pack(run_dir)
    result = pack["results"]
    out = {"80": result[f"acc_{kind}"]}
    for epoch in (5, 20):
        out[str(epoch)] = result["milestones"][str(epoch)][f"acc_{kind}"]
    out["0"] = result["acc_init"]
    return out


def _validate_compatible(zero_recipe, other_recipe, label):
    excluded = {
        "target_scale",
        "target_match_kl",
        "output_dir",
        "checkpoint_dir",
        "run_name",
        "max_seconds",
        "student_loss_mode",
    }
    left = zero_recipe.to_json()
    right = other_recipe.to_json()
    differences = {
        key: [left.get(key), right.get(key)]
        for key in sorted((set(left) | set(right)) - excluded)
        if left.get(key) != right.get(key)
    }
    if differences:
        raise ValueError(f"{label} recipe mismatch: {differences}")


def run_bridge(
    zero_run,
    ordinary_run,
    matched_run,
    teacher_checkpoint,
    out_dir,
    checkpoint_dir,
):
    zero_recipe, zero_receipt = _recipe_from_receipt(zero_run)
    ordinary_recipe, _ = _recipe_from_receipt(ordinary_run)
    matched_recipe, _ = _recipe_from_receipt(matched_run)
    _validate_compatible(zero_recipe, ordinary_recipe, "ordinary")
    _validate_compatible(zero_recipe, matched_recipe, "matched")
    if zero_recipe.student_epochs != 80 or zero_recipe.student_eval_epochs != [
        5,
        20,
        80,
    ]:
        raise ValueError("bridge requires the archived 0/5/20/80 horizon")
    if zero_recipe.weight_decay != 0 or zero_recipe.protocol != "per_example_v1":
        raise ValueError("bridge requires archived zero-decay per-example recipe")
    zero_recipe.student_loss_mode = EXACT_CANCEL_KL
    recipe = zero_recipe
    keys = recipe.model_keys
    archived = _pack(zero_run)
    if archived["keys"] != keys:
        raise ValueError("archived keys differ from recipe")
    matched_pack = _pack(matched_run)
    matched_scale = matched_pack["results"]["target_a"]

    os.makedirs(out_dir, exist_ok=False)
    os.makedirs(checkpoint_dir, exist_ok=False)
    progress_path = os.path.join(out_dir, "progress.jsonl")
    device = pick_device(recipe)
    torch.backends.cudnn.benchmark = False
    count = len(keys)
    w0 = build(recipe, count).to(device)
    init_model(w0, keys, "init")
    w0.eval()
    expected = archived["results"]
    w0_digests = [state_digest(w0, row) for row in range(count)]
    if w0_digests != expected["w0_digest"]:
        raise ValueError("reconstructed W0 differs from archived receipt")

    teacher = build(recipe, count).to(device)
    teacher.load_state_dict(
        torch.load(teacher_checkpoint, map_location=device, weights_only=True)
    )
    teacher.eval().requires_grad_(False)
    teacher_digests = [state_digest(teacher, row) for row in range(count)]
    if teacher_digests != expected["teacher_digest"]:
        raise ValueError("teacher state differs from archived receipt")
    teacher_sha = _sha256(teacher_checkpoint)
    if teacher_sha != _manifest(zero_run)["pack00_teacher.pt"]["sha256"]:
        raise ValueError("teacher checkpoint byte digest differs from manifest")

    noise_cpu = torch.stack([make_noise(key, recipe.noise_samples) for key in keys])
    noise_digests = [tensor_digest(noise_cpu[row]) for row in range(count)]
    if noise_digests != expected["noise_digest"]:
        raise ValueError("noise differs from archived receipt")
    noise = noise_cpu.to(device)
    del noise_cpu
    heldout = torch.stack(
        [make_noise(key, recipe.heldout_noise, stream="heldout_noise") for key in keys]
    ).to(device)
    data_dir = os.path.expanduser(recipe.data_dir)
    _train_x, _train_y, test_x, test_y = load_mnist(data_dir)
    test_x, test_y = test_x.to(device), test_y.to(device)

    with torch.no_grad():
        w0_heldout = logits_on(w0, heldout)
        teacher_heldout = logits_on(teacher, heldout)

    scales = {
        "zero": torch.zeros(count, device=device),
        "one": torch.ones(count, device=device),
        "matched": torch.tensor(matched_scale, device=device),
    }
    legacy_runs = {"zero": zero_run, "one": ordinary_run, "matched": matched_run}
    source = git_identity()
    result = {
        "schema": "subliminal-repair-bridge.v1",
        "source": source,
        "environment": environment(device),
        "recipe": recipe.to_json(),
        "milestones": list(MILESTONES),
        "provenance_checks": {
            "keys": keys,
            "w0_digest": w0_digests,
            "teacher_digest": teacher_digests,
            "noise_digest": noise_digests,
            "teacher_checkpoint": _file_receipt(teacher_checkpoint),
            "archived_receipts": {
                label: {
                    name: _file_receipt(os.path.join(path, name))
                    for name in (
                        "recipe.json",
                        "results.json",
                        "status.json",
                        "manifest.json",
                    )
                }
                for label, path in legacy_runs.items()
            },
        },
        "target_scales": {name: value.cpu().tolist() for name, value in scales.items()},
        "initial": {
            "digit_scoring": _digit_scoring(w0, test_x, test_y),
            "parameter_values": _parameter_values(w0),
        },
        "repaired": {},
        "legacy": {},
        "checkpoint_manifest": {},
    }
    started = time.time()
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in w0.named_parameters()
    }
    for condition, scale in scales.items():
        target_heldout = w0_heldout + scale.view(-1, 1, 1) * (
            teacher_heldout - w0_heldout
        )
        result["repaired"][condition] = {}
        for kind in KINDS:
            student = copy.deepcopy(w0)
            optimizer = make_optimizer(student, recipe)
            points = {}
            cumulative_steps = 0
            condition_start = time.time()
            for epoch in range(0, recipe.student_epochs + 1):
                if epoch in MILESTONES:
                    checkpoint_name = f"{condition}-{kind}-e{epoch}.pt"
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                    _checkpoint(
                        checkpoint_path,
                        condition,
                        kind,
                        epoch,
                        student,
                        optimizer,
                        source,
                    )
                    points[str(epoch)] = {
                        **_evaluate(
                            student,
                            w0,
                            initial_parameters,
                            optimizer,
                            test_x,
                            test_y,
                            heldout,
                            w0_heldout,
                            target_heldout,
                            recipe,
                            kind,
                        ),
                        "training_steps": cumulative_steps,
                        "checkpoint": checkpoint_name,
                    }
                if epoch == recipe.student_epochs:
                    break
                sync(device)
                epoch_start = time.time()
                kl_mean, optimized_loss, steps = train_student_epoch(
                    student,
                    teacher,
                    optimizer,
                    kind,
                    keys,
                    epoch,
                    noise,
                    recipe,
                    w0,
                    scale,
                )
                sync(device)
                cumulative_steps += steps
                row = {
                    "condition": condition,
                    "kind": kind,
                    "epoch": epoch + 1,
                    "steps": steps,
                    "cumulative_steps": cumulative_steps,
                    "kl_mean": kl_mean.tolist(),
                    "optimized_loss": optimized_loss.tolist(),
                    "epoch_seconds": time.time() - epoch_start,
                }
                with open(progress_path, "a") as progress:
                    progress.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            result["repaired"][condition][kind] = {
                "points": points,
                "elapsed_s": time.time() - condition_start,
            }

    for condition, run_dir in legacy_runs.items():
        scale = scales[condition]
        target_heldout = w0_heldout + scale.view(-1, 1, 1) * (
            teacher_heldout - w0_heldout
        )
        result["legacy"][condition] = {}
        for kind in KINDS:
            model, checkpoint = _load_legacy_model(run_dir, kind, recipe, device)
            final = _evaluate(
                model,
                w0,
                initial_parameters,
                make_optimizer(model, recipe),
                test_x,
                test_y,
                heldout,
                w0_heldout,
                target_heldout,
                recipe,
                kind,
            )
            result["legacy"][condition][kind] = {
                "accuracy_by_epoch": _legacy_accuracy(run_dir, kind),
                "final_common_evaluation": final,
                "checkpoint": checkpoint,
            }

    for name in sorted(os.listdir(checkpoint_dir)):
        path = os.path.join(checkpoint_dir, name)
        result["checkpoint_manifest"][name] = _file_receipt(path)
    result["elapsed_s"] = time.time() - started
    result["validation"] = _validate_bridge(result)
    _write_json(os.path.join(out_dir, "bridge.json"), result)
    return result


def _validate_bridge(result):
    initial = result["initial"]["digit_scoring"]["correct_counts"]
    zero = result["repaired"]["zero"]
    for kind in KINDS:
        points = zero[kind]["points"]
        for epoch in map(str, MILESTONES):
            assert points[epoch]["digit_scoring"]["correct_counts"] == initial
            assert points[epoch]["model_digest"] == points["0"]["model_digest"]
        moments = points["80"]["optimizer_moments"]
        assert moments["exp_avg"]["nonzero"] == [0] * 5
        assert moments["exp_avg_sq"]["nonzero"] == [0] * 5
    for condition in ("one", "matched"):
        for kind in KINDS:
            points = result["repaired"][condition][kind]["points"]
            assert points["80"]["model_digest"] != points["0"]["model_digest"]
            if kind.startswith("aux"):
                assert all(points["80"]["digit_head_rows_unchanged"])
    assert len(result["checkpoint_manifest"]) == 24
    return {
        "all_gates_pass": True,
        "active_zero_stationary_through_80": True,
        "nonzero_conditions_move": True,
        "aux_digit_head_rows_unchanged": True,
        "checkpoint_count": 24,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-run", required=True)
    parser.add_argument("--ordinary-run", required=True)
    parser.add_argument("--matched-run", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args(argv)
    result = run_bridge(
        args.zero_run,
        args.ordinary_run,
        args.matched_run,
        args.teacher_checkpoint,
        args.out_dir,
        args.checkpoint_dir,
    )
    print(json.dumps({"complete": True, "elapsed_s": result["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
