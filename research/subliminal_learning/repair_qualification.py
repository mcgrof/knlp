# SPDX-License-Identifier: MIT
"""Qualify the active exact-cancellation distillation path on archived inputs."""

import argparse
import copy
import hashlib
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

from .data import make_noise, tensor_digest
from .models import build, init_model
from .null_audit import (
    _archived_pack,
    _displacement_stats,
    _gradient_stats,
    _manifest_entry,
    _ordered_batch,
    _per_model_stats,
    _recipe_from_receipt,
    _sha256,
)
from .numerical_kl import EXACT_CANCEL_KL
from .run import (
    distill_targets,
    environment,
    git_identity,
    make_optimizer,
    output_index,
    pick_device,
    rows_unchanged,
    state_digest,
    student_loss,
    sync,
)


def _optimizer_digest(model, optimizer):
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        h.update(name.encode())
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            value = state[key]
            h.update(key.encode())
            if torch.is_tensor(value):
                h.update(tensor_digest(value).encode())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def _moment_stats(model, optimizer):
    rows = next(model.parameters()).shape[0]
    avg = [[] for _ in range(rows)]
    sq = [[] for _ in range(rows)]
    steps = []
    for parameter in model.parameters():
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        steps.append(int(state["step"].item()))
        for target, key in ((avg, "exp_avg"), (sq, "exp_avg_sq")):
            flat = state[key].detach().reshape(rows, -1).double()
            for row in range(rows):
                target[row].append(flat[row])

    def summarize(parts):
        joined = [torch.cat(values) for values in parts]
        return {
            "l2": [float(torch.linalg.vector_norm(value)) for value in joined],
            "nonzero": [int(torch.count_nonzero(value)) for value in joined],
            "finite": [bool(torch.isfinite(value).all()) for value in joined],
        }

    return {
        "step_values": sorted(set(steps)),
        "exp_avg": summarize(avg),
        "exp_avg_sq": summarize(sq),
    }


def _fp64_kl_mean(target_logits, student_logits):
    target = F.softmax(target_logits.double(), dim=-1)
    rows = F.kl_div(
        F.log_softmax(student_logits.double(), dim=-1),
        target,
        reduction="none",
    ).sum(-1)
    return rows.mean(1).cpu().tolist()


def _logit_reference(student_logits, target_logits, recipe):
    logits = student_logits.detach().requires_grad_(True)
    loss, _, _ = student_loss(
        logits,
        target_logits.detach(),
        recipe.protocol,
        recipe.cohort,
        EXACT_CANCEL_KL,
    )
    loss.backward()
    scale = (
        1 / recipe.cohort
        if recipe.protocol == "upstream_batched_v1"
        else 1 / student_logits.shape[1]
    )
    reference = (
        F.softmax(student_logits.detach().double(), -1)
        - F.softmax(target_logits.detach().double(), -1)
    ) * scale
    error = logits.grad.double() - reference
    return {
        "repaired": _per_model_stats(logits.grad),
        "fp64_reference": _per_model_stats(reference),
        "error": _per_model_stats(error),
    }


def _condition_trace(name, initial, teacher, w0, noise, keys, recipe, scale, steps):
    student = copy.deepcopy(initial)
    optimizer = make_optimizer(student, recipe)
    initial_parameters = {
        key: value.detach().cpu().clone() for key, value in student.named_parameters()
    }
    initial_digests = [state_digest(student, row) for row in range(len(keys))]
    out_idx = output_index("aux_shared", recipe)
    trace = []
    calls = {"loss": 0, "backward": 0, "optimizer_step": 0}

    for step in range(steps):
        batch = _ordered_batch(noise, keys, recipe.batch_size, step)
        with torch.no_grad():
            targets = distill_targets(teacher, w0, scale, batch)[..., out_idx]
        logits = student(batch)[..., out_idx]
        before_fit = _fp64_kl_mean(targets, logits.detach())
        reference = _logit_reference(logits.detach(), targets, recipe)
        loss, _kl_sum, kl_mean = student_loss(
            logits,
            targets,
            recipe.protocol,
            recipe.cohort,
            EXACT_CANCEL_KL,
        )
        calls["loss"] += 1
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        calls["backward"] += 1
        gradients = _gradient_stats(student)
        optimizer.step()
        calls["optimizer_step"] += 1
        sync(noise.device)
        with torch.no_grad():
            after_logits = student(batch)[..., out_idx]
        trace.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "kl_mean_fp32": kl_mean.detach().cpu().tolist(),
                "target_fit_fp64_before": before_fit,
                "target_fit_fp64_after": _fp64_kl_mean(targets, after_logits),
                "logit_gradient_reference": reference,
                "parameter_gradient": gradients,
                "displacement_from_initial": _displacement_stats(
                    student, initial_parameters
                ),
                "model_digest": [
                    state_digest(student, row) for row in range(len(keys))
                ],
                "optimizer_digest": _optimizer_digest(student, optimizer),
                "optimizer_moments": _moment_stats(student, optimizer),
                "digit_head_rows_unchanged": rows_unchanged(
                    student, initial, list(range(10))
                ).tolist(),
            }
        )

    return {
        "name": name,
        "target_mode": "direct_teacher_v1" if scale is None else "interpolated_v1",
        "target_scale": None if scale is None else scale.detach().cpu().tolist(),
        "calls": calls,
        "initial_model_digest": initial_digests,
        "trace": trace,
    }


def _summary_nonzero(group_stats, key="nonzero"):
    groups = group_stats.values()
    return [sum(group["summary"][key][row] for group in groups) for row in range(5)]


def _validate(result, steps):
    conditions = {item["name"]: item for item in result["conditions"]}
    for condition in conditions.values():
        assert condition["calls"] == {
            "loss": steps,
            "backward": steps,
            "optimizer_step": steps,
        }
        for point in condition["trace"]:
            assert all(point["digit_head_rows_unchanged"])
            assert max(point["logit_gradient_reference"]["error"]["max_abs"]) < 2e-7

    zero = conditions["shared_zero"]
    for point in zero["trace"]:
        assert _summary_nonzero(point["parameter_gradient"]) == [0] * 5
        assert point["model_digest"] == zero["initial_model_digest"]
        assert point["optimizer_moments"]["exp_avg"]["nonzero"] == [0] * 5
        assert point["optimizer_moments"]["exp_avg_sq"]["nonzero"] == [0] * 5

    for name in ("different_init_zero", "shared_one"):
        condition = conditions[name]
        for point in condition["trace"]:
            assert all(
                value > 0 for value in _summary_nonzero(point["parameter_gradient"])
            )
            assert all(
                after != before
                for before, after in zip(
                    condition["initial_model_digest"], point["model_digest"]
                )
            )
            assert all(
                value > 0 for value in point["optimizer_moments"]["exp_avg"]["nonzero"]
            )
            assert all(
                value > 0
                for value in point["optimizer_moments"]["exp_avg_sq"]["nonzero"]
            )
    return {
        "all_gates_pass": True,
        "shared_zero_exact_stationarity": True,
        "different_init_zero_learns": True,
        "shared_nonzero_learns": True,
        "digit_head_rows_unchanged": True,
    }


def run_qualification(run_dir, teacher_checkpoint, steps=4):
    recipe, receipt = _recipe_from_receipt(run_dir)
    pack = _archived_pack(run_dir)
    keys = pack["keys"]
    if keys != recipe.model_keys:
        raise ValueError("receipt keys differ from recipe model keys")
    if recipe.target_scale != 0 or recipe.target_match_kl != 0:
        raise ValueError("qualification requires the archived target-zero receipt")
    if recipe.weight_decay != 0:
        raise ValueError("qualification requires zero weight decay")
    if steps < 1 or steps > math.ceil(recipe.noise_samples / recipe.batch_size):
        raise ValueError("steps must fit within the first archived epoch")
    recipe.student_loss_mode = EXACT_CANCEL_KL

    device = pick_device(recipe)
    torch.backends.cudnn.benchmark = False
    count = len(keys)
    w0 = build(recipe, count).to(device)
    init_model(w0, keys, "init")
    w0_other = build(recipe, count).to(device)
    init_model(w0_other, keys, "init_other")
    expected = pack["results"]
    got_w0 = [state_digest(w0, row) for row in range(count)]
    got_other = [state_digest(w0_other, row) for row in range(count)]
    if got_w0 != expected["w0_digest"]:
        raise ValueError("reconstructed W0 digest differs from archived run")
    if got_other != expected["w0_other_digest"]:
        raise ValueError("reconstructed W0_other digest differs from archived run")

    teacher = build(recipe, count).to(device)
    teacher.load_state_dict(
        torch.load(teacher_checkpoint, map_location=device, weights_only=True)
    )
    teacher.eval().requires_grad_(False)
    teacher_digests = [state_digest(teacher, row) for row in range(count)]
    if teacher_digests != expected["teacher_digest"]:
        raise ValueError("teacher state digest differs from archived run")
    manifest = _manifest_entry(run_dir, os.path.basename(teacher_checkpoint))
    checkpoint_sha = _sha256(teacher_checkpoint)
    if checkpoint_sha != manifest["sha256"]:
        raise ValueError("teacher checkpoint byte digest differs from manifest")

    noise_cpu = torch.stack([make_noise(key, recipe.noise_samples) for key in keys])
    noise_digests = [tensor_digest(noise_cpu[row]) for row in range(count)]
    if noise_digests != expected["noise_digest"]:
        raise ValueError("reconstructed noise digest differs from archived run")
    noise = noise_cpu.to(device)
    del noise_cpu

    zero = torch.zeros(count, device=device)
    one = torch.ones(count, device=device)
    result = {
        "schema": "subliminal-repair-qualification.v1",
        "source": {
            "git": git_identity(),
            "archived_git": receipt.get("git"),
            "run_dir": os.path.abspath(run_dir),
            "teacher_checkpoint": os.path.abspath(teacher_checkpoint),
            "teacher_checkpoint_sha256": checkpoint_sha,
            "teacher_checkpoint_bytes": os.path.getsize(teacher_checkpoint),
        },
        "environment": environment(device),
        "recipe": recipe.to_json(),
        "provenance_checks": {
            "keys": keys,
            "w0_digest": got_w0,
            "w0_other_digest": got_other,
            "teacher_digest": teacher_digests,
            "noise_digest": noise_digests,
        },
        "conditions": [
            _condition_trace(
                "shared_zero", w0, teacher, w0, noise, keys, recipe, zero, steps
            ),
            _condition_trace(
                "different_init_zero",
                w0_other,
                teacher,
                w0,
                noise,
                keys,
                recipe,
                zero,
                steps,
            ),
            _condition_trace(
                "shared_one", w0, teacher, w0, noise, keys, recipe, one, steps
            ),
        ],
    }
    result["qualification"] = _validate(result, steps)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_qualification(args.run_dir, args.teacher_checkpoint, args.steps)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
