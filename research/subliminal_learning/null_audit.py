# SPDX-License-Identifier: MIT
"""Audit the target-scale-zero distillation path before a production step.

This command reconstructs W0 and the production noise stream from an archived
run receipt, loads the run's frozen teacher, and reproduces the first few
student batches without changing the archived run. It is intentionally a
diagnostic sidecar: the normal runner and historical receipts stay unchanged.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import fields

import torch
import torch.nn.functional as F

from .data import epoch_order, make_noise, tensor_digest
from .models import build, init_model
from .recipe import Recipe
from .run import (
    distill_targets,
    environment,
    git_identity,
    make_optimizer,
    output_index,
    pick_device,
    state_digest,
    student_loss,
    sync,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _recipe_from_receipt(run_dir):
    with open(os.path.join(run_dir, "recipe.json")) as f:
        receipt = json.load(f)
    raw = receipt["recipe"]
    allowed = {f.name for f in fields(Recipe)}
    values = {name: raw[name] for name in allowed if name in raw}
    if "adam_betas" in values:
        values["adam_betas"] = tuple(values["adam_betas"])
    return Recipe(**values), receipt


def _archived_pack(run_dir):
    with open(os.path.join(run_dir, "results.json")) as f:
        results = json.load(f)
    if sorted(results) != ["0"]:
        raise ValueError("null audit currently requires one archived pack")
    pack = results["0"]
    if pack["keys"] == []:
        raise ValueError("archived pack has no model keys")
    return pack


def _manifest_entry(run_dir, basename):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    entries = manifest.get("files", manifest)
    if basename not in entries:
        raise ValueError(f"{basename} is absent from archived manifest")
    return entries[basename]


def _tensor_comparison(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {
            "equal": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
        }
    d = (a.detach() - b.detach()).abs()
    return {
        "equal": bool(torch.equal(a.detach(), b.detach())),
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(d.double()).item()),
    }


def _per_model_stats(tensor):
    if tensor is None:
        return None
    x = tensor.detach().reshape(tensor.shape[0], -1).double()
    return {
        "max_abs": x.abs().amax(1).cpu().tolist(),
        "l2": torch.linalg.vector_norm(x, dim=1).cpu().tolist(),
        "nonzero": (x != 0).sum(1).cpu().tolist(),
        "finite": torch.isfinite(x).all(1).cpu().tolist(),
    }


def _parameter_groups(model):
    groups = {}
    for name, prm in model.named_parameters():
        group = name.split(".", 1)[0]
        groups.setdefault(group, []).append((name, prm))
    return groups


def _group_tensor_stats(model, value):
    """Summarize parameter-shaped tensors per packed model and top group."""
    out = {}
    for group, named in _parameter_groups(model).items():
        rows = [[] for _ in range(named[0][1].shape[0])]
        names = {}
        for name, prm in named:
            tensor = value(name, prm)
            names[name] = _per_model_stats(tensor)
            if tensor is None:
                continue
            flat = tensor.detach().reshape(tensor.shape[0], -1).double()
            for model_i in range(flat.shape[0]):
                rows[model_i].append(flat[model_i])
        joined = [torch.cat(parts) for parts in rows]
        out[group] = {
            "summary": {
                "max_abs": [float(x.abs().max().item()) for x in joined],
                "l2": [float(torch.linalg.vector_norm(x).item()) for x in joined],
                "nonzero": [int((x != 0).sum().item()) for x in joined],
                "finite": [bool(torch.isfinite(x).all().item()) for x in joined],
            },
            "parameters": names,
        }
    return out


def _gradient_stats(model):
    return _group_tensor_stats(model, lambda _name, prm: prm.grad)


def _displacement_stats(model, initial):
    return _group_tensor_stats(
        model, lambda name, prm: prm.detach() - initial[name].to(prm.device)
    )


def _loss_and_logit_gradient(student_logits, target_logits, recipe, dtype):
    logits = student_logits.detach().to(dtype).requires_grad_(True)
    targets = target_logits.detach().to(dtype)
    loss, kl_sum, kl_mean = student_loss(
        logits, targets, recipe.protocol, recipe.cohort
    )
    loss.backward()
    return {
        "dtype": str(dtype),
        "loss": float(loss.detach().cpu()),
        "kl_sum": kl_sum.detach().cpu().tolist(),
        "kl_mean": kl_mean.detach().cpu().tolist(),
        "d_loss_d_logits": _per_model_stats(logits.grad),
    }


def _ordered_batch(noise, keys, batch_size, step):
    p, n = noise.shape[:2]
    order = torch.stack([epoch_order(k, "student_order", 0, n) for k in keys])
    start = step * batch_size
    if start >= n:
        raise ValueError(f"step {step} is outside an epoch with {n} examples")
    idx = order[:, start : start + batch_size].to(noise.device)
    rows = torch.arange(p, device=noise.device)[:, None]
    return noise[rows, idx]


def _optimizer_trace(kind, initial_model, teacher, w0, noise, keys, recipe, steps):
    student = copy.deepcopy(initial_model)
    if kind == "adam":
        opt = make_optimizer(student, recipe)
    elif kind == "sgd":
        opt = torch.optim.SGD(student.parameters(), lr=recipe.lr)
    else:
        raise ValueError(kind)
    initial = {
        name: prm.detach().cpu().clone() for name, prm in student.named_parameters()
    }
    out = []
    out_idx = output_index("aux_shared", recipe)
    a = torch.zeros(len(keys), device=noise.device)
    for step in range(steps):
        bx = _ordered_batch(noise, keys, recipe.batch_size, step)
        with torch.no_grad():
            targets = distill_targets(teacher, w0, a, bx)[..., out_idx]
        logits = student(bx)[..., out_idx]
        loss, _kl_sum, kl_mean = student_loss(
            logits, targets, recipe.protocol, recipe.cohort
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = _gradient_stats(student)
        opt.step()
        sync(noise.device)
        out.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "kl_mean": kl_mean.detach().cpu().tolist(),
                "gradient": grad,
                "displacement_from_w0": _displacement_stats(student, initial),
                "state_digest": [state_digest(student, m) for m in range(len(keys))],
            }
        )
    return out


def run_audit(run_dir, teacher_checkpoint, steps=4):
    recipe, receipt = _recipe_from_receipt(run_dir)
    pack = _archived_pack(run_dir)
    keys = pack["keys"]
    if keys != recipe.model_keys:
        raise ValueError(f"receipt keys differ: {keys} != {recipe.model_keys}")
    if recipe.target_scale != 0 or recipe.target_match_kl != 0:
        raise ValueError("null audit requires archived target_scale=0")
    if recipe.weight_decay != 0:
        raise ValueError("null audit requires zero weight decay")
    if steps < 1 or steps > math.ceil(recipe.noise_samples / recipe.batch_size):
        raise ValueError("steps must fit within the first student epoch")

    device = pick_device(recipe)
    torch.backends.cudnn.benchmark = False
    p = len(keys)
    w0 = build(recipe, p).to(device)
    init_model(w0, keys, "init")
    expected_w0 = pack["results"]["w0_digest"]
    got_w0 = [state_digest(w0, m) for m in range(p)]
    if got_w0 != expected_w0:
        raise ValueError("reconstructed W0 digest differs from archived run")

    teacher = build(recipe, p).to(device)
    teacher.load_state_dict(
        torch.load(teacher_checkpoint, map_location=device, weights_only=True)
    )
    teacher.eval()
    teacher.requires_grad_(False)
    expected_teacher = pack["results"]["teacher_digest"]
    got_teacher = [state_digest(teacher, m) for m in range(p)]
    if got_teacher != expected_teacher:
        raise ValueError("teacher state digest differs from archived run")

    manifest = _manifest_entry(run_dir, os.path.basename(teacher_checkpoint))
    checkpoint_sha = _sha256(teacher_checkpoint)
    if checkpoint_sha != manifest["sha256"]:
        raise ValueError("teacher checkpoint byte digest differs from manifest")

    noise_cpu = torch.stack([make_noise(k, recipe.noise_samples) for k in keys])
    noise_digests = [tensor_digest(noise_cpu[m]) for m in range(p)]
    if noise_digests != pack["results"]["noise_digest"]:
        raise ValueError("reconstructed noise digest differs from archived run")
    noise = noise_cpu.to(device)
    del noise_cpu

    student = copy.deepcopy(w0)
    bx = _ordered_batch(noise, keys, recipe.batch_size, 0)
    out_idx = output_index("aux_shared", recipe)
    a = torch.zeros(p, device=device)
    with torch.no_grad():
        w0_logits_1 = w0(bx)[..., out_idx]
        w0_logits_2 = w0(bx)[..., out_idx]
        student_logits_1 = student(bx)[..., out_idx]
        student_logits_2 = student(bx)[..., out_idx]
        teacher_logits = teacher(bx)
        attenuated = distill_targets(teacher, w0, a, bx)[..., out_idx]
        target_probs = F.softmax(attenuated, dim=-1)
        student_probs = F.softmax(student_logits_1, dim=-1)

    before_backward = [state_digest(student, m) for m in range(p)]
    loss, _kl_sum, _kl_mean = student_loss(
        student(bx)[..., out_idx], attenuated, recipe.protocol, recipe.cohort
    )
    student.zero_grad(set_to_none=True)
    loss.backward()
    after_backward = [state_digest(student, m) for m in range(p)]

    return {
        "schema": "subliminal-null-audit.v1",
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
            "teacher_digest": got_teacher,
            "noise_digest": noise_digests,
            "first_batch_digest": [tensor_digest(bx[m].cpu()) for m in range(p)],
        },
        "forward": {
            "w0_repeated": _tensor_comparison(w0_logits_1, w0_logits_2),
            "student_repeated": _tensor_comparison(student_logits_1, student_logits_2),
            "w0_vs_student": _tensor_comparison(w0_logits_1, student_logits_1),
            "w0_vs_attenuated_target": _tensor_comparison(w0_logits_1, attenuated),
            "target_vs_student_probabilities": _tensor_comparison(
                target_probs, student_probs
            ),
            "teacher_logits": _per_model_stats(teacher_logits[..., out_idx]),
            "w0_logits": _per_model_stats(w0_logits_1),
        },
        "loss_gradient": {
            "fp32_saved_logits": _loss_and_logit_gradient(
                student_logits_1, attenuated, recipe, torch.float32
            ),
            "fp64_saved_logits": _loss_and_logit_gradient(
                student_logits_1, attenuated, recipe, torch.float64
            ),
            "production_parameter_gradient": _gradient_stats(student),
            "backward_changed_parameters": before_backward != after_backward,
        },
        "optimizer_traces": {
            "sgd": _optimizer_trace("sgd", w0, teacher, w0, noise, keys, recipe, steps),
            "adam": _optimizer_trace(
                "adam", w0, teacher, w0, noise, keys, recipe, steps
            ),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="archived null run receipt")
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args(argv)
    result = run_audit(args.run_dir, args.teacher_checkpoint, args.steps)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
