#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the literal no-loss baseline for an archived target-zero run.

This sidecar reconstructs the archived W0 models, scores their digit outputs,
then executes the same first noise forwards without constructing a loss or an
optimizer.  It proves what the zero-signal reference looks like when no
training operation can move the model.
"""

import argparse
import copy
import json
import os
import sys

import torch

from .data import load_mnist, make_noise, tensor_digest
from .models import build, init_model
from .null_audit import _archived_pack, _ordered_batch, _recipe_from_receipt
from .recipe import NUM_DIGITS, resolve_path
from .run import environment, git_identity, pick_device, state_digest, sync


@torch.no_grad()
def _digit_logits(model, test_x, chunk=2000):
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, test_x.shape[0], chunk):
        out.append(model(test_x[start : start + chunk])[..., :NUM_DIGITS].cpu())
    model.train(was_training)
    return torch.cat(out, dim=1)


def _digit_summary(logits, labels):
    predictions = logits.argmax(-1)
    labels = labels.cpu()
    correct_counts = [
        int((predictions[m] == labels).sum().item())
        for m in range(predictions.shape[0])
    ]
    return {
        "accuracy": [correct / labels.numel() for correct in correct_counts],
        "correct_counts": correct_counts,
        "predicted_digit_counts": [
            torch.bincount(predictions[m], minlength=NUM_DIGITS).tolist()
            for m in range(predictions.shape[0])
        ],
        "true_digit_counts": torch.bincount(labels, minlength=NUM_DIGITS).tolist(),
        "examples": int(labels.numel()),
        "uniform_random_accuracy": 1.0 / NUM_DIGITS,
    }


def run_baseline(run_dir, steps=4):
    recipe, receipt = _recipe_from_receipt(run_dir)
    pack = _archived_pack(run_dir)
    keys = pack["keys"]
    if keys != recipe.model_keys:
        raise ValueError(f"receipt keys differ: {keys} != {recipe.model_keys}")
    if recipe.target_scale != 0 or recipe.target_match_kl != 0:
        raise ValueError("no-loss baseline requires archived target_scale=0")
    if steps < 1:
        raise ValueError("steps must be positive")

    device = pick_device(recipe)
    torch.backends.cudnn.benchmark = False
    w0 = build(recipe, len(keys)).to(device)
    init_model(w0, keys, "init")
    expected_w0 = pack["results"]["w0_digest"]
    initial_digest = [state_digest(w0, m) for m in range(len(keys))]
    if initial_digest != expected_w0:
        raise ValueError("reconstructed W0 digest differs from archived run")

    data_dir = resolve_path(recipe.data_dir)
    _train_x, _train_y, test_x, test_y = load_mnist(data_dir)
    if recipe.test_limit:
        test_x, test_y = test_x[: recipe.test_limit], test_y[: recipe.test_limit]
    test_x = test_x.to(device)
    initial_logits = _digit_logits(w0, test_x)
    initial = _digit_summary(initial_logits, test_y)
    archived_accuracy = pack["results"]["acc_init"]
    archived_correct_counts = [
        int(round(value * initial["examples"])) for value in archived_accuracy
    ]
    if initial["correct_counts"] != archived_correct_counts:
        raise ValueError("reconstructed initial accuracy differs from archived run")

    noise_cpu = torch.stack([make_noise(k, recipe.noise_samples) for k in keys])
    noise_digest = [tensor_digest(noise_cpu[m]) for m in range(len(keys))]
    if noise_digest != pack["results"]["noise_digest"]:
        raise ValueError("reconstructed noise digest differs from archived run")
    noise = noise_cpu.to(device)
    del noise_cpu

    student = copy.deepcopy(w0)
    digest_after_forward = []
    with torch.no_grad():
        for step in range(steps):
            student(_ordered_batch(noise, keys, recipe.batch_size, step))
            sync(device)
            digest_after_forward.append(
                [state_digest(student, m) for m in range(len(keys))]
            )

    final_logits = _digit_logits(student, test_x)
    final = _digit_summary(final_logits, test_y)
    final_digest = [state_digest(student, m) for m in range(len(keys))]
    return {
        "schema": "subliminal-no-loss-baseline.v1",
        "source": {
            "git": git_identity(),
            "archived_git": receipt.get("git"),
            "run_dir": os.path.abspath(run_dir),
        },
        "environment": environment(device),
        "recipe": recipe.to_json(),
        "provenance_checks": {
            "keys": keys,
            "w0_digest": initial_digest,
            "noise_digest": noise_digest,
            "archived_accuracy": archived_accuracy,
            "archived_correct_counts": archived_correct_counts,
        },
        "operations": {
            "forward_only_batches": steps,
            "teacher_loaded": False,
            "loss_function_called": False,
            "backward_called": False,
            "optimizer_created": False,
            "optimizer_step_called": False,
        },
        "initial_digit_scoring": initial,
        "final_digit_scoring": final,
        "checks": {
            "state_equal_after_each_forward": [
                digest == initial_digest for digest in digest_after_forward
            ],
            "final_state_equal": final_digest == initial_digest,
            "digit_logits_bitwise_equal": bool(
                torch.equal(initial_logits, final_logits)
            ),
            "digit_predictions_equal": bool(
                torch.equal(initial_logits.argmax(-1), final_logits.argmax(-1))
            ),
            "digit_accuracy_equal": final["accuracy"] == initial["accuracy"],
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="archived null run receipt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args(argv)
    result = run_baseline(args.run_dir, args.steps)
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
