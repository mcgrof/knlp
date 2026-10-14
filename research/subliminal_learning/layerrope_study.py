# SPDX-License-Identifier: MIT
"""Matched 2x2 MNIST study: residual/HC, without/with LayerRoPE.

The primary endpoint is epoch 80; epochs 5 and 20 are diagnostic. Both
LayerRoPE arms are compared to HC with 97.5% paired bootstrap intervals.
Residual contrasts and the factorial interaction are supporting evidence.
This is the paper's vision-style pre-LayerNorm adaptation, not its full
input/output RMSNorm language-model implementation.
"""

import argparse
import copy
import hashlib
import io
import json
import math
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import analyze
from .data import epoch_order, make_noise
from .models import build, init_model
from .recipe import Recipe, validate
from .run import (
    cmd_probe,
    cmd_run,
    environment,
    git_identity,
    load_data,
    make_optimizer,
    pick_device,
    rows_unchanged,
    state_digest,
    student_loss,
    teacher_loss,
)

ARMS = {
    "residual": ("residual_mlp", False),
    "hc": ("hc", False),
    "layerrope": ("residual_mlp", True),
    "hc_layerrope": ("hc", True),
}


def recipes(args):
    result = {}
    for name, (arch, layerrope) in ARMS.items():
        recipe = Recipe(
            arch=arch,
            layerrope=layerrope,
            protocol="per_example_v1",
            student_loss_mode="exact_cancel_v1",
            seeds=[int(s) for s in args.seeds.split(",")],
            pack_size=args.pack_size,
            student_epochs=80,
            student_eval_epochs=[5, 20, 80],
            output_dir=str(args.root / "runs"),
            checkpoint_dir=args.checkpoint_root,
            data_dir=args.data_dir,
            device=args.device,
            run_name=name,
            max_seconds=7200,
        )
        validate(recipe)
        result[name] = recipe
    return result


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def qualify(recipe):
    """Active null, direct signal and alternate-init control on real GPU batches.

    Disposable teachers take three MNIST steps. Each of three disposable
    auxiliary students then takes three actual loss/backward/Adam steps.
    No qualification state is used by the measured cohort.
    """
    device = pick_device(recipe)
    keys = recipe.model_keys[: recipe.pack_size or len(recipe.model_keys)]
    count = len(keys)
    initial = build(recipe, count).to(device)
    init_model(initial, keys, "init")
    other = build(recipe, count).to(device)
    init_model(other, keys, "init_other")
    teacher = copy.deepcopy(initial)
    teacher_opt = make_optimizer(teacher, recipe)
    train_x, train_y, _, _ = load_data(recipe, device)
    order = torch.stack(
        [epoch_order(k, "teacher_order", 0, len(train_x)) for k in keys]
    ).to(device)
    batches = []
    for step in range(3):
        idx = order[:, step * recipe.batch_size : (step + 1) * recipe.batch_size]
        loss, _ = teacher_loss(
            teacher(train_x[idx]), train_y[idx], recipe.protocol, recipe.cohort
        )
        teacher_opt.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            torch.isfinite(p.grad).all()
            for p in teacher.parameters()
            if p.grad is not None
        )
        teacher_opt.step()
    assert rows_unchanged(teacher, initial, list(range(10, 13))).all()
    teacher.eval().requires_grad_(False)
    teacher_hash = state_digest(teacher)
    noise = torch.stack([make_noise(k, recipe.batch_size * 3) for k in keys]).to(device)
    for step in range(3):
        batches.append(
            noise[:, step * recipe.batch_size : (step + 1) * recipe.batch_size]
        )
    checks = {}
    for name, base, target in (
        ("shared_null", initial, initial),
        ("different_init_null", other, initial),
        ("shared_signal", initial, teacher),
    ):
        student = copy.deepcopy(base)
        opt = make_optimizer(student, recipe)
        before = state_digest(student)
        step_rows = []
        for batch in batches:
            with torch.no_grad():
                target_logits = target(batch)[..., 10:]
            loss, _, _ = student_loss(
                student(batch)[..., 10:],
                target_logits,
                recipe.protocol,
                recipe.cohort,
                recipe.student_loss_mode,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in student.parameters() if p.grad is not None]
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads)
            nonzero = sum(int(g.count_nonzero()) for g in grads)
            opt.step()
            assert rows_unchanged(student, base, list(range(10))).all()
            stationary = state_digest(student) == before
            if name == "shared_null":
                assert nonzero == 0 and stationary
            else:
                assert nonzero > 0 and not stationary
            step_rows.append(
                {"nonzero_gradient_entries": nonzero, "stationary": stationary}
            )
        checks[name] = step_rows
    assert state_digest(teacher) == teacher_hash
    return {
        "all_pass": True,
        "conditions": checks,
        "environment": environment(device),
        "recipe": asdict(recipe),
        "git": git_identity(),
    }


def analyze_study(root):
    runs = {name: analyze.load(str(root / "runs" / name)) for name in ARMS}
    checks = {}
    for name, run in runs.items():
        recipe = run["recipe"]
        for field, value in runs["residual"]["recipe"].items():
            if field not in ("arch", "layerrope", "run_name", "effective_streams"):
                assert recipe[field] == value, (name, field)
        assert run["git"] == runs["residual"]["git"]
        assert recipe["student_loss_mode"] == "exact_cancel_v1" and not recipe["smoke"]
        assert recipe["student_epochs"] == 80 and recipe["teacher_epochs"] == 5
        assert recipe["target_scale"] == 1 and recipe["target_match_kl"] == 0
        expected_arch, expected_layerrope = ARMS[name]
        assert (recipe["arch"], recipe["layerrope"]) == (
            expected_arch,
            expected_layerrope,
        )
        packets = json.loads((root / "runs" / name / "results.json").read_text())
        for packet in packets.values():
            result = packet["results"]
            assert result["teacher_frozen"]
            assert all(
                result[field][m]
                for field in (
                    "teacher_aux_rows_unchanged",
                    "aux_shared_digit_rows_unchanged",
                    "aux_other_digit_rows_unchanged",
                )
                for m in range(len(packet["keys"]))
            )
        curves = run["curves"]
        assert all(
            len(curves[k][stage]) == (5 if stage == "teacher" else 80)
            for k in run["keys"]
            for stage in ("teacher",) + analyze.STAGES
        )
        assert all(
            math.isfinite(acc) and math.isfinite(seconds)
            for curve in curves.values()
            for points in curve.values()
            for acc, seconds in points.values()
        )
        checks[name] = True
    # Match common initialization, independent controls and noise across all arms.
    baseline_packets = json.loads(
        (root / "runs" / "residual" / "results.json").read_text()
    )
    for name in ARMS:
        candidate_packets = json.loads(
            (root / "runs" / name / "results.json").read_text()
        )
        for pack_id, baseline in baseline_packets.items():
            candidate = candidate_packets[pack_id]
            assert baseline["keys"] == candidate["keys"]
            for field in ("common_digest", "common_digest_other", "noise_digest"):
                assert baseline["results"][field] == candidate["results"][field]
    comparisons = [
        analyze.compare(runs["hc"], runs[name], 0.975, 95.0, 100000)
        for name in ("layerrope", "hc_layerrope")
    ]
    supporting = analyze.compare(
        runs["residual"], runs["layerrope"], 0.95, 95.0, 100000
    )
    summaries, interaction = {}, {}
    for name, run in runs.items():
        summaries[name] = {}
        for epoch, table in run["milestones"].items():
            summaries[name][str(epoch)] = {
                field: analyze.boot(
                    [100 * float(table[k][field]) for k in run["keys"]], 0.95, 100000
                )
                for field in (
                    "acc_init",
                    "acc_teacher",
                    "acc_aux_shared",
                    "acc_full_shared",
                    "acc_aux_other",
                    "acc_full_other",
                    "gain_aux_shared",
                    "gain_aux_other",
                )
            }
    for epoch in (5, 20, 80):
        gains = {
            name: {
                k: 100 * float(run["milestones"][epoch][k]["gain_aux_shared"])
                for k in run["keys"]
            }
            for name, run in runs.items()
        }
        differences = [
            gains["hc_layerrope"][k]
            - gains["hc"][k]
            - gains["layerrope"][k]
            + gains["residual"][k]
            for k in runs["hc"]["keys"]
        ]
        interaction[str(epoch)] = analyze.boot(differences, 0.95, 100000)
    out = {
        "schema": "subliminal-layerrope-study.v1",
        "checks": checks,
        "summaries": summaries,
        "primary_ci_level": 0.975,
        "primary_comparisons": comparisons,
        "supporting_residual_contrast": supporting,
        "factorial_interaction_pp": interaction,
    }
    write(root / "analysis.json", out)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, run in runs.items():
        for ax, stage in zip(axes, ("aux_shared", "full_shared")):
            values = np.array(
                [
                    [run["curves"][k][stage][e][0] for e in range(1, 81)]
                    for k in run["keys"]
                ]
            )
            if stage == "aux_shared":
                values -= np.array(
                    [
                        100 * float(run["milestones"][80][k]["acc_init"])
                        for k in run["keys"]
                    ]
                )[:, None]
            ax.plot(range(1, 81), values.mean(0), label=name.replace("_", " + "))
    axes[0].set_ylabel("Auxiliary student gain from W0 (percentage points)")
    axes[1].set_ylabel("All-output student MNIST accuracy (%)")
    axes[1].axhline(95, color="gray", ls=":")
    for ax in axes:
        ax.set_xlabel("Student epoch")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=180)
    fig.savefig(root / "learning_curves.pdf")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("action", choices=("probe", "qualify", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root", default="~/subliminal-learning-checkpoints/layerrope"
    )
    parser.add_argument("--data-dir", default="~/.cache/knlp/mnist")
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in range(74001, 74011))
    )
    parser.add_argument("--pack-size", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    cells = recipes(args)
    if args.action == "analyze":
        analyze_study(args.root)
        return
    if args.action in ("probe", "qualify"):
        out = {}
        for name, recipe in cells.items():
            if args.action == "qualify":
                out[name] = qualify(recipe)
            else:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    cmd_probe(recipe, steps=10)
                out[name] = json.loads(stdout.getvalue())
            print(name, json.dumps(out[name]), flush=True)
        write(args.root / f"{args.action}.json", out)
        return
    qualification = json.loads((args.root / "qualify.json").read_text())
    assert set(qualification) == set(ARMS) and all(
        q["all_pass"] for q in qualification.values()
    )
    for name, recipe in cells.items():
        assert qualification[name]["recipe"] == json.loads(json.dumps(asdict(recipe)))
        assert qualification[name]["git"] == git_identity()
    cache = (
        Path(cells["residual"].data_dir).expanduser()
        / "subliminal_mnist_28x28_norm05.pt"
    )
    data_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    plan = {
        "recipes": {name: asdict(recipe) for name, recipe in cells.items()},
        "git": git_identity(),
        "dataset_cache_sha256": data_hash,
        "qualification_sha256": hashlib.sha256(
            (args.root / "qualify.json").read_bytes()
        ).hexdigest(),
        "primary_epoch": 80,
        "primary_pairs": [["hc", "layerrope"], ["hc", "hc_layerrope"]],
        "primary_ci_level": 0.975,
        "bootstrap_resamples": 100000,
        "screens": {
            "transfer_mean_pp": -5,
            "transfer_ci_upper_lt": 0,
            "utility_noninferiority_pp": 2,
            "speed_epoch_ratio_upper": 1.10,
        },
    }
    plan_path = args.root / "plan.json"
    if plan_path.exists():
        raise RuntimeError("plan already exists; use a new output root")
    write(plan_path, plan)
    source = {field: "layerrope-study" for field in asdict(next(iter(cells.values())))}
    for name, recipe in cells.items():
        code = cmd_run(recipe, source, ["layerrope_study", "run", str(plan_path)])
        if code:
            raise RuntimeError(
                f"{name} stopped with status {code}; see its resume receipt"
            )
    analyze_study(args.root)


if __name__ == "__main__":
    main()
