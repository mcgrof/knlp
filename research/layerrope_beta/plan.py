# SPDX-License-Identifier: MIT
"""Freeze paired input/output-gain comparisons before measured training."""

from dataclasses import asdict
from pathlib import Path

from research.subliminal_learning.recipe import Recipe

BETAS = (-1.0, -0.5, 0.0)
LM_SEEDS = (77311, 77312, 77313)
MNIST_SEEDS = tuple(range(77401, 77411))
LR_GRID = (0.0003, 0.001, 0.003, 0.01)


def tag(beta):
    return str(beta).replace("-", "minus").replace(".", "p")


def language_rows():
    rows = [dict(id=arm, arm=arm, beta=None) for arm in ("R", "M")]
    rows += [
        dict(id=f"{arm}-beta-{tag(beta)}", arm=arm, beta=beta)
        for arm in ("L", "ML")
        for beta in BETAS
    ]
    return rows


def mnist_recipes(root, data, checkpoints, device="cuda", smoke=False):
    rows = {}
    for label, arch in (("residual", "residual_mlp"), ("hc", "hc"), ("mhc", "mhc")):
        configurations = [(label, False, False, 0.0, -0.5)]
        configurations += [
            (f"{label}-full-beta-{tag(beta)}", True, True, beta, beta) for beta in BETAS
        ]
        if arch in ("residual_mlp", "hc"):
            configurations.append((label + "-input-only", True, False, 0.0, -0.5))
        for name, enabled, output, magnitude, rotation in configurations:
            rows[name] = Recipe(
                arch=arch,
                layerrope=enabled,
                layerrope_output_gain=output,
                layerrope_mag_slope=magnitude,
                layerrope_rot_slope=rotation,
                seeds=list(MNIST_SEEDS),
                protocol="per_example_v1",
                student_loss_mode="exact_cancel_v1",
                teacher_epochs=5,
                student_epochs=80,
                student_eval_epochs=[5, 20, 80],
                pack_size=10,
                output_dir=str(Path(root) / "mnist/runs"),
                checkpoint_dir=str(Path(checkpoints) / "mnist"),
                data_dir=str(Path(data) / "mnist"),
                device=device,
                run_name=name,
                max_seconds=14400,
            )
    return rows


def freeze(root, data, checkpoints):
    root, data, checkpoints = map(Path, (root, data, checkpoints))
    return dict(
        schema="layerrope-input-output-beta.v1",
        frozen_before_outcomes=True,
        question="Do full magnitude/rotation input and output gains improve learning, residual distributions or quantization tolerance?",
        reference="https://www.cs.rochester.edu/u/ssrivas9/layer-rope/#code",
        betas=list(BETAS),
        beta_semantics="Equal magnitude and rotation INITIAL slopes at every site; all schedule parameters remain trainable",
        language=dict(
            rows=language_rows(),
            depth=32,
            width=128,
            heads=2,
            ffn_width=352,
            pilot_seed=77301,
            seeds=list(LM_SEEDS),
            pilot_steps=500,
            steps=2000,
            lr_grid=list(LR_GRID),
            lr_selection="Lowest mean curve CE at steps 300/400/500; ties choose lower LR; invalid pilots remain recorded and ineligible",
            training_tokens_per_update=2048,
            warmup_steps=100,
            optimizer=dict(
                type="AdamW",
                betas=[0.9, 0.95],
                eps=1e-8,
                weight_decay=0.0,
                clipping=None,
            ),
            primary="Mean curve CE at steps 1000,1100,...,2000; final own-clean/quantized CE on the separate final array",
            quantization="Every final checkpoint, all 49 signed/affine KV cells plus FP4/FP8 KV and weight/activation/residual interventions; own-clean CE <=0.1 nats per seed; no selection by final quantized loss",
            residual_statistics="Same fixed eight-window batch, at initialization and saved checkpoints; per-stream and collapsed-mean distributions separately; learned gains and linear weights retained",
            checkpoints=[0, 1000, 2000],
            pilot_checkpoint=500,
        ),
        mnist=dict(
            recipes={
                k: asdict(v) for k, v in mnist_recipes(root, data, checkpoints).items()
            },
            primary_epoch=80,
            diagnostic_epochs=[5, 20],
            primary_contrasts="Nine full-gain variants versus their same-architecture no-LayerRoPE baseline; input-only rows supporting only",
            confidence_level=1 - 0.05 / 9,
            bootstrap_resamples=100000,
            suppression_rule=dict(
                mean_gain_change_pp_at_most=-5.0,
                upper_ci_below=0.0,
                utility_lower_ci_pp_at_least=-2.0,
                upper_epoch_to_95_ratio_at_most=1.10,
            ),
            exact_null="Before measured runs: active identical-target stationarity, direct teacher signal, independent-init control, immutable unused head rows",
            quantization="Teacher, auxiliary-shared and all-output-shared endpoints; twelve FP4/FP8 and matched INT4/INT8 weight/activation policies; digit accuracy and CE on all 10000 test images; no calibration or retraining",
            residual_statistics="Same first 256 test images for every retained endpoint; per-model and per-stream distributions, linear weights and both learned gains; no residual-stream rounding intervention in this MNIST screen",
        ),
        precision="FP32 weights/optimizer/norms/gains/routing/CE; BF16 language-model matrix operations; MNIST FP32; no TF32 or compilation",
        limits=[
            "Small scratch-trained models and short contexts",
            "Three language-model seeds are descriptive",
            "Evaluation arrays reused from the original story preparation",
            "FP4/FP8 are scaled numerical simulations, not packed inference or hardware throughput",
            "Variance alone does not establish a quantization benefit",
            "No math-accuracy claim from TinyStories perplexity",
        ],
        source_data=str(data),
        checkpoint_directory=str(checkpoints),
    )
