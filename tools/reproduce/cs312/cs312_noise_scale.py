#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the gradient noise scale across a trained run's checkpoints.

The noise scale B_noise = tr(S) / |G|^2 (McCandlish et al. 2018) is the batch
size at which the curvature penalty and the gradient-noise penalty on a step
are equal. Below it training is noise limited, above it curvature limited.
It is not a constant: as the loss falls, |G|^2 shrinks faster than tr(S), so
B_noise rises through training.

That matters for schedules. A warmup-stable-decay schedule holds the peak
learning rate for most of the run and anneals at the end. The excess loss
from gradient noise goes as learning_rate / batch, so holding a high rate is
cheap exactly while the run is curvature limited, and expensive once it is
noise limited. The crossing of B_noise past the training batch size is
therefore the point after which a schedule ought to be decaying, which
predicts the decay fraction that a run actually wants.

This app trains nothing. It loads the model at each kept checkpoint of an
existing run and estimates |G|^2 and tr(S) from several independent batches,
using the standard unbiased pair of estimators: for K batches of size b with
per-batch gradients g_i and their mean ghat,

    E|g_i|^2 = |G|^2 + tr(S)/b      E|ghat|^2 = |G|^2 + tr(S)/(K b)

so  |G|^2 = (K |ghat|^2 - <|g_i|^2>) / (K - 1)
    tr(S) = b K / (K - 1) * (<|g_i|^2> - |ghat|^2)

Gradients are taken exactly as training takes them: the same loss, the same
bf16 autocast, fp32 parameters, and no gradient clipping (clipping would bias
both terms).
"""

import json
from pathlib import Path

import modal

from modal_utils import (
    MODAL_MODEL_DIR,
    VOLUME_MOUNTS,
    app,
    build_image,
)


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="H100",
    timeout=60 * 60,
)
def measure(
    run_name: str,
    steps: list[int],
    batch_size: int = 64,
    num_batches: int = 16,
    repeats: int = 4,
    seed: int = 0,
):
    import torch

    from data import dclm_train_dataset, load_token_dataset
    from model_config import LMConfig
    from train import CausalLMTrainingLoss, TrainConfig, build_model
    from utils import autocast_context, create_batches

    device = "cuda:0"
    run_dir = Path(str(MODAL_MODEL_DIR)) / run_name
    meta = json.loads((run_dir / "metadata.json").read_text())
    known = {f for f in TrainConfig.__dataclass_fields__}
    drop = {
        "model_config", "train_dataset", "val_dataset", "metric_loggers",
        "keep_checkpoint_steps", "model_dir", "wandb_tags",
        "init_checkpoint_path", "fork_from_run", "fork_from_step",
    }
    kept = {k: v for k, v in meta.items() if k in known and k not in drop}
    config = TrainConfig(model_config=LMConfig.from_dict(meta["model_config"]), **kept)

    train_dataset = load_token_dataset(dclm_train_dataset(), role="train")
    batches = create_batches(train_dataset, batch_size)
    num_available = len(batches)
    print(f"{num_available:,} batches of {batch_size} available", flush=True)

    generator = torch.Generator().manual_seed(seed)
    results = []

    for step in steps:
        model = build_model(config, device=device)
        if step > 0:
            path = run_dir / f"step_{step}.pt"
            if not path.exists() and (run_dir / "model.pt").exists():
                path = run_dir / "model.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload["model_state"] if "model_state" in payload else payload
            model.load_state_dict(state)
        # training mode, exactly as the gradients were taken during the run
        model.train()
        training_loss = CausalLMTrainingLoss(model)
        params = [p for p in model.parameters() if p.requires_grad]

        per_repeat = []
        for rep in range(repeats):
            # disjoint batches within a repeat, so the draws are independent
            picks = torch.randperm(num_available, generator=generator)[:num_batches]
            accum = [torch.zeros_like(p, dtype=torch.float32) for p in params]
            sq_sum = 0.0
            for idx in picks.tolist():
                model.zero_grad(set_to_none=True)
                input_ids = batches[idx].to(device)
                with autocast_context(config.precision, device=device):
                    loss = training_loss(input_ids)
                loss.backward()
                sq = 0.0
                for buf, p in zip(accum, params):
                    g = p.grad.detach().float()
                    sq += float((g * g).sum())
                    buf += g
                sq_sum += sq
            mean_sq = sq_sum / num_batches                       # <|g_i|^2>
            ghat_sq = float(sum(((b / num_batches) ** 2).sum() for b in accum))
            k = num_batches
            g_true_sq = (k * ghat_sq - mean_sq) / (k - 1)
            tr_sigma = (batch_size * k / (k - 1)) * (mean_sq - ghat_sq)
            per_repeat.append(
                {
                    "mean_grad_sq": mean_sq,
                    "mean_batch_grad_sq": ghat_sq,
                    "g_true_sq": g_true_sq,
                    "tr_sigma": tr_sigma,
                    "b_noise": (tr_sigma / g_true_sq) if g_true_sq > 0 else None,
                }
            )
            del accum
        model.zero_grad(set_to_none=True)
        del model, training_loss
        torch.cuda.empty_cache()

        valid = [r["b_noise"] for r in per_repeat if r["b_noise"] is not None]
        median = sorted(valid)[len(valid) // 2] if valid else None
        row = {
            "step": step,
            "tokens": step * batch_size * 1024,
            "b_noise_median": median,
            "b_noise_all": valid,
            "g_true_sq_mean": sum(r["g_true_sq"] for r in per_repeat) / repeats,
            "tr_sigma_mean": sum(r["tr_sigma"] for r in per_repeat) / repeats,
            "repeats": per_repeat,
        }
        results.append(row)
        shown = f"{median:,.0f}" if median else "n/a"
        print(f"step {step:>6}  B_noise {shown:>12} sequences", flush=True)

    return {
        "run_name": run_name,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "repeats": repeats,
        "context_length": 1024,
        "results": results,
    }


@app.local_entrypoint()
def main(
    run_name: str = "model-d8-lr0.003-tok614M-noise",
    steps: str = "0,10,50,150,400,900,1900,3400,5200,7200,9375",
    out: str = "",
):
    step_list = [int(s) for s in steps.split(",") if s.strip()]
    payload = measure.remote(run_name, step_list)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text[:2000])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
        print(f"wrote {out}")
