#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Eval-only step 0: quantize a trained checkpoint's KV cache at eval.

Loads one already-trained d8 checkpoint from the user volume, then
measures its validation loss under a ladder of KV-cache quantization
settings, applying the same straight-through fake-quant hook the
quantization-aware training path uses. This establishes the post-hoc
degradation curve that QAT has to beat; it trains nothing. qk_norm is a
property of the trained model and cannot be toggled here, so that axis
belongs to the QAT path.
"""

import json
import os
from pathlib import Path

import modal

from modal_utils import (
    MODAL_MODEL_DIR,
    VOLUME_MOUNTS,
    app,
    build_image,
)

# K/V ladder x granularity. 8-bit uses fp8 e4m3; <=4-bit uses symmetric
# integers (int4/int3/int2). per_token is the realistic cache granularity;
# per_tensor is the harsher one-scale setting.
def _ladder(gran):
    levels = [
        ("K16V16", 16, 16), ("K8V8", 8, 8), ("K16V8", 16, 8),
        ("K16V4", 16, 4), ("K4V16", 4, 16), ("K8V4", 8, 4),
        ("K4V4", 4, 4), ("K4V2", 4, 2), ("K2V16", 2, 16), ("K2V2", 2, 2),
    ]
    return [
        {"label": f"{lab}-{gran}", "k_bits": k, "v_bits": v, "granularity": gran}
        for lab, k, v in levels
    ]


FRONTIER = _ladder("per_token") + _ladder("per_tensor")


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="H100",
    timeout=60 * 60,
)
def eval_quant(run_name: str, specs: list[dict]):
    import torch

    import cs312_quant
    from data import dclm_val_dataset, load_token_dataset
    from model_io import load_model
    from train import TrainConfig, evaluate
    from utils import create_batches

    device = "cuda:0"
    config = TrainConfig()  # defaults; evaluate only reads precision
    model_path = Path(str(MODAL_MODEL_DIR)) / run_name
    model = load_model(model_path, device=device, precision=config.precision)

    val_dataset = load_token_dataset(dclm_val_dataset(), role="val")
    val_batches = create_batches(val_dataset, config.batch_size)
    # create_batches is a one-pass iterable; materialize so every spec
    # evaluates the identical val set.
    val_batches = list(val_batches)

    results = {}
    for spec in specs:
        cs312_quant.configure(
            k_bits=spec["k_bits"],
            v_bits=spec["v_bits"],
            k_format=spec.get("k_format", "e4m3"),
            v_format=spec.get("v_format", "e4m3"),
            granularity=spec.get("granularity", "per_token"),
        )
        loss = evaluate(model, val_batches, config=config, device=device).item()
        results[spec["label"]] = loss
        print(f"{spec['label']:8s} val_loss {loss:.4f}", flush=True)
    cs312_quant.configure(k_bits=16, v_bits=16)
    return {
        "run_name": run_name,
        "checkpoint": str(model_path),
        "val_sequences": len(val_dataset),
        "results": results,
    }


@app.local_entrypoint()
def main(run_name: str = "model-d8-lr0.003-tok614M-modal", out: str = ""):
    payload = eval_quant.remote(run_name, FRONTIER)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
        print(f"wrote {out}")
