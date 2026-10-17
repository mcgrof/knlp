# SPDX-License-Identifier: MIT
"""Synthetic CPU qualification of resume, routing and matched endpoint paths."""

import argparse
from pathlib import Path
import numpy as np
import torch
from research.layerrope_beta.time_budget.train import train
from research.layerrope_beta.time_budget.evaluation import endpoint
from research.layerrope_beta.time_budget.protocol import freeze
from research.layerrope_beta.time_budget.state import DataCursor, json_hash, restore
from research.layerrope_beta.model import Model
from research.mo_variants.stage_c_common import sha, write, read
from research.tinystories_depth.train import state_hash


def run(root, device="cpu"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data = root / "training"
    data.mkdir()
    tokens = np.random.default_rng(77690).integers(0, 97, 32 * 16 + 1, dtype=np.uint16)
    for n in ("train", "validation"):
        np.save(data / (n + ".npy"), tokens[:97] if n == "train" else tokens)
    write(
        data / "TRAINING.json",
        dict(files={n: sha(data / n) for n in ("train.npy", "validation.npy")}),
    )
    checks = []
    for arm in ("R", "L", "M", "ML"):
        job = dict(
            arm=arm,
            seed=77690,
            stage="qualification",
            lr=0.001,
            depth=2,
            vocab=97,
            window=16,
            batch_size=2,
            max_process_seconds=180,
        )
        whole = train(
            data,
            job,
            root / (arm + "-whole"),
            device=device,
            stop_after_updates=4,
            smoke=True,
        )
        partial = train(
            data,
            job,
            root / (arm + "-partial"),
            device=device,
            stop_after_updates=2,
            smoke=True,
        )
        receipt = partial["last_checkpoint"]
        resumed = train(
            data,
            job,
            root / (arm + "-resumed"),
            device=device,
            resume=receipt["path"],
            resume_sha=receipt["sha256"],
            stop_after_updates=4,
            smoke=True,
        )
        a = torch.load(
            whole["last_checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        b = torch.load(
            resumed["last_checkpoint"]["path"], map_location="cpu", weights_only=False
        )
        if state_hash(a["model"]) != state_hash(b["model"]) or state_hash(
            a["optimizer"]
        ) != state_hash(b["optimizer"]):
            raise RuntimeError(
                "Interrupted/uninterrupted parameter or optimizer mismatch"
            )
        if state_hash(a["rng"]) != state_hash(b["rng"]) or a["cursor"] != b["cursor"]:
            raise RuntimeError("Interrupted/uninterrupted RNG/data mismatch")
        if b["training_seconds"] < partial["training_seconds"]:
            raise RuntimeError("Resume clock regressed")
        model = Model(arm, 2, 77690, beta=0.0, vocab=97).to(device)
        model.load_state_dict(b["model"])
        model.eval()
        result = endpoint(model, tokens, device, window=16, batch_size=32)
        if len(result["cache"]) != 8 or len(result["weight_activation_residual"]) != 8:
            raise RuntimeError("Incomplete reduced menu")
        write(root / (arm + "-ENDPOINT.json"), result)
        checks.append(
            dict(
                arm=arm,
                model_optimizer_rng_and_cursor_identical=True,
                whole_training_seconds=a["training_seconds"],
                resumed_training_seconds=b["training_seconds"],
                clock_semantics="Wall times may differ; resume equivalence is at the same update/cursor",
                parity=result["parity"],
                cache_cases=8,
                other_cases=8,
            )
        )
    write(
        root / "REPORT.json",
        dict(
            status="complete",
            scientific_results=False,
            scope="Synthetic four-update CPU/explicit-device qualification only",
            checks=checks,
        ),
    )
    return read(root / "REPORT.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = p.parse_args()
    run(a.root, a.device)
