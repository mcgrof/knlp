# SPDX-License-Identifier: MIT
"""Exercise real gain, optimizer, checkpoint and quantization paths on synthetic tokens."""

import argparse
from pathlib import Path

import numpy as np
import torch

from research.layerrope_beta.model import Model
from research.layerrope_beta.measure import quantization, distributions
from research.mo_variants.stage_c_common import write, sha
from research.tinystories_depth.model import loss
from research.tinystories_depth.train import state_hash


def run(root, device="cpu"):
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    tokens = np.random.default_rng(77390).integers(0, 97, 33, dtype=np.uint16)
    model = Model("ML", 2, 77390, beta=-1.0, vocab=97).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids = torch.tensor(tokens[:17].astype(np.int64), device=device)[None]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            value = loss(model(ids[:, :-1]), ids[:, 1:])
        value.backward()
        optimizer.step()
    before = state_hash(model.state_dict())
    checkpoint = root / "step2.pt"
    torch.save(
        dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=2),
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state_hash(saved["model"]) == before
    model.eval()
    statistics = distributions(model, tokens, device, window=16, batch_size=2)
    write(root / "STATISTICS.json", statistics)
    results = quantization(model, tokens, device, window=16, batch_size=2)
    write(root / "QUANTIZATION.json", results)
    write(
        root / "REPORT.json",
        dict(
            status="complete",
            scientific_confirmation_allowed=False,
            scope="Synthetic two-update numerical smoke only",
            checkpoint_sha256=sha(checkpoint),
            cache_cases=len(results["cache"]),
            other_interventions=len(results["weight_activation_residual"]),
            model_state_unchanged=state_hash(model.state_dict()) == before,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.root, args.device)
