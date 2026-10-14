"""Small analysis/quantization checks and disposable CPU evaluation timing."""

import os
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .analyze import crossing
from .model import Model
from .probe import evaluate
from .quantize import quantize_rows


def run(data, checkpoint, root):
    curve = [
        {
            "ce": ce,
            "step": i * 100,
            "tokens": i * 204800,
            "training_seconds": i,
            "end_to_end_seconds": i * 2,
        }
        for i, ce in enumerate([5.0, 2.9, 3.1, 2.8, 2.7])
    ]
    assert crossing(curve, 3.0, 10)["step"] == 400
    assert crossing(curve[:4], 3.0, 10)["censored"]
    w = torch.tensor([[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [-0.7, 0.2, 1.0]])
    for bits in [8, 4, 3, 2]:
        q, scale = quantize_rows(w, bits)
        assert torch.isfinite(q).all() and torch.equal(q[0], w[0])
        assert torch.all((q - w).abs() <= scale / 2 + 1e-6)
        assert torch.equal(q[1], w[1])
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Model("ML", 32, 75290)
    model.load_state_dict(payload["model"])
    del payload
    model.requires_grad_(False)
    tokens = np.load(Path(data, "curve.npy"), mmap_mode="r")[:2049]
    started = time.perf_counter()
    ce = evaluate(model, tokens, "cpu")
    record = {
        "status": "pass",
        "analysis_censoring": True,
        "quantization_row_checks": True,
        "cpu_2048_tokens_seconds": time.perf_counter() - started,
        "disposable_ce": ce,
        "checkpoint": checkpoint,
        "purpose": "CPU backend functionality/cost; disposable weights; no final model outcome",
    }
    Path(root, "CPU-SUPPORT-CHECK.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    run(a.data, a.checkpoint, a.root)
