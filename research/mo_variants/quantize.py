"""Inherited primary weight quantizer, on the declared discovery array only."""

import os
import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import digest
from research.tinystories_depth.probe import evaluate
from research.tinystories_depth.quantize import quantize_rows
from .preflight import write
from .model import Model


def endpoint(result_path, array_path, destination, fixture=False):
    result_path, array_path, destination = map(
        Path, (result_path, array_path, destination)
    )
    assert not destination.exists(), "Never overwrite an endpoint precision record"
    result = json.loads(result_path.read_text())
    assert result["status"] == "complete"
    job = result["metadata"]["job"]
    if fixture:
        assert (
            job["vocab"] == 128
            and job["steps"] == 3
            and job["id"].startswith("runner-fixture-")
        )
    else:
        assert job["stage"] in (
            "A-discovery",
            "B",
        ), "Confirmation secondary quantizers need the selection-specific protocol"
        assert job.get("vocab", 50257) == 50257
    assert sha(array_path) == result["metadata"]["curve_sha256"]
    receipt = result["checkpoint"]
    assert sha(receipt["path"]) == receipt["sha256"]
    torch.set_num_threads(min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    saved = torch.load(receipt["path"], map_location="cpu", weights_only=False)
    model = Model(
        job["variant"],
        job["depth"],
        job["seed"],
        vocab=job.get("vocab", 50257),
        mapping=job.get("mapping", "ordered"),
    )
    model.load_state_dict(saved["model"], strict=True)
    model.step = saved["step"]
    del saved
    model.requires_grad_(False)
    tokens = np.load(array_path, mmap_mode="r")
    originals = {
        name + ".weight": module.weight.detach().clone()
        for name, module in model.lm.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    }
    parameters = dict(model.lm.named_parameters())
    protected = {
        n: digest(p)
        for n, p in model.named_parameters()
        if not n.startswith("lm.") or n.removeprefix("lm.") not in originals
    }
    count = sum(p.numel() for p in originals.values())
    rows = sum(p.shape[0] for p in originals.values())
    total = sum(p.numel() for p in model.parameters())
    named = dict(model.named_parameters())
    buffer_bytes = sum(
        t.numel() * t.element_size()
        for n, t in model.state_dict().items()
        if n not in named
    )
    record = {
        "id": job["id"],
        "variant": job["variant"],
        "seed": job["seed"],
        "scope": (
            "disposable toy fixture"
            if fixture
            else "discovery precision screen, not independent confirmation"
        ),
        "method": "inherited rowwise symmetric max-absolute, nearest rounding; FP32 controls/norms, BF16 activations",
        "checkpoint_sha256": receipt["sha256"],
        "array_sha256": sha(array_path),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "quantized_parameters": count,
        "total_parameters": total,
        "coverage": count / total,
        "row_scales": rows,
        "persistent_buffer_bytes": buffer_bytes,
        "bits": [],
        "protected_parameter_sha256": protected,
    }
    before = time.perf_counter()
    record["clean_ce"] = evaluate(model, tokens, "cpu")
    record["clean_seconds"] = time.perf_counter() - before
    gpu = result["validation"][-1]["ce"] if result["validation"] else None
    record["paired_execution_gpu_clean_ce"] = gpu
    record["cpu_minus_gpu_clean_ce"] = (
        record["clean_ce"] - gpu if gpu is not None else None
    )
    write(destination, record)
    for bits in (4, 3):
        before = time.perf_counter()
        hashes = {}
        with torch.no_grad():
            for name, weight in originals.items():
                dequantized, _ = quantize_rows(weight, bits)
                parameters[name].copy_(dequantized)
                hashes[name] = digest(dequantized)
        ce = evaluate(model, tokens, "cpu")
        delta = ce - record["clean_ce"]
        record["bits"].append(
            {
                "bits": bits,
                "ce": ce,
                "delta_ce": delta,
                "perplexity_ratio": math.exp(delta),
                "within_point1_nats": delta <= 0.1,
                "seconds_including_dequantization": time.perf_counter() - before,
                "dequantized_sha256": hashes,
                "scale_bytes": rows * 4,
                "unquantized_parameter_bytes": (total - count) * 4,
                "ideal_packed_backbone_bytes": (count * bits + 7) // 8,
                "ideal_packed_total_bytes": (count * bits + 7) // 8
                + rows * 4
                + (total - count) * 4
                + buffer_bytes,
            }
        )
        with torch.no_grad():
            for name, weight in originals.items():
                parameters[name].copy_(weight)
        assert all(digest(parameters[n]) == digest(w) for n, w in originals.items())
        assert all(
            digest(dict(model.named_parameters())[n]) == h for n, h in protected.items()
        )
        write(destination, record)
    record["status"] = "complete"
    write(destination, record)
    del model, parameters, originals
    gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--fixture", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    out = root / "records/quantization"
    out.mkdir(exist_ok=True)
    endpoint(
        root / "records/runs" / args.job_id / "RESULT.json",
        root / "data/curve.npy",
        out / (args.job_id + ".json"),
        args.fixture,
    )
