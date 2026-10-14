"""Actual capacity, mapping-control checks, and immutable pattern receipts."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from research.tinystories_depth.data import sha
from research.tinystories_depth.model import digest
from .preflight import write
from .model import Model
from .qualify import code_parameters


def inspect(root, device="cpu"):
    if device == "cuda":
        assert os.environ.get("GPUQ_RUN_ID"), "GPU fixtures require gpuq"
    root = Path(root)
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    record = {
        "device": device,
        "gpuq_run_id": os.environ.get("GPUQ_RUN_ID"),
        "rows": [],
        "controls": [],
        "source_files": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    }
    common = None
    for i in range(32):
        v = f"V{i:02}"
        # Use the full, declared vocabulary and depth for actual counts.
        model = Model(v, 32, 76390)
        if common is None:
            common = model.common_hashes
        assert common == model.common_hashes
        backbone = sum(p.numel() for p in model.lm.parameters())
        total = sum(p.numel() for p in model.parameters())
        code = code_parameters(model)
        controls = {
            n: {
                "elements": p.numel(),
                "requires_grad": p.requires_grad,
                "initial_sha256": digest(p),
            }
            for n, p in code.items()
        }
        energies = []
        if v != "V00":
            for site in (0, 1):
                for layer in range(32):
                    c = model.code(layer, site).detach()
                    assert abs(c.sum().item()) < 1e-6 and c.abs().max() <= 1 + 1e-7
                    energies.append(c.square().sum().item())
        record["rows"].append(
            {
                "id": v,
                "depth": 32,
                "vocab": 50257,
                "total_parameters": total,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "backbone_parameters": backbone,
                "routing_parameters": sum(p.numel() for p in model.routes.parameters()),
                "code_parameters": controls,
                "fixed_buffers": {
                    n: {
                        "elements": t.numel(),
                        "bytes": t.numel() * t.element_size(),
                        "sha256": digest(t),
                    }
                    for n, t in model.named_buffers()
                    if n == "fixed_angles"
                },
                "code_squared_energy_range": (
                    [min(energies), max(energies)] if energies else [0, 0]
                ),
                "code_map": model.code_map,
                "common_backbone_hashes_sha256": hashlib.sha256(
                    json.dumps(model.common_hashes, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
        del model
        gc.collect()
        base = Model(v, 4, 76390, vocab=128).to(device)
        if v == "V00":
            continue
        for mapping in ("shuffle", "constant"):
            control = Model(v, 4, 76390, vocab=128, mapping=mapping).to(device)
            assert base.common_hashes == control.common_hashes
            assert list(base.state_dict()) == list(control.state_dict())
            assert all(
                torch.equal(a, b)
                for a, b in zip(
                    base.state_dict().values(), control.state_dict().values()
                )
            )
            active = (
                list(range(2))
                if v == "V13"
                else list(range(2, 4)) if v == "V14" else list(range(4))
            )
            for site in (0, 1):
                lookup = control.mapping_lookup[site]
                if mapping == "shuffle":
                    assert sorted(lookup[l] for l in active) == active
                    assert [lookup[l] for l in active] != active
                else:
                    assert [lookup[l] for l in active] == [1] * len(active)
                for layer in range(4):
                    assert torch.equal(
                        control.code(layer, site), base.code(lookup[layer], site)
                    )
                    assert control.strength(layer, site) == base.strength(layer, site)
                    if layer not in active:
                        assert lookup[layer] == layer
            # Full LayerRoPE gains stay at their original block indices.
            if v == "V31":
                for l in range(4):
                    for gain in ("in_attn", "in_mlp", "out_attn", "out_mlp"):
                        assert torch.equal(
                            base.lm.model.layers[l]._gain(gain),
                            control.lm.model.layers[l]._gain(gain),
                        )
            record["controls"].append(
                {
                    "id": v,
                    "mapping": mapping,
                    "mapping_lookup": control.mapping_lookup,
                    "code_map": control.code_map,
                    "mapped_codes_and_placement_pass": True,
                    "equivalent_to_candidate": v == "V03",
                    "capacity_note": (
                        "constant lookup can reduce functional capacity; free table reads one entry per site"
                        if mapping == "constant"
                        else "lookup permutation retains free-table function class"
                    ),
                }
            )
        if v == "V27":
            with torch.no_grad():
                for route in base.routes:
                    route.base[4:8].copy_(
                        torch.logit(
                            torch.tensor([0.33, 0.74, 1.14, 1.61], device=device) / 2
                        )
                    )
            base.capture = True
            ids = torch.randint(
                0, 128, (1, 8), generator=torch.Generator().manual_seed(76390)
            ).to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ):
                base(ids)
            assert any(s["immediate_write_sum_change_max"] > 1e-3 for s in base.samples)
            assert all(
                s["base_write"].shape == s["coded_write"].shape for s in base.samples
            )
            record["V27_capture"] = (
                "baseline coefficients precede logit injection; nonzero write-sum change retained"
            )
        print(v, "capacity and controls passed", flush=True)
    record["status"] = "complete"
    record["V22_definition"] = (
        "exact explicit fixed blend (c(2piu)+c(4piu))/2; already centered and bounded; energy varies. No extra phase-dependent normalization changes the frozen row formula. No arbitrary non-sinusoidal draws are introduced."
    )
    write(root / f"MATRIX-IMPLEMENTATION-{device}.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args()
    inspect(args.root, args.device)
