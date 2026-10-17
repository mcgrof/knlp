# SPDX-License-Identifier: MIT
"""Retain attention operands and numerical evidence before enforcing tolerances."""

import hashlib
from pathlib import Path

import torch

from .instrument import save, tensor_inventory


def record_attention_check(
    output, stage, *, wrapper, result, reference, tensors, cache, configuration
):
    """Save the physical fixture for a passing or failing attention check."""
    directory = Path(output) / stage
    directory.mkdir(parents=True, exist_ok=True)
    error = (result.float() - reference.float()).abs()
    finite = {
        name: bool(torch.isfinite(value.float()).all())
        for name, value in {
            "output": result,
            "reference": reference,
            "error": error,
        }.items()
    }
    all_finite = all(finite.values())
    maximum = float(error.max()) if all_finite else None
    mean = float(error.mean()) if all_finite else None
    bad = (
        error.flatten().argmax()
        if all_finite
        else (~torch.isfinite(error)).flatten().nonzero()[0, 0]
    )
    index = [int(x) for x in torch.unravel_index(bad, error.shape)]
    artifact = {name: tensor.detach().cpu() for name, tensor in tensors.items()}
    artifact.update(
        output=result.detach().cpu(),
        reference=reference.detach().cpu(),
        absolute_error=error.detach().cpu(),
    )
    seen = set()
    for tensor in cache:
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in seen:
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            raw.set_(
                tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,)
            )
            artifact[f"physical_cache_storage_{len(seen)}"] = raw.cpu()
            seen.add(pointer)
    payload = directory / "tensors.pt"
    torch.save(artifact, payload)
    libraries = []
    maps = Path("/proc/self/maps")
    if maps.exists():
        paths = {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if ".so" in line and "flashinfer" in line
        }
        for filename in sorted(paths):
            path = Path(filename)
            if path.is_file():
                libraries.append(
                    {
                        "path": filename,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    passed = all_finite and maximum <= 0.005 and mean <= 0.001
    receipt = {
        "passed": passed,
        "configuration": configuration,
        "backend": str(getattr(wrapper, "_backend", None)),
        "use_tensor_cores": getattr(wrapper, "_use_tensor_cores", None),
        "cached_module": repr(getattr(wrapper, "_cached_module", None)),
        "loaded_flashinfer_libraries": libraries,
        "physical_cache_views": tensor_inventory(cache),
        "tensors_file": payload.name,
        "tensors_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "finite": finite,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
        "offending_index": index,
        "maximum_tolerance": 0.005,
        "mean_tolerance": 0.001,
    }
    save(directory / "check.json", receipt)
    if not passed:
        raise RuntimeError(
            f"{stage} differs from FP32 reference; retained {directory / 'check.json'}"
        )
    return receipt
