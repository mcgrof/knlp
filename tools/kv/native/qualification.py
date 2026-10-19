# SPDX-License-Identifier: MIT
"""Retain attention operands and numerical evidence before enforcing tolerances."""

import hashlib
import os
import re
import sys
import tempfile
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
    libraries = loaded_attention_libraries()
    retained_libraries = snapshot_attention_libraries(
        libraries, directory / "libraries"
    )
    passed = all_finite and maximum <= 0.005 and mean <= 0.001
    receipt = {
        "passed": passed,
        "configuration": configuration,
        "backend": str(getattr(wrapper, "_backend", None)),
        "use_tensor_cores": getattr(wrapper, "_use_tensor_cores", None),
        "cached_module": repr(getattr(wrapper, "_cached_module", None)),
        "loaded_flashinfer_libraries": libraries,
        "retained_flashinfer_libraries": retained_libraries,
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


def snapshot_attention_libraries(libraries, directory):
    """Retain each recorded binary before a later JIT build reuses its path.

    Return filenames relative to directory. A recorded hash must match the
    current source bytes and any existing retained object. Numerical callers
    should invoke this when making the receipt, before further compilation.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    retained = []
    for row in libraries:
        digest = row["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("invalid recorded library SHA-256")
        data = Path(row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("library changed after its hash was recorded")
        target = directory / (digest + ".so")
        if target.exists():
            if target.read_bytes() != data:
                raise RuntimeError("retained library differs from its recorded hash")
        else:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        retained.append({"sha256": digest, "file": target.name})
    return retained


def loaded_attention_libraries(maps_path=Path("/proc/self/maps")):
    """Find mapped attention binaries even in explicitly isolated JIT roots."""
    jit = sys.modules.get("flashinfer.jit.env")
    roots = [
        Path(getattr(jit, name)).resolve()
        for name in ("FLASHINFER_JIT_DIR", "FLASHINFER_AOT_DIR")
        if jit is not None and hasattr(jit, name)
    ]
    libraries = []
    if maps_path.exists():
        paths = {
            line.split()[-1]
            for line in maps_path.read_text().splitlines()
            if ".so" in line
        }
        for filename in sorted(paths):
            path = Path(filename)
            if (
                path.is_absolute()
                and path.is_file()
                and (
                    "flashinfer" in filename
                    or any(path.resolve().is_relative_to(root) for root in roots)
                )
            ):
                libraries.append(
                    {
                        "path": filename,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    return libraries
