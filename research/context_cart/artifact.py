# SPDX-License-Identifier: MIT
"""Prefix artifact export and strict reload through the common schema.

An exported cartridge is a typed ``prefix_kv`` artifact from
``tools/reproduce/context_artifacts``: a safetensors payload with
``key.N``/``value.N`` tensors in THD layout and a canonical manifest that
binds the model identity, tokenizer and rotary digests, geometry, and exact
logical positions. Extra provenance that the common schema has no field for
(config and weight digests, source digest, writer checkpoint, world id) is
stored in the payload metadata and checked on load. A loader rejects any
identity, geometry, position, or digest mismatch instead of recomputing.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file

from tools.reproduce.context_artifacts.artifacts import (
    ArtifactError,
    ConsumerIdentity,
    ContextArtifactManifest,
    PrefixMetadata,
    TensorSpec,
    canonical,
    digest_file,
    inspect_payload,
)

from . import model as M

PRODUCER = "knlp.research.context_cart"
PAYLOAD = "payload.safetensors"
MANIFEST = "manifest.json"


@dataclasses.dataclass
class ExportResult:
    artifact_id: str
    payload_digest: str
    payload_bytes: int
    tensor_bytes: int
    directory: str


def check_positions(positions: Sequence[int], reader_start: int, rows: int) -> None:
    """The stored rows carry exact logical positions; the reader's first token
    sits strictly after every stored row."""
    values = [int(v) for v in positions]
    if len(values) != rows:
        raise ArtifactError("one position per stored row is required")
    if any(v < 0 for v in values):
        raise ArtifactError("positions must be non-negative")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ArtifactError("positions must be strictly increasing")
    if int(reader_start) <= values[-1]:
        raise ArtifactError("reader_start must follow every stored position")


def export_prefix(
    directory: Path,
    prefix: list[tuple[torch.Tensor, torch.Tensor]],
    identity: dict[str, Any],
    positions: Sequence[int],
    reader_start: int,
    source_revision: str,
    checkpoint_digest: str,
    namespace: str,
    extra: dict[str, Any],
    index: int = 0,
) -> ExportResult:
    """Write one cartridge: payload, manifest, and return its identity.

    ``positions`` are the logical positions the stored keys were produced at
    (post-rotary keys are position bound); ``reader_start`` is the logical
    position of the first reader token. The two are recorded separately.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tensors = M.to_thd(prefix, index)
    rows = next(iter(tensors.values())).shape[0]
    check_positions(positions, reader_start, rows)
    positions = tuple(int(v) for v in positions)
    geometry = identity["geometry"]
    for layer in range(geometry["layers"]):
        for plane in ("key", "value"):
            tensor = tensors[f"{plane}.{layer}"]
            if tuple(tensor.shape) != (
                rows,
                geometry["kv_heads"],
                geometry["head_dim"],
            ):
                raise ArtifactError(
                    "exported tensor geometry does not match the reader"
                )
    metadata = {
        "config_digest": identity["config_digest"],
        "weights_digest": identity["weights_digest"],
        "reader_start": str(int(reader_start)),
        "rows": str(rows),
        "dtype": identity["dtype"],
        "extra": canonical(extra).decode(),
    }
    payload = directory / PAYLOAD
    save_file(tensors, str(payload), metadata=metadata)
    payload_digest = digest_file(payload)
    specs = inspect_payload(payload)
    consumer = ConsumerIdentity(model=identity["model"], revision=identity["revision"])
    manifest = ContextArtifactManifest(
        schema=1,
        namespace=namespace,
        representation="prefix_kv",
        producer=PRODUCER,
        source_revision=source_revision,
        checkpoint_digest=checkpoint_digest,
        consumers=(consumer,),
        payload_digest=payload_digest,
        tensors=specs,
        metadata=PrefixMetadata(
            model=consumer,
            tokenizer_digest=identity["tokenizer_digest"],
            rope_digest=identity["rope_digest"],
            layers=geometry["layers"],
            kv_heads=geometry["kv_heads"],
            head_dim=geometry["head_dim"],
            logical_tokens=rows,
            physical_tokens=rows,
            block_size=rows,
            positions=positions,
            layout="THD",
            quantization="none",
        ),
    )
    manifest.validate()
    (directory / MANIFEST).write_bytes(canonical(manifest.to_dict()))
    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    return ExportResult(
        artifact_id=manifest.artifact_id,
        payload_digest=payload_digest,
        payload_bytes=payload.stat().st_size,
        tensor_bytes=tensor_bytes,
        directory=str(directory),
    )


@dataclasses.dataclass
class LoadedPrefix:
    prefix: list[tuple[torch.Tensor, torch.Tensor]]
    manifest: ContextArtifactManifest
    first_position: int  # the reader's first logical position (reader_start)
    positions: tuple[int, ...]  # logical positions of the stored rows
    rows: int
    payload_digest: str
    tensor_digest: str
    extra: dict[str, Any]
    read_seconds: float
    validate_seconds: float
    copy_seconds: float


def load_prefix(
    directory: Path, identity: dict[str, Any], device: torch.device | str
) -> LoadedPrefix:
    """Validate and load one cartridge for the reader whose identity is given.

    Every mismatch raises ``ArtifactError``. Nothing is recomputed.
    """
    t0 = time.perf_counter()
    manifest_path, payload = directory / MANIFEST, directory / PAYLOAD
    if not manifest_path.exists() or not payload.exists():
        raise ArtifactError(f"artifact files missing under {directory}")
    raw = manifest_path.read_bytes()
    manifest = ContextArtifactManifest.from_dict(json.loads(raw))
    if canonical(manifest.to_dict()) != raw:
        raise ArtifactError("manifest is not canonical")
    read_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    meta = manifest.metadata
    if not isinstance(meta, PrefixMetadata):
        raise ArtifactError("artifact is not a prefix")
    consumer = ConsumerIdentity(model=identity["model"], revision=identity["revision"])
    if consumer not in manifest.consumers or meta.model != consumer:
        raise ArtifactError("reader identity does not match the artifact")
    if meta.tokenizer_digest != identity["tokenizer_digest"]:
        raise ArtifactError("tokenizer digest mismatch")
    if meta.rope_digest != identity["rope_digest"]:
        raise ArtifactError("rotary convention mismatch")
    geometry = identity["geometry"]
    if (meta.layers, meta.kv_heads, meta.head_dim) != (
        geometry["layers"],
        geometry["kv_heads"],
        geometry["head_dim"],
    ):
        raise ArtifactError("head geometry mismatch")
    if digest_file(payload) != manifest.payload_digest:
        raise ArtifactError("payload digest mismatch")
    if inspect_payload(payload) != manifest.tensors:
        raise ArtifactError("payload tensor metadata mismatch")
    with open(payload, "rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_size))
    file_meta = header.get("__metadata__", {})
    if file_meta.get("config_digest") != identity["config_digest"]:
        raise ArtifactError("model configuration digest mismatch")
    if file_meta.get("weights_digest") != identity["weights_digest"]:
        raise ArtifactError("model weights digest mismatch")
    if file_meta.get("dtype") != identity["dtype"]:
        raise ArtifactError("reader dtype mismatch")
    if "reader_start" not in file_meta:
        raise ArtifactError("artifact predates the reader_start contract")
    first_position = int(file_meta["reader_start"])
    rows = int(file_meta["rows"])
    if rows != meta.logical_tokens or rows != meta.physical_tokens:
        raise ArtifactError("row count mismatch")
    check_positions(meta.positions, first_position, rows)
    validate_seconds = time.perf_counter() - t1
    t2 = time.perf_counter()
    tensors = load_file(str(payload), device="cpu")
    expected_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[identity["dtype"]]
    for name, tensor in tensors.items():
        if tensor.dtype != expected_dtype:
            raise ArtifactError(
                f"{name} has dtype {tensor.dtype}, expected {expected_dtype}"
            )
        if tuple(tensor.shape) != (rows, meta.kv_heads, meta.head_dim):
            raise ArtifactError(f"{name} has an incompatible shape")
    prefix = M.from_thd(tensors, meta.layers, device)
    if device != "cpu" and torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    copy_seconds = time.perf_counter() - t2
    return LoadedPrefix(
        prefix=prefix,
        manifest=manifest,
        first_position=first_position,
        positions=tuple(int(v) for v in meta.positions),
        rows=rows,
        payload_digest=manifest.payload_digest,
        tensor_digest=M.prefix_digest(prefix),
        extra=json.loads(file_meta.get("extra", "{}")),
        read_seconds=read_seconds,
        validate_seconds=validate_seconds,
        copy_seconds=copy_seconds,
    )


def tensor_specs(manifest: ContextArtifactManifest) -> tuple[TensorSpec, ...]:
    return manifest.tensors
