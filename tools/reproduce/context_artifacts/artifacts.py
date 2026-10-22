# SPDX-License-Identifier: MIT
"""Typed, immutable tensor artifacts with durable file publication.

This module owns payload identity and persistence. Consumers retain ownership
of tensor allocation, prompt assembly, cache insertion, and inference.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any

from safetensors import SafetensorError, safe_open


class ArtifactError(ValueError):
    """The artifact is missing, corrupt, or incompatible."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value: str, name: str = "digest") -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ArtifactError(f"{name} must be a SHA-256 digest")


def revision(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ArtifactError("source revision must be a complete Git SHA")


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ArtifactError(f"{name} must be a positive integer")


@dataclasses.dataclass(frozen=True)
class ConsumerIdentity:
    model: str
    revision: str

    def validate(self) -> None:
        if not self.model:
            raise ArtifactError("consumer model is required")
        revision(self.revision)


@dataclasses.dataclass(frozen=True)
class DocumentMetadata:
    encoder: ConsumerIdentity
    preprocessing_digest: str
    source_digest: str
    rows: int
    width: int

    def validate(self) -> None:
        self.encoder.validate()
        digest(self.preprocessing_digest)
        digest(self.source_digest)
        _positive(self.rows, "rows")
        _positive(self.width, "width")


@dataclasses.dataclass(frozen=True)
class InputMetadata:
    reader_checkpoint_digest: str
    bank_id: str
    query_digest: str
    template_digest: str
    preceding_prompt_digest: str
    placement: str
    positions: tuple[int, ...]
    width: int

    def validate(self) -> None:
        for value in (
            self.reader_checkpoint_digest,
            self.bank_id,
            self.query_digest,
            self.template_digest,
            self.preceding_prompt_digest,
        ):
            digest(value)
        if self.placement != "system_message":
            raise ArtifactError("only explicit system_message placement is supported")
        _positive(self.width, "width")
        _positions(self.positions)


@dataclasses.dataclass(frozen=True)
class PrefixMetadata:
    model: ConsumerIdentity
    tokenizer_digest: str
    rope_digest: str
    layers: int
    kv_heads: int
    head_dim: int
    logical_tokens: int
    physical_tokens: int
    block_size: int
    positions: tuple[int, ...]
    layout: str
    quantization: str

    def validate(self) -> None:
        self.model.validate()
        digest(self.tokenizer_digest)
        digest(self.rope_digest)
        for name in (
            "layers",
            "kv_heads",
            "head_dim",
            "logical_tokens",
            "physical_tokens",
            "block_size",
        ):
            _positive(getattr(self, name), name)
        if self.layout != "THD" or self.quantization != "none":
            raise ArtifactError("only unquantized THD prefix tensors are supported")
        if (
            self.physical_tokens
            != math.ceil(self.logical_tokens / self.block_size) * self.block_size
        ):
            raise ArtifactError("physical length must include exactly one aligned tail")
        _positions(self.positions)
        if len(self.positions) != self.logical_tokens:
            raise ArtifactError("positions must preserve the exact logical length")


def _positions(values: tuple[int, ...]) -> None:
    if not values or any(type(v) is not int or v < 0 for v in values):
        raise ArtifactError("positions must be nonnegative integers")
    if tuple(values) != tuple(range(values[0], values[0] + len(values))):
        raise ArtifactError("positions must be ordered and contiguous")


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


_TYPES = {"F16": 2, "BF16": 2, "F32": 4}
_METADATA = {
    "document_embeddings": DocumentMetadata,
    "input_embeddings": InputMetadata,
    "prefix_kv": PrefixMetadata,
}


def inspect_payload(path: Path) -> tuple[TensorSpec, ...]:
    """Inspect tensor geometry through the safetensors format validator."""
    tensors = []
    try:
        with safe_open(str(path), framework="np") as payload:
            for name in payload.keys():
                tensor = payload.get_slice(name)
                shape, dtype = tuple(tensor.get_shape()), tensor.get_dtype()
                if not shape or any(size < 1 for size in shape) or dtype not in _TYPES:
                    raise ArtifactError("unsupported tensor shape or dtype")
                tensors.append(TensorSpec(name, shape, dtype))
    except SafetensorError as error:
        raise ArtifactError(f"invalid safetensors payload: {error}") from error
    if not tensors:
        raise ArtifactError("payload must contain tensors")
    return tuple(sorted(tensors, key=lambda spec: spec.name))


@dataclasses.dataclass(frozen=True)
class ContextArtifactManifest:
    schema: int
    namespace: str
    representation: str
    producer: str
    source_revision: str
    checkpoint_digest: str
    consumers: tuple[ConsumerIdentity, ...]
    payload_digest: str
    tensors: tuple[TensorSpec, ...]
    metadata: DocumentMetadata | InputMetadata | PrefixMetadata

    @property
    def artifact_id(self) -> str:
        return hashlib.sha256(canonical(self.to_dict())).hexdigest()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if self.schema != 1 or not self.namespace or not self.producer:
            raise ArtifactError("schema, namespace, and producer are required")
        expected = _METADATA.get(self.representation)
        if expected is None or type(self.metadata) is not expected:
            raise ArtifactError("representation requires matching typed metadata")
        revision(self.source_revision)
        digest(self.checkpoint_digest)
        digest(self.payload_digest)
        if not self.consumers:
            raise ArtifactError("at least one exact consumer identity is required")
        for consumer in self.consumers:
            consumer.validate()
        self.metadata.validate()
        specs = {t.name: t for t in self.tensors}
        if len(specs) != len(self.tensors):
            raise ArtifactError("duplicate tensor names")
        if isinstance(self.metadata, DocumentMetadata):
            expected_shapes = {"embeddings": (self.metadata.rows, self.metadata.width)}
        elif isinstance(self.metadata, InputMetadata):
            expected_shapes = {
                "embeddings": (len(self.metadata.positions), self.metadata.width)
            }
        else:
            m = self.metadata
            if m.model not in self.consumers:
                raise ArtifactError("prefix model must be a compatible consumer")
            expected_shapes = {
                f"{plane}.{layer}": (m.physical_tokens, m.kv_heads, m.head_dim)
                for layer in range(m.layers)
                for plane in ("key", "value")
            }
        if set(specs) != set(expected_shapes):
            raise ArtifactError("tensor names do not match the representation")
        for name, shape in expected_shapes.items():
            if specs[name].shape != shape or specs[name].dtype not in _TYPES:
                raise ArtifactError(f"incompatible shape or dtype for {name}")

    @classmethod
    def from_dict(cls, value: dict) -> ContextArtifactManifest:
        try:
            data = dict(value)
            data["consumers"] = tuple(ConsumerIdentity(**c) for c in data["consumers"])
            data["tensors"] = tuple(
                TensorSpec(t["name"], tuple(t["shape"]), t["dtype"])
                for t in data["tensors"]
            )
            meta = dict(data["metadata"])
            kind = _METADATA[data["representation"]]
            for key in ("encoder", "model"):
                if key in meta:
                    meta[key] = ConsumerIdentity(**meta[key])
            if "positions" in meta:
                meta["positions"] = tuple(meta["positions"])
            data["metadata"] = kind(**meta)
            result = cls(**data)
            result.validate()
            return result
        except (KeyError, TypeError) as error:
            raise ArtifactError("invalid artifact manifest") from error


def attachment_key(
    manifests: list[ContextArtifactManifest],
    consumer: ConsumerIdentity,
    template_digest: str,
    preceding_prompt_digest: str,
) -> str:
    """Bind ordered payload composition and placement into a cache identity."""
    consumer.validate()
    digest(template_digest)
    digest(preceding_prompt_digest)
    for manifest in manifests:
        manifest.validate()
        if consumer not in manifest.consumers:
            raise ArtifactError("incompatible attachment consumer")
    return hashlib.sha256(
        canonical(
            {
                "artifacts": [m.artifact_id for m in manifests],
                "consumer": dataclasses.asdict(consumer),
                "template": template_digest,
                "preceding_prompt": preceding_prompt_digest,
            }
        )
    ).hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ContextArtifactRegistry:
    """Publish a verified payload before making immutable metadata discoverable.

    File and directory fsync precede SQLite FULL-synchronous publication.
    Eviction removes only this process's working bytes. Restore verifies the
    durable file again. This does not assert hardware power-loss guarantees.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        missing = []
        parent = self.root
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(exist_ok=True)
            _fsync_dir(directory.parent)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        _fsync_dir(self.root)
        self.db = sqlite3.connect(self.root / "registry.sqlite3")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS artifacts "
            "(id TEXT PRIMARY KEY, manifest TEXT NOT NULL)"
        )
        self.db.commit()
        _fsync_dir(self.root)
        self.resident: dict[str, bytes] = {}

    def close(self) -> None:
        self.resident.clear()
        self.db.close()

    def register(self, manifest: ContextArtifactManifest, source: str | Path) -> str:
        manifest.validate()
        artifact_id = manifest.artifact_id
        with (self.root / ".publish.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            target = self.objects / artifact_id
            if not target.exists():
                stage = Path(tempfile.mkdtemp(prefix=".import-", dir=self.objects))
                try:
                    payload = stage / "payload.safetensors"
                    with Path(source).open("rb") as src, payload.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    self._verify(manifest, payload)
                    with (stage / "manifest.json").open("wb") as stream:
                        stream.write(canonical(manifest.to_dict()))
                        stream.flush()
                        os.fsync(stream.fileno())
                    _fsync_dir(stage)
                    stage.rename(target)
                    _fsync_dir(self.objects)
                finally:
                    if stage.exists():
                        shutil.rmtree(stage)
            self._verify(manifest, target / "payload.safetensors")
            if (target / "manifest.json").read_bytes() != canonical(manifest.to_dict()):
                raise ArtifactError("existing durable manifest mismatch")
            _fsync_dir(self.objects)
            self.db.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?)",
                (artifact_id, canonical(manifest.to_dict()).decode()),
            )
            self.db.commit()
        return artifact_id

    @staticmethod
    def _verify(manifest: ContextArtifactManifest, payload: Path) -> None:
        if digest_file(payload) != manifest.payload_digest:
            raise ArtifactError("payload digest mismatch")
        if inspect_payload(payload) != manifest.tensors:
            raise ArtifactError("payload tensor metadata mismatch")

    def resolve(
        self, artifact_id: str, consumer: ConsumerIdentity
    ) -> ContextArtifactManifest:
        digest(artifact_id, "artifact ID")
        consumer.validate()
        row = self.db.execute(
            "SELECT manifest FROM artifacts WHERE id=?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise ArtifactError(f"required artifact is missing: {artifact_id}")
        manifest = ContextArtifactManifest.from_dict(json.loads(row[0]))
        if manifest.artifact_id != artifact_id or consumer not in manifest.consumers:
            raise ArtifactError("manifest identity or consumer mismatch")
        if (self.objects / artifact_id / "manifest.json").read_bytes() != canonical(
            manifest.to_dict()
        ):
            raise ArtifactError("durable manifest mismatch")
        self._verify(manifest, self.objects / artifact_id / "payload.safetensors")
        return manifest

    def load(self, artifact_id: str, consumer: ConsumerIdentity) -> bytes:
        manifest = self.resolve(artifact_id, consumer)
        payload = (self.objects / artifact_id / "payload.safetensors").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.payload_digest:
            raise ArtifactError("payload changed during load")
        self.resident[artifact_id] = payload
        return payload

    def evict(self, artifact_id: str) -> bool:
        return self.resident.pop(artifact_id, None) is not None

    def restore(self, artifact_id: str, consumer: ConsumerIdentity) -> bytes:
        if artifact_id in self.resident:
            raise ArtifactError("evict the working copy before restore")
        return self.load(artifact_id, consumer)
