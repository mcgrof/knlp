# SPDX-License-Identifier: MIT
"""Typed artifact leases through the cartridge catalog and LMCache storage."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import threading
from typing import Any

from .artifacts import ArtifactError, ContextArtifactManifest, inspect_payload


@dataclass(frozen=True)
class _TensorLayout:
    header: bytes
    start: int
    tensors: tuple


@dataclass
class _PreparedArtifact:
    manifest: ContextArtifactManifest
    read_plan: Any
    layout: _TensorLayout | None = None


class LMCacheArtifactRegistry:
    """Use an existing cartridge registry for metadata and LMCache for bytes.

    Consumers must keep a lease until all tensor views and asynchronous
    transfers finish. Borrowed CPU tensors must not be modified in place.
    Publication accepts existing durable immutable files; it does not make
    an independently mutable source safe to share.
    """

    def __init__(self, catalog, allocator, max_prepared=1024):
        from lmcache.v1.context_artifacts import ArtifactStore

        self.catalog = catalog
        self.storage = ArtifactStore(catalog, allocator, max_read_plans=max_prepared)
        self._max_prepared = max_prepared
        self._prepared = OrderedDict()
        self._lock = threading.RLock()

    def register(self, manifest, source, offset=0, length=None):
        manifest.validate()
        source = Path(source).resolve(strict=True)
        length = source.stat().st_size - offset if length is None else length
        if offset == 0 and length == source.stat().st_size:
            if inspect_payload(source) != manifest.tensors:
                raise ArtifactError("payload tensor metadata mismatch")
        self.storage.publish(
            manifest.artifact_id,
            manifest.to_dict(),
            source,
            manifest.payload_digest,
            length,
            offset,
        )
        return manifest.artifact_id

    def resolve(self, artifact_id, consumer):
        """Return an admitted immutable manifest after exact consumer checking."""
        return self._admit(artifact_id, consumer).manifest

    @contextmanager
    def lease(self, artifact_id, consumer, *, timeout=0.0):
        """Borrow tensor views without copying the complete file into bytes."""
        import torch

        prepared = self._admit(artifact_id, consumer)
        with self.storage.lease(
            artifact_id, plan=prepared.read_plan, timeout=timeout
        ) as obj:
            buffer = memoryview(obj.byte_array).cast("B")
            if len(buffer) != prepared.read_plan.length:
                raise ArtifactError("payload length differs from admitted read plan")
            with self._lock:
                if prepared.layout is None:
                    prepared.layout = self._layout(buffer, prepared.manifest, torch)
                layout = prepared.layout
            # Cache the parsing and geometry, not successful verification of
            # future bytes. Misses still hash the complete newly loaded range.
            if buffer[: layout.start] != layout.header:
                raise ArtifactError("safetensors header differs from admitted layout")
            tensors = {}
            for name, dtype, count, offset, shape in layout.tensors:
                tensors[name] = torch.frombuffer(
                    buffer, dtype=dtype, count=count, offset=offset
                ).reshape(shape)
            yield tensors

    def evict(self, artifact_id):
        return self.storage.evict(artifact_id)

    def close(self):
        self.storage.close()
        with self._lock:
            self._prepared.clear()

    def _admit(self, artifact_id, consumer):
        with self._lock:
            prepared = self._prepared.get(artifact_id)
            if prepared is None:
                plan = self.storage.prepare(artifact_id)
                manifest = ContextArtifactManifest.from_dict(
                    json.loads(plan.manifest_json)
                )
                if manifest.artifact_id != artifact_id:
                    raise ArtifactError("artifact identity or consumer mismatch")
                if plan.sha256 != manifest.payload_digest:
                    raise ArtifactError("artifact digest metadata mismatch")
                prepared = _PreparedArtifact(manifest, plan)
                self._prepared[artifact_id] = prepared
                if len(self._prepared) > self._max_prepared:
                    self._prepared.popitem(last=False)
            if consumer not in prepared.manifest.consumers:
                raise ArtifactError("artifact identity or consumer mismatch")
            self._prepared.move_to_end(artifact_id)
            return prepared

    @staticmethod
    def _layout(buffer, manifest, torch):
        if len(buffer) < 8:
            raise ArtifactError("short safetensors header")
        start = 8 + struct.unpack_from("<Q", buffer)[0]
        if start > len(buffer):
            raise ArtifactError("safetensors header exceeds payload")
        header_bytes = bytes(buffer[:start])
        try:
            header = json.loads(header_bytes[8:])
            specs = {s.name: s for s in manifest.tensors}
            if set(header) - {"__metadata__"} != set(specs):
                raise ArtifactError("tensor names differ from manifest")
            tensors, occupied = [], []
            for name, spec in specs.items():
                entry = header[name]
                begin, end = entry["data_offsets"]
                dtype = {
                    "F16": torch.float16,
                    "BF16": torch.bfloat16,
                    "F32": torch.float32,
                }[spec.dtype]
                count = 1
                for dim in spec.shape:
                    count *= dim
                if (
                    type(begin) is not int
                    or type(end) is not int
                    or entry["dtype"] != spec.dtype
                    or tuple(entry["shape"]) != spec.shape
                    or begin < 0
                    or end > len(buffer) - start
                    or end - begin != count * dtype.itemsize
                ):
                    raise ArtifactError("invalid tensor range or geometry")
                occupied.append((begin, end))
                tensors.append((name, dtype, count, start + begin, spec.shape))
            ranges = sorted(occupied)
            if (
                not ranges
                or ranges[0][0] != 0
                or ranges[-1][1] != len(buffer) - start
                or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))
            ):
                raise ArtifactError("tensor ranges overlap or leave unclaimed bytes")
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError("invalid safetensors layout") from error
        return _TensorLayout(header_bytes, start, tuple(tensors))
