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
from types import MappingProxyType
from weakref import WeakKeyDictionary

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


class PreparedArtifactReader:
    """Bind a checked consumer once; each borrow still acquires backing ownership."""

    def __init__(self, manifest, borrow, storage, backing):
        self.manifest = manifest
        self.lease = borrow
        self.storage = storage
        self.backing = backing
        self._views = None
        self._lock = threading.Lock()

    def views(self):
        """Borrow mapped tensors whose references retain their backing memory.

        An allocator-owned arena cannot supply this interface because it may
        reuse a buffer after lease release. Mapped tensors retain their mmap
        owner independently of cache eviction or registry close. Borrowing a
        new reference after close fails; existing references remain valid.
        The bounded prepared-reader cache can retain mappings outside the
        backing's lookup-entry budget, without pinning physical pages.
        Consumers must not modify borrowed tensor values.
        """
        if self.backing != "mmap":
            raise ValueError("unleased views require immutable mapped backing")
        if self.storage.closed:
            raise RuntimeError("artifact store is closed")
        views = self._views
        if views is None:
            with self._lock:
                if self._views is None:
                    with self.lease() as tensors:
                        self._views = tensors
                views = self._views
        return views

    def release_views(self):
        """Drop this reader's cached references; consumer tensors stay valid."""
        with self._lock:
            self._views = None


class LMCacheArtifactRegistry:
    """Use an existing cartridge registry for metadata and LMCache for bytes.

    Consumers must keep a lease until all tensor views and asynchronous
    transfers finish. Borrowed CPU tensors must not be modified in place.
    Publication accepts existing durable immutable files; it does not make
    an independently mutable source safe to share.
    """

    def __init__(
        self,
        catalog,
        allocator=None,
        max_prepared=1024,
        *,
        backing="arena",
        verification="on-map",
        max_mappings=512,
    ):
        from lmcache.v1.context_artifacts import ArtifactStore

        self.catalog = catalog
        if backing == "mmap":
            from lmcache.v1.mapped_artifacts import (
                ArtifactVerification,
                MappedArtifactStore,
            )

            self.storage = MappedArtifactStore(
                catalog,
                max_mappings=max_mappings,
                max_read_plans=max_prepared,
                verification=ArtifactVerification(verification),
            )
        elif backing == "arena":
            if allocator is None or verification != "on-map":
                raise ValueError(
                    "arena backing requires an allocator and load verification"
                )
            self.storage = ArtifactStore(
                catalog, allocator, max_read_plans=max_prepared
            )
        else:
            raise ValueError("unknown artifact backing")
        self.backing = backing
        self.verification = verification
        self._max_prepared = max_prepared
        self._prepared = OrderedDict()
        # Values retain mmap buffers, not their weak-key backing wrappers.
        # Eviction therefore drops cached views; escaped tensors retain only
        # their own mapping lifetime, never an allocator-reusable arena slot.
        self._mapped_views = WeakKeyDictionary()
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

    def prepare_reader(self, artifact_id, consumer, *, timeout=0.0):
        """Check identity/consumer once and return a reusable borrowed-view reader.

        The reader remains bound to this registry. Borrowing after close fails.
        Mapped tensors must not be modified, and async consumers must retain
        the lease until their transfer finishes.
        """
        from functools import partial

        prepared = self._admit(artifact_id, consumer)
        return PreparedArtifactReader(
            prepared.manifest,
            partial(self._lease_prepared, prepared, timeout=timeout),
            self.storage,
            self.backing,
        )

    @contextmanager
    def lease(self, artifact_id, consumer, *, timeout=0.0):
        """Borrow typed views with backing-specific loading and verification."""
        prepared = self._admit(artifact_id, consumer)
        with self._lease_prepared(prepared, timeout=timeout) as tensors:
            yield tensors

    @contextmanager
    def _lease_prepared(self, prepared, *, timeout):
        import torch

        with self.storage.lease(
            prepared.manifest.artifact_id, plan=prepared.read_plan, timeout=timeout
        ) as obj:
            if self.backing == "mmap":
                with self._lock:
                    cached = self._mapped_views.get(obj)
                if cached is not None:
                    if self.verification == "on-map":
                        buffer = obj.byte_array
                        if buffer[: prepared.layout.start] != prepared.layout.header:
                            raise ArtifactError(
                                "safetensors header differs from admitted layout"
                            )
                    yield cached
                    return
            buffer = memoryview(obj.byte_array).cast("B")
            if len(buffer) != prepared.read_plan.length:
                raise ArtifactError("payload length differs from admitted read plan")
            with self._lock:
                if prepared.layout is None:
                    prepared.layout = self._layout(buffer, prepared.manifest, torch)
                layout = prepared.layout
            # Cache parsing and geometry. The backing's policy controls full
            # payload verification; a fresh buffer still checks this header.
            if buffer[: layout.start] != layout.header:
                raise ArtifactError("safetensors header differs from admitted layout")
            tensors = {}
            for name, dtype, count, offset, shape in layout.tensors:
                tensors[name] = torch.frombuffer(
                    buffer, dtype=dtype, count=count, offset=offset
                ).reshape(shape)
            if self.backing == "mmap":
                tensors = MappingProxyType(tensors)
                with self._lock:
                    self._mapped_views[obj] = tensors
            yield tensors

    def evict(self, artifact_id):
        return self.storage.evict(artifact_id)

    def close(self):
        self.storage.close()
        with self._lock:
            self._prepared.clear()
            self._mapped_views.clear()

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
