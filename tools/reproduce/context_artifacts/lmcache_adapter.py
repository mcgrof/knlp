# SPDX-License-Identifier: MIT
"""Typed artifact leases through the cartridge catalog and LMCache storage."""

from contextlib import contextmanager
import json
from pathlib import Path
import struct

from .artifacts import ArtifactError, ContextArtifactManifest, inspect_payload


class LMCacheArtifactRegistry:
    """Use an existing cartridge registry for metadata and LMCache for bytes.

    Consumers must keep a lease until all tensor views and asynchronous
    transfers finish. Borrowed CPU tensors must not be modified in place.
    Publication accepts existing durable immutable files; it does not make
    an independently mutable source safe to share.
    """

    def __init__(self, catalog, allocator):
        from lmcache.v1.context_artifacts import ArtifactStore

        self.catalog = catalog
        self.storage = ArtifactStore(catalog, allocator)

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
        record = self.storage.describe(artifact_id)
        manifest = ContextArtifactManifest.from_dict(record["manifest"])
        if manifest.artifact_id != artifact_id or consumer not in manifest.consumers:
            raise ArtifactError("artifact identity or consumer mismatch")
        if record["sha256"] != manifest.payload_digest:
            raise ArtifactError("artifact digest metadata mismatch")
        return manifest

    @contextmanager
    def lease(self, artifact_id, consumer):
        """Borrow tensor views without copying the complete file into bytes."""
        import torch

        manifest = self.resolve(artifact_id, consumer)
        with self.storage.lease(artifact_id) as obj:
            buffer = memoryview(obj.byte_array).cast("B")
            if len(buffer) < 8:
                raise ArtifactError("short safetensors header")
            header_size = struct.unpack_from("<Q", buffer)[0]
            start = 8 + header_size
            if start > len(buffer):
                raise ArtifactError("safetensors header exceeds payload")
            header = json.loads(bytes(buffer[8:start]))
            specs = {s.name: s for s in manifest.tensors}
            if set(header) - {"__metadata__"} != set(specs):
                raise ArtifactError("tensor names differ from manifest")
            tensors = {}
            occupied = []
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
                    entry["dtype"] != spec.dtype
                    or tuple(entry["shape"]) != spec.shape
                    or begin < 0
                    or end > len(buffer) - start
                    or end - begin != count * dtype.itemsize
                ):
                    raise ArtifactError("invalid tensor range or geometry")
                occupied.append((begin, end))
                tensors[name] = torch.frombuffer(
                    buffer, dtype=dtype, count=count, offset=start + begin
                ).reshape(spec.shape)
            ranges = sorted(occupied)
            if (
                not ranges
                or ranges[0][0] != 0
                or ranges[-1][1] != len(buffer) - start
                or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:]))
            ):
                raise ArtifactError("tensor ranges overlap or leave unclaimed bytes")
            yield tensors

    def evict(self, artifact_id):
        return self.storage.evict(artifact_id)

    def close(self):
        self.storage.close()
