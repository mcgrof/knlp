# SPDX-License-Identifier: MIT
"""Typed prepared leases retain exact identity, geometry and ownership checks."""

from copy import deepcopy

import pytest

pytest.importorskip("lmcache.v1.context_artifacts")
torch = pytest.importorskip("torch")

from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator

from ..artifacts import ArtifactError, ConsumerIdentity
from ..lmcache_adapter import LMCacheArtifactRegistry
from .test_artifacts import CONSUMER, fixture as artifact_fixture


class Catalog:
    def __init__(self):
        self.records = {}

    def register_artifact(self, key, record):
        if key in self.records and self.records[key] != record:
            raise ValueError("conflicting immutable identity")
        self.records[key] = deepcopy(record)

    def lookup_artifact(self, key):
        return deepcopy(self.records.get(key))


@pytest.fixture
def registry():
    allocator = HostMemoryAllocator(65536)
    result = LMCacheArtifactRegistry(Catalog(), allocator, max_prepared=2)
    yield result
    result.close()
    assert allocator.memcheck()


@pytest.mark.parametrize(
    "kind", ["document_embeddings", "input_embeddings", "prefix_kv"]
)
def test_cached_layout_reloads_and_owned_copies(tmp_path, registry, kind):
    path, manifest = artifact_fixture(tmp_path, kind)
    key = registry.register(manifest, path)
    for _ in range(3):
        assert registry.resolve(key, CONSUMER) == manifest
        with registry.lease(key, CONSUMER) as tensors:
            owned = {name: value.clone() for name, value in tensors.items()}
        assert registry.evict(key)
    assert registry.storage.read_count == 3
    for value in owned.values():
        assert torch.all(value == 1)
    before = registry.storage.read_count
    with pytest.raises(ArtifactError, match="consumer"):
        with registry.lease(key, ConsumerIdentity("wrong", "a" * 40)):
            pass
    assert registry.storage.read_count == before
    path.write_bytes(path.read_bytes()[:-1] + b"!")
    with pytest.raises(ValueError, match="checksum"):
        with registry.lease(key, CONSUMER):
            pass


def test_cached_header_is_compared_on_every_lease(tmp_path, registry):
    path, manifest = artifact_fixture(tmp_path)
    key = registry.register(manifest, path)
    with registry.lease(key, CONSUMER):
        pass
    # Deliberately violate the borrowed-view contract to check the cached
    # layout cannot conceal a modified resident header.
    with registry.storage.lease(key) as obj:
        buffer = memoryview(obj.byte_array).cast("B")
        buffer[8] ^= 1
    with pytest.raises(ArtifactError, match="header"):
        with registry.lease(key, CONSUMER):
            pass
    assert registry.evict(key)
    with registry.lease(key, CONSUMER) as tensors:
        assert torch.all(tensors["embeddings"] == 1)


def test_metadata_cache_eviction_rechecks_identity(tmp_path, registry):
    ids = []
    for value in [1, 2, 3]:
        path, manifest = artifact_fixture(tmp_path, value=value)
        key = registry.register(manifest, path)
        ids.append(key)
        assert registry.resolve(key, CONSUMER) == manifest
    registry.catalog.records[ids[0]]["manifest"]["namespace"] = "changed"
    with pytest.raises(ArtifactError, match="identity"):
        registry.resolve(ids[0], CONSUMER)


def test_changed_durable_catalog_is_rejected_after_eviction(tmp_path, registry):
    path, manifest = artifact_fixture(tmp_path)
    key = registry.register(manifest, path)
    with registry.lease(key, CONSUMER):
        pass
    assert registry.evict(key)
    registry.catalog.records[key]["offset"] = 1
    with pytest.raises(ValueError, match="catalog record"):
        with registry.lease(key, CONSUMER):
            pass
