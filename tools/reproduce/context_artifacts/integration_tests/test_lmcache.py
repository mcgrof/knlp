# SPDX-License-Identifier: MIT
"""Real cartridge catalog and LMCache allocator/storage integration checks."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch
from safetensors.torch import load_file
from lmcache.v1.memory_allocators.host_memory_allocator import HostMemoryAllocator
from vllm.cartridge_registry import CartridgeRegistry

from tools.reproduce.context_artifacts.artifacts import ArtifactError, ConsumerIdentity, ContextArtifactManifest
from tools.reproduce.context_artifacts.lmcache_adapter import LMCacheArtifactRegistry
from tools.reproduce.context_artifacts.tests.test_artifacts import fixture, CONSUMER


def store_at(tmp_path, size=1 << 20):
    catalog = CartridgeRegistry(tmp_path / "catalog.sqlite3")
    allocator = HostMemoryAllocator(size)
    return LMCacheArtifactRegistry(catalog, allocator), catalog, allocator


@pytest.mark.parametrize(
    "kind", ["document_embeddings", "input_embeddings", "prefix_kv"]
)
def test_shared_catalog_roundtrip(tmp_path, kind):
    path, manifest = fixture(tmp_path, kind=kind)
    store, catalog, allocator = store_at(tmp_path)
    key = store.register(manifest, path)
    assert catalog.lookup_artifact(key)["manifest"]["representation"] == kind
    original = load_file(path)
    with store.lease(key, CONSUMER) as tensors:
        assert not store.evict(key)
        for name in original:
            assert torch.equal(tensors[name], original[name])
        with pytest.raises(RuntimeError):
            store.close()
    assert store.evict(key)
    with store.lease(key, CONSUMER) as tensors:
        assert all(torch.equal(tensors[k], v) for k, v in original.items())
    assert store.storage.read_count == 2
    store.close()
    assert allocator.memcheck()
    catalog.close()


def test_bounded_range_and_resident_reuse(tmp_path):
    path, manifest = fixture(tmp_path)
    packed = tmp_path / "packed.bin"
    packed.write_bytes(b"left" * 123 + path.read_bytes() + b"right" * 100)
    store, catalog, _ = store_at(tmp_path)
    key = store.register(manifest, packed, offset=492, length=path.stat().st_size)
    with store.lease(key, CONSUMER) as first:
        pointer = first["embeddings"].data_ptr()
    with store.lease(key, CONSUMER) as second:
        assert second["embeddings"].data_ptr() == pointer
    assert store.storage.read_count == 1
    assert store.storage.cache_hits == 1
    assert store.storage.read_bytes == path.stat().st_size
    store.close()
    catalog.close()


def test_concurrent_misses_share_one_object(tmp_path):
    path, manifest = fixture(tmp_path)
    store, catalog, _ = store_at(tmp_path)
    key = store.register(manifest, path)
    barrier = threading.Barrier(8)

    def read(_):
        barrier.wait()
        with store.lease(key, CONSUMER) as tensors:
            return tensors["embeddings"].data_ptr(), float(tensors["embeddings"].sum())

    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(read, range(8)))
    assert len(set(values)) == 1
    assert store.storage.read_count == 1
    assert store.storage.cache_hits == 7
    store.close()
    catalog.close()


def test_wrong_consumer_rejects_before_io(tmp_path):
    path, manifest = fixture(tmp_path)
    store, catalog, _ = store_at(tmp_path)
    key = store.register(manifest, path)
    with pytest.raises(ArtifactError):
        with store.lease(key, ConsumerIdentity("other", "b" * 40)):
            pytest.fail("incompatible consumer acquired bytes")
    assert store.storage.read_count == 0
    store.close()
    catalog.close()


@pytest.mark.parametrize("damage", ["corrupt", "truncate"])
def test_damaged_storage_fails_and_releases_allocation(tmp_path, damage):
    path, manifest = fixture(tmp_path)
    store, catalog, allocator = store_at(tmp_path, size=4096)
    key = store.register(manifest, path)
    raw = path.read_bytes()
    path.write_bytes(
        raw[:-1] if damage == "truncate" else raw[:-1] + bytes([raw[-1] ^ 1])
    )
    with pytest.raises((ValueError, EOFError)):
        with store.lease(key, CONSUMER):
            pytest.fail("damaged payload acquired")
    path.write_bytes(raw)
    with store.lease(key, CONSUMER) as tensors:
        assert torch.equal(tensors["embeddings"], load_file(path)["embeddings"])
    store.close()
    assert allocator.memcheck()
    catalog.close()


def test_allocator_pressure_evicts_idle_only(tmp_path):
    store, catalog, allocator = store_at(tmp_path, size=4096)
    keys = []
    for value in range(12):
        path, manifest = fixture(tmp_path, value=value)
        key = store.register(manifest, path)
        keys.append(key)
        with store.lease(key, CONSUMER) as tensors:
            assert bool((tensors["embeddings"] == value).all())
    assert len(store.storage.resident) < len(keys)
    with store.lease(keys[0], CONSUMER) as tensors:
        assert bool((tensors["embeddings"] == 0).all())
    store.close()
    assert allocator.memcheck()
    catalog.close()


def test_publication_identity_cannot_change_location_silently(tmp_path):
    path, manifest = fixture(tmp_path)
    store, catalog, _ = store_at(tmp_path)
    key = store.register(manifest, path)
    other = tmp_path / "copy.safetensors"
    other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        store.register(manifest, other)
    assert catalog.lookup_artifact(key)["path"] == str(path)
    store.close()
    catalog.close()


def test_legacy_cartridges_and_embeddings_share_catalog(tmp_path):
    from vllm.cartridge_manifest import CartridgeManifest
    store, catalog, _ = store_at(tmp_path)
    native = CartridgeManifest("legacy", "decoder", 2, 2, 4, "float16",
        16, 16, 16, 1, False, labels={"topic": "test"})
    catalog.register(native)
    assert catalog.lookup("legacy").to_dict() == native.to_dict()
    path, manifest = fixture(tmp_path)
    key = store.register(manifest, path)
    assert catalog.lookup_by_label("topic", "test")[0].cartridge_id == "legacy"
    assert ContextArtifactManifest.from_dict(catalog.lookup_artifact(key)["manifest"]).artifact_id == key
    native.labels = {"topic": "updated"}
    catalog.register(native)
    assert catalog.lookup_by_label("topic", "updated")[0].cartridge_id == "legacy"
    with store.lease(key, CONSUMER) as tensors:
        assert bool((tensors["embeddings"] == 1).all())
    assert catalog.unregister("legacy")
    assert catalog.lookup_artifact(key) is not None
    store.close()
    catalog.close()


def test_active_lease_prevents_pressure_reuse(tmp_path):
    store, catalog, allocator = store_at(tmp_path, size=4096)
    first, first_manifest = fixture(tmp_path, value=1)
    second, second_manifest = fixture(tmp_path, value=2)
    one = store.register(first_manifest, first)
    two = store.register(second_manifest, second)
    with store.lease(one, CONSUMER) as held:
        with pytest.raises(MemoryError):
            with store.lease(two, CONSUMER):
                pytest.fail("active lease was overwritten")
        assert bool((held["embeddings"] == 1).all())
    with store.lease(two, CONSUMER) as loaded:
        assert bool((loaded["embeddings"] == 2).all())
    store.close()
    assert allocator.memcheck()
    catalog.close()
