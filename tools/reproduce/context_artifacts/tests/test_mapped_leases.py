# SPDX-License-Identifier: MIT
"""Typed mapped views preserve consumer identity and backing lifetime."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("lmcache.v1.mapped_artifacts")
torch = pytest.importorskip("torch")

from ..artifacts import ArtifactError, ConsumerIdentity
from ..lmcache_adapter import LMCacheArtifactRegistry
from .test_artifacts import CONSUMER, fixture as artifact_fixture
from .test_prepared_leases import Catalog


@pytest.mark.parametrize("verification", ["published", "on-map"])
def test_prepared_reader_reuses_views_and_escaped_tensor_keeps_backing(
    tmp_path, verification
):
    path, manifest = artifact_fixture(tmp_path)
    registry = LMCacheArtifactRegistry(
        Catalog(), backing="mmap", verification=verification
    )
    key = registry.register(manifest, path)
    with pytest.raises(ArtifactError, match="consumer"):
        registry.prepare_reader(key, ConsumerIdentity("wrong", "a" * 40))
    reader = registry.prepare_reader(key, CONSUMER)
    with reader.lease() as first:
        tensor = first["embeddings"]
    for _ in range(32):
        with reader.lease() as second:
            assert second["embeddings"].data_ptr() == tensor.data_ptr()
    assert registry.storage.map_count == 1
    assert registry.storage.read_bytes == 0
    assert registry.evict(key)
    with reader.lease() as after:
        assert torch.equal(after["embeddings"], tensor)
    registry.close()
    del first, second, after
    assert torch.all(tensor == 1)
    with pytest.raises(RuntimeError, match="closed"):
        with reader.lease():
            pass


def test_parallel_prepared_borrows_and_owned_batch_survive_cache_close(tmp_path):
    path, manifest = artifact_fixture(tmp_path)
    registry = LMCacheArtifactRegistry(
        Catalog(), backing="mmap", verification="published"
    )
    key = registry.register(manifest, path)
    reader = registry.prepare_reader(key, CONSUMER)

    def consume(_):
        with reader.lease() as values:
            return values["embeddings"].to(torch.bfloat16)

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(consume, range(128)))
    assert registry.storage.map_count == 1
    registry.close()
    assert all(torch.all(row == 1) for row in rows)
