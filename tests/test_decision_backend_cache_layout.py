# SPDX-License-Identifier: MIT
"""Use the pinned backend's explicit KV axis, including ambiguous shapes."""

from types import SimpleNamespace

import pytest
import torch

from tools.kv.decision.paged_memory import inventory
from tools.kv.decision.production_numerics import planes
from tools.kv.native import instrument


def cache_fixture(blocks):
    # The current FlashAttention/FlashInfer backend contract is B,2,N,H,D.
    # Distinct block and component values expose choosing the wrong axis even
    # when both leading dimensions are two. No CUDA allocation is required.
    cache = torch.empty(blocks, 2, 16, 3, 8, dtype=torch.float16)
    for block in range(blocks):
        for component in range(2):
            cache[block, component].fill_(100 * block + 10 * component + 1)
    return cache


@pytest.mark.parametrize("blocks", [2, 3])
@pytest.mark.parametrize(
    "implementation", ["FlashAttentionImpl", "FlashInferImpl", "TritonAttentionImpl"]
)
def test_numerical_views_keep_block_and_component_identity(blocks, implementation):
    cache = cache_fixture(blocks)
    key, value = planes(cache, implementation)
    assert key.shape == value.shape == (blocks, 16, 3, 8)
    for block in range(blocks):
        assert torch.all(key[block] == 100 * block + 1)
        assert torch.all(value[block] == 100 * block + 11)
    for component, view in enumerate((key, value)):
        assert view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert view.storage_offset() == component * 16 * 3 * 8
        assert view.stride() == (2 * 16 * 3 * 8, 3 * 8, 8, 1)


@pytest.mark.parametrize("blocks", [2, 3])
def test_memory_inventory_uses_current_fa3_views_before_cuda_accounting(
    monkeypatch, blocks
):
    cache = cache_fixture(blocks)
    implementation = type("FlashAttentionImpl", (), {})()
    layer = SimpleNamespace(
        kv_cache=cache,
        impl=implementation,
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.attn", layer)])
    worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))
    observed = []

    class ReachedScaleInventory(Exception):
        pass

    def capture(value):
        if isinstance(value, tuple):
            raise ReachedScaleInventory
        observed.append(value)
        return [{"dtype": str(value.dtype), "shape": list(value.shape)}]

    monkeypatch.setattr(instrument, "tensor_inventory", capture)
    with pytest.raises(ReachedScaleInventory):
        inventory(
            worker,
            "native_fa3",
            geometry={"layers": 1, "block_size": 16, "kv_heads": 3, "head_dim": 8},
            persistent_budget_bytes=4096,
        )
    assert len(observed) == 2
    for block in range(blocks):
        assert torch.all(observed[0][block] == 100 * block + 1)
        assert torch.all(observed[1][block] == 100 * block + 11)


@pytest.mark.parametrize(
    "cache,implementation",
    [
        (torch.empty(2, 3, 16, 3, 8), "FlashAttentionImpl"),
        (torch.empty(3, 2, 16, 3, 8), "UnknownImpl"),
        (torch.empty(3, 16, 3, 8), "FlashAttentionImpl"),
        ([torch.empty(3, 16, 3, 8)] * 2, "FlashAttentionImpl"),
    ],
)
def test_unqualified_tensor_layouts_are_not_guessed(cache, implementation):
    with pytest.raises(ValueError):
        planes(cache, implementation)


def test_asymmetric_tuple_preserves_its_two_existing_planes():
    cache = (torch.empty(3, 16, 3, 8), torch.empty(3, 16, 3, 8))
    assert planes(cache, "FlashInferImpl") is cache


def test_native_memory_inventory_does_not_admit_an_asymmetric_tuple():
    layer = SimpleNamespace(
        kv_cache=(torch.empty(3, 16, 3, 8), torch.empty(3, 16, 3, 8)),
        impl=type("FlashAttentionImpl", (), {})(),
        _k_scale=torch.tensor(1.0),
    )
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.attn", layer)])
    worker = SimpleNamespace(model_runner=SimpleNamespace(get_model=lambda: model))
    with pytest.raises(ValueError, match="five-dimensional"):
        inventory(worker, "native_fa3", geometry={}, persistent_budget_bytes=4096)
