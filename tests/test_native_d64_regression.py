# SPDX-License-Identifier: MIT
"""Check the independent reference and physical fixture assembly on CPU."""

import pytest
import torch

from tools.kv.native.d64_regression import populate_cache, reference


def test_uniform_grouped_attention_has_known_causal_means():
    query = torch.zeros(2, 4, 2)
    key = torch.zeros(3, 2, 2)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    full = reference(query, key, value, causal=False)
    causal = reference(query, key, value, causal=True)
    expected_full = value.mean(0).repeat_interleave(2, dim=0)
    assert torch.equal(full[0], expected_full)
    assert torch.equal(full[1], expected_full)
    assert torch.equal(causal[0], value[:2].mean(0).repeat_interleave(2, dim=0))
    torch.testing.assert_close(causal[1], expected_full)


@pytest.mark.parametrize("mixed", [True, False])
def test_fixture_updates_only_its_remapped_physical_slots(mixed):
    shape = (4, 16, 2, 64)
    n = 4 * 16 * 2 * 64
    storage = torch.zeros(n * (3 if mixed else 4), dtype=torch.uint8)
    key = storage[: n * 2].view(torch.float16).view(shape)
    value = (
        storage[n * 2 :]
        .view(torch.float8_e4m3fn if mixed else torch.float16)
        .view(shape)
    )
    slots = torch.tensor(list(range(48, 64)) + list(range(16)) + [32])
    positions = torch.arange(33 * 2 * 64).float().reshape(33, 2, 64)
    k0, v0 = populate_cache(key, value, slots, positions, 1 / 64 if mixed else 1, 0)
    before = storage.clone()
    k1, v1 = populate_cache(key, value, slots, positions, 1 / 64 if mixed else 1, 1)
    assert not torch.equal(before, storage)
    assert not torch.equal(k0, k1) and not torch.equal(v0, v1)
    assert torch.equal(key.flatten(0, 1)[slots], k1)
    observed = value.float().flatten(0, 1)[slots] * (1 / 64 if mixed else 1)
    assert torch.equal(observed, v1)
    unused = torch.ones(64, dtype=torch.bool)
    unused[slots] = False
    assert torch.count_nonzero(key.flatten(0, 1)[unused]) == 0
    assert torch.count_nonzero(value.float().flatten(0, 1)[unused]) == 0
