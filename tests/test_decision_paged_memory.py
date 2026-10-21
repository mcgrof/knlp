# SPDX-License-Identifier: MIT
"""Do not mistake shared tensor views or a whole CUDA segment for cache bytes."""

import copy

import pytest

from tools.kv.decision.paged_memory import allocation_sizes


def inputs():
    stores = [
        {"device": "cuda:0", "storage_pointer": 1000, "storage_bytes": 4},
        {"device": "cuda:0", "storage_pointer": 1512, "storage_bytes": 1024},
    ]
    snapshot = [
        {
            "device": 0,
            "address": 1000,
            "total_size": 4096,
            "blocks": [
                {"state": "active_allocated", "requested_size": 4, "size": 512},
                {"state": "active_allocated", "requested_size": 1024, "size": 1024},
                {"state": "inactive", "requested_size": 0, "size": 2560},
            ],
        }
    ]
    return stores, snapshot


def test_scale_rounding_alias_deduplication_and_inactive_space():
    stores, snapshot = inputs()
    result = allocation_sizes([*stores, stores[1]], snapshot)
    assert result["allocation_count"] == 2
    assert result["requested_bytes"] == 1028
    assert result["allocated_block_bytes"] == 1536
    assert result["rounding_bytes"] == 508
    assert result["allocated_block_bytes"] != snapshot[0]["total_size"]


@pytest.mark.parametrize(
    "bad", ["missing", "size", "alias_size", "address", "segment", "state", "device"]
)
def test_unknown_or_inconsistent_allocations_fail_closed(bad):
    stores, snapshot = inputs()
    if bad == "missing":
        stores[0]["storage_pointer"] += 1
    elif bad == "size":
        stores[0]["storage_bytes"] = 8
    elif bad == "alias_size":
        stores.append(copy.deepcopy(stores[0]))
        stores[-1]["storage_bytes"] = 8
    elif bad == "address":
        snapshot[0]["blocks"][0]["address"] = 1001
    elif bad == "segment":
        snapshot[0]["total_size"] += 1
    elif bad == "state":
        snapshot[0]["blocks"][0]["state"] = "active_awaiting_free"
    else:
        stores[0]["device"] = "cuda:1"
    with pytest.raises(ValueError):
        allocation_sizes(stores, snapshot)
