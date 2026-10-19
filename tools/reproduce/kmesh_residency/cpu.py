# SPDX-License-Identifier: MIT
"""Qualify bounded family references through the pinned real vLLM allocator."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .lease import FamilyLeases

VLLM_COMMIT = "1b76bf9dd9d52efbd75fb89bf2619898d8f01233"


def authenticate(source):
    source = Path(source).resolve()
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != VLLM_COMMIT
    ):
        raise ValueError("vLLM source does not match the qualified serving branch")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise ValueError("vLLM source is dirty")
    import vllm.v1.core.kv_cache_manager as allocator

    if (
        Path(allocator.__file__).resolve()
        != source / "vllm/v1/core/kv_cache_manager.py"
    ):
        raise ValueError("allocator did not load from the authenticated source")
    return source


def load_helpers(source):
    spec = importlib.util.spec_from_file_location(
        "kmesh_allocator_helpers", source / "benchmarks/kmesh/fork_block_accounting.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.init_none_hash(module.sha256)
    return module


def qualify(helpers):
    """Use a synthetic request order to check lifetimes and displaced hits."""
    make_manager, make_request = helpers.make_manager, helpers.make_request

    def admit(manager, name, tokens):
        request = make_request(name, tokens)
        cached, hits, _ = manager.get_computed_blocks(request)
        if (
            manager.allocate_slots(request, request.num_tokens - hits, hits, cached)
            is None
        ):
            raise RuntimeError("controlled metadata request did not fit")
        return request, hits

    manager = make_manager(33, True)
    pool = manager.block_pool
    leases = FamilyLeases(pool, 16, 8, 2)
    parent, _ = admit(manager, "parent", list(range(128)))
    blocks = manager.get_blocks("parent").blocks[0][:8]
    token = leases.acquire_completed("a", blocks, now_ns=0, ttl_ns=100)
    second = leases.acquire_completed("b", blocks, now_ns=0, ttl_ns=100)
    assert leases.snapshot()["unique_held_blocks"] == 8
    assert leases.snapshot()["family_block_references"] == 16
    assert all(b.ref_cnt == 3 for b in blocks)
    assert leases.acquire_completed("third", blocks, now_ns=0, ttl_ns=100) is None
    manager.free(parent)
    assert all(b.ref_cnt == 2 for b in blocks)
    assert leases.release(token, "cancel")
    assert not leases.release(token, "cancel")
    assert all(b.ref_cnt == 1 for b in blocks)
    leases.expire(100)
    assert all(b.ref_cnt == 0 for b in blocks)
    parent, hits = admit(manager, "again", list(range(128)) + [999])
    assert hits == 128
    fresh = leases.acquire_completed("a", blocks, now_ns=101, ttl_ns=100)
    assert not leases.release(token)
    assert leases.snapshot()["families"] == 1
    other, _ = admit(manager, "other", list(range(1000, 1128)))
    other_blocks = manager.get_blocks("other").blocks[0][:8]
    assert (
        leases.acquire_completed("too-large", other_blocks, now_ns=101, ttl_ns=100)
        is None
    )
    assert all(b.ref_cnt == 1 for b in other_blocks)
    for bad in ([pool.null_block], [blocks[0], blocks[0]], [blocks[1]], []):
        try:
            leases.acquire_completed("bad", bad, now_ns=101, ttl_ns=100)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid block hold was accepted")
    leases.release(fresh)
    manager.free(parent)
    manager.free(other)
    leases.close()
    assert pool.get_num_free_blocks() == 32
    lifecycle = leases.snapshot()

    from .serving import capture
    from vllm.v1.request import RequestStatus

    manager = make_manager(129, True)
    leases = FamilyLeases(manager.block_pool, 16, 32, 2)
    identities = {"observed-parent": {"role": "parent", "family_id": "observed"}}
    with capture(manager, leases, identities, True) as (events, held):
        request, _ = admit(manager, "observed-parent", list(range(513)))
        request.num_computed_tokens = 513
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        manager.free(request)
        assert leases.snapshot()["unique_held_blocks"] == 32
        child, hits = admit(manager, "observed-child", list(range(513)) + [999])
        assert hits >= 512
        manager.free(child)
    assert manager.block_pool.get_num_free_blocks() == 128
    assert any(e["action"] == "lease-acquired" for e in events)
    assert not any(b.ref_cnt for b in manager.block_pool.blocks if not b.is_null)

    scenarios = []
    # Eight ancestor blocks in a 32-block usable pool. Four unrelated eight-block
    # requests replace inactive prefixes. Replay that competing working set too.
    for enabled in (False, True):
        manager = make_manager(33, True)
        leases = FamilyLeases(manager.block_pool, 16, 8, 2)
        parent_tokens = list(range(128))
        parent, _ = admit(manager, "parent", parent_tokens + [999])
        token = None
        if enabled:
            token = leases.acquire_completed(
                "family",
                manager.get_blocks("parent").blocks[0][:8],
                now_ns=0,
                ttl_ns=100,
            )
        manager.free(parent)
        competitor_hits = []
        for cycle in range(2):
            for index in range(4):
                tokens = list(range(1000 + 200 * index, 1128 + 200 * index)) + [999]
                request, hits = admit(manager, f"competitor-{cycle}-{index}", tokens)
                competitor_hits.append(hits)
                manager.free(request)
        child, child_hits = admit(manager, "child", parent_tokens + [999, 888])
        if enabled:
            assert child_hits == 128
            assert manager.get_block_ids("child")[0][:8] == [
                b.block_id for b in leases.active["family"][1]
            ]
            leases.release(token)
        else:
            assert child_hits == 0
        manager.free(child)
        leases.close()
        assert manager.block_pool.get_num_free_blocks() == 32
        scenarios.append(
            {
                "lease": enabled,
                "child_hit_tokens": child_hits,
                "competitor_hit_tokens": competitor_hits,
                "lease_state": leases.snapshot(),
            }
        )
    return {
        "schema": "knlp.kmesh.lease-cpu.v1",
        "synthetic_metadata_only": True,
        "gpu_execution": False,
        "usable_pool_blocks": 32,
        "ancestor_blocks": 8,
        "block_size_tokens": 16,
        "lifecycle": lifecycle,
        "scenarios": scenarios,
        "limits": [
            "Allocator metadata only: successful allocations stand in for completed model execution.",
            "The controlled request order is synthetic; it is not a measured production trace.",
            "No model correctness, GPU synchronization, storage traffic or latency claim.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        parser.error("qualification requires Python assertions enabled")
    if args.output.exists():
        parser.error("output must be fresh")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error(
            "hide GPUs with CUDA_VISIBLE_DEVICES='' for metadata qualification"
        )
    source = authenticate(args.source)
    import torch
    import vllm

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU qualification initialized CUDA")
    result = qualify(load_helpers(source))
    result["source_revision"] = VLLM_COMMIT
    result["runtime"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "vllm": vllm.__version__,
    }
    args.output.mkdir(parents=True)
    (args.output / "cpu.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
