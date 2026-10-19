# SPDX-License-Identifier: MIT
"""Capture a controlled family hold through synchronous real vLLM serving.

The request order deliberately stresses capacity; it is not a production trace.
No storage connector is installed in this test. Compare generated continuations
and actual cache events, without inferring a DMA-BUF or latency benefit.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .cpu import authenticate, VLLM_COMMIT
from .lease import FamilyLeases

BLOCK_SIZE = 16
ANCESTOR = 512
POOL_BLOCKS = 129


@contextmanager
def capture(manager, leases, identities, acquire, on_finish=None):
    """Observe allocator boundaries and acquire only at completed request free."""
    events, held = [], {}
    pool = manager.block_pool
    allocate, free, evict = (
        manager.allocate_slots,
        manager.free,
        pool._maybe_evict_cached_block,
    )

    def snapshot(action, request=None, **extra):
        groups = [] if request is None else manager.get_block_ids(request.request_id)
        ids = [] if not groups else list(groups[0])
        events.append(
            {
                "time_ns": time.perf_counter_ns(),
                "action": action,
                "request_id": None if request is None else request.request_id,
                "block_ids": ids,
                "ref_counts": [pool.blocks[i].ref_cnt for i in ids],
                "computed_tokens": (
                    None if request is None else request.num_computed_tokens
                ),
                "free_blocks": pool.get_num_free_blocks(),
                "lease_state": leases.snapshot(),
                **extra,
            }
        )

    def on_allocate(request, *args, **kwargs):
        result = allocate(request, *args, **kwargs)
        snapshot("allocated" if result is not None else "allocation-failed", request)
        return result

    def on_free(request):
        identity = identities.get(request.request_id)
        if acquire and identity is not None and identity["role"] == "parent":
            if (
                request.status.name
                not in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
                or request.num_computed_tokens < ANCESTOR
            ):
                raise RuntimeError("cannot hold an unfinished ancestor")
            groups = manager.get_blocks(request.request_id).blocks
            if len(groups) != 1:
                raise RuntimeError("controlled hold requires one full-attention group")
            token = leases.acquire_completed(
                identity["family_id"],
                groups[0][: ANCESTOR // BLOCK_SIZE],
                now_ns=time.perf_counter_ns(),
                ttl_ns=60_000_000_000,
            )
            if token is None:
                raise RuntimeError("controlled ancestor exceeded its lease budget")
            held[identity["family_id"]] = token
            snapshot("lease-acquired", request)
        snapshot("before-free", request)
        free(request)
        snapshot("after-free", freed_request_id=request.request_id)

    def on_evict(block):
        block_id, refs = block.block_id, block.ref_cnt
        was_cached = block.block_hash is not None
        result = evict(block)
        if result:
            snapshot(
                "evicted",
                block_id=block_id,
                prior_ref_count=refs,
                was_cached=was_cached,
            )
        return result

    manager.allocate_slots, manager.free, pool._maybe_evict_cached_block = (
        on_allocate,
        on_free,
        on_evict,
    )
    try:
        yield events, held
    finally:
        manager.allocate_slots, manager.free, pool._maybe_evict_cached_block = (
            allocate,
            free,
            evict,
        )
        leases.close()
        if on_finish is not None:
            on_finish(events)


def run(llm, output, samples):
    from vllm import SamplingParams

    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    if scheduler.scheduler_config.async_scheduling:
        raise RuntimeError("capture requires synchronous scheduling")
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    if len(pool.blocks) != POOL_BLOCKS:
        raise RuntimeError("unexpected controlled cache pool size")
    params = SamplingParams(
        temperature=0, max_tokens=8, ignore_eos=True, seed=0, logprobs=1
    )
    # Deterministic token sequences have no imported user conversation content.
    ancestor = [100 + i % 300 for i in range(ANCESTOR)]
    competitors = [
        [2000 + k * 1000 + i % 400 for i in range(496)] + [7] for k in range(4)
    ]
    clock = "monotonic-" + os.urandom(8).hex()
    records, cells = [], []
    expected = {}
    for sample in range(samples):
        for enabled in ((False, True) if sample % 2 == 0 else (True, False)):
            if not llm.reset_prefix_cache():
                raise RuntimeError("cache reset refused after drained leases")
            leases = FamilyLeases(pool, BLOCK_SIZE, ANCESTOR // BLOCK_SIZE, 2)
            identities = {}
            label = f"sample-{sample}-{'lease' if enabled else 'stock'}"
            family = label + "-family"
            with capture(
                manager,
                leases,
                identities,
                enabled,
                lambda events: (output / (label + "-events.json")).write_text(
                    json.dumps(events, indent=2, allow_nan=False) + "\n"
                ),
            ) as (events, held):

                def invoke(role, tokens, parent=None, ordinal=0):
                    name = label + f"-{role}-{ordinal}"
                    start, first_event = time.perf_counter_ns(), len(events)
                    leases.expire(start)
                    internal = engine.add_request(
                        name, {"prompt_token_ids": tokens}, params
                    )
                    identities[internal] = {"role": role, "family_id": family}
                    result = None
                    while engine.has_unfinished_requests():
                        if (time.perf_counter_ns() - start) > 60_000_000_000:
                            raise TimeoutError("controlled invocation exceeded bound")
                        for row in engine.step():
                            if row.finished:
                                if row.request_id != name or result is not None:
                                    raise RuntimeError("unexpected completed request")
                                result = row
                    end = time.perf_counter_ns()
                    if result is None or len(result.outputs) != 1:
                        raise RuntimeError("request did not finish")
                    sequence = result.outputs[0]
                    if (
                        len(sequence.token_ids) != 8
                        or sequence.finish_reason == "error"
                    ):
                        raise RuntimeError("bounded continuation failed")
                    scores = [
                        p[t].logprob
                        for p, t in zip(
                            sequence.logprobs, sequence.token_ids, strict=True
                        )
                    ]
                    key = (sample, role, ordinal)
                    value = (list(sequence.token_ids), scores)
                    if key in expected and expected[key] != value:
                        raise RuntimeError("stock and held continuations differ")
                    expected[key] = value
                    observation = {
                        "schema": "knlp.kmesh.invocation.v1",
                        "request_id": name,
                        "family_id": (
                            family
                            if role in ("parent", "child")
                            else label + f"-competitor-{ordinal % 4}"
                        ),
                        "parent_request_id": parent,
                        "engine_id": "controlled-engine",
                        "clock_id": clock,
                        "start_ns": start,
                        "end_ns": end,
                        "input_token_ids": tokens,
                        "output_token_ids": list(sequence.token_ids),
                        "status": "complete",
                        "cache_observations": events[first_event:].copy(),
                        "chosen_logprobs": scores,
                        "cached_tokens": result.num_cached_tokens,
                        "role": role,
                        "sample": sample,
                        "lease": enabled,
                    }
                    records.append(observation)
                    with (output / "invocations.jsonl").open("a") as stream:
                        stream.write(json.dumps(observation, allow_nan=False) + "\n")
                    return observation

                parent = invoke("parent", ancestor + [7])
                if enabled and (
                    family not in held
                    or leases.snapshot()["unique_held_blocks"] != ANCESTOR // BLOCK_SIZE
                ):
                    raise RuntimeError(
                        "completed parent did not leave bounded family refs"
                    )
                for cycle in range(2):
                    for index, prompt in enumerate(competitors):
                        invoke("competitor", prompt, ordinal=cycle * 4 + index)
                child = invoke(
                    "child",
                    parent["input_token_ids"]
                    + parent["output_token_ids"]
                    + [9000 + i for i in range(64)],
                    parent["request_id"],
                )
                if enabled:
                    if child["cached_tokens"] < ANCESTOR:
                        raise RuntimeError("held ancestor did not serve a local hit")
                    parent_ids = next(
                        e["block_ids"]
                        for e in parent["cache_observations"]
                        if e["action"] == "lease-acquired"
                    )[: ANCESTOR // BLOCK_SIZE]
                    child_ids = next(
                        e["block_ids"]
                        for e in child["cache_observations"]
                        if e["action"] == "allocated"
                    )[: ANCESTOR // BLOCK_SIZE]
                    if parent_ids != child_ids:
                        raise RuntimeError(
                            "child did not alias the held physical ancestor"
                        )
                    leases.release(held[family])
                leases.close()
                if any(b.ref_cnt for b in pool.blocks if not b.is_null):
                    raise RuntimeError("request or family references leaked")
                cells.append(
                    {
                        "sample": sample,
                        "lease": enabled,
                        "pool_blocks": len(pool.blocks),
                        "ancestor_blocks": ANCESTOR // BLOCK_SIZE,
                        "child_cached_tokens": child["cached_tokens"],
                        "competitor_cached_tokens": [
                            r["cached_tokens"]
                            for r in records[-10:]
                            if r["role"] == "competitor"
                        ],
                        "evictions": sum(e["action"] == "evicted" for e in events),
                        "lease_state": leases.snapshot(),
                        "final_free_blocks": pool.get_num_free_blocks(),
                        "final_nonnull_references": sum(
                            b.ref_cnt for b in pool.blocks if not b.is_null
                        ),
                        "events": events,
                    }
                )
                (output / "cells.json").write_text(
                    json.dumps(cells, indent=2, allow_nan=False) + "\n"
                )
    return {
        "schema": "knlp.kmesh.residency-serving.v1",
        "controlled_synthetic": True,
        "storage_connector": None,
        "samples_per_arm": samples,
        "pool_blocks": POOL_BLOCKS,
        "ancestor_tokens": ANCESTOR,
        "generated_tokens": 8,
        "exact_continuations": True,
        "cells": cells,
        "limits": [
            "This fixed synthetic request order establishes mechanism behavior, not a production-workload benefit.",
            "Allocator instrumentation is enabled: invocation durations are not a latency benchmark.",
            "No external storage, DMA-BUF route or cross-engine sharing is tested.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 2:
        parser.error("use fresh output and at least two samples per arm")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    authenticate(args.source)
    from vllm import LLM
    import torch
    import vllm

    args.output.mkdir(parents=True)
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        num_gpu_blocks_override=POOL_BLOCKS,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        async_scheduling=False,
        seed=0,
        gpu_memory_utilization=0.3,
    )
    try:
        result = run(llm, args.output, args.samples)
        result["source_revision"] = VLLM_COMMIT
        result["runtime"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "vllm": vllm.__version__,
        }
        config = args.model / "config.json"
        result["model_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        (args.output / "serving.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    finally:
        llm.llm_engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    main()
