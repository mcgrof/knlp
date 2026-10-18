# SPDX-License-Identifier: MIT
"""Run ordinary fork requests through vLLM and the LMCache MP connector."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from restore_common import (
    ANCESTOR,
    REVISION,
    cold_l1,
    compact_core_pair,
    http,
    prompts,
    write_json,
)


@contextmanager
def record_prefix_blocks(llm):
    """Observe allocation IDs for an untimed integrity control, then unhook."""
    manager = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    allocate = manager.allocate_slots
    tables = {}

    def observe(request, *args, **kwargs):
        result = allocate(request, *args, **kwargs)
        if result is not None:
            groups = manager.get_block_ids(request.request_id)
            if len(groups) != 1:
                raise RuntimeError(
                    "the integrity control requires one full-attention KV group"
                )
            if len(groups[0]) >= ANCESTOR // 16:
                tables[request.request_id] = list(groups[0][: ANCESTOR // 16])
        return result

    manager.allocate_slots = observe
    try:
        yield tables
    finally:
        manager.allocate_slots = allocate


def prefix_checksums(config, tables, producer_ids=()):
    """Hash actual model KV through the public MP endpoint outside timing."""
    status = http(config["http_port"], "status")
    ids = set(status["registered_gpu_ids"]) - set(producer_ids)
    if len(ids) != 1 or not tables:
        raise RuntimeError("cannot identify the live model's registered KV context")
    instance_id = ids.pop()
    return {
        "instance_id": instance_id,
        "members": [
            {
                "request_id": name,
                "block_ids": blocks,
                "checksums": http(
                    config["http_port"],
                    "cache/checksums",
                    {
                        "instance_id": instance_id,
                        "block_ids": blocks,
                        "chunk_size": 16,
                        "layerwise": True,
                    },
                ),
            }
            for name, blocks in tables.items()
        ],
    }


def main(config: dict, role: str) -> None:
    """Store the ancestor or measure cold requests with an empty engine cache."""
    # Keep allocator metadata in this process only for the separate integrity
    # control. Measured requests have no allocation hook or checksum call.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    output = Path(config["output"])
    ancestor, children = prompts()
    kwargs = dict(
        model=config["model"],
        revision=REVISION,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=512 << 20,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        seed=0,
        block_size=16,
    )
    if role != "oracle":
        kwargs["kv_transfer_config"] = {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_producer" if role == "producer" else "kv_consumer",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": config["mq_port"],
                "lmcache.mp.isolated_ipc": True,
                "lmcache.mp.mq_timeout": 30.0,
                "lmcache.mp.heartbeat_interval": 1.0,
            },
        }
    llm = LLM(**kwargs)
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        seed=0,
        logprobs=1,
        extra_args=(
            None
            if role == "oracle"
            else {"kv_transfer_params": {"cached_token_stats": {}}}
        ),
    )

    def generate(batch: list[list[int]]) -> tuple[list[dict], float]:
        start = time.perf_counter_ns()
        results = llm.generate(
            [{"prompt_token_ids": p} for p in batch], params, use_tqdm=False
        )
        elapsed = (time.perf_counter_ns() - start) / 1e6
        rows = []
        for result in results:
            sequence = result.outputs[0]
            if sequence.finish_reason == "error" or len(sequence.token_ids) != 1:
                raise RuntimeError("vLLM failed to produce the bounded continuation")
            if result.metrics is None or result.metrics.first_token_latency <= 0:
                raise RuntimeError("vLLM did not return a first-token latency")
            rows.append(
                {
                    "request_id": result.request_id,
                    "token_ids": sequence.token_ids,
                    "logprob": sequence.logprobs[0][sequence.token_ids[0]].logprob,
                    "ttft_ms": result.metrics.first_token_latency * 1000,
                    "kv_transfer_params": getattr(result, "kv_transfer_params", None),
                    "num_cached_tokens": result.num_cached_tokens,
                }
            )
        return rows, elapsed

    records = []
    try:
        # Exercise the same model path before any measured request.
        generate([[44, 55, 66, 77, 88] * 20])
        if role == "producer":
            # One extra token lets the connector store all sixteen complete
            # ancestor chunks while retaining its final-token recompute rule.
            with record_prefix_blocks(llm) as tables:
                rows, elapsed = generate([ancestor + [7]])
            checksums = prefix_checksums(config, tables)
            if len(checksums["members"]) != 1:
                raise RuntimeError("producer prefix integrity control lost its request")
            write_json(output / "producer-prefix.json", checksums)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = http(config["http_port"], "status")["storage_manager"]
                if (
                    status["l2_adapters"][0]["core"]["indexed_key_count"]
                    == ANCESTOR // 256
                ):
                    cold_l1(config["http_port"])
                    write_json(
                        output / "producer.json",
                        {"rows": rows, "elapsed_ms": elapsed, "status": status},
                    )
                    return
                time.sleep(0.02)
            raise TimeoutError("ancestor did not reach the NVMe-backed index")
        for fanout in (1, 5):
            for sample in range(config["serving_samples"] + 3):
                if not llm.reset_prefix_cache():
                    raise RuntimeError("engine prefix-cache reset was refused")
                before = None if role == "oracle" else cold_l1(config["http_port"])
                if role == "consumer" and sample == 0:
                    with record_prefix_blocks(llm) as tables:
                        rows, elapsed = generate(children[:fanout])
                    producer = json.loads((output / "producer-prefix.json").read_text())
                    checksums = prefix_checksums(
                        config, tables, [producer["instance_id"]]
                    )
                    if len(checksums["members"]) != fanout:
                        raise RuntimeError(
                            "consumer prefix integrity control lost a family member"
                        )
                    expected = producer["members"][0]["checksums"]["chunk_checksums"]
                    if any(
                        m["checksums"]["chunk_checksums"] != expected
                        for m in checksums["members"]
                    ):
                        raise AssertionError(
                            "restored model KV differs from the producer ancestor"
                        )
                    write_json(output / f"consumer-prefix-n{fanout}.json", checksums)
                else:
                    rows, elapsed = generate(children[:fanout])
                after = (
                    None
                    if role == "oracle"
                    else http(config["http_port"], "status")["storage_manager"]
                )
                if before is not None:
                    (
                        before["l2_adapters"][0]["core"],
                        after["l2_adapters"][0]["core"],
                    ) = compact_core_pair(
                        before["l2_adapters"][0]["core"],
                        after["l2_adapters"][0]["core"],
                    )
                records.append(
                    {
                        "fanout": fanout,
                        "sample": sample - 3,
                        "warmup": sample < 3,
                        "rows": rows,
                        "batch_wall_ms": elapsed,
                        "before": before,
                        "after": after,
                    }
                )
    finally:
        if role != "producer":
            write_json(output / f"{role}.json", records)
        llm.llm_engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("role", choices=("oracle", "producer", "consumer"))
    args = parser.parse_args()
    main(json.loads(args.config.read_text()), args.role)
