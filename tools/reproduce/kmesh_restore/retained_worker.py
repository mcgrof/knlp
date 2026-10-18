# SPDX-License-Identifier: MIT
"""Compare engine preloads with and without retained external-L1 chunks."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

from family_worker import replay
from restore_common import (
    ANCESTOR,
    CHUNK_BYTES,
    CHUNKS,
    cold_l1,
    compact_core_pair,
    http,
    prompts,
    write_json,
)
from worker import (
    create_llm,
    prefix_checksums,
    record_live_prefix_blocks,
    record_prefix_blocks,
)

GAP_MS = 250


def warm_external(config, ancestor):
    """Complete the stock retained L2-to-L1 prefetch before warming the engine."""
    start = time.perf_counter_ns()
    submitted = http(
        config["http_port"],
        "cache/prefetches",
        {
            "model_name": config["model"],
            "world_size": 1,
            "token_ids": ancestor,
            "cache_salt": "",
            "source_tier": "l2",
            "target_tier": "l1",
        },
    )
    if submitted["status"] != "submitted" or submitted["chunks"] != CHUNKS:
        raise ValueError("external warm did not submit the complete ancestor")
    deadline = time.monotonic() + 60
    polls = 0
    while time.monotonic() < deadline:
        result = http(
            config["http_port"], "cache/prefetches/" + submitted["request_id"]
        )
        polls += 1
        if result["status"] == "completed":
            if result["found_keys"] != CHUNKS or result["total_keys"] != CHUNKS:
                raise ValueError("external warm missed ancestor chunks")
            return {
                "found_keys": result["found_keys"],
                "total_keys": result["total_keys"],
                "polls": polls,
                "completion_ms": (time.perf_counter_ns() - start) / 1e6,
                "l1": http(config["http_port"], "status")["storage_manager"][
                    "l1_manager"
                ],
            }
        if result["status"] != "pending":
            raise ValueError("unexpected external warm status")
        time.sleep(0.001)
    raise TimeoutError("external warm exceeded its completion bound")


def main(config):
    """Charge external prefetch and engine warm work to the same fork clock."""
    llm = create_llm(config, "consumer")
    from vllm import SamplingParams

    engine = llm.llm_engine
    connector = engine.engine_core.engine_core.scheduler.connector
    shared = {
        "temperature": 0,
        "ignore_eos": True,
        "seed": 0,
        "logprobs": 1,
        "extra_args": {"kv_transfer_params": {"cached_token_stats": {}}},
    }
    params = SamplingParams(max_tokens=8, **shared)
    warm_params = SamplingParams(max_tokens=1, **shared)
    ancestor, children = prompts(8)
    fanout = config["fanouts"][0]
    output = Path(config["output"])
    producer = json.loads((output / "producer-prefix.json").read_text())
    expected = producer["members"][0]["checksums"]["chunk_checksums"]
    records = []

    def family(arm, sample, unused=False):
        if not llm.reset_prefix_cache():
            raise RuntimeError("engine prefix reset refused")
        before = cold_l1(config["http_port"])
        stats_before = connector.get_family_prefetch_stats()
        label = "unused" if unused else f"n{fanout}-{arm}-s{sample}"
        observe = sample == 0 and not unused
        with record_live_prefix_blocks(llm) if observe else nullcontext(None) as live, (
            record_prefix_blocks(llm) if observe else nullcontext(None)
        ) as tables:
            row = replay(
                engine,
                ancestor,
                [] if unused else children[:fanout],
                params,
                warm_params,
                "family",
                GAP_MS,
                label,
                declare=lambda warm_id: connector.declare_family_prefetch(
                    ancestor,
                    ANCESTOR,
                    fanout,
                    bytes_per_token=CHUNK_BYTES // 256,
                    prefetch_request_id=warm_id,
                ),
                prepare=(
                    (lambda: warm_external(config, ancestor))
                    if arm == "retained"
                    else None
                ),
            )
        if tables is not None:
            checksums = prefix_checksums(config, tables, [producer["instance_id"]])
            if len(checksums["members"]) != fanout + 1 or any(
                m["checksums"]["chunk_checksums"] != expected
                for m in checksums["members"]
            ):
                raise ValueError("retained preload KV integrity failed")
            write_json(output / (label + "-prefix.json"), checksums)
        connector.release_family_prefetch(row["declaration_identity"])
        after = http(config["http_port"], "status")["storage_manager"]
        before["l2_adapters"][0]["core"], after["l2_adapters"][0]["core"] = (
            compact_core_pair(
                before["l2_adapters"][0]["core"], after["l2_adapters"][0]["core"]
            )
        )
        return row | {
            "fanout": fanout,
            "arm": arm,
            "policy": "family",
            "requested_gap_ms": GAP_MS,
            "sample": sample - 3,
            "warmup": sample < 3,
            "before": before,
            "after": after,
            "live_prefix": live,
            "singleflight": connector.get_singleflight_stats(),
            "family_before": stats_before,
            "family_after": connector.get_family_prefetch_stats(),
        }

    try:
        llm.generate(
            [{"prompt_token_ids": [44, 55, 66, 77, 88] * 20}], params, use_tqdm=False
        )
        for sample in range(config["serving_samples"] + 3):
            # Alternate paired arms while resetting both resident caches each time.
            for arm in (
                ("engine", "retained") if sample % 2 == 0 else ("retained", "engine")
            ):
                records.append(family(arm, sample))
        write_json(output / "wasted.json", family("retained", 0, unused=True))
    finally:
        write_json(output / "consumer.json", records)
        engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
