# SPDX-License-Identifier: MIT
"""Release children during a real ancestor preload on one vLLM engine."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

from restore_common import (
    ANCESTOR,
    CHUNK_BYTES,
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
    result_rows,
)


def replay(
    engine, ancestor, children, params, warm_params, policy, gap_ms, label, declare=None
):
    """Drive the public engine loop, adding children at the attained fork gap."""
    start = time.perf_counter_ns()
    release = None
    warm_done = None
    warm_output = None
    completed = {}
    warm_name = label + "-warm"
    names = [label + f"-child-{i}" for i in range(len(children))]
    identity = None
    if policy != "reactive":
        warm_internal_id = engine.add_request(
            warm_name, {"prompt_token_ids": ancestor + [7]}, warm_params
        )
        if declare is not None:
            identity = declare(warm_internal_id)
            if identity is None:
                raise RuntimeError("family declaration was not admitted")
    while release is None or engine.has_unfinished_requests():
        now = time.perf_counter_ns()
        if (now - start) / 1e9 > 120:
            raise TimeoutError("family engine replay exceeded its bound")
        due = (
            policy == "reactive"
            or (policy == "oracle" and warm_done is not None)
            or (policy == "family" and (now - start) / 1e6 >= gap_ms)
        )
        if release is None and due:
            release = time.perf_counter_ns()
            for name, tokens in zip(names, children, strict=True):
                engine.add_request(name, {"prompt_token_ids": tokens}, params)
        if engine.has_unfinished_requests():
            for output in engine.step():
                if not output.finished:
                    continue
                if output.request_id == warm_name:
                    warm_done = time.perf_counter_ns()
                    warm_output = output
                else:
                    completed[output.request_id] = output
        elif release is None:
            time.sleep(min(0.001, max(0, gap_ms / 1000 - (now - start) / 1e9)))
    end = time.perf_counter_ns()
    if set(completed) != set(names):
        raise RuntimeError("engine lost or duplicated a family child")
    return {
        "declaration_identity": identity,
        "rows": result_rows([completed[name] for name in names], 8),
        "warm_rows": [] if warm_output is None else result_rows([warm_output], 1),
        "actual_gap_ms": (release - start) / 1e6,
        "warm_complete_ms": None if warm_done is None else (warm_done - start) / 1e6,
        "warm_completed_before_release": warm_done is not None and warm_done <= release,
        "children_wall_ms": (end - release) / 1e6,
        "declaration_to_completion_ms": (end - start) / 1e6,
    }


def main(config: dict) -> None:
    """Compare reactive, fully warm and declared-preload families on owned NVMe."""
    llm = create_llm(config, "consumer")
    from vllm import SamplingParams

    engine = llm.llm_engine
    connector = engine.engine_core.engine_core.scheduler.connector
    params = SamplingParams(
        temperature=0,
        max_tokens=8,
        ignore_eos=True,
        seed=0,
        logprobs=1,
        extra_args={"kv_transfer_params": {"cached_token_stats": {}}},
    )
    warm_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        seed=0,
        logprobs=1,
        extra_args={"kv_transfer_params": {"cached_token_stats": {}}},
    )
    ancestor, children = prompts(8)
    output = Path(config["output"])
    records = []
    cases = [("reactive", 0), ("oracle", 0)] + [
        ("family", gap) for gap in (0, 10, 25, 100)
    ]
    try:
        llm.generate(
            [{"prompt_token_ids": [44, 55, 66, 77, 88] * 20}], params, use_tqdm=False
        )
        for fanout in config["fanouts"]:
            for policy, gap_ms in cases:
                for sample in range(config["serving_samples"] + 3):
                    if not llm.reset_prefix_cache():
                        raise RuntimeError("engine prefix reset refused")
                    before = cold_l1(config["http_port"])
                    stats_before = connector.get_family_prefetch_stats()

                    def declare(warm_id):
                        return connector.declare_family_prefetch(
                            ancestor,
                            ANCESTOR,
                            fanout,
                            bytes_per_token=CHUNK_BYTES // 256,
                            prefetch_request_id=warm_id,
                        )

                    label = f"n{fanout}-{policy}-g{gap_ms}-s{sample}"
                    # Observe ownership in the first warmup only; all other
                    # samples use the same driver without allocation hooks.
                    observation = (
                        record_live_prefix_blocks(llm)
                        if sample == 0
                        else nullcontext(None)
                    )
                    integrity = (
                        record_prefix_blocks(llm) if sample == 0 else nullcontext(None)
                    )
                    with observation as live, integrity as tables:
                        record = replay(
                            engine,
                            ancestor,
                            children[:fanout],
                            params,
                            warm_params,
                            policy,
                            gap_ms,
                            label,
                            declare=declare if policy == "family" else None,
                        )
                    if tables is not None:
                        producer = json.loads(
                            (output / "producer-prefix.json").read_text()
                        )
                        checksums = prefix_checksums(
                            config, tables, [producer["instance_id"]]
                        )
                        expected = producer["members"][0]["checksums"][
                            "chunk_checksums"
                        ]
                        expected_members = fanout + int(policy != "reactive")
                        if len(checksums["members"]) != expected_members or any(
                            m["checksums"]["chunk_checksums"] != expected
                            for m in checksums["members"]
                        ):
                            raise ValueError("family preload KV integrity failed")
                        write_json(output / (label + "-prefix.json"), checksums)
                    if record["declaration_identity"] is not None:
                        connector.release_family_prefetch(
                            record["declaration_identity"]
                        )
                    after = http(config["http_port"], "status")["storage_manager"]
                    (
                        before["l2_adapters"][0]["core"],
                        after["l2_adapters"][0]["core"],
                    ) = compact_core_pair(
                        before["l2_adapters"][0]["core"],
                        after["l2_adapters"][0]["core"],
                    )
                    records.append(
                        record
                        | {
                            "fanout": fanout,
                            "policy": policy,
                            "requested_gap_ms": gap_ms,
                            "sample": sample - 3,
                            "warmup": sample < 3,
                            "before": before,
                            "after": after,
                            "live_prefix": live,
                            "singleflight": connector.get_singleflight_stats(),
                            "family_before": stats_before,
                            "family_after": connector.get_family_prefetch_stats(),
                        }
                    )
        if not llm.reset_prefix_cache():
            raise RuntimeError("wasted control prefix reset refused")
        before = cold_l1(config["http_port"])
        stats_before = connector.get_family_prefetch_stats()
        wasted = replay(
            engine,
            ancestor,
            [],
            params,
            warm_params,
            "family",
            0,
            "unused",
            declare=lambda warm_id: connector.declare_family_prefetch(
                ancestor, ANCESTOR, config["fanouts"][0], prefetch_request_id=warm_id
            ),
        )
        connector.release_family_prefetch(wasted["declaration_identity"])
        after = http(config["http_port"], "status")["storage_manager"]
        before["l2_adapters"][0]["core"], after["l2_adapters"][0]["core"] = (
            compact_core_pair(
                before["l2_adapters"][0]["core"], after["l2_adapters"][0]["core"]
            )
        )
        write_json(
            output / "wasted.json",
            wasted
            | {
                "before": before,
                "after": after,
                "family_before": stats_before,
                "family_after": connector.get_family_prefetch_stats(),
            },
        )
    finally:
        write_json(output / "consumer.json", records)
        engine.engine_core.shutdown(timeout=30)


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
