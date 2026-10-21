# SPDX-License-Identifier: MIT
"""Record actual synchronous engine calls and physical cache observations."""

import json
import os
from pathlib import Path
import time

from .audit import check_invocation


class InvocationCapture:
    """Bind caller lineage to scheduler requests without inventing KV equality."""

    def __init__(self, llm, output):
        self.engine = llm.llm_engine
        scheduler = self.engine.engine_core.engine_core.scheduler
        if scheduler.scheduler_config.async_scheduling:
            raise ValueError("capture requires synchronous scheduling")
        self.manager = scheduler.kv_cache_manager
        self.pool = self.manager.block_pool
        if len(self.manager.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("capture requires one full-attention cache group")
        spec = self.manager.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if type(spec).__name__ != "FullAttentionSpec" or spec.block_size != 16:
            raise ValueError("capture requires full attention with 16-token blocks")
        self.output = Path(output)
        self.clock_id = "monotonic-" + os.urandom(8).hex()
        self.engine_id = "engine-" + os.urandom(8).hex()
        self.events = []
        self.internal = {}
        self.ready = {}
        self.records = []
        self.lookups = {}
        self.original = {}
        self.tables = {}
        self.initial_free = self.pool.get_num_free_blocks()
        self.pool_blocks = len(self.pool.blocks)
        self.null_id = self.pool.null_block.block_id

    def event(self, action, request=None, **extra):
        value = {
            "action": action,
            "time_ns": time.perf_counter_ns(),
            "free_blocks": self.pool.get_num_free_blocks(),
            **extra,
        }
        if request is not None:
            groups = self.manager.get_blocks(request.request_id).blocks
            value.update(
                {
                    "request_id": self.internal.get(
                        request.request_id, request.request_id
                    ),
                    "computed_tokens": request.num_computed_tokens,
                    "blocks": [
                        {
                            "block_id": block.block_id,
                            "ref_count": block.ref_cnt,
                            "hash": (
                                None
                                if block.block_hash is None
                                else repr(block.block_hash)
                            ),
                            "hash_tokens": block.block_hash_num_tokens,
                        }
                        for block in groups[0]
                    ],
                }
            )
        self.events.append(value)
        return value

    def __enter__(self):
        manager, pool = self.manager, self.pool
        self.original = {
            "allocate": manager.allocate_slots,
            "free": manager.free,
            "lookup": manager.get_computed_blocks,
            "evict": pool._maybe_evict_cached_block,
        }

        def lookup(request):
            result = self.original["lookup"](request)
            self.lookups[request.request_id] = result[1]
            self.event(
                "lookup",
                request_id=self.internal.get(request.request_id, request.request_id),
                hit_tokens=result[1],
            )
            return result

        def allocate(request, *args, **kwargs):
            result = self.original["allocate"](request, *args, **kwargs)
            table = self.manager.get_block_ids(request.request_id)
            identity = tuple(tuple(group) for group in table)
            if result is None or self.tables.get(request.request_id) != identity:
                self.event(
                    "allocated" if result is not None else "allocation-failed", request
                )
                self.tables[request.request_id] = identity
            return result

        def free(request):
            event = self.event("before-free", request)
            canonical = self.internal.get(request.request_id)
            if canonical is not None:
                self.ready[canonical] = {
                    "computed_tokens": request.num_computed_tokens,
                    "all_token_ids": list(request.all_token_ids),
                    "cached_tokens": self.lookups.get(request.request_id, 0),
                    "blocks": event["blocks"],
                    "status": request.status.name,
                }
            self.original["free"](request)
            self.event("after-free", request_id=canonical)

        def evict(block):
            identity, references = block.block_id, block.ref_cnt
            old_hash = repr(block.block_hash)
            result = self.original["evict"](block)
            if result:
                self.event(
                    "evicted",
                    block_id=identity,
                    prior_ref_count=references,
                    prior_hash=old_hash,
                )
            return result

        manager.get_computed_blocks, manager.allocate_slots, manager.free = (
            lookup,
            allocate,
            free,
        )
        pool._maybe_evict_cached_block = evict
        return self

    def __exit__(self, *exception):
        manager = self.manager
        manager.get_computed_blocks = self.original["lookup"]
        manager.allocate_slots = self.original["allocate"]
        manager.free = self.original["free"]
        self.pool._maybe_evict_cached_block = self.original["evict"]
        (self.output / "events.json").write_text(
            json.dumps(self.events, indent=2, allow_nan=False) + "\n"
        )

    def invoke(self, name, family, parent, tokens, params, *, purpose):
        """Submit one caller request and retain its actual terminal evidence."""
        if self.engine.has_unfinished_requests():
            raise RuntimeError("the first capture profile requires sequential calls")
        first_event = len(self.events)
        start = time.perf_counter_ns()
        internal = self.engine.add_request(name, {"prompt_token_ids": tokens}, params)
        self.internal[internal] = name
        result = None
        failure = None
        try:
            while self.engine.has_unfinished_requests():
                if time.perf_counter_ns() - start > 120_000_000_000:
                    raise TimeoutError("model invocation exceeded its bound")
                for value in self.engine.step():
                    if value.finished:
                        if value.request_id != name or result is not None:
                            raise RuntimeError("unexpected completed request")
                        result = value
            if result is None or len(result.outputs) != 1:
                raise RuntimeError("missing model output")
            sequence = result.outputs[0]
            if sequence.finish_reason == "error":
                raise RuntimeError("engine reported a generation error")
        except BaseException as error:
            failure = error
            self.engine.abort_request([name])
        end = time.perf_counter_ns()
        ready = self.ready.get(name)
        record = {
            "schema": "knlp.kmesh.invocation.v1",
            "request_id": name,
            "family_id": family,
            "parent_request_id": parent,
            "engine_id": self.engine_id,
            "clock_id": self.clock_id,
            "start_ns": start,
            "end_ns": end,
            "input_token_ids": list(tokens),
            "output_token_ids": (
                [] if result is None else list(result.outputs[0].token_ids)
            ),
            "status": "complete" if failure is None else "error",
            "purpose": purpose,
            "cache_observations": self.events[first_event:].copy(),
            "ready": ready,
        }
        if failure is None:
            expected = record["input_token_ids"] + record["output_token_ids"]
            if (
                ready is None
                or ready["all_token_ids"] != expected
                or ready["cached_tokens"] != result.num_cached_tokens
            ):
                failure = ValueError(
                    "scheduler and caller token/cache identities disagree"
                )
                record["status"] = "error"
            record["finish_reason"] = result.outputs[0].finish_reason
        check_invocation(record)
        with (self.output / "invocations.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)
        if failure is not None:
            raise failure
        return result.outputs[0].text, record

    def manifest(self):
        return {
            "engine_id": self.engine_id,
            "clock_id": self.clock_id,
            "physical_pool_blocks": self.pool_blocks,
            "reserved_null_block_id": self.null_id,
            "initial_free_blocks": self.initial_free,
            "final_free_blocks": self.pool.get_num_free_blocks(),
            "final_nonnull_references": sum(
                block.ref_cnt for block in self.pool.blocks if not block.is_null
            ),
            "block_tokens": 16,
            "completed_requests": sum(
                row["status"] == "complete" for row in self.records
            ),
            "evictions": sum(event["action"] == "evicted" for event in self.events),
        }
