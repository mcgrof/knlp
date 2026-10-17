# SPDX-License-Identifier: Apache-2.0
"""Measure exact-prefix reasoning forks through a real vLLM serving engine.

Replay warm ancestors, concurrent arrivals, eviction and decode-produced
prefixes using the same seeded tree as the allocator experiment. Each case
runs in a fresh process. Pool sampling observes scheduler metadata, not GPU
memory traffic or storage offload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import threading
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_PATH = None

BLOCK_SIZE = 16
VOCAB_LOW, VOCAB_HIGH = 4, 50000
SEED = 20260805


def build_fork_tree(seed, ancestor_len, branching, tail_len):
    rng = random.Random(seed)

    def fresh(n):
        return [rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(n)]

    root = {"label": "d0", "tokens": fresh(ancestor_len), "divergence": 0}
    nodes = [root]
    for child_idx in range(branching):
        nodes.append(
            {
                "label": f"d0.b{child_idx}",
                "tokens": root["tokens"] + fresh(tail_len),
                "divergence": ancestor_len,
            }
        )
    return nodes


def prompt_digest(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


class PoolSampler:
    """Observe real allocation boundaries and sample the scheduler pool."""

    def __init__(self, llm):
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        self.manager = scheduler.kv_cache_manager
        self.pool = self.manager.block_pool
        self.total = len(self.pool.blocks)
        self.peak = 0
        self.boundary_peak = 0
        self.events = []
        self.active = {}
        self._tables = {}
        self._allocate = self.manager.allocate_slots
        self._free = self.manager.free
        self.resolved = {
            name: getattr(llm.llm_engine.vllm_config.scheduler_config, name)
            for name in (
                "max_num_batched_tokens",
                "max_num_seqs",
                "enable_chunked_prefill",
                "async_scheduling",
            )
        }
        runner = (
            llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        )
        self.attention_backends = sorted(
            {
                group.backend.get_name()
                for groups in runner.attn_groups
                for group in groups
            }
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used = self.total - self.pool.free_block_queue.num_free_blocks
            if used > self.peak:
                self.peak = used
            time.sleep(0.005)

    def __enter__(self):
        def allocate(request, *args, **kwargs):
            result = self._allocate(request, *args, **kwargs)
            if result is not None:
                if request.request_id not in self.active:
                    self.active[request.request_id] = prompt_digest(
                        request.prompt_token_ids
                    )
                self._observe("allocate", request.request_id)
            return result

        def free(request):
            self._observe("before_free", request.request_id)
            result = self._free(request)
            self.active.pop(request.request_id, None)
            self._observe("free", request.request_id)
            return result

        self.manager.allocate_slots = allocate
        self.manager.free = free
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.manager.allocate_slots = self._allocate
        self.manager.free = self._free

    def _observe(self, action, request_id):
        used = self.total - self.pool.free_block_queue.num_free_blocks
        self.boundary_peak = max(self.boundary_peak, used)
        tables = {
            key: [list(group) for group in self.manager.get_block_ids(key)]
            for key in self.active
        }
        if tables != self._tables or action != "allocate":
            self.events.append(
                {
                    "action": action,
                    "request_id": request_id,
                    "used_blocks": used,
                    "tables": tables,
                    "prompt_sha256": dict(self.active),
                }
            )
            self._tables = tables

    def evidence(self):
        return {
            "allocation_boundary_peak_used_blocks": self.boundary_peak,
            "sampled_peak_used_blocks": self.peak,
            "allocation_events": self.events,
            "resolved_scheduler": self.resolved,
            "attention_backends": self.attention_backends,
            "null_block_id": self.pool.null_block.block_id,
        }


def boot(enable_caching, blocks_override=None, max_model_len=8192):
    from vllm import LLM

    kwargs = dict(
        model=MODEL_PATH,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=enable_caching,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.45,
        disable_log_stats=True,
        seed=0,
        block_size=BLOCK_SIZE,
    )
    if blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = blocks_override
    return LLM(**kwargs)


def generate(llm, token_lists, max_tokens, labels):
    from vllm import SamplingParams, TokensPrompt

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    started = time.perf_counter()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=t) for t in token_lists],
        params,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started
    rows = []
    out_by_index = list(outputs)
    for label, tokens, output in zip(labels, token_lists, out_by_index):
        rows.append(
            {
                "label": label,
                "prompt_tokens": len(tokens),
                "num_cached_tokens": output.num_cached_tokens,
                "output_tokens": len(output.outputs[0].token_ids),
                "output_token_ids": list(output.outputs[0].token_ids),
                "request_id": output.request_id,
                "prompt_sha256": prompt_digest(tokens),
            }
        )
    return rows, elapsed, out_by_index


def churn_prompts(total_tokens, request_tokens, seed):
    rng = random.Random(seed)
    prompts = []
    remaining = total_tokens
    while remaining > 0:
        n = min(request_tokens, remaining)
        prompts.append([rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(n)])
        remaining -= n
    return prompts


def scenario_children_after_parent(nodes, enable_caching):
    """Parent completes, then all children in one batch."""
    llm = boot(enable_caching)
    result = {}
    with PoolSampler(llm) as sampler:
        parent_rows, parent_s, _ = generate(
            llm, [nodes[0]["tokens"]], 1, [nodes[0]["label"]]
        )
        child_rows, child_s, _ = generate(
            llm,
            [n["tokens"] for n in nodes[1:]],
            64,
            [n["label"] for n in nodes[1:]],
        )
        result = {
            "rows": parent_rows + child_rows,
            "parent_seconds": round(parent_s, 3),
            "children_seconds": round(child_s, 3),
            "peak_used_blocks": sampler.peak,
            "total_blocks": sampler.total,
            **sampler.evidence(),
        }
    del llm
    gc.collect()
    return result


def scenario_burst(nodes, include_parent):
    """Everything submitted in one call: the scheduler decides ordering.
    Without the parent this is the cold sibling burst — the case the
    allocator model cannot represent, because prefix-cache lookup
    happens when each request is first scheduled."""
    llm = boot(True)
    members = nodes if include_parent else nodes[1:]
    with PoolSampler(llm) as sampler:
        rows, seconds, _ = generate(
            llm,
            [n["tokens"] for n in members],
            8,
            [n["label"] for n in members],
        )
        result = {
            "rows": rows,
            "seconds": round(seconds, 3),
            "peak_used_blocks": sampler.peak,
            "total_blocks": sampler.total,
            **sampler.evidence(),
        }
    del llm
    gc.collect()
    return result


def scenario_churn(nodes, pool_blocks, churn_fraction):
    """Parent, then unrelated churn through a bounded pool, then the
    children: the eviction-window cliff at serving level."""
    llm = boot(True, blocks_override=pool_blocks)
    with PoolSampler(llm) as sampler:
        parent_rows, _, _ = generate(llm, [nodes[0]["tokens"]], 1, ["d0"])
        churn_tokens = int(pool_blocks * BLOCK_SIZE * churn_fraction)
        prompts = churn_prompts(churn_tokens, 2048, SEED + 7)
        churn_rows, churn_s, _ = generate(
            llm, prompts, 1, [f"churn-{i}" for i in range(len(prompts))]
        )
        child_rows, child_s, _ = generate(
            llm,
            [n["tokens"] for n in nodes[1:]],
            8,
            [n["label"] for n in nodes[1:]],
        )
        hits = [r["num_cached_tokens"] for r in child_rows]
        result = {
            "rows": parent_rows + child_rows,
            "churn_fraction": churn_fraction,
            "churn_tokens": churn_tokens,
            "churn_requests": len(prompts),
            "churn_seconds": round(churn_s, 3),
            "children_seconds": round(child_s, 3),
            "child_hit_tokens": hits,
            "ancestor_survival": round(
                sum(hits) / (len(hits) * len(nodes[0]["tokens"])), 4
            ),
            "peak_used_blocks": sampler.peak,
            "total_blocks": sampler.total,
            **sampler.evidence(),
        }
    del llm
    gc.collect()
    return result


def scenario_fork_from_decode(nodes, decode_tokens):
    """Children fork from the parent's prompt PLUS its generated output:
    tests whether decode-produced blocks are cached and shareable, which
    the allocator model never exercised."""
    llm = boot(True)
    root = nodes[0]
    rng = random.Random(SEED + 11)
    with PoolSampler(llm) as sampler:
        parent_rows, _, outputs = generate(llm, [root["tokens"]], decode_tokens, ["d0"])
        generated = list(outputs[0].outputs[0].token_ids)
        base = root["tokens"] + generated
        children = []
        for child_idx in range(3):
            tail = [rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(512)]
            children.append((f"d0.gen.b{child_idx}", base + tail))
        child_rows, child_s, _ = generate(
            llm, [t for _, t in children], 8, [lbl for lbl, _ in children]
        )
        result = {
            "rows": parent_rows + child_rows,
            "parent_generated": len(generated),
            "fork_point": len(base),
            "fork_point_block_floor": (len(base) // BLOCK_SIZE) * BLOCK_SIZE,
            "children_seconds": round(child_s, 3),
            "peak_used_blocks": sampler.peak,
            "total_blocks": sampler.total,
            **sampler.evidence(),
        }
    del llm
    gc.collect()
    return result


SCENARIOS = {
    "children_after_parent_cached": lambda n: scenario_children_after_parent(n, True),
    "children_after_parent_uncached": lambda n: scenario_children_after_parent(
        n, False
    ),
    "burst_with_parent": lambda n: scenario_burst(n, True),
    "cold_sibling_burst": lambda n: scenario_burst(n, False),
    "churn_050": lambda n: scenario_churn(n, 1024, 0.5),
    "churn_100": lambda n: scenario_churn(n, 1024, 1.0),
    "fork_from_decode": lambda n: scenario_fork_from_decode(n, 128),
}


def main():
    # Each fresh process gives its engine sole ownership of the CUDA context.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS))
    args = parser.parse_args()
    global MODEL_PATH
    MODEL_PATH = args.model

    if args.smoke:
        nodes = build_fork_tree(SEED, 512, 2, 128)
        results = {
            "smoke_children_after_parent": scenario_children_after_parent(nodes, True)
        }
    else:
        if not args.scenario:
            parser.error("--scenario is required unless --smoke")
        nodes = build_fork_tree(SEED, 4096, 5, 512)
        results = {
            "tree": "a4096-b5-t512",
            args.scenario: SCENARIOS[args.scenario](nodes),
        }

    import torch
    import vllm

    results["vllm_version"] = vllm.__version__
    results["torch_version"] = torch.__version__
    results["python_version"] = sys.version.split()[0]
    results["cuda_version"] = torch.version.cuda
    results["schema_version"] = 1
    results["measurement"] = "serving-fork-replay"
    results["gpu_name"] = torch.cuda.get_device_name(0)
    results["block_size"] = BLOCK_SIZE
    results["sampler_interval_seconds"] = 0.005
    results["model_revision"] = MODEL_REVISION
    results["seed"] = SEED
    with open(args.output, "w") as sink:
        json.dump(results, sink, indent=2, sort_keys=True)
    print(json.dumps({"wrote": args.output}, indent=2))


if __name__ == "__main__":
    main()
