# SPDX-License-Identifier: MIT
"""Replay a cached reasoning family with an explicit cascade policy."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    mode = sys.argv[2]
    profile = json.loads((HERE / "profile.json").read_text())
    w = profile["workload"]
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    import vllm.v1.attention.backends.flash_attn as fa
    import vllm.v1.worker.gpu_model_runner as gmr

    if Path(vllm.__file__).resolve().parent != Path(config["vllm_source"]) / "vllm":
        raise ValueError("imported vLLM does not match the authenticated source")
    stats = {
        "heuristic_calls": 0,
        "natural_true": 0,
        "forced_true": 0,
        "runner_calls": 0,
        "runner_nonzero": 0,
        "kernel_calls": 0,
        "common_prefix_lens": {},
    }
    heuristic = fa.use_cascade_attention
    signature = inspect.signature(heuristic)
    runner_method = gmr.GPUModelRunner._compute_cascade_attn_prefix_len
    kernel = fa.cascade_attention

    def policy(*args, **kwargs):
        decision = heuristic(*args, **kwargs)
        if mode != "forced" or decision:
            return decision
        values = signature.bind(*args, **kwargs).arguments
        supported = (
            values["common_prefix_len"] >= 256
            and not values["use_alibi"]
            and not values["use_sliding_window"]
            and not values["use_local_attention"]
            and values["dcp_world_size"] == 1
        )
        return supported

    def observed_heuristic(*args, **kwargs):
        natural = heuristic(*args, **kwargs)
        selected = policy(*args, **kwargs)
        prefix = signature.bind(*args, **kwargs).arguments["common_prefix_len"]
        stats["heuristic_calls"] += 1
        stats["natural_true"] += int(natural)
        stats["forced_true"] += int(selected and not natural)
        key = str(prefix)
        stats["common_prefix_lens"][key] = stats["common_prefix_lens"].get(key, 0) + 1
        return selected

    def observed_runner(self, *args, **kwargs):
        result = runner_method(self, *args, **kwargs)
        stats["runner_calls"] += 1
        stats["runner_nonzero"] += int(result > 0)
        return result

    def observed_kernel(*args, **kwargs):
        stats["kernel_calls"] += 1
        return kernel(*args, **kwargs)

    # Patches stay inside this worker; installed vLLM sources remain clean.
    fa.use_cascade_attention = policy
    llm = LLM(
        model=config["model"],
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        disable_cascade_attn=mode == "disabled",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        max_model_len=w["ancestor"] + w["tail"] + w["decode"] + 64,
        kv_cache_memory_bytes=w["kv_cache_bytes"],
        max_num_batched_tokens=w["max_num_batched_tokens"],
        max_num_seqs=w["fanout"],
        async_scheduling=False,
        gpu_memory_utilization=0.92,
        disable_log_stats=True,
        seed=0,
    )
    runner = (
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    )
    backends = sorted(
        {group.backend.get_name() for groups in runner.attn_groups for group in groups}
    )
    rng = random.Random(w["seed"])

    def fresh(count):
        return [rng.randrange(4, 50000) for _ in range(count)]

    ancestor = fresh(w["ancestor"])
    prompts = [ancestor + fresh(w["tail"]) for _ in range(w["fanout"])]
    tp = [TokensPrompt(prompt_token_ids=t) for t in prompts]
    params = SamplingParams(temperature=0, max_tokens=w["decode"], ignore_eos=True)
    llm.generate(tp, SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    fa.use_cascade_attention = observed_heuristic
    gmr.GPUModelRunner._compute_cascade_attn_prefix_len = observed_runner
    fa.cascade_attention = observed_kernel
    llm.generate(tp, params, use_tqdm=False)
    fa.use_cascade_attention = policy
    gmr.GPUModelRunner._compute_cascade_attn_prefix_len = runner_method
    fa.cascade_attention = kernel
    elapsed, outputs, cached = [], [], []
    for _ in range(w["repeats"]):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = llm.generate(tp, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
        outputs.append([list(row.outputs[0].token_ids) for row in result])
        cached.append([row.num_cached_tokens for row in result])
    record = {
        "mode": mode,
        "workload": w,
        "source_commit": profile["vllm_commit"],
        "runner": type(runner).__module__ + "." + type(runner).__name__,
        "backends": backends,
        "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "engagement": stats,
        "generation_wall_ms": elapsed,
        "outputs": outputs,
        "cached_tokens": cached,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "flash_attn_version": 2,
            "batch_invariant": os.environ["VLLM_BATCH_INVARIANT"],
            "threads": torch.get_num_threads(),
        },
    }
    from run import sha256

    record["native_sha256"] = {
        str(path): sha256(path) for path in Path(vllm.__file__).parent.rglob("*.so")
    }
    with (Path(config["output"]) / (mode + ".json")).open("x") as stream:
        json.dump(record, stream, indent=2)


if __name__ == "__main__":
    main()
