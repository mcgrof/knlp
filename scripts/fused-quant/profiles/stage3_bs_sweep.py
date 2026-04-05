#!/usr/bin/env python3
"""Stage 3: Tiny BS=1/4/8 decode sweep — FP16 graph vs fused graph.

Purpose: determine if the gap scales with batch (kernel/memory tax),
stays constant (fixed overhead), or shrinks at larger batch (BS=1 scheduling).
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time, json, sys, gc

RUN_ROOT = "/tmp/acp-runs/20260404T2339Z-residgap-h100-cldopus-a1"
ARTIFACTS = f"{RUN_ROOT}/artifacts"
MODEL = "marin-community/marin-8b-base"

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{RUN_ROOT}/progress.log", "a") as f:
        f.write(line + "\n")

def append_event(etype, phase, note):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ev = {"ts": ts, "run_id": "20260404T2339Z-residgap-h100-cldopus-a1",
          "attempt": 1, "actor": "worker", "host": "lewwsrdy631hun",
          "type": etype, "phase": phase, "note": note}
    with open(f"{RUN_ROOT}/events.jsonl", "a") as f:
        f.write(json.dumps(ev) + "\n")

def sweep_arm(arm_name, kv_cache_dtype, batch_sizes=[1, 4, 8]):
    import torch
    from vllm import LLM, SamplingParams

    log(f"--- ARM: {arm_name} (kv_cache_dtype={kv_cache_dtype}) ---")
    log(f"  creating LLM...")

    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=False,
    )

    prompt = " ".join(["token"] * 512)
    sp = SamplingParams(max_tokens=256, temperature=0)

    # Warmup with different batch sizes
    log(f"  warmup...")
    for bs in batch_sizes:
        llm.generate([prompt] * bs, sp)
    for _ in range(2):
        llm.generate([prompt], sp)
    log(f"  warmup done")

    results = {}
    for bs in batch_sizes:
        log(f"  BS={bs}...")
        trials = []
        prompts = [prompt] * bs
        for trial in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            outputs = llm.generate(prompts, sp)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            # Per-sequence ITL = total_time / max_tokens (all seqs finish simultaneously)
            n_tokens = len(outputs[0].outputs[0].token_ids)
            itl = (elapsed * 1000) / max(n_tokens, 1)
            tps = (n_tokens * bs) / elapsed  # total throughput
            trials.append({
                "trial": trial, "bs": bs, "elapsed_s": round(elapsed, 4),
                "tokens_per_seq": n_tokens, "itl_ms": round(itl, 3),
                "total_tps": round(tps, 1),
            })
            log(f"    trial {trial}: {elapsed:.3f}s, ITL={itl:.3f}ms, TPS={tps:.1f}")
        
        avg_itl = sum(t["itl_ms"] for t in trials) / len(trials)
        avg_tps = sum(t["total_tps"] for t in trials) / len(trials)
        results[f"bs{bs}"] = {
            "bs": bs, "avg_itl_ms": round(avg_itl, 3),
            "avg_tps": round(avg_tps, 1), "trials": trials,
        }
        log(f"  BS={bs}: avg ITL={avg_itl:.3f}ms, avg TPS={avg_tps:.1f}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results

def main():
    arm_name = sys.argv[1] if len(sys.argv) > 1 else "fp16_graph"
    kv_dtype = "auto" if arm_name == "fp16_graph" else "int4_fused"

    log(f"=== Stage 3 BS sweep: {arm_name} ===")
    append_event("PROCESS_STARTED", f"S3_{arm_name}", f"BS sweep for {arm_name}")

    os.makedirs(ARTIFACTS, exist_ok=True)
    results = sweep_arm(arm_name, kv_dtype)

    with open(f"{ARTIFACTS}/stage3_{arm_name}_sweep.json", "w") as f:
        json.dump({"arm": arm_name, "results": results}, f, indent=2)

    append_event("CHECKPOINT", f"S3_{arm_name}_DONE", f"BS sweep for {arm_name} complete")
    log(f"=== {arm_name} sweep complete ===")
    os._exit(0)

if __name__ == "__main__":
    main()
