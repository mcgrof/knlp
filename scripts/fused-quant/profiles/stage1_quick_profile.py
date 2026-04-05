#!/usr/bin/env python3
"""Quick NSYS decode profile — single-process, no vLLM server.
Uses direct model forward pass with torch CUDA graphs to match vLLM decode behavior.
"""
import time, json, os, sys
import torch

RUN_ROOT = "/tmp/acp-runs/20260404T2339Z-residgap-h100-cldopus-a1"

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(f"{RUN_ROOT}/progress.log", "a") as f:
        f.write(line + "\n")

def main():
    arm_name = sys.argv[1] if len(sys.argv) > 1 else "fp16_graph"
    log(f"=== Quick profile: {arm_name} ===")

    from vllm import LLM, SamplingParams

    kv_dtype = "auto" if arm_name == "fp16_graph" else "int4_fused"
    log(f"  creating LLM (kv_cache_dtype={kv_dtype})...")
    llm = LLM(
        model="marin-community/marin-8b-base",
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=kv_dtype,
        enforce_eager=False,
    )

    # Warmup
    log("  warmup...")
    sp = SamplingParams(max_tokens=256, temperature=0)
    prompt = " ".join(["token"] * 512)
    for i in range(3):
        llm.generate([prompt], sp)
        log(f"  warmup {i+1}/3")

    # Timed decode runs (nsys captures everything)
    log("  starting timed decode (3 trials)...")
    results = []
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n = len(outputs[0].outputs[0].token_ids)
        itl = (elapsed * 1000) / max(n, 1)
        results.append({"trial": trial, "elapsed_s": round(elapsed, 4), "tokens": n, "itl_ms": round(itl, 3)})
        log(f"  trial {trial}: {elapsed:.3f}s, {n} tok, ITL={itl:.3f}ms")

    # Save
    with open(f"{RUN_ROOT}/artifacts/stage1_{arm_name}_v3_results.json", "w") as f:
        json.dump({"arm": arm_name, "trials": results}, f, indent=2)

    itls = [r["itl_ms"] for r in results]
    avg = sum(itls) / len(itls)
    log(f"  avg ITL: {avg:.3f} ms")
    log(f"  done, exiting")

    # Force exit to avoid hanging cleanup
    os._exit(0)

if __name__ == "__main__":
    main()
