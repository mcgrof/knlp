#!/usr/bin/env python3
"""Tier 0b: Baseline autoregressive decode — no speculation, no offload."""

import json, time, torch, sys, os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
]
SEQ_LENS = [512, 2048, 8192]
NUM_PROMPTS = 20
MAX_TOKENS = 256

BASE_PROMPTS = [
    "Explain the theory of general relativity and its implications for modern physics in great detail.",
    "Write a detailed analysis of sorting algorithms including quicksort mergesort heapsort and their time complexities.",
    "Describe the history of computing from Charles Babbage to modern GPUs covering all major milestones.",
    "Explain how neural networks learn through backpropagation including the chain rule and gradient descent.",
    "Discuss the economic implications of artificial intelligence on global labor markets and policy responses.",
    "Analyze the environmental impact of large language models and potential mitigation strategies.",
    "Compare and contrast different database architectures including relational document and graph databases.",
    "Explain quantum computing principles including superposition entanglement and quantum error correction.",
    "Describe the evolution of programming languages from assembly to modern high-level languages.",
    "Discuss the ethical considerations in deploying AI systems in healthcare and criminal justice.",
]


def make_prompts(seq_len, n=20):
    prompts = []
    for i in range(n):
        base = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        if seq_len <= 512:
            prompts.append(base)
        else:
            repeat = max(1, seq_len // 40)
            prompts.append((base + " ") * repeat)
    return prompts


all_results = []

for model in MODELS:
    model_short = model.split("/")[-1]
    for seq_len in SEQ_LENS:
        print(f"\n{'='*60}")
        print(f"Model: {model_short}, seq_len: {seq_len}")
        print(f"{'='*60}", flush=True)

        max_model_len = min(seq_len + MAX_TOKENS + 512, 32768)
        try:
            llm = LLM(
                model=model,
                max_model_len=max_model_len,
                gpu_memory_utilization=0.85,
                dtype="auto",
                enforce_eager=True,
            )
        except Exception as e:
            print(f"Failed to load {model}: {e}")
            all_results.append({"model": model, "seq_len": seq_len, "error": str(e)})
            continue

        sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
        prompts = make_prompts(seq_len, NUM_PROMPTS)

        # Warmup
        print("Warmup...", flush=True)
        try:
            llm.generate(prompts[:2], sampling_params)
        except Exception as e:
            print(f"Warmup failed: {e}")

        # Benchmark
        print("Benchmarking...", flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling_params)
        except Exception as e:
            print(f"Generate failed: {e}")
            all_results.append({"model": model, "seq_len": seq_len, "error": str(e)})
            del llm
            torch.cuda.empty_cache()
            continue
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_in = sum(len(o.prompt_token_ids) for o in outputs)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        result = {
            "tier": "0b",
            "type": "baseline_decode",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": "H100-80GB",
            "model": model,
            "seq_len": seq_len,
            "num_prompts": len(prompts),
            "max_tokens": MAX_TOKENS,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "wall_time_sec": round(elapsed, 3),
            "tokens_per_sec": round(total_out / elapsed, 2),
            "avg_decode_latency_ms": round(elapsed / max(total_out, 1) * 1000, 2),
            "peak_vram_gb": round(peak_mem, 2),
        }
        all_results.append(result)

        out_path = f"/root/spev01/json/tier0_baseline_{model_short}_{seq_len}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"  => {result['tokens_per_sec']} tok/s, latency={result['avg_decode_latency_ms']}ms, VRAM={result['peak_vram_gb']}GB"
        )

        del llm
        torch.cuda.empty_cache()

# Save combined
with open("/root/spev01/json/tier0_baseline_all.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n\n=== ALL BASELINE RESULTS ===")
for r in all_results:
    if "error" not in r:
        print(
            f"{r['model'].split('/')[-1]:40s} seq={r['seq_len']:5d}  {r['tokens_per_sec']:8.1f} tok/s  latency={r['avg_decode_latency_ms']:6.2f}ms  VRAM={r['peak_vram_gb']:.1f}GB"
        )
    else:
        print(
            f"{r['model'].split('/')[-1]:40s} seq={r['seq_len']:5d}  ERROR: {r['error'][:80]}"
        )
