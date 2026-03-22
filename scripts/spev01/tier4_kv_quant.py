#!/usr/bin/env python3
"""Tier 4: KV quantization impact analysis.
Test vLLM with FP8 KV cache to measure size reduction and quality impact.
"""

import json, time, torch, os, sys, shutil, subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams

# --- portability helpers ------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get(
    "SPEV01_OUTPUT_DIR",
    os.path.join(_SCRIPT_DIR, "json"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    sys.exit("ERROR: tier4_kv_quant requires a CUDA GPU (torch.cuda unavailable)")


def _detect_gpu_name():
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            return f"{parts[0]}-{int(parts[1])//1024}GB"
        except Exception:
            pass
    return torch.cuda.get_device_name(0)


GPU_TAG = _detect_gpu_name()

MODEL = "Qwen/Qwen2.5-7B-Instruct"
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

for kv_dtype in ["auto", "fp8"]:
    for seq_len in SEQ_LENS:
        print(f"\n{'='*60}")
        print(f"KV dtype={kv_dtype}, seq_len={seq_len}")
        print(f"{'='*60}", flush=True)

        max_model_len = min(seq_len + MAX_TOKENS + 512, 32768)
        try:
            llm = LLM(
                model=MODEL,
                max_model_len=max_model_len,
                gpu_memory_utilization=0.85,
                dtype="auto",
                enforce_eager=True,
                kv_cache_dtype=kv_dtype,
            )
        except Exception as e:
            print(f"Failed to load: {e}")
            all_results.append(
                {
                    "model": MODEL,
                    "kv_dtype": kv_dtype,
                    "seq_len": seq_len,
                    "error": str(e)[:200],
                }
            )
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
            all_results.append(
                {
                    "model": MODEL,
                    "kv_dtype": kv_dtype,
                    "seq_len": seq_len,
                    "error": str(e)[:200],
                }
            )
            del llm
            torch.cuda.empty_cache()
            continue
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_in = sum(len(o.prompt_token_ids) for o in outputs)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # Get first few outputs for quality comparison
        sample_outputs = [o.outputs[0].text[:200] for o in outputs[:3]]

        result = {
            "tier": "4",
            "type": "kv_quantization",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu": GPU_TAG,
            "model": MODEL,
            "kv_cache_dtype": kv_dtype,
            "seq_len": seq_len,
            "num_prompts": len(prompts),
            "max_tokens": MAX_TOKENS,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "wall_time_sec": round(elapsed, 3),
            "tokens_per_sec": round(total_out / elapsed, 2),
            "avg_decode_latency_ms": round(elapsed / max(total_out, 1) * 1000, 2),
            "peak_vram_gb": round(peak_mem, 2),
            "sample_outputs": sample_outputs,
        }
        all_results.append(result)

        out_path = os.path.join(OUTPUT_DIR, f"tier4_kvquant_{kv_dtype}_{seq_len}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"  => {result['tokens_per_sec']} tok/s, latency={result['avg_decode_latency_ms']}ms, VRAM={result['peak_vram_gb']}GB"
        )

        del llm
        torch.cuda.empty_cache()

# Save combined
with open(os.path.join(OUTPUT_DIR, "tier4_kvquant_all.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\n\n=== KV QUANTIZATION RESULTS ===")
for r in all_results:
    if "error" not in r:
        print(
            f"  kv={r['kv_cache_dtype']:6s} seq={r['seq_len']:5d}  {r['tokens_per_sec']:8.1f} tok/s  VRAM={r['peak_vram_gb']:.1f}GB"
        )
    else:
        print(
            f"  kv={r.get('kv_dtype','?'):6s} seq={r.get('seq_len','?'):>5}  ERROR: {r['error'][:60]}"
        )
