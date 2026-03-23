# Fused KV Quantization Benchmark Documentation

This directory contains the evaluation documentation for validating
fused INT4 KV quantization against FP16 baselines. The benchmarks
produce paper-grade evidence for serving throughput, latency, model
accuracy, and long-context retrieval quality.

## Documents

| Document | Purpose |
|----------|---------|
| [Smoke Test Plan](smoke-test.md) | Validate plumbing and artifact generation across the full evaluation stack in 15-25 minutes before committing to real runs |
| [Quickstart](quickstart.md) | Get running in 30 minutes with a minimal FP16-vs-FUSED comparison |
| [Full Runbook](../fused_kv_benchmark_runbook.md) | The canonical evaluation protocol: all 7 benchmark tools, execution order, thresholds, and result directory layout |
| [Reproducibility Checklist](reproducibility.md) | Pre-submission checklist, required artifacts, environment pinning, and result archival procedure |

## What This Evaluates

Every benchmark compares exactly two configurations:

| Label   | Description                                   |
|---------|-----------------------------------------------|
| `FP16`  | Standard FP16 KV cache (vLLM default)         |
| `FUSED` | Fused INT4 dequant inside the attention kernel |

All other variables (model weights, vLLM commit, tensor-parallel
degree, max-model-len, GPU type, driver version, **attention
backend**) are held identical. Any deviation invalidates the
comparison.

**Attention backend dispatch is implicit.** vLLM and HuggingFace
Transformers select the attention kernel (FlashAttention, SDPA,
paged attention, etc.) at runtime based on installed libraries and
hardware. Two runs with identical CLI flags can hit different
kernels. Every benchmark run must record the actual backend in a
`backend_manifest.json` artifact. See the
[runbook Section 0](../fused_kv_benchmark_runbook.md) for the
manifest generation script and required fields.

## Benchmark Tools

Seven tools form the evaluation stack. Each targets a different
failure mode of KV cache quantization.

### Serving and Latency

| Tool | What It Measures | When It Matters |
|------|-----------------|-----------------|
| **GuideLLM** | End-to-end serving under open-loop traffic (TTFT, ITL, goodput at varying request rates) | Production deployment: does the fused kernel help under realistic load? |
| **vLLM `benchmark_latency.py`** | Raw decode latency at fixed batch sizes, no scheduling overhead | Kernel-level validation: is the fused decode path faster per token? |
| **vLLM `benchmark_throughput.py`** | Peak token throughput with engine saturated | Capacity planning: how many tokens/s can the system sustain? |
| **vLLM `benchmark_serving.py`** | Online serving with Poisson-arrival clients | SLO compliance: does latency hold under bursty traffic? |

### Accuracy

| Tool | What It Measures | When It Matters |
|------|-----------------|-----------------|
| **lm-eval** | Standard NLP benchmarks (MMLU, HellaSwag, ARC, WinoGrande, GSM8K, TruthfulQA) via vLLM backend | Quality gate: does quantization degrade model accuracy beyond 1%? |

### Long-Context Retrieval

| Tool | What It Measures | When It Matters |
|------|-----------------|-----------------|
| **NIAH** (Needle-in-a-Haystack) | Retrieval accuracy across document depth and context length | Cache corruption detection: do quantization errors compound over long sequences? |
| **RULER** | Controlled synthetic long-context tasks (multi-key retrieval, variable tracking, aggregation, QA) | Structured long-context: does the model degrade on tasks requiring precise long-range recall? |
| **LongBench** | Real-world long-document tasks (single/multi-doc QA, summarization, few-shot, code) | Ecological validity: does quantization hurt on actual long-document workloads? |
| **InfiniteBench** | Extreme-length tasks at 100K+ tokens (passkey, KV retrieval, code debug) | Stress test: does the fused cache survive at maximum context length? |

## Result Directory Structure

All benchmarks write JSON artifacts into a single results tree:

```
<RESULTS_DIR>/
├── backend_manifest.json    # REQUIRED: attention backend, versions, GPU
├── backend_manifest_runtime.txt  # Runtime backend from server log
├── collect_env.txt          # vLLM version, torch version, GPU info
├── config_diff.txt          # Exact flag difference between FP16 and FUSED
├── common.env               # Shared launch config (model, TP, max-model-len)
├── guidellm/
│   ├── fp16_sweep.json      # GuideLLM sweep at prompt_tokens=512
│   ├── fp16_sweep_long.json # GuideLLM sweep at prompt_tokens=4096
│   ├── fused_sweep.json
│   └── fused_sweep_long.json
├── bench/
│   ├── fp16_latency_b1.log  # Latency at batch=1
│   ├── fp16_latency_b8.log  # Latency at batch=8
│   ├── fused_latency_b1.log
│   ├── fused_latency_b8.log
│   ├── fp16_throughput.log
│   ├── fused_throughput.log
│   ├── fp16_serving_rr4.json
│   ├── fused_serving_rr4.json
│   ├── fp16_startup.log
│   └── fused_startup.log
├── lm_eval/
│   ├── fp16_standard.json
│   └── fused_standard.json
├── niah/
│   ├── fp16/                # Per-cell retrieval results
│   └── fused/
├── ruler/
│   ├── fp16/
│   └── fused/
├── longbench/
│   ├── fp16/
│   └── fused/
└── infinitebench/
    ├── fp16/
    └── fused/
```

After the run, archive results to the key-results repo:

```bash
cp -a <RESULTS_DIR>/ /data/knlp-key-results/fused_kv_bench/
cd /data/knlp-key-results && git add fused_kv_bench/ && git commit
```

## Thresholds

These are the pass/fail criteria from the runbook:

- **Accuracy (lm-eval)**: FUSED within 1% of FP16 on all tasks is
  PASS. Within 3% is acceptable with justification. Beyond 3% is
  FAIL.
- **NIAH retrieval**: 100% at all tested depth/length pairs.
  Any cell below 95% warrants investigation.
- **Latency improvement**: Gains below 5% at any operating point
  should not be claimed as meaningful.
- **Throughput**: Report both output and total token throughput.
  Serving improvement claims require measurable gains at realistic
  request rates, not just saturated throughput.

## Related Documentation

- [Fused KV Quantization overview](../fused_kv_quantization.md) ---
  what "fused" means, why it matters, implementation pointers,
  paper results summary
- [BPA overview](../bpa.md) --- the broader research arc
- [Comparison axes](../comparison_axes.md) --- alignment with
  literature benchmarking standards (Palu, MiniCache, PyramidKV)
- [Calibration guide](../kv_plugin/calibration_guide.md) --- ratio
  classifier for per-model KV precision policy
