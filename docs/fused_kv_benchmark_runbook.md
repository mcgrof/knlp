# Fused KV Quantization: Benchmark Runbook

> **Integration status**: This runbook is a protocol specification
> for evaluating fused INT4 KV quantization inside a serving engine.
> Stock vLLM (through 0.18.0) does not support the
> `--kv-cache-dtype int4_fused` flag used in examples below. The
> flag is a placeholder for a future vLLM branch or plugin. Until
> that branch exists, the fused decode proof artifacts come from
> standalone Triton kernel benchmarks
> (`scripts/spev01/tier5_fused_decode.py`), not from a vLLM serving
> path. See [fused_kv_quantization.md](fused_kv_quantization.md)
> for the full serving integration gap analysis.

This document specifies the evaluation protocol for validating fused
INT4 KV quantization against FP16 baselines in a paper-grade setting.
It covers serving throughput, latency profiling, accuracy evaluation,
and long-context retrieval benchmarks. The structure draws on
established long-context and serving evaluation practice in the open
LLM community, and is partially informed by the Lookahead Q-Cache
evaluation methodology.

All benchmarks compare exactly two configurations that differ only in
the KV cache format:

| Label    | Description                                    |
|----------|------------------------------------------------|
| `FP16`   | Standard FP16 KV cache (vLLM default)          |
| `FUSED`  | Fused INT4 dequant inside the attention kernel  |

Every other variable (model weights, vLLM commit, tensor-parallel
degree, max-model-len, GPU type, driver version) must be identical
between the two runs. Any deviation invalidates the comparison.

---

## 0. Reproducibility Requirements

Before running anything, lock the environment and record the
attention backend.

### Pin the vLLM commit

```bash
cd <VLLM_DIR>
git log --oneline -1 > /tmp/vllm_commit.txt
cat /tmp/vllm_commit.txt
# e.g. a1b2c3d  some commit message
```

Record this hash in every results directory. If you build from
source, record the full `git describe --dirty --always`.

### Collect environment

```bash
python -c "import vllm; vllm.utils.collect_env()" \
  > <RESULTS_DIR>/collect_env.txt 2>&1
python -c "import torch; print(torch.__version__)" \
  >> <RESULTS_DIR>/collect_env.txt
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader \
  >> <RESULTS_DIR>/collect_env.txt
```

### Generate the attention backend manifest

**Every benchmark run must produce `backend_manifest.json`.**
Attention dispatch in vLLM and HuggingFace is implicit: the
framework selects FlashAttention, SDPA, paged attention, or a
fallback based on hardware, library versions, and config flags
at runtime. Two runs with identical flags can hit different
kernels if the installed FlashAttention version differs. Record
the actual dispatch path, not just what you requested.

Generate the manifest at the start of every run:

```bash
python3 -c "
import json, os, sys, importlib

manifest = {}

# --- Serving engine ---
import vllm
manifest['serving_engine'] = 'vLLM'
manifest['vllm_version'] = vllm.__version__

# --- PyTorch ---
import torch
manifest['torch_version'] = torch.__version__
manifest['cuda_version'] = torch.version.cuda or 'N/A'
manifest['rocm_version'] = getattr(torch.version, 'hip', None) or 'N/A'

# --- AMD/ROCm experimental SDPA flag ---
# On AMD GPUs, PyTorch warns that Flash Efficient and Mem
# Efficient SDPA paths are experimental unless this is set.
# On prune this is exported in ~/.bashrc.
manifest['torch_rocm_aotriton_enable_experimental'] = os.environ.get(
    'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'unset')

# --- FlashAttention ---
try:
    import flash_attn
    manifest['flash_attn_version'] = flash_attn.__version__
except ImportError:
    manifest['flash_attn_version'] = 'not installed'

# --- vLLM attention backend ---
try:
    from vllm.config import get_attn_backend
    manifest['vllm_attn_backend'] = str(get_attn_backend())
except Exception:
    manifest['vllm_attn_backend'] = 'could not detect'

# --- GPU / platform ---
if torch.cuda.is_available():
    manifest['gpu_name'] = torch.cuda.get_device_name(0)
    manifest['gpu_count'] = torch.cuda.device_count()
    manifest['gpu_memory_gb'] = round(
        torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
else:
    manifest['gpu_name'] = 'N/A'

# --- KV cache mode ---
# Filled in by the caller after server starts.
manifest['kv_cache_dtype'] = '${KV_CACHE_DTYPE:-fp16}'

# --- Config flags ---
manifest['tensor_parallel_size'] = int('${TP:-1}')
manifest['max_model_len'] = int('${MAX_MODEL_LEN:-0}')
manifest['dtype'] = '${DTYPE:-auto}'
manifest['enforce_eager'] = '${ENFORCE_EAGER:-false}'

json.dump(manifest, sys.stdout, indent=2)
print()
" > <RESULTS_DIR>/backend_manifest.json
```

After the vLLM server starts, update the manifest with the
runtime attention backend from the server log:

```bash
# Extract the actual attention backend from server startup log
grep -i "attention backend\|Using.*attention\|flash.*attn\|PagedAttention" \
  <RESULTS_DIR>/guidellm/server_<TAG>.log \
  | head -5 \
  >> <RESULTS_DIR>/backend_manifest_runtime.txt
```

#### Required manifest fields

Every `backend_manifest.json` must contain at minimum:

| Field | Example | Why |
|-------|---------|-----|
| `serving_engine` | `vLLM`, `HF Transformers`, `TGI` | Different engines use different attention dispatch |
| `vllm_version` | `0.7.3` | Attention backend selection changes between versions |
| `torch_version` | `2.5.1+cu124` | SDPA backend availability depends on torch build |
| `cuda_version` / `rocm_version` | `12.4` / `6.2` | Kernel availability depends on compute platform |
| `flash_attn_version` | `2.7.3` or `not installed` | FlashAttention availability changes dispatch |
| `vllm_attn_backend` | `FLASH_ATTN`, `XFORMERS`, `ROCM_FLASH` | The actual backend vLLM selected |
| `gpu_name` | `NVIDIA H100 80GB HBM3` | Kernel dispatch is GPU-dependent |
| `gpu_count` | `1` | TP degree affects attention path |
| `kv_cache_dtype` | `fp16`, `int4_fused` | The quantization mode under test |
| `tensor_parallel_size` | `1` | Affects KV cache layout |
| `max_model_len` | `32768` | Affects paged attention block allocation |
| `enforce_eager` | `false` | torch.compile changes attention dispatch |
| `torch_rocm_aotriton_enable_experimental` | `1` or `unset` | On AMD/ROCm: enables experimental Flash Efficient and Mem Efficient SDPA paths. Without it, PyTorch silently falls back to slower attention. On `prune` this is exported in `~/.bashrc`. |

If the run uses HuggingFace Transformers directly (not vLLM),
record `attn_implementation` (`eager`, `sdpa`, `flash_attention_2`)
from the model config. If using vLLM with paged attention, record
whether PagedAttention V1 or V2 is active.

### Save all JSON artifacts

Every benchmark tool below produces JSON. Save it:

```bash
mkdir -p <RESULTS_DIR>/{guidellm,bench,lm_eval,niah,ruler,longbench,infinitebench}
```

### Identical launch configs

Write a single shell function or env file that both FP16 and FUSED
runs source, so the only delta is the quantization flag:

```bash
# common.env — source this in every run script
export MODEL=<MODEL_ID>
export TP=<TENSOR_PARALLEL_DEGREE>
export MAX_MODEL_LEN=<MAX_SEQ_LEN>
export GPU_MEMORY_UTILIZATION=0.90
export DTYPE=auto
export SEED=42
```

The FUSED run adds exactly one flag (e.g.
`--kv-cache-dtype fp8` or a custom fused-int4 flag depending on
the vLLM branch). Note: stock vLLM does not support
`int4_fused`; this requires a custom vLLM branch with fused
INT4 paged attention support. Document the exact flag
difference in `<RESULTS_DIR>/config_diff.txt`.

---

## 1. GuideLLM Headline Serving Sweeps

GuideLLM drives realistic open-loop traffic patterns against a vLLM
OpenAI-compatible endpoint and reports TTFT, ITL, throughput, and
goodput at varying request rates.

### Start the vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --max-model-len <MAX_MODEL_LEN> \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype $DTYPE \
  --port 8000 \
  <FUSED_FLAG_OR_EMPTY> \
  2>&1 | tee <RESULTS_DIR>/guidellm/server_<TAG>.log &

# Wait for server readiness
until curl -s http://localhost:8000/health | grep -q ok; do sleep 2; done
```

### Run GuideLLM sweep

```bash
guidellm \
  --target http://localhost:8000 \
  --model <MODEL_ID> \
  --data-type emulated \
  --data "prompt_tokens=512,generated_tokens=128" \
  --rate-type sweep \
  --max-seconds 300 \
  --output-path <RESULTS_DIR>/guidellm/<TAG>_sweep.json
```

Repeat with longer prompts to stress KV cache pressure:

```bash
guidellm \
  --target http://localhost:8000 \
  --model <MODEL_ID> \
  --data-type emulated \
  --data "prompt_tokens=4096,generated_tokens=256" \
  --rate-type sweep \
  --max-seconds 300 \
  --output-path <RESULTS_DIR>/guidellm/<TAG>_sweep_long.json
```

### Key metrics to extract

- Request throughput (req/s) at each offered rate
- Output token throughput (tok/s)
- TTFT p50, p95, p99
- ITL (inter-token latency) p50, p95, p99
- Goodput (successful requests within SLO)

---

## 2. vLLM Benchmark Suite

The vLLM repository ships four benchmark scripts under
`benchmarks/`. Run all four for each configuration.

### 2a. Latency (`benchmark_latency.py`)

Fixed batch, measures raw decode latency without scheduling
overhead.

```bash
python <VLLM_DIR>/benchmarks/benchmark_latency.py \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --input-len 512 \
  --output-len 128 \
  --batch-size 1 \
  --num-iters-warmup 3 \
  --num-iters 10 \
  <FUSED_FLAG_OR_EMPTY> \
  2>&1 | tee <RESULTS_DIR>/bench/<TAG>_latency_b1.log
```

Repeat for batch sizes 1, 4, 8, 16, 32 and input lengths
512, 2048, 8192.

### 2b. Throughput (`benchmark_throughput.py`)

Saturates the engine to find peak throughput.

```bash
python <VLLM_DIR>/benchmarks/benchmark_throughput.py \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --input-len 512 \
  --output-len 128 \
  --num-prompts 1000 \
  <FUSED_FLAG_OR_EMPTY> \
  2>&1 | tee <RESULTS_DIR>/bench/<TAG>_throughput.log
```

### 2c. Online serving (`benchmark_serving.py`)

Simulates Poisson-arrival clients against the OpenAI endpoint
(requires the server from Section 1 to be running).

```bash
python <VLLM_DIR>/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model <MODEL_ID> \
  --endpoint /v1/completions \
  --host localhost \
  --port 8000 \
  --dataset-name sharegpt \
  --dataset-path <SHAREGPT_JSON> \
  --request-rate 4 \
  --num-prompts 500 \
  --save-result \
  --result-dir <RESULTS_DIR>/bench/ \
  --result-filename <TAG>_serving_rr4.json
```

Sweep request rates: 1, 2, 4, 8, 16 req/s.

### 2d. Startup time

Engine initialization time matters for deployment. Measure
cold-start:

```bash
time python -c "
from vllm import LLM
llm = LLM(model='<MODEL_ID>',
           tensor_parallel_size=<TP>,
           max_model_len=<MAX_MODEL_LEN>,
           gpu_memory_utilization=$GPU_MEMORY_UTILIZATION)
print('Engine ready')
" 2>&1 | tee <RESULTS_DIR>/bench/<TAG>_startup.log
```

Run three times per configuration, report median.

---

## 3. lm-eval with vLLM Backend

Accuracy evaluation ensures fused quantization does not degrade
model quality beyond acceptable thresholds.

### Standard benchmarks

```bash
lm_eval --model vllm \
  --model_args pretrained=<MODEL_ID>,tensor_parallel_size=<TP>,dtype=auto,gpu_memory_utilization=0.90,add_bos_token=True,max_model_len=<MAX_MODEL_LEN> \
  --tasks mmlu,hellaswag,arc_challenge,winogrande,gsm8k,truthfulqa_mc2 \
  --batch_size auto \
  --num_fewshot 5 \
  --output_path <RESULTS_DIR>/lm_eval/<TAG>_standard.json \
  --log_samples
```

The `add_bos_token=True` flag is required for correct results
on models that condition on BOS (most causal LMs). Omitting it
silently degrades scores.

### Report format

For each task, report:

| Task            | FP16 acc | FUSED acc | Delta |
|-----------------|----------|-----------|-------|
| mmlu (5-shot)   |          |           |       |
| hellaswag       |          |           |       |
| arc_challenge   |          |           |       |
| winogrande      |          |           |       |
| gsm8k           |          |           |       |
| truthfulqa_mc2  |          |           |       |

Delta beyond 1% on any task warrants investigation. Delta beyond
3% is a red flag.

---

## 4. Long-Context Evaluations

These benchmarks stress the KV cache at extreme sequence lengths,
which is where fused quantization errors can compound.

### 4a. Needle-in-a-Haystack (NIAH)

Retrieval accuracy across depth and context length.

```bash
python <NIAH_DIR>/run_needle_in_haystack.py \
  --model_name <MODEL_ID> \
  --provider vllm \
  --context_lengths 4096,8192,16384,32768,65536,131072 \
  --document_depth_percents 0,10,20,30,40,50,60,70,80,90,100 \
  --results_dir <RESULTS_DIR>/niah/<TAG>/ \
  <FUSED_FLAG_OR_EMPTY>
```

If using the LLMTest_NeedleInAHaystack framework, configure
the vLLM OpenAI endpoint as the provider:

```bash
export NIAH_MODEL_API_URL=http://localhost:8000/v1
python -m needlehaystack.run \
  --model_name <MODEL_ID> \
  --evaluator_model_name gpt-4o \
  --context_lengths_min 4096 \
  --context_lengths_max 131072 \
  --context_lengths_num_intervals 6 \
  --document_depth_percent_intervals 11 \
  --results_dir <RESULTS_DIR>/niah/<TAG>/
```

Plot the retrieval heatmap (depth x length). Any cell where
FUSED retrieval drops below FP16 indicates a quantization
failure at that operating point.

### 4b. RULER

RULER provides controlled synthetic tasks that isolate specific
long-context capabilities: multi-key retrieval, variable tracking,
aggregation, and question answering.

```bash
# Generate RULER task data
python <RULER_DIR>/scripts/data/prepare_data.py \
  --task niah_single,niah_multikey,vt,cwe,qa \
  --model_name <MODEL_ID> \
  --tokenizer_name <MODEL_ID> \
  --max_seq_length 131072 \
  --output_dir <RESULTS_DIR>/ruler/<TAG>/data/

# Evaluate via vLLM
python <RULER_DIR>/scripts/eval/evaluate_vllm.py \
  --model_name <MODEL_ID> \
  --data_dir <RESULTS_DIR>/ruler/<TAG>/data/ \
  --output_dir <RESULTS_DIR>/ruler/<TAG>/results/ \
  --tensor_parallel_size <TP> \
  --max_model_len <MAX_MODEL_LEN> \
  <FUSED_FLAG_OR_EMPTY>
```

Report per-task accuracy at each context length. Focus on the
slope of degradation: fused quantization should not steepen the
accuracy falloff compared to FP16.

### 4c. LongBench

LongBench evaluates on real-world long-document tasks (single-doc
QA, multi-doc QA, summarization, few-shot learning, code
completion, synthetic retrieval).

```bash
python <LONGBENCH_DIR>/pred.py \
  --model <MODEL_ID> \
  --backend vllm \
  --tensor_parallel_size <TP> \
  --max_length <MAX_MODEL_LEN> \
  --output_dir <RESULTS_DIR>/longbench/<TAG>/predictions/ \
  <FUSED_FLAG_OR_EMPTY>

python <LONGBENCH_DIR>/eval.py \
  --pred_dir <RESULTS_DIR>/longbench/<TAG>/predictions/ \
  --output_dir <RESULTS_DIR>/longbench/<TAG>/scores/
```

Report the per-dataset F1/ROUGE/accuracy and the aggregate
score. Compare FP16 vs FUSED.

### 4d. InfiniteBench

InfiniteBench targets 100K+ token contexts with tasks that
require reasoning over the full document (passkey retrieval,
number retrieval, KV retrieval, math find, code debug).

```bash
python <INFINITEBENCH_DIR>/eval_vllm.py \
  --model_name <MODEL_ID> \
  --task_name passkey,number_string,kv_retrieval,math_find,code_debug \
  --max_seq_length <MAX_MODEL_LEN> \
  --tensor_parallel_size <TP> \
  --output_dir <RESULTS_DIR>/infinitebench/<TAG>/ \
  <FUSED_FLAG_OR_EMPTY>
```

Passkey and KV retrieval are the most sensitive to cache
corruption. Any score drop on these tasks is a strong signal
that quantization errors are accumulating at long range.

---

## 5. Recommended Execution Order

If this is a new machine, new vLLM build, or changed model
target, run the [smoke test suite](benchmarks/smoke-test.md)
first (15-25 minutes) to validate plumbing across all benchmark
tools before investing in the phases below.

Run benchmarks in this order so that early failures prevent
wasted compute on downstream tasks.

| Phase | Benchmark             | Time Est. | Purpose                          |
|-------|-----------------------|-----------|----------------------------------|
| 1     | Startup time          | 10 min    | Verify both configs load cleanly |
| 2     | lm-eval (standard)    | 1-3 hr    | Catch accuracy regressions early |
| 3     | Latency (batch=1)     | 20 min    | Sanity-check decode path         |
| 4     | NIAH                  | 1-2 hr    | Detect long-range cache errors   |
| 5     | Latency (full sweep)  | 1-2 hr    | Batch x length latency matrix    |
| 6     | Throughput            | 30 min    | Peak token throughput             |
| 7     | GuideLLM sweep        | 1-2 hr    | End-to-end serving profile       |
| 8     | Serving (rate sweep)  | 1-2 hr    | Online serving under load        |
| 9     | RULER                 | 2-4 hr    | Structured long-context tasks    |
| 10    | LongBench             | 2-4 hr    | Real-world long-doc tasks        |
| 11    | InfiniteBench         | 2-4 hr    | Extreme-length stress tests      |

Rationale: phases 1-4 are fast sanity checks that catch showstopper
issues. If accuracy drops >3% or NIAH retrieval fails, stop and
debug before investing in the full serving and long-context sweep.

---

## 6. Minimal Respectable Subset

If GPU budget is tight, run this reduced set. It covers accuracy,
latency, throughput, and one long-context probe in roughly 4-6
hours total.

1. **lm-eval** with mmlu + hellaswag + arc_challenge (3 tasks,
   5-shot). Confirms no quality loss.
2. **Latency** at batch=1 and batch=8, input_len=512 and
   input_len=8192 (4 data points per config). Confirms decode
   speedup exists and scales.
3. **Throughput** at input_len=512, output_len=128, 1000 prompts.
   Confirms peak throughput improvement.
4. **NIAH** at context lengths 4096, 16384, 32768 with 5 depth
   intervals. Confirms no retrieval degradation.

This subset is sufficient for a workshop paper or technical report.
The full suite is expected for a top-venue submission.

---

## 7. Report Metrics

### Serving and Latency

For every benchmark that measures latency or throughput, report:

| Metric                     | Unit     | Source             |
|----------------------------|----------|--------------------|
| TTFT p50                   | ms       | GuideLLM / serving |
| TTFT p95                   | ms       | GuideLLM / serving |
| TTFT p99                   | ms       | GuideLLM / serving |
| ITL p50                    | ms       | GuideLLM / serving |
| ITL p95                    | ms       | GuideLLM / serving |
| ITL p99                    | ms       | GuideLLM / serving |
| Request throughput         | req/s    | GuideLLM / serving |
| Output token throughput    | tok/s    | throughput bench    |
| Total token throughput     | tok/s    | throughput bench    |
| Decode latency (batch=N)   | ms/token | latency bench      |

### Accuracy

| Metric           | Unit | Source     |
|------------------|------|------------|
| Accuracy delta   | %    | lm-eval    |
| NIAH retrieval   | %    | NIAH       |
| RULER per-task   | %    | RULER      |
| LongBench agg.   | F1/% | LongBench  |
| InfiniteBench    | %    | InfiniteBench |

### Thresholds

- **Accuracy**: FUSED accuracy within 1% of FP16 on all standard
  tasks is a PASS. Within 3% is acceptable with documented
  justification. Beyond 3% is a FAIL.
- **NIAH**: 100% retrieval at all tested depths and lengths is
  expected. Any cell with <95% retrieval warrants investigation.
- **Latency**: Report absolute numbers and percentage change vs
  FP16. Improvements <5% at any operating point should not be
  claimed as meaningful.
- **Throughput**: Report both output and total token throughput.
  A serving improvement claim requires measurable gains at
  realistic request rates (not just saturated throughput).

---

## 8. Results Directory Structure

```
<RESULTS_DIR>/
├── backend_manifest.json       # REQUIRED: attention backend + versions
├── backend_manifest_runtime.txt # Attention backend from server log
├── collect_env.txt
├── config_diff.txt
├── common.env
├── guidellm/
│   ├── fp16_sweep.json
│   ├── fp16_sweep_long.json
│   ├── fused_sweep.json
│   └── fused_sweep_long.json
├── bench/
│   ├── fp16_latency_b1.log
│   ├── fp16_latency_b8.log
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
│   ├── fp16/
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

Every JSON artifact must be committed to the key-results repo
for reproducibility:

```bash
cp -a <RESULTS_DIR>/ /data/knlp-key-results/fused_kv_bench/
cd /data/knlp-key-results && git add fused_kv_bench/ && git commit
```

---

## 9. Attention Backend Testing Matrix

Do not leave attention backend dispatch implicit. Test both
FlashAttention-backed and paged-attention paths where the hardware
and vLLM build support them. At minimum, run each benchmark
configuration with the backend explicitly set and recorded.

### Required backend coverage

| Backend | How to force | When to test |
|---------|-------------|--------------|
| FlashAttention V2 | Install `flash-attn`, vLLM auto-selects | Default on NVIDIA GPUs with flash-attn installed |
| SDPA (torch native) | `export VLLM_ATTENTION_BACKEND=TORCH_SDPA` | Fallback path; test when flash-attn is absent or on non-NVIDIA hardware |
| FlashInfer | `export VLLM_ATTENTION_BACKEND=FLASHINFER` | Paged-attention path used in production vLLM deployments |
| ROCm Flash | Auto-selected on AMD GPUs | Required for AMD GPU benchmarks |
| Eager (no fusion) | `--enforce-eager` + `export VLLM_ATTENTION_BACKEND=TORCH_SDPA` | Baseline for isolating kernel-level effects |

For each backend tested, generate a separate
`backend_manifest.json` and store results in backend-specific
subdirectories (e.g. `bench_flash/`, `bench_sdpa/`). If a backend
is unavailable on the test hardware, document this in the manifest
rather than silently omitting it.

### Preventing silent backend fallback

vLLM silently falls back to a slower backend when the preferred
one is unavailable (e.g. FlashAttention not installed, or
incompatible GPU). To detect this:

1. Set `VLLM_ATTENTION_BACKEND` explicitly in `common.env`.
2. After server startup, grep the log for the actual backend.
3. Compare the requested backend against the runtime log. If
   they differ, the run is invalid and must be re-done with the
   correct backend installed.

```bash
# Verify backend matches request
REQUESTED="FLASH_ATTN"
ACTUAL=$(grep -oP 'Using attention backend: \K\S+' \
  <RESULTS_DIR>/guidellm/server_<TAG>.log)
if [ "$ACTUAL" != "$REQUESTED" ]; then
  echo "ERROR: Requested $REQUESTED but got $ACTUAL"
  echo "Install the correct backend or update common.env"
  exit 1
fi
```

---

## 10. Checklist Before Submission

- [ ] `backend_manifest.json` present with all required fields
- [ ] Attention backend verified against server startup log
- [ ] vLLM commit hash recorded
- [ ] `collect_env.txt` saved
- [ ] `config_diff.txt` documents the exact flag difference
- [ ] FP16 and FUSED use identical model weights, TP, max-model-len
- [ ] FP16 and FUSED use the same attention backend
- [ ] All JSON results committed to key-results repo
- [ ] lm-eval accuracy delta <1% on all standard tasks
- [ ] NIAH retrieval 100% at all tested (depth, length) pairs
- [ ] Latency and throughput numbers reported with p50/p95/p99
- [ ] No cherry-picked operating points; full sweep data available
- [ ] Results reproducible from committed configs and pinned vLLM
- [ ] Multiple attention backends tested where hardware permits
