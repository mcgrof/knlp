# Smoke Test Plan

A smoke test validates that every benchmark tool starts, runs,
produces the expected output artifacts, and exits cleanly. It does
NOT produce publishable numbers. The goal is to catch configuration
errors, missing dependencies, broken paths, incompatible API
changes, and artifact generation failures before committing hours
of GPU time to a full evaluation run.

Run the smoke suite whenever:

- You set up a new machine or environment
- You upgrade vLLM, lm-eval, or any benchmark dependency
- You change the model target or tensor-parallel degree
- You modify the KV cache quantization flag
- You want a quick sanity check before a multi-hour run

For the full evaluation protocol, see the
[runbook](../fused_kv_benchmark_runbook.md). For the minimal
publishable subset, see the [quickstart](quickstart.md).

---

## Confirmed Smoke-Pass Set on W7900

The following phases pass end-to-end on an AMD Radeon Pro W7900
(48 GB) with `marin-community/marin-8b-base`, TP=1:

| Phase | Benchmark | Status |
|-------|-----------|--------|
| S2 | vLLM latency | PASS |
| S3 | vLLM throughput | PASS |
| S4 | vLLM serving | PASS |
| S6 | GuideLLM serving | PASS |
| S9 | LongBench | PASS |
| S7 | NIAH | PASS |
| S8 | RULER (variable tracking) | PASS |
| S10 | InfiniteBench smoke | PASS |

InfiniteBench smoke proves harness execution, not benchmark
quality. The passkey score at `MAX_MODEL_LEN=4096` is
meaningless for quality purposes because the task requires
100K+ token contexts; the smoke only validates that the eval
loop runs to completion without error.

---

## AMD / ROCm Environment Prerequisites

On AMD GPUs, PyTorch warns that the Flash Efficient and Mem
Efficient SDPA attention paths are experimental unless the
following environment variable is set:

```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

On the `prune` benchmark host, this variable is already
exported in `~/.bashrc`. Verify it is active before every
benchmark session:

```bash
echo $TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
# Must print: 1
```

Without this variable, vLLM may silently fall back to a
slower attention path or refuse to use the preferred SDPA
backend. Log the state of this variable in the backend
manifest (see S-1 below) for every run.

---

## What Smoke Tests Validate

Each smoke test targets one tool in the evaluation stack and
confirms four properties:

1. **Plumbing**: The tool launches, connects to the server (if
   applicable), and completes without error.
2. **Artifact generation**: The expected JSON, log, or directory
   output is created and non-empty.
3. **Metrics output**: The tool emits parseable metrics (latency
   numbers, accuracy scores, retrieval results) rather than empty
   or malformed output.
4. **Config correctness**: The model loads with the intended flags,
   the KV cache dtype is what you specified, and tensor-parallel
   degree matches.

Smoke tests explicitly do NOT validate:

- Statistical significance of latency differences
- Accuracy deltas (sample counts are too small)
- Long-context degradation slopes
- Production throughput claims

---

## Prerequisites

Run all smoke tests inside the `knlp-rocm-bench` container (see
[container.md](container.md)). The container provides vLLM, lm-eval,
GuideLLM, and sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
automatically.  Python dependencies for xKV and InfiniteBench eval
paths are baked into the image — no `pip install` at container start
is needed.  The xKV and InfiniteBench repositories themselves are
bind-mounted from `/data` (see container.md for what is baked in
vs. bind-mounted and why).

These examples use `marin-community/marin-8b-base` on an AMD
Radeon Pro W7900 (48 GB), but substitute any model your pipeline
targets.

```bash
# Shared environment for smoke tests
export MODEL=marin-community/marin-8b-base
export TP=1
export MAX_MODEL_LEN=4096       # Short context is fine for smoke
export GPU_MEMORY_UTILIZATION=0.90
export DTYPE=auto
export SEED=42

SMOKE_DIR=./smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p $SMOKE_DIR/{guidellm,bench,lm_eval,niah,ruler,longbench,infinitebench}
```

The `MAX_MODEL_LEN=4096` is deliberate. Smoke tests should run
in minutes, not hours. Long-context benchmarks use the shortest
lengths that exercise the code path.

---

## Recommended Execution Order

Run smoke tests in this order. Each phase gates the next: if a
phase fails, fix it before proceeding.

| Phase | Benchmark | Est. Time | What It Catches |
|-------|-----------|-----------|-----------------|
| S-1 | Backend manifest | <1 min | Attention backend detection, library versions, GPU info |
| S0 | Server startup | 1-2 min | Model loading, VRAM, flag parsing |
| S1 | Startup time | 1 min | Cold-start initialization, engine construction |
| S2 | vLLM latency smoke | 1 min | Decode path, batch dispatch |
| S3 | vLLM throughput smoke | 1 min | Engine saturation path, prompt handling |
| S4 | vLLM serving smoke | 1-2 min | Poisson-arrival client, result serialization |
| S5 | lm-eval smoke | 2-5 min | Accuracy harness, vLLM backend integration |
| S6 | GuideLLM serving smoke | 2-3 min | Server endpoint, open-loop client, JSON output |
| S7 | NIAH smoke | 2-3 min | Retrieval harness, server interaction, heatmap data |
| S8 | RULER smoke | 2-3 min | Synthetic task generation, eval loop, scoring |
| S9 | LongBench smoke | 2-3 min | Prediction + scoring pipeline, dataset download |
| S10 | InfiniteBench smoke | 2-3 min | Extreme-length eval loop, passkey retrieval path |

**Total estimated time**: 15-25 minutes for all phases.

Rationale: S-1 catches library and backend detection issues
before any GPU work. S0 catches fatal issues (wrong model path,
OOM, bad flags). S1 measures cold-start. S2-S3 validate the
offline engine without a server. S4-S6 require a running server
and validate online paths. S7-S10 validate long-context
harnesses against the server.

---

## S-1. Backend Manifest Smoke

**Validates**: Attention backend libraries are installed and
detectable, the manifest generation script runs without error,
and all required fields are populated. This catches missing
FlashAttention installs, wrong CUDA/ROCm builds, and GPU
detection failures before any benchmark runs.

**Artifacts**: `$SMOKE_DIR/backend_manifest.json`

```bash
python3 -c "
import json, os, sys, torch
m = {}
m['torch_version'] = torch.__version__
m['cuda_version'] = torch.version.cuda or 'N/A'
m['rocm_version'] = getattr(torch.version, 'hip', None) or 'N/A'
m['torch_rocm_aotriton_enable_experimental'] = os.environ.get(
    'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'unset')
try:
    import flash_attn
    m['flash_attn_version'] = flash_attn.__version__
except ImportError:
    m['flash_attn_version'] = 'not installed'
import vllm
m['vllm_version'] = vllm.__version__
try:
    from vllm.config import get_attn_backend
    m['vllm_attn_backend'] = str(get_attn_backend())
except Exception:
    m['vllm_attn_backend'] = 'could not detect'
if torch.cuda.is_available():
    m['gpu_name'] = torch.cuda.get_device_name(0)
    m['gpu_count'] = torch.cuda.device_count()
    m['gpu_memory_gb'] = round(
        torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
else:
    m['gpu_name'] = 'N/A'
json.dump(m, sys.stdout, indent=2); print()
" > $SMOKE_DIR/backend_manifest.json
```

**Pass criteria**: Script exits 0. JSON file contains
`vllm_attn_backend`, `flash_attn_version`, `torch_version`,
and `gpu_name`. On AMD/ROCm hosts, verify
`torch_rocm_aotriton_enable_experimental` is `1`. If it is
`unset`, add `export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
to `~/.bashrc` and re-source before proceeding. If
`flash_attn_version` is `not installed` and you expected
FlashAttention, stop and install it before proceeding.

**Verify artifact**:
```bash
python3 -c "
import json, sys
with open('$SMOKE_DIR/backend_manifest.json') as f:
    m = json.load(f)
required = ['torch_version', 'flash_attn_version',
            'vllm_version', 'gpu_name']
missing = [k for k in required if k not in m]
if missing:
    print(f'SMOKE S-1: FAIL — missing fields: {missing}')
    sys.exit(1)
print(f'SMOKE S-1: PASS')
print(f'  Attention backend: {m.get(\"vllm_attn_backend\", \"unknown\")}')
print(f'  FlashAttention: {m[\"flash_attn_version\"]}')
print(f'  GPU: {m[\"gpu_name\"]}')
"
```

---

## S0. Server Startup Smoke

**Validates**: Model loads, health endpoint responds, VRAM
allocation succeeds, KV cache dtype flag is parsed correctly.

**Artifacts**: `$SMOKE_DIR/bench/smoke_startup.log`

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size $TP \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype $DTYPE \
  --port 8000 \
  2>&1 | tee $SMOKE_DIR/bench/smoke_startup.log &
SERVER_PID=$!

# Wait for readiness (timeout after 120s)
for i in $(seq 1 60); do
  curl -s http://localhost:8000/health | grep -q ok && break
  sleep 2
done

# Verify
curl -s http://localhost:8000/health | grep -q ok \
  && echo "SMOKE S0: PASS — server healthy" \
  || echo "SMOKE S0: FAIL — server not healthy after 120s"
```

**Pass criteria**: Server responds to `/health` within 120
seconds. Log file is non-empty and contains no Python
tracebacks.

Leave the server running for S4-S10.

---

## S1. Startup Time Smoke

**Validates**: Engine cold-start completes, LLM constructor
returns without error, weight loading and KV cache allocation
succeed. This corresponds to runbook Section 2d (startup time)
and catches issues like missing model files, incompatible
config, or silent OOM during cache pre-allocation.

**Artifacts**: `$SMOKE_DIR/bench/smoke_startup_time.log`

```bash
{ time python -c "
from vllm import LLM
llm = LLM(model='$MODEL',
           tensor_parallel_size=$TP,
           max_model_len=$MAX_MODEL_LEN,
           gpu_memory_utilization=$GPU_MEMORY_UTILIZATION)
print('Engine ready')
" ; } 2>&1 | tee $SMOKE_DIR/bench/smoke_startup_time.log
```

**Pass criteria**: Script exits 0 and prints "Engine ready".
Log contains a `real` time line from the shell `time` builtin.
The time value itself does not matter for smoke purposes; only
that the engine initializes without error.

**Verify artifact**:
```bash
grep -q "Engine ready" $SMOKE_DIR/bench/smoke_startup_time.log \
  && echo "SMOKE S1: PASS" \
  || echo "SMOKE S1: FAIL — engine did not initialize"
```

---

## S2. vLLM Latency Smoke

**Validates**: Offline decode path works end-to-end. The
engine accepts a batch, runs prefill + decode, and reports
latency. Catches kernel dispatch errors, dtype mismatches,
and attention backend failures.

**Artifacts**: `$SMOKE_DIR/bench/smoke_latency.log`

```bash
python $VLLM_DIR/benchmarks/benchmark_latency.py \
  --model $MODEL \
  --tensor-parallel-size $TP \
  --input-len 128 \
  --output-len 16 \
  --batch-size 1 \
  --num-iters-warmup 1 \
  --num-iters 3 \
  2>&1 | tee $SMOKE_DIR/bench/smoke_latency.log
```

**Pass criteria**: Script exits 0. Log file contains a latency
number (e.g. `avg latency: X.XX ms`). The number itself does
not matter for smoke purposes.

**Verify artifact**:
```bash
test -s $SMOKE_DIR/bench/smoke_latency.log \
  && grep -qi "latency" $SMOKE_DIR/bench/smoke_latency.log \
  && echo "SMOKE S2: PASS" \
  || echo "SMOKE S2: FAIL — missing or empty output"
```

---

## S3. vLLM Throughput Smoke

**Validates**: Engine saturation path works. The throughput
benchmark feeds many prompts and measures aggregate token
throughput. Catches prompt handling errors, scheduler bugs,
and output token counting failures.

**Artifacts**: `$SMOKE_DIR/bench/smoke_throughput.log`

```bash
python $VLLM_DIR/benchmarks/benchmark_throughput.py \
  --model $MODEL \
  --tensor-parallel-size $TP \
  --input-len 128 \
  --output-len 16 \
  --num-prompts 10 \
  2>&1 | tee $SMOKE_DIR/bench/smoke_throughput.log
```

**Pass criteria**: Script exits 0. Log contains throughput
numbers (e.g. `Throughput: X.XX requests/s`).

**Verify artifact**:
```bash
test -s $SMOKE_DIR/bench/smoke_throughput.log \
  && grep -qi "throughput" $SMOKE_DIR/bench/smoke_throughput.log \
  && echo "SMOKE S3: PASS" \
  || echo "SMOKE S3: FAIL — missing or empty output"
```

---

## S4. vLLM Serving Smoke

**Validates**: Online serving path through the OpenAI-compatible
endpoint. The Poisson-arrival client sends requests, the
server schedules them, and the benchmark collects per-request
latency metrics. Catches endpoint routing errors, request
serialization bugs, and result file generation failures.

Requires the server from S0 to be running. Use the completions endpoint (`/v1/completions`) for Marin in this stack. Do not use the chat endpoint unless the tokenizer provides a valid chat template. Uses synthetic data to avoid needing the ShareGPT dataset.

**Artifacts**: `$SMOKE_DIR/bench/smoke_serving.json`

```bash
python $VLLM_DIR/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model $MODEL \
  --endpoint /v1/completions \
  --host localhost \
  --port 8000 \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 16 \
  --request-rate 2 \
  --num-prompts 10 \
  --save-result \
  --result-dir $SMOKE_DIR/bench/ \
  --result-filename smoke_serving.json
```

**Pass criteria**: Script exits 0. JSON result file exists with
latency metrics.

**Verify artifact**:
```bash
test -s $SMOKE_DIR/bench/smoke_serving.json \
  && echo "SMOKE S4: PASS" \
  || echo "SMOKE S4: FAIL — serving benchmark produced no output"
```

Note: `--dataset-name random` avoids needing to download
ShareGPT. 10 prompts at rate=2 finishes in about 5 seconds of
request generation plus decode time.

---

## S5. lm-eval Smoke

**Validates**: The lm-eval harness loads, connects to the vLLM
backend, runs evaluation on a small sample, and writes a JSON
results file. Catches backend integration errors (wrong
model_args format, missing `add_bos_token`), task loading
failures, and output serialization bugs.

**Artifacts**: `$SMOKE_DIR/lm_eval/smoke_standard.json`

Run a single lightweight task with minimal samples. The
`arc_easy` task is fast and catches vLLM backend integration
issues.

```bash
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.90,add_bos_token=True,max_model_len=$MAX_MODEL_LEN \
  --tasks arc_easy \
  --batch_size auto \
  --num_fewshot 0 \
  --limit 20 \
  --output_path $SMOKE_DIR/lm_eval/smoke_standard.json \
  --log_samples
```

**Pass criteria**: Script exits 0. JSON output file exists and
contains an `"results"` key with accuracy scores.

**Verify artifact**:
```bash
python3 -c "
import json, sys
with open('$SMOKE_DIR/lm_eval/smoke_standard.json') as f:
    d = json.load(f)
assert 'results' in d, 'no results key'
print('SMOKE S5: PASS — lm-eval produced results')
" || echo "SMOKE S5: FAIL"
```

Note: `--limit 20` restricts to 20 samples per task so it
finishes in under a minute. This is far too few for meaningful
accuracy numbers but sufficient to verify the pipeline.

---

## S6. GuideLLM Serving Smoke

**Validates**: GuideLLM connects to the vLLM OpenAI-compatible
endpoint, drives open-loop traffic, collects TTFT/ITL/throughput
metrics, and writes a JSON report. Catches endpoint
compatibility issues, emulated data generation failures, and
rate sweep logic errors.

Requires the server from S0 to be running.

**Artifacts**: `$SMOKE_DIR/guidellm/smoke_sweep.json`

```bash
guidellm \
  --target http://localhost:8000 \
  --model $MODEL \
  --data-type emulated \
  --data "prompt_tokens=128,generated_tokens=16" \
  --rate-type sweep \
  --max-seconds 30 \
  --output-path $SMOKE_DIR/guidellm/smoke_sweep.json
```

**Pass criteria**: Script exits 0. JSON output file exists and
contains request-level metrics.

**Verify artifact**:
```bash
test -s $SMOKE_DIR/guidellm/smoke_sweep.json \
  && echo "SMOKE S6: PASS" \
  || echo "SMOKE S6: FAIL — GuideLLM produced no output"
```

Note: `--max-seconds 30` caps the sweep to 30 seconds total.
This produces only 1-2 rate points, enough to validate
plumbing.

---

## S7. NIAH Smoke

**Validates**: The Needle-in-a-Haystack retrieval harness
sends prompts with an embedded needle to the vLLM server,
collects model responses, scores retrieval accuracy, and writes
per-cell result files. Catches prompt construction errors,
server interaction failures, result parsing bugs, and output
directory creation issues.

Requires the server from S0 to be running. Uses the shortest
context length and fewest depth intervals that exercise the
retrieval pipeline. The Marin-compatible xKV smoke path
supports NIAH on W7900. Local NIAH data must be materialized
under `evaluate/data/ruler/data/llama-3/` for smoke use (see
S8 for directory setup).

**Quick path**: Use the built-in `smoke-xkv` helper from the
container to run xKV NIAH smoke with one command (see
[container.md](container.md#xkv-smoke-canonical-container-workflow)).

**Artifacts**: `$SMOKE_DIR/niah/smoke/` directory with
per-cell retrieval result files.

```bash
python $NIAH_DIR/run_needle_in_haystack.py \
  --model_name $MODEL \
  --provider vllm \
  --context_lengths 1024 \
  --document_depth_percents 0,50,100 \
  --results_dir $SMOKE_DIR/niah/smoke/
```

**Pass criteria**: Script exits 0. Results directory contains
per-cell retrieval files. At least one result file is non-empty.

**Verify artifact**:
```bash
test -d $SMOKE_DIR/niah/smoke/ \
  && [ "$(ls -A $SMOKE_DIR/niah/smoke/)" ] \
  && echo "SMOKE S7: PASS" \
  || echo "SMOKE S7: FAIL — NIAH produced no results"
```

Note: A single context length with 3 depth points is the
minimum that exercises the retrieval loop. This runs in
1-2 minutes vs 1-2 hours for the full NIAH grid.

---

## S8. RULER Smoke

**Validates**: RULER data preparation generates synthetic task
inputs, the vLLM evaluation script processes them, and scored
results are written to disk. Catches tokenizer incompatibilities,
task data generation errors, vLLM eval script integration
failures, and output directory issues.

Requires a running server or offline vLLM access depending on
the RULER version. The Marin-compatible xKV smoke path now
supports the `vt` (variable tracking) task on W7900 in
addition to NIAH-style tasks.

### Local data materialization

RULER and NIAH require pre-generated task data on disk. For
smoke use with the xKV path, materialize data under:

```
evaluate/data/ruler/data/llama-3/
```

This directory must exist before running RULER smoke. The
`prepare_data.py` script writes tokenized task inputs here.
If the directory is missing, the eval script fails silently
or with a confusing file-not-found error.

**Artifacts**:
- `$SMOKE_DIR/ruler/smoke/data/` — generated task inputs
- `$SMOKE_DIR/ruler/smoke/results/` — scored evaluation output

```bash
# Step 0: Ensure local data directory exists
mkdir -p evaluate/data/ruler/data/llama-3/

# Step 1: Generate task data for one task at short length
python $RULER_DIR/scripts/data/prepare_data.py \
  --task vt \
  --model_name $MODEL \
  --tokenizer_name $MODEL \
  --max_seq_length 4096 \
  --num_samples 5 \
  --output_dir $SMOKE_DIR/ruler/smoke/data/

# Step 2: Evaluate
python $RULER_DIR/scripts/eval/evaluate_vllm.py \
  --model_name $MODEL \
  --data_dir $SMOKE_DIR/ruler/smoke/data/ \
  --output_dir $SMOKE_DIR/ruler/smoke/results/ \
  --tensor_parallel_size $TP \
  --max_model_len $MAX_MODEL_LEN
```

**Pass criteria**: Both scripts exit 0. The data directory
contains generated input files. The results directory contains
scored output for `vt`.

**Verify artifact**:
```bash
test -d $SMOKE_DIR/ruler/smoke/data/ \
  && [ "$(ls -A $SMOKE_DIR/ruler/smoke/data/)" ] \
  && test -d $SMOKE_DIR/ruler/smoke/results/ \
  && [ "$(ls -A $SMOKE_DIR/ruler/smoke/results/)" ] \
  && echo "SMOKE S8: PASS" \
  || echo "SMOKE S8: FAIL — RULER data or results missing"
```

Note: The `vt` (variable tracking) task with 5 samples at
4096 tokens is confirmed to pass on W7900 via the xKV smoke
path. This runs in 2-3 minutes vs 2-4 hours for the full
RULER suite.

---

## S9. LongBench Smoke

**Validates**: The LongBench prediction script downloads (or
loads cached) dataset samples, sends them through the vLLM
backend, writes predictions to disk, and the scoring script
produces a results file. Catches dataset download failures,
backend incompatibilities, prediction file format errors, and
evaluation metric computation bugs.

The Marin-compatible xKV smoke path supports LongBench on
W7900. Sample limiting (`--max_samples`) was fixed in the
smoke harness to correctly truncate the dataset before
inference rather than after, preventing full-dataset runs
that waste GPU time during smoke.

**Artifacts**:
- `$SMOKE_DIR/longbench/smoke/predictions/` — model predictions
- `$SMOKE_DIR/longbench/smoke/scores/` — evaluated scores

```bash
# Step 1: Generate predictions on a single short dataset
python $LONGBENCH_DIR/pred.py \
  --model $MODEL \
  --backend vllm \
  --tensor_parallel_size $TP \
  --max_length $MAX_MODEL_LEN \
  --datasets qasper \
  --max_samples 5 \
  --output_dir $SMOKE_DIR/longbench/smoke/predictions/

# Step 2: Score predictions
python $LONGBENCH_DIR/eval.py \
  --pred_dir $SMOKE_DIR/longbench/smoke/predictions/ \
  --output_dir $SMOKE_DIR/longbench/smoke/scores/
```

**Pass criteria**: Both scripts exit 0. Predictions directory
contains a non-empty file for `qasper`. Scores directory
contains a non-empty results file. Verify the prediction
file contains exactly `--max_samples` entries (not the full
dataset).

**Verify artifact**:
```bash
test -d $SMOKE_DIR/longbench/smoke/predictions/ \
  && [ "$(ls -A $SMOKE_DIR/longbench/smoke/predictions/)" ] \
  && test -d $SMOKE_DIR/longbench/smoke/scores/ \
  && [ "$(ls -A $SMOKE_DIR/longbench/smoke/scores/)" ] \
  && echo "SMOKE S9: PASS" \
  || echo "SMOKE S9: FAIL — LongBench predictions or scores missing"
```

Note: The `qasper` dataset is short (single-doc QA, typically
under 8K tokens) and fast to evaluate. 5 samples exercise
the full predict-then-score pipeline without long waits.
This runs in 2-3 minutes vs 2-4 hours for the full LongBench
suite.

---

## S10. InfiniteBench Smoke

**Validates**: The InfiniteBench evaluation script loads
extreme-length task data, sends it through vLLM, and writes
scored results. The passkey task is the simplest (needle
retrieval in a long document) and the most sensitive to cache
corruption. Catches task data loading errors, context length
truncation bugs, vLLM integration failures, and scoring logic
errors.

**Quick path**: Use the built-in `smoke-ib` helper from the
container to run InfiniteBench smoke with one command (see
[container.md](container.md#infinitebench-smoke-canonical-container-workflow)).
This loads the model directly via HuggingFace Transformers
(no vLLM server required) and runs a single passkey sample.

**Important**: InfiniteBench smoke proves harness execution,
not benchmark quality. The passkey score at
`MAX_MODEL_LEN=4096` is meaningless for quality purposes
because the task requires 100K+ token contexts. The smoke
only validates that the eval loop, scoring logic, and artifact
generation run to completion without error.

**Artifacts**: `$SMOKE_DIR/infinitebench/smoke/` directory
with passkey result files.

```bash
python $INFINITEBENCH_DIR/eval_vllm.py \
  --model_name $MODEL \
  --task_name passkey \
  --max_seq_length $MAX_MODEL_LEN \
  --tensor_parallel_size $TP \
  --max_samples 1 \
  --output_dir $SMOKE_DIR/infinitebench/smoke/
```

**Pass criteria**: Script exits 0. Output directory contains
result files with passkey retrieval scores.

**Verify artifact**:
```bash
test -d $SMOKE_DIR/infinitebench/smoke/ \
  && [ "$(ls -A $SMOKE_DIR/infinitebench/smoke/)" ] \
  && echo "SMOKE S10: PASS" \
  || echo "SMOKE S10: FAIL — InfiniteBench produced no results"
```

Note: With `--max_samples 1` and `MAX_MODEL_LEN=4096`, this
exercises the full eval loop without requiring 100K+ token
contexts. For a proper stress test of extreme lengths, run
the full InfiniteBench suite from the runbook.

---

## Cleanup

After the smoke suite completes, kill the server and check
results:

```bash
kill $SERVER_PID 2>/dev/null

echo "=== Smoke Test Summary ==="
for f in \
  $SMOKE_DIR/backend_manifest.json \
  $SMOKE_DIR/bench/smoke_startup.log \
  $SMOKE_DIR/bench/smoke_startup_time.log \
  $SMOKE_DIR/bench/smoke_latency.log \
  $SMOKE_DIR/bench/smoke_throughput.log \
  $SMOKE_DIR/bench/smoke_serving.json \
  $SMOKE_DIR/lm_eval/smoke_standard.json \
  $SMOKE_DIR/guidellm/smoke_sweep.json; do
  if [ -s "$f" ]; then
    echo "  OK  $f"
  else
    echo "  MISSING  $f"
  fi
done

# Check directory-based outputs
for d in \
  $SMOKE_DIR/niah/smoke/ \
  $SMOKE_DIR/ruler/smoke/results/ \
  $SMOKE_DIR/longbench/smoke/scores/ \
  $SMOKE_DIR/infinitebench/smoke/; do
  if [ -d "$d" ] && [ "$(ls -A $d)" ]; then
    echo "  OK  $d"
  else
    echo "  MISSING  $d"
  fi
done
```

The smoke directory can be deleted after review. Smoke results
are not archived to the key-results repo since they have no
evidentiary value.

---

## FUSED Config Smoke

The examples above run with default FP16 KV cache. To smoke-test
the FUSED configuration, repeat S0-S10 with the fused flag added
to each command. The key difference is one extra flag on the
server and offline benchmarks:

```bash
# Server with fused KV
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size $TP \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype $DTYPE \
  --kv-cache-dtype int4_fused \
  --port 8000 &

# Offline benchmarks with fused KV
python $VLLM_DIR/benchmarks/benchmark_latency.py \
  --model $MODEL ... \
  --kv-cache-dtype int4_fused \
  2>&1 | tee $SMOKE_DIR/bench/smoke_latency_fused.log
```

The critical thing the FUSED smoke validates beyond FP16: the
fused quantization kernel loads, the KV cache dtype is reported
correctly in the server log, and no runtime errors occur from
the quantization path.

---

## Relationship to Other Documents

- **This document** (smoke test): validates plumbing in
  15-25 minutes. No publishable numbers.
- **[Quickstart](quickstart.md)**: minimal publishable results
  in 4-6 hours GPU time.
- **[Full runbook](../fused_kv_benchmark_runbook.md)**: paper-grade
  results across all 11 phases in 14-25 hours.
- **[Reproducibility checklist](reproducibility.md)**: pre-
  submission gate for any results you intend to publish.
