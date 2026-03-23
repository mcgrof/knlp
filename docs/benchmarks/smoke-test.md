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

## What Smoke Tests Validate

Smoke tests confirm these properties for each benchmark tool:

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

Same as the [quickstart prerequisites](quickstart.md#prerequisites),
plus a small model that loads fast. These examples use
`Qwen/Qwen2.5-7B` (fits on a single 80 GB GPU), but substitute
any model your pipeline targets.

```bash
# Shared environment for smoke tests
export MODEL=Qwen/Qwen2.5-7B
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
| S0 | Server startup | 1-2 min | Model loading, VRAM, flag parsing |
| S1 | vLLM latency smoke | 1 min | Decode path, batch dispatch |
| S2 | vLLM throughput smoke | 1 min | Engine saturation path, prompt handling |
| S3 | lm-eval smoke | 2-5 min | Accuracy harness, vLLM backend integration |
| S4 | GuideLLM serving smoke | 2-3 min | Server endpoint, open-loop client, JSON output |
| S5 | vLLM serving smoke | 1-2 min | Poisson-arrival client, result serialization |
| S6 | NIAH smoke | 2-3 min | Retrieval harness, server interaction, heatmap data |
| S7 | RULER smoke (placeholder) | TBD | Synthetic task generation, eval loop |
| S8 | LongBench smoke (placeholder) | TBD | Prediction + scoring pipeline |
| S9 | InfiniteBench smoke (placeholder) | TBD | Extreme-length eval loop |

**Total estimated time**: 10-20 minutes for phases S0-S6
(concrete commands below). Phases S7-S9 are placeholders pending
framework integration.

Rationale: S0 catches fatal issues (wrong model path, OOM,
bad flags). S1-S2 validate the offline engine without a server.
S3 validates accuracy tooling. S4-S6 require a running server
and validate online paths. S7-S9 validate long-context harnesses.

---

## S0. Server Startup Smoke

Verify the model loads and the health endpoint responds.

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

**Pass criteria**: Server responds to `/health` within 120 seconds.
Log file is non-empty and contains no Python tracebacks.

Leave the server running for S4-S6.

---

## S1. vLLM Latency Smoke

Minimal latency benchmark: single batch size, short sequence,
few iterations.

```bash
python <VLLM_DIR>/benchmarks/benchmark_latency.py \
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
  && echo "SMOKE S1: PASS" \
  || echo "SMOKE S1: FAIL — missing or empty output"
```

---

## S2. vLLM Throughput Smoke

Minimal throughput benchmark: few prompts, short sequences.

```bash
python <VLLM_DIR>/benchmarks/benchmark_throughput.py \
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
  && echo "SMOKE S2: PASS" \
  || echo "SMOKE S2: FAIL — missing or empty output"
```

---

## S3. lm-eval Smoke

Run a single lightweight task with minimal samples. The `arc_easy`
task is fast and catches vLLM backend integration issues.

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
print('SMOKE S3: PASS — lm-eval produced results')
" || echo "SMOKE S3: FAIL"
```

Note: `--limit 20` restricts to 20 samples per task so it
finishes in under a minute. This is far too few for meaningful
accuracy numbers but sufficient to verify the pipeline.

---

## S4. GuideLLM Serving Smoke

Requires the server from S0 to be running.

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
  && echo "SMOKE S4: PASS" \
  || echo "SMOKE S4: FAIL — GuideLLM produced no output"
```

Note: `--max-seconds 30` caps the sweep to 30 seconds total.
This produces only 1-2 rate points, enough to validate
plumbing.

---

## S5. vLLM Serving Smoke

Requires the server from S0 to be running. Uses synthetic data
to avoid needing the ShareGPT dataset.

```bash
python <VLLM_DIR>/benchmarks/benchmark_serving.py \
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
  && echo "SMOKE S5: PASS" \
  || echo "SMOKE S5: FAIL — serving benchmark produced no output"
```

Note: `--dataset-name random` avoids needing to download
ShareGPT. 10 prompts at rate=2 finishes in about 5 seconds of
request generation plus decode time.

---

## S6. NIAH Smoke

Requires the server from S0 to be running. Uses the shortest
context length and fewest depth intervals that exercise the
retrieval pipeline.

```bash
python <NIAH_DIR>/run_needle_in_haystack.py \
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
  && echo "SMOKE S6: PASS" \
  || echo "SMOKE S6: FAIL — NIAH produced no results"
```

Note: A single context length with 3 depth points is the
minimum that exercises the retrieval loop. This runs in
1-2 minutes vs 1-2 hours for the full NIAH grid.

---

## S7. RULER Smoke (Placeholder)

RULER integration depends on the framework setup for your
environment. The smoke test should:

1. Generate task data for a single task (`niah_single`) at the
   shortest supported sequence length.
2. Run evaluation on that single task.
3. Verify the output directory contains scored results.

```bash
# Template — adapt paths to your RULER installation
python <RULER_DIR>/scripts/data/prepare_data.py \
  --task niah_single \
  --model_name $MODEL \
  --tokenizer_name $MODEL \
  --max_seq_length 4096 \
  --num_samples 5 \
  --output_dir $SMOKE_DIR/ruler/smoke/data/

python <RULER_DIR>/scripts/eval/evaluate_vllm.py \
  --model_name $MODEL \
  --data_dir $SMOKE_DIR/ruler/smoke/data/ \
  --output_dir $SMOKE_DIR/ruler/smoke/results/ \
  --tensor_parallel_size $TP \
  --max_model_len $MAX_MODEL_LEN
```

**Pass criteria**: Both scripts exit 0. Results directory
contains scored output for the single task.

---

## S8. LongBench Smoke (Placeholder)

The smoke test should run prediction and evaluation on a single
short dataset subset.

```bash
# Template — adapt paths to your LongBench installation
python <LONGBENCH_DIR>/pred.py \
  --model $MODEL \
  --backend vllm \
  --tensor_parallel_size $TP \
  --max_length $MAX_MODEL_LEN \
  --datasets qasper \
  --max_samples 5 \
  --output_dir $SMOKE_DIR/longbench/smoke/predictions/

python <LONGBENCH_DIR>/eval.py \
  --pred_dir $SMOKE_DIR/longbench/smoke/predictions/ \
  --output_dir $SMOKE_DIR/longbench/smoke/scores/
```

**Pass criteria**: Both scripts exit 0. Scores directory
contains a non-empty results file.

---

## S9. InfiniteBench Smoke (Placeholder)

The smoke test should run the simplest task (passkey) at the
shortest feasible length.

```bash
# Template — adapt paths to your InfiniteBench installation
python <INFINITEBENCH_DIR>/eval_vllm.py \
  --model_name $MODEL \
  --task_name passkey \
  --max_seq_length $MAX_MODEL_LEN \
  --tensor_parallel_size $TP \
  --max_samples 3 \
  --output_dir $SMOKE_DIR/infinitebench/smoke/
```

**Pass criteria**: Script exits 0. Output directory contains
result files with passkey retrieval scores.

---

## Cleanup

After the smoke suite completes, kill the server and check
results:

```bash
kill $SERVER_PID 2>/dev/null

echo "=== Smoke Test Summary ==="
for f in \
  $SMOKE_DIR/bench/smoke_startup.log \
  $SMOKE_DIR/bench/smoke_latency.log \
  $SMOKE_DIR/bench/smoke_throughput.log \
  $SMOKE_DIR/lm_eval/smoke_standard.json \
  $SMOKE_DIR/guidellm/smoke_sweep.json \
  $SMOKE_DIR/bench/smoke_serving.json; do
  if [ -s "$f" ]; then
    echo "  OK  $f"
  else
    echo "  MISSING  $f"
  fi
done

# Check NIAH directory
if [ -d "$SMOKE_DIR/niah/smoke/" ] && [ "$(ls -A $SMOKE_DIR/niah/smoke/)" ]; then
  echo "  OK  $SMOKE_DIR/niah/smoke/"
else
  echo "  MISSING  $SMOKE_DIR/niah/smoke/"
fi
```

The smoke directory can be deleted after review. Smoke results
are not archived to the key-results repo since they have no
evidentiary value.

---

## FUSED Config Smoke

The examples above run with default FP16 KV cache. To smoke-test
the FUSED configuration, repeat S0-S6 with the fused flag added
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
python <VLLM_DIR>/benchmarks/benchmark_latency.py \
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
  10-20 minutes. No publishable numbers.
- **[Quickstart](quickstart.md)**: minimal publishable results
  in 4-6 hours GPU time.
- **[Full runbook](../fused_kv_benchmark_runbook.md)**: paper-grade
  results across all 11 phases in 14-25 hours.
- **[Reproducibility checklist](reproducibility.md)**: pre-
  submission gate for any results you intend to publish.
