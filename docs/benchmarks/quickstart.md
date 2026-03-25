# Quickstart: Fused KV Quantization Benchmarks

> **Integration status (updated 2026-03-25)**: The
> `--kv-cache-dtype int4_fused` flag is now implemented in the vLLM
> branch `20250325-fused-quantization` on `/data/vllm`. This branch
> adds a real `FusedInt4AttentionBackend` with a Triton fused decode
> kernel that reads packed INT4 KV cache and dequantizes in-register.
> The commands below are now runnable against that branch. Kernel-level
> validation was done on AMD W7900 (ROCm 6.4, Triton 3.5.1) with
> 2.5x-5.4x decode speedup and cosine similarity = 1.0. H100
> validation is the next step.
>
> For the standalone Triton kernel ablations (no vLLM needed), use
> `scripts/spev01/tier5_fused_decode.py`. For the vLLM kernel-level
> smoke benchmark, use `benchmarks/fused_int4_smoke.py` in the vLLM
> branch. See [fused_kv_quantization.md](../fused_kv_quantization.md)
> for the full technical overview.

This page gets you from zero to a minimal FP16-vs-FUSED comparison
in about 30 minutes of active setup time plus benchmark runtime.
If this is your first run on a new machine or after upgrading
dependencies, run the [smoke tests](smoke-test.md) first (10-20
minutes) to validate that every tool works end-to-end before
committing GPU hours here. For the full evaluation protocol, see
the [runbook](../fused_kv_benchmark_runbook.md).

---

## Prerequisites

Run all benchmarks inside the `knlp-rocm-bench` container (see
[container.md](container.md)). The container provides vLLM, lm-eval,
GuideLLM, and FlashAttention with ROCm support. Direct host Python
environments are for debugging only.

If not using the container, you need:

1. **A GPU with enough VRAM** for the target model (e.g. 80 GB for
   a 7B model at FP16 with long-context KV cache).
2. **vLLM** from the `20250325-fused-quantization` branch on
   `/data/vllm` (or a descendant). This branch adds the
   `FusedInt4AttentionBackend` and the `--kv-cache-dtype int4_fused`
   flag. Install in development mode:
   ```bash
   cd /data/vllm
   git checkout 20250325-fused-quantization
   VLLM_USE_PRECOMPILED=1 pip install -e .
   ```
   Pin the commit with `git log --oneline -1`.
3. **lm-eval-harness** (`pip install lm-eval`).
4. **GuideLLM** (`pip install guidellm`) --- optional for the
   quickstart, required for the full runbook.

## Step 1: Lock the Environment

Record exactly what you are running, including the attention
backend. This takes 2 minutes and prevents wasted debugging later.

```bash
RESULTS_DIR=./fused_kv_results_$(date +%Y%m%d)
mkdir -p $RESULTS_DIR/{bench,lm_eval,niah}

# Pin vLLM commit
cd /data/vllm
git log --oneline -1 > $RESULTS_DIR/vllm_commit.txt

# Collect environment
python -c "import vllm; vllm.utils.collect_env()" \
  > $RESULTS_DIR/collect_env.txt 2>&1
python -c "import torch; print(torch.__version__)" \
  >> $RESULTS_DIR/collect_env.txt
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader \
  >> $RESULTS_DIR/collect_env.txt
```

Generate the attention backend manifest. This records which
attention kernel the framework will actually dispatch, not just
what you requested. See the
[runbook Section 0](../fused_kv_benchmark_runbook.md) for the
full manifest generation script. At minimum:

```bash
python3 -c "
import json, sys, torch
m = {}
m['torch_version'] = torch.__version__
m['cuda_version'] = torch.version.cuda or 'N/A'
try:
    import flash_attn; m['flash_attn_version'] = flash_attn.__version__
except ImportError:
    m['flash_attn_version'] = 'not installed'
import vllm; m['vllm_version'] = vllm.__version__
if torch.cuda.is_available():
    m['gpu_name'] = torch.cuda.get_device_name(0)
    m['gpu_count'] = torch.cuda.device_count()
json.dump(m, sys.stdout, indent=2); print()
" > $RESULTS_DIR/backend_manifest.json
```

After the vLLM server starts (Step 3d), capture the runtime
attention backend from the server log and append it to the
manifest directory:

```bash
grep -i "attention backend\|Using.*attention" \
  $RESULTS_DIR/niah/server_fp16.log \
  > $RESULTS_DIR/backend_manifest_runtime.txt
```

## Step 2: Write a Shared Config

Create a file that both runs source so the only difference is the
KV quantization flag:

```bash
cat > $RESULTS_DIR/common.env <<'EOF'
export MODEL=Qwen/Qwen2.5-7B
export TP=1
export MAX_MODEL_LEN=32768
export GPU_MEMORY_UTILIZATION=0.90
export DTYPE=auto
export SEED=42
EOF

source $RESULTS_DIR/common.env
```

Document the exact flag difference:

```bash
echo "FUSED adds: --kv-cache-dtype int4_fused" \
  > $RESULTS_DIR/config_diff.txt
```

The `--kv-cache-dtype int4_fused` flag is the correct flag for
the `20250325-fused-quantization` vLLM branch. Stock vLLM
(main) does not support this flag yet.

## Step 3: Run the Minimal Benchmark Set

The quickstart runs four benchmarks from the runbook's "Minimal
Respectable Subset" (Section 6). Each benchmark runs twice: once
for FP16, once for FUSED.

### 3a. lm-eval (Accuracy Gate)

This is the most important check. If accuracy degrades beyond
3%, stop and debug before investing in latency benchmarks.

```bash
# FP16 baseline
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.90,add_bos_token=True,max_model_len=$MAX_MODEL_LEN \
  --tasks mmlu,hellaswag,arc_challenge \
  --batch_size auto \
  --num_fewshot 5 \
  --output_path $RESULTS_DIR/lm_eval/fp16_standard.json \
  --log_samples

# FUSED
lm_eval --model vllm \
  --model_args pretrained=$MODEL,tensor_parallel_size=$TP,dtype=auto,gpu_memory_utilization=0.90,add_bos_token=True,max_model_len=$MAX_MODEL_LEN,kv_cache_dtype=int4_fused \
  --tasks mmlu,hellaswag,arc_challenge \
  --batch_size auto \
  --num_fewshot 5 \
  --output_path $RESULTS_DIR/lm_eval/fused_standard.json \
  --log_samples
```

**Pass criteria**: FUSED accuracy within 1% of FP16 on all three
tasks. Within 3% is acceptable with justification.

### 3b. Latency (Decode Path Check)

```bash
for TAG in fp16 fused; do
  if [ "$TAG" = "fused" ]; then EXTRA="--kv-cache-dtype int4_fused"; else EXTRA=""; fi
  for BS in 1 8; do
    for IL in 512 8192; do
      python /data/vllm/benchmarks/benchmark_latency.py \
        --model $MODEL \
        --tensor-parallel-size $TP \
        --input-len $IL \
        --output-len 128 \
        --batch-size $BS \
        --num-iters-warmup 3 \
        --num-iters 10 \
        $EXTRA \
        2>&1 | tee $RESULTS_DIR/bench/${TAG}_latency_b${BS}_il${IL}.log
    done
  done
done
```

**Pass criteria**: FUSED decode latency at least 5% lower than
FP16 at one or more operating points. Improvements below 5% should
not be claimed as meaningful.

### 3c. Throughput (Peak Capacity)

```bash
for TAG in fp16 fused; do
  if [ "$TAG" = "fused" ]; then EXTRA="--kv-cache-dtype int4_fused"; else EXTRA=""; fi
  python /data/vllm/benchmarks/benchmark_throughput.py \
    --model $MODEL \
    --tensor-parallel-size $TP \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 1000 \
    $EXTRA \
    2>&1 | tee $RESULTS_DIR/bench/${TAG}_throughput.log
done
```

### 3d. NIAH (Long-Context Retrieval)

Start the vLLM server, then run NIAH against it.

```bash
# Start server (FP16 first, then repeat for FUSED)
python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size $TP \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype $DTYPE \
  --port 8000 \
  2>&1 | tee $RESULTS_DIR/niah/server_fp16.log &

until curl -s http://localhost:8000/health | grep -q ok; do sleep 2; done

python <NIAH_DIR>/run_needle_in_haystack.py \
  --model_name $MODEL \
  --provider vllm \
  --context_lengths 4096,16384,32768 \
  --document_depth_percents 0,25,50,75,100 \
  --results_dir $RESULTS_DIR/niah/fp16/

# Kill server, repeat with FUSED flag
```

**Pass criteria**: 100% retrieval at all (depth, length) pairs.
Any cell below 95% warrants investigation.

## Step 4: Review Results

Compare FP16 and FUSED numbers side by side. The key table to
produce:

| Metric | FP16 | FUSED | Delta |
|--------|------|-------|-------|
| MMLU (5-shot) | | | |
| HellaSwag | | | |
| ARC-Challenge | | | |
| Latency b=1 il=512 (ms/tok) | | | |
| Latency b=8 il=8192 (ms/tok) | | | |
| Throughput (tok/s) | | | |
| NIAH worst-cell retrieval (%) | | | |

If all deltas are within thresholds, proceed to the full runbook
for the remaining benchmarks (GuideLLM serving sweeps, RULER,
LongBench, InfiniteBench).

## Step 5: Archive Results

```bash
cp -a $RESULTS_DIR/ /data/knlp-key-results/fused_kv_bench/
cd /data/knlp-key-results && git add fused_kv_bench/ && git commit \
  -m "bench: quickstart results $(date +%Y%m%d)"
```

## Step 0 (Optional): Kernel-Level Smoke Test

Before running the full benchmark protocol, you can verify the fused
INT4 kernels work on your GPU with a self-contained smoke test that
does not require a model download:

```bash
cd /data/vllm
python benchmarks/fused_int4_smoke.py 2>smoke_stderr.log
```

This runs both the FP16 SDPA baseline and the fused INT4 decode
kernel with synthetic tensors and emits a JSON manifest to stdout.
Check for:

- All `cosine_similarity` values should be > 0.99
- `backend_manifest.fused_selected` should be `FUSED_INT4_TRITON`
- `backend_manifest.fallback` should be `none`

The stderr output shows per-point latencies with WIN/LOSE tags.
Expected results on W7900: 2.5x-5.4x speedup at B >= 2.

You can also run the unit tests:

```bash
python -m pytest tests/kernels/attention/test_fused_int4.py -xvs
```

### Validated results (2026-03-25)

Kernel-level smoke benchmark on AMD Radeon Pro W7900 (ROCm 6.4,
Triton 3.5.1), Qwen2.5-7B config (heads=28, kv_heads=4, dim=128):

| B | T | Baseline (ms) | Fused (ms) | Speedup |
|---|---|---------------|------------|---------|
| 1 | 2048 | 3.073 | 1.157 | 2.66x |
| 1 | 4096 | 5.998 | 2.239 | 2.68x |
| 2 | 2048 | 4.747 | 1.862 | 2.55x |
| 2 | 4096 | 9.645 | 3.395 | 2.84x |
| 4 | 2048 | 8.483 | 2.872 | 2.95x |
| 4 | 4096 | 17.775 | 5.209 | 3.41x |
| 8 | 2048 | 16.363 | 3.572 | 4.58x |
| 8 | 4096 | 35.119 | 6.518 | 5.39x |

Cosine similarity = 1.000000 across all points.

## What Next

- **Smoke tests**: If anything failed unexpectedly, run the
  [smoke test suite](smoke-test.md) to isolate the broken tool.
- **Full evaluation**: Follow the
  [runbook](../fused_kv_benchmark_runbook.md) from Section 5
  (Recommended Execution Order). The quickstart covers phases 2-4
  of 11 total phases.
- **Reproducibility**: Before publishing or submitting, walk
  through the [reproducibility checklist](reproducibility.md).
- **Calibration**: If accuracy drops >3% on any task, run the
  [ratio classifier](../kv_plugin/calibration_guide.md) to check
  whether the model needs asymmetric key/value precision.
