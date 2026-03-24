# Unified ROCm Benchmark Container

The canonical benchmark lane for prune/W7900 runs from a single
container image (`knlp-rocm-bench`) derived from `rocm/vllm:latest`.
This replaces the previous split-host-env approach where vLLM, lm-eval,
GuideLLM, and long-context harnesses each required separate Python
environments.

## What the container provides

| Tool | Source | Path inside container |
|------|--------|-----------------------|
| vLLM (ROCm) | base image | `python -m vllm` |
| vLLM benchmarks | base image | `$VLLM_DIR/benchmarks/` |
| lm-eval | Dockerfile pip | `lm_eval` CLI |
| GuideLLM | Dockerfile pip | `guidellm` CLI |
| PyTorch + ROCm 7.0 | base image | `import torch` |
| FlashAttention (ROCm) | base image | `import flash_attn` |
| xKV eval deps | Dockerfile pip | termcolor, pandas, jieba, fuzzywuzzy, rouge, etc. |
| InfiniteBench eval deps | Dockerfile pip | openai, xopen, python-dotenv |
| smoke-xkv helper | COPY in Dockerfile | `/usr/local/bin/smoke-xkv` |
| smoke-ib helper | COPY in Dockerfile | `/usr/local/bin/smoke-ib` |

### What is baked in vs. bind-mounted

**Baked into the image** — Python libraries that xKV and InfiniteBench
import at eval time.  These are lightweight pip packages (termcolor,
pandas, jieba, fuzzywuzzy, rouge, rouge_score, wonderwords, tabulate,
tiktoken, einops, omegaconf, loguru, tenacity, protobuf, nltk, openai,
xopen, python-dotenv).  Baking them in eliminates the need to run
`pip install` at container start.

The Dockerfile deliberately excludes `nemo_toolkit[all]`,
`pytorch-lightning`, `hydra-core`, `fastchat`, and other heavy
packages from xKV's `requirements.txt`.  Those are only needed for
xKV's training and patching paths, not the evaluation smoke paths
this container targets.

**Bind-mounted at runtime** — The xKV and InfiniteBench repositories
themselves (`/data/xKV`, `/data/InfiniteBench`), the HuggingFace
model cache, and the results directory.  These are bind-mounted
because:

| Host path | Container path | Why bind-mounted |
|-----------|---------------|------------------|
| `/data/xKV` | `/data/xKV` | Contains model code, eval scripts, and pre-tokenized RULER/NIAH data that change independently of the container |
| `/data/InfiniteBench` | `/data/InfiniteBench` | Contains eval scripts and task templates that change independently |
| `/data/knlp` | `/data/knlp` | Main code repo; avoids rebuilding container for code changes |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | 100+ GB model cache; vendoring would make the image unusable |
| `$BENCH_RESULTS_DIR` | `/results` | Output must persist after container exits |
| `/dev/kfd`, `/dev/dri` | device passthrough | GPU access (ROCm) |

## Build the image

```bash
cd /data/knlp
docker build -t knlp-rocm-bench:latest -f container/Dockerfile .
```

Rebuild after upgrading `rocm/vllm:latest` or changing benchmark
dependency versions.

## Run the container

The `container/run-bench.sh` script handles device passthrough,
HuggingFace cache mounts, results mounts, and environment variables:

```bash
# Interactive shell
./container/run-bench.sh

# Run a specific command
./container/run-bench.sh python3 -c "import vllm; print(vllm.__version__)"
```

### Environment variables set by the script

| Variable | Value | Purpose |
|----------|-------|---------|
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | `1` | Enable Flash/MemEfficient SDPA on AMD |
| `HIP_VISIBLE_DEVICES` | `0` | Select GPU (override with env var) |
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace cache path |
| `VLLM_DIR` | `/app/vllm` | vLLM source tree (benchmarks live here) |

### Override defaults

```bash
# Use a different GPU
HIP_VISIBLE_DEVICES=1 ./container/run-bench.sh

# Write results to a specific directory
BENCH_RESULTS_DIR=/data/knlp-key-results/fused_kv_bench \
  ./container/run-bench.sh

# Use a different image tag
KNLP_BENCH_IMAGE=knlp-rocm-bench:v2 ./container/run-bench.sh
```

## Run smoke tests inside the container

### S-1. Backend manifest

```bash
./container/run-bench.sh python3 -c "
import json, os, sys, torch, vllm
m = {
    'torch_version': torch.__version__,
    'hip_version': getattr(torch.version, 'hip', 'N/A'),
    'vllm_version': vllm.__version__,
    'aotriton_experimental': os.environ.get(
        'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'unset'),
}
try:
    import flash_attn
    m['flash_attn_version'] = flash_attn.__version__
except ImportError:
    m['flash_attn_version'] = 'not installed'
if torch.cuda.is_available():
    m['gpu_name'] = torch.cuda.get_device_name(0)
    m['gpu_count'] = torch.cuda.device_count()
    m['gpu_memory_gb'] = round(
        torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
json.dump(m, sys.stdout, indent=2); print()
"
```

### S5. lm-eval smoke

```bash
./container/run-bench.sh lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen2.5-0.5B,dtype=auto,gpu_memory_utilization=0.90,add_bos_token=True,max_model_len=4096 \
  --tasks arc_easy \
  --batch_size auto \
  --num_fewshot 0 \
  --limit 20 \
  --output_path /results/lm_eval_smoke.json \
  --log_samples
```

### S2. vLLM latency smoke

```bash
./container/run-bench.sh python \
  /app/vllm/benchmarks/benchmark_latency.py \
  --model Qwen/Qwen2.5-0.5B \
  --input-len 128 --output-len 16 \
  --batch-size 1 \
  --num-iters-warmup 1 --num-iters 3
```

### vLLM serve (for online benchmarks S4, S6, S7-S10)

Start the server inside the container, then run online benchmarks
in the same container session:

```bash
./container/run-bench.sh bash -c '
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --port 8000 &
  for i in $(seq 1 60); do
    curl -s http://localhost:8000/health | grep -q ok && break
    sleep 2
  done
  echo "Server ready"
  # Run online benchmarks here, e.g.:
  python /app/vllm/benchmarks/benchmark_serving.py \
    --backend vllm --model Qwen/Qwen2.5-0.5B \
    --endpoint /v1/completions --host localhost --port 8000 \
    --dataset-name random --random-input-len 128 --random-output-len 16 \
    --request-rate 2 --num-prompts 10
'
```

### xKV smoke (canonical container workflow)

Run xKV long-context smoke directly via the built-in helper
script.  No `pip install` needed — all eval dependencies are
baked into the image.

```bash
# One-command smoke (defaults to Qwen/Qwen2.5-0.5B)
./container/run-bench.sh smoke-xkv

# Override model
./container/run-bench.sh smoke-xkv --model marin-community/marin-8b-base
```

Or run manually for more control:

```bash
./container/run-bench.sh bash -c '
  export PYTHONPATH=/data/xKV:$PYTHONPATH
  python /data/xKV/evaluate/eval_acc.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --datalen 4096 \
    --dataset_name "ruler/niah_single_1" \
    --num_samples 5 \
    --result_dir /results/smoke_xkv/
'
```

### InfiniteBench smoke (canonical container workflow)

Run the minimal HuggingFace InfiniteBench passkey smoke.  Loads
the model directly via Transformers — no vLLM server needed.

```bash
# One-command smoke (defaults to Qwen/Qwen2.5-0.5B)
./container/run-bench.sh smoke-ib

# Override model
./container/run-bench.sh smoke-ib --model marin-community/marin-8b-base
```

Or run manually:

```bash
./container/run-bench.sh python /data/InfiniteBench/src/eval_hf_smoke.py \
  --model Qwen/Qwen2.5-0.5B \
  --task passkey \
  --limit 1 \
  --output /results/smoke_infinitebench/passkey_smoke.json
```

InfiniteBench smoke proves harness execution, not benchmark
quality.  The passkey score at short context lengths is
meaningless — the task requires 100K+ tokens for a real signal.

## Split host envs (debug-only)

Direct host installs of vLLM, lm-eval, and other benchmark tools
are no longer the canonical benchmark lane. Use them only for:

- Debugging container build failures
- Testing unreleased vLLM branches before container rebuild
- Quick one-off experiments that do not produce archival results

Any results intended for publication or archival to
`/data/knlp-key-results/` must come from the container path.

## Rebuilding after upstream updates

When `rocm/vllm:latest` is updated:

```bash
docker pull rocm/vllm:latest
docker build --no-cache -t knlp-rocm-bench:latest \
  -f container/Dockerfile .
```

Verify with the backend manifest smoke (S-1 above) before running
benchmarks.
