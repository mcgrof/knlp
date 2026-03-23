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
| lm-eval | pip install in Dockerfile | `lm_eval` CLI |
| GuideLLM | pip install in Dockerfile | `guidellm` CLI |
| PyTorch + ROCm 7.0 | base image | `import torch` |
| FlashAttention (ROCm) | base image | `import flash_attn` |

Long-context harnesses (xKV, InfiniteBench, RULER, LongBench, NIAH)
are bind-mounted from the host at `/data` rather than vendored into
the image. This avoids bloating the image with multi-GB repos and
datasets while keeping the repos updatable independently.

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

### What the launch script mounts

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `/dev/kfd`, `/dev/dri` | device passthrough | GPU access |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | Model cache |
| `/data` | `/data` | xKV, InfiniteBench, knlp repos |
| `$BENCH_RESULTS_DIR` | `/results` | Benchmark output |

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

### Long-context harnesses (xKV, InfiniteBench)

These repos live on the host at `/data/xKV` and `/data/InfiniteBench`
and are accessible inside the container via the `/data` mount:

```bash
./container/run-bench.sh python /data/xKV/evaluate/eval_acc.py --help
```

Install any repo-specific Python deps at container start if needed:

```bash
./container/run-bench.sh bash -c '
  pip install -r /data/xKV/requirements.txt
  python /data/xKV/evaluate/eval_acc.py ...
'
```

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
