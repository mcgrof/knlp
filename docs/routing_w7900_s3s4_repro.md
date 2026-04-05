# Routing W7900 S3/S4 reproduction guide

## Overview
S3 (serving smoke) and S4 (TTFT comparison) were run on prune/W7900 on 2026-04-04.
There was no standalone orchestration script; the steps below reproduce the results.

## Prerequisites
- Host: prune (AMD Radeon Pro W7900, 48 GB VRAM, gfx1100)
- Routing env: `/home/mcgrof/envs/w7900-routing`
- Routing repo: `/data/vllm-routing`, branch `routing-on-spf-base` @ `c66bdb0fd`
- Model: `Qwen/Qwen2.5-7B-Instruct` (fp16, cached on prune)
- Cartridge: citation-check/default-50pct at `/tmp/routing-s3-smoke/cartridge_converted.pt`
  with prefix token IDs at `/tmp/routing-s3-smoke/prefix_token_ids.json`

## S3: Serving smoke

### 1. Start vLLM with CartridgeConnector

```bash
export PATH=/opt/rocm/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export VLLM_USE_V1=1

KV_CONFIG='{"kv_connector": "CartridgeConnector", "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector", "kv_connector_extra_config": {"cartridge_path": "/tmp/routing-s3-smoke/cartridge_converted.pt", "prefix_token_ids_path": "/tmp/routing-s3-smoke/prefix_token_ids.json"}, "kv_role": "kv_both"}'

/home/mcgrof/envs/w7900-routing/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --enforce-eager \
    --kv-transfer-config "$KV_CONFIG"
```

### 2. Test control request (non-matching prefix)

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "What is 2+2?", "max_tokens": 16, "temperature": 0}'
```
Expected: HTTP 200, short coherent answer, no cartridge match in server log.

### 3. Test matching request (cartridge prefix)

Send the exact prefix token IDs from the cartridge. The server log should show:
```
Cartridge prefix match: 4128 tokens matched, 4128 new (beyond 0 computed)
Injecting cartridge KV (4128 tokens) into GPU cache
```

## S4: TTFT comparison

With the same server running:

1. Run `ttft_comparison.py` (archived in knlp-key-results under `s4-ttft/`).
2. It sends two streaming `/v1/completions` requests:
   - **Dense**: reversed prefix token IDs (4136 tokens, won't match → full prefill)
   - **Routed**: exact prefix token IDs (4136 tokens, matches → KV injection)
3. Measures wall-clock time to first SSE chunk for each arm.

### Expected result
- Dense TTFT: ~1.6s
- Routed TTFT: ~0.1s
- Speedup: ~15x
- Gate: routed TTFT < dense TTFT

## V1 fix (c66bdb0fd)
S3 initially hit `AttributeError: 'ForwardContext' object has no attribute 'virtual_engine'`.
The fix: `cartridge_connector.py:379` — use `kv_cache_attr` directly instead of indexing by
`forward_context.virtual_engine`. V1 stores KV cache as a direct tensor, not indexed by
virtual_engine like V0.

## Artifacts
Results archived in: `/data/knlp-key-results/routing-w7900-quality-20260404T131800Z/`
