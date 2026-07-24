#!/bin/bash
# vLLM teacher for CAS self-study synth. flashinfer disabled (a JIT/CUB bug on
# sm_90a fails engine init); CUDA graphs ON (no --enforce-eager) for ~2x
# throughput; 16384 context so a ~7.7K-token LongHealth record + 512 output fits
# (Qwen3-8B native 32768, no YaRN). Honors VLLM_GPU / VLLM_PORT so several can be
# launched one-per-GPU. Launch instances SEQUENTIALLY with spaced ports
# (8005/8105/8205): the EngineCore's torch.distributed rendezvous port is chosen
# racily, so simultaneous launches collide on EADDRINUSE.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER=0
export CUDA_VISIBLE_DEVICES=${VLLM_GPU:-6}
exec /home/mcgrof/kvio/venv/bin/vllm serve Qwen/Qwen3-8B \
  --port ${VLLM_PORT:-8006} --max-model-len 16384 \
  --gpu-memory-utilization 0.85
