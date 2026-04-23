#!/bin/bash
# Phase 1b H100 setup v2 — VLLM_USE_PRECOMPILED=1 path.
#
# Installs stock vLLM 0.19.0 for its compiled CUDA kernels, then
# overlays the asymmetric-kv-plumbing branch's Python side only.
# Takes ~5 min vs the 20-40 min CMake rebuild.
#
# The branch's changes (commits on asymmetric-kv-plumbing) are
# essentially Python-only (config/, v1/, model_executor/), so the
# precompiled kernels link cleanly.
set -e
set -x

: "${HF_TOKEN:?HF_TOKEN required}"

# HF cache on the volume
mkdir -p /runpod-volume/hf_cache/huggingface
ln -sfn /runpod-volume/hf_cache/huggingface /root/.cache/huggingface

# Step 1: stock vLLM wheel (provides compiled .so kernels)
python3.12 -m pip install --quiet vllm==0.19.0 ray setuptools_scm 2>&1 | tail -3
python3.12 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Step 2: clone the asymmetric-kv-plumbing branch
cd /root
if [ ! -d vllm-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-plumbing \
    https://github.com/mcgrof/vllm.git vllm-asym
fi

# Step 3: overlay Python side only (VLLM_USE_PRECOMPILED reuses stock wheel's .so)
python3.12 -m pip uninstall -y vllm
cd vllm-asym
VLLM_USE_PRECOMPILED=1 python3.12 -m pip install --no-build-isolation -e . 2>&1 | tail -5

# Step 4: FlashInfer from asymmetric-kv-dtype branch (source build required —
# the asymmetric plan() signature touches CUDA bindings)
python3.12 -m pip uninstall -y flashinfer-python flashinfer
cd /root
if [ ! -d flashinfer-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-dtype \
    https://github.com/mcgrof/flashinfer.git flashinfer-asym
fi
cd flashinfer-asym
git submodule update --init --recursive
export TORCH_CUDA_ARCH_LIST='9.0+PTX'
python3.12 -m pip install --no-build-isolation --quiet . 2>&1 | tail -5

# Step 5: lm-eval + deps
python3.12 -m pip install --quiet lm-eval==0.4.11 wonderwords nltk datasets scipy 2>&1 | tail -3
python3.12 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# Step 6: verify the branch is live
python3.12 << 'PYEOF'
from vllm.config.cache import cache_dtype_k, cache_dtype_v, is_asymmetric_kv
from vllm.model_executor.layers.attention.attention import Attention
import inspect
src = inspect.getsource(Attention.calc_kv_scales)
assert 'k_quantized' in src or 'k_is_quantized' in src, \
    "calc_kv_scales not asymmetric-aware — branch did not install"
print("OK: asymmetric-kv-plumbing live (calc_kv_scales asym-aware)")

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
sig = inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan)
assert 'k_data_type' in sig.parameters and 'v_data_type' in sig.parameters
print("OK: asymmetric-kv-dtype live (FlashInfer plan() has k/v_data_type)")
PYEOF

# Step 7: download models
for M in \
  'meta-llama/Llama-3.1-8B-Instruct' \
  'mistralai/Mistral-7B-Instruct-v0.3' \
  'microsoft/phi-4' \
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
do
  python3.12 -c "
import os
os.environ['HF_HOME'] = '/runpod-volume/hf_cache/huggingface'
from huggingface_hub import snapshot_download
snapshot_download('$M', allow_patterns=['*.safetensors','*.json','*.py','tokenizer*','*.txt','*.model'])
print('downloaded: $M')
"
done

mkdir -p /workspace/results
echo "Phase 1b v2 setup complete (VLLM_USE_PRECOMPILED + asymmetric-kv-plumbing)"
