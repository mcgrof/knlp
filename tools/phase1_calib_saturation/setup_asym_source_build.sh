#!/bin/bash
# Phase 1b pod setup — install vLLM and FlashInfer from source off the
# asymmetric-kv-plumbing and asymmetric-kv-dtype branches respectively,
# so the full calibration infrastructure is asymmetric-aware (not just
# calc_kv_scales() via a surgical patch).
#
# Use this instead of setup_h100_pod.sh when asym_calib correctness matters.
# Takes 30-50 min because both builds compile CUDA kernels from source.
set -e
set -x

: "${HF_TOKEN:?HF_TOKEN required}"

# HF cache on the runpod volume (root disk too small for 5 models)
mkdir -p /runpod-volume/hf_cache/huggingface
ln -sfn /runpod-volume/hf_cache/huggingface /root/.cache/huggingface

# Step 1: stock vLLM wheel for matching torch+cuda, then uninstall vllm
python3.12 -m pip install --quiet vllm==0.19.0 ray 2>&1 | tail -3
python3.12 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Step 2: vLLM from asymmetric-kv-plumbing branch (source, editable)
python3.12 -m pip uninstall -y vllm
cd /root
if [ ! -d vllm-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-plumbing \
    https://github.com/mcgrof/vllm.git vllm-asym
fi
cd vllm-asym
export TORCH_CUDA_ARCH_LIST='9.0+PTX'
python3.12 -m pip install --no-build-isolation --quiet -e . 2>&1 | tail -5

# Step 3: FlashInfer from asymmetric-kv-dtype branch
python3.12 -m pip uninstall -y flashinfer-python flashinfer
cd /root
if [ ! -d flashinfer-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-dtype \
    https://github.com/mcgrof/flashinfer.git flashinfer-asym
fi
cd flashinfer-asym
git submodule update --init --recursive
python3.12 -m pip install --no-build-isolation --quiet . 2>&1 | tail -5

# Step 4: eval and calibration deps
python3.12 -m pip install --quiet lm-eval==0.4.11 wonderwords nltk datasets scipy 2>&1 | tail -3
python3.12 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# Step 5: Verify the full asymmetric-aware calibration is installed
python3.12 << 'PYEOF'
from vllm.config.cache import cache_dtype_k, cache_dtype_v, is_asymmetric_kv
from vllm.model_executor.layers.attention.attention import Attention
import inspect
src = inspect.getsource(Attention.calc_kv_scales)
assert 'k_quantized' in src or 'k_is_quantized' in src, \
    "calc_kv_scales not asymmetric-aware — source build did not land"
print("OK: vLLM asymmetric-kv-plumbing installed with calibration fix")

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
sig = inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan)
assert 'k_data_type' in sig.parameters and 'v_data_type' in sig.parameters
print("OK: FlashInfer asymmetric-kv-dtype installed")
PYEOF

# Step 6: download the four tolerant models used by Phase 1 saturation
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
echo "Setup complete"
