#!/bin/bash
# H200 pod setup — same VLLM_USE_PRECOMPILED=1 overlay path as H100.
#
# Installs stock vLLM 0.19.0 for its compiled CUDA kernels, overlays
# the asymmetric-kv-plumbing Python side, then source-builds
# FlashInfer asymmetric-kv-dtype branch.
#
# H200 is Hopper (sm_90) like H100, so the H100 precompiled kernels
# work unchanged.
set -e
set -x

: "${HF_TOKEN:?HF_TOKEN required}"

# HF cache to the volume (plenty of room for 72B weights)
mkdir -p /runpod-volume/hf_cache/huggingface
ln -sfn /runpod-volume/hf_cache/huggingface /root/.cache/huggingface
export HF_HOME=/runpod-volume/hf_cache/huggingface
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Step 1: stock vLLM wheel (provides compiled .so kernels)
python3.12 -m pip install --quiet vllm==0.19.0 ray setuptools_scm 2>&1 | tail -3
python3.12 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Step 2: clone + overlay asymmetric-kv-plumbing branch
cd /root
if [ ! -d vllm-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-plumbing \
    https://github.com/mcgrof/vllm.git vllm-asym
fi
python3.12 -m pip uninstall -y vllm
cd vllm-asym
VLLM_USE_PRECOMPILED=1 python3.12 -m pip install --no-build-isolation -e . 2>&1 | tail -5

# Step 3: FlashInfer asymmetric-kv-dtype branch (source build required)
python3.12 -m pip uninstall -y flashinfer-python flashinfer
cd /root
if [ ! -d flashinfer-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-dtype \
    https://github.com/mcgrof/flashinfer.git flashinfer-asym
fi
cd flashinfer-asym
git submodule update --init --recursive
# H200 is sm_90 like H100
export TORCH_CUDA_ARCH_LIST='9.0+PTX'
python3.12 -m pip install --no-build-isolation --quiet . 2>&1 | tail -5

# Step 4: lm-eval + deps
python3.12 -m pip install --quiet lm-eval==0.4.11 wonderwords nltk datasets scipy 2>&1 | tail -3
python3.12 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# Step 5: persist env vars for all subprocesses
cat > /root/.asym_env <<EOF
export HF_HOME=/runpod-volume/hf_cache/huggingface
export HF_TOKEN=$HF_TOKEN
export FLASHINFER_DISABLE_VERSION_CHECK=1
EOF

# Step 6: verify install
python3.12 << 'PYEOF'
from vllm.config.cache import cache_dtype_k, cache_dtype_v, is_asymmetric_kv
from vllm.model_executor.layers.attention.attention import Attention
import inspect
src = inspect.getsource(Attention.calc_kv_scales)
assert 'k_quantized' in src or 'k_is_quantized' in src
print("OK: asymmetric-kv-plumbing live")

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
sig = inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan)
assert 'k_data_type' in sig.parameters and 'v_data_type' in sig.parameters
print("OK: asymmetric-kv-dtype live")
PYEOF

mkdir -p /workspace/results
echo "H200 setup complete"
