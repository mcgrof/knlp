#!/bin/bash
# A100 80GB (sm_80) pod setup — same stack as H100/H200 but with
# sm_80 in TORCH_CUDA_ARCH_LIST for FlashInfer.
#
# Stock vllm==0.19.0 wheel ships kernels for sm_70/75/80/86/89/90, so
# VLLM_USE_PRECOMPILED=1 for the asymmetric-kv-plumbing overlay
# works on A100 unchanged.  FlashInfer is source-built and MUST target
# sm_80 (not sm_90).
set -e
set -x

: "${HF_TOKEN:?HF_TOKEN required}"

mkdir -p /runpod-volume/hf_cache/huggingface
ln -sfn /runpod-volume/hf_cache/huggingface /root/.cache/huggingface
export HF_HOME=/runpod-volume/hf_cache/huggingface
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Step 1: stock vLLM wheel (has sm_80 kernels)
python3.12 -m pip install --quiet vllm==0.19.0 ray setuptools_scm 2>&1 | tail -3
python3.12 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Step 2: overlay asymmetric-kv-plumbing branch (Python only)
cd /root
if [ ! -d vllm-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-plumbing \
    https://github.com/mcgrof/vllm.git vllm-asym
fi
python3.12 -m pip uninstall -y vllm
cd vllm-asym
VLLM_USE_PRECOMPILED=1 python3.12 -m pip install --no-build-isolation -e . 2>&1 | tail -5

# Step 3: FlashInfer asymmetric-kv-dtype for sm_80 (A100)
python3.12 -m pip uninstall -y flashinfer-python flashinfer
cd /root
if [ ! -d flashinfer-asym ]; then
  git clone --depth 50 --branch asymmetric-kv-dtype \
    https://github.com/mcgrof/flashinfer.git flashinfer-asym
fi
cd flashinfer-asym
git submodule update --init --recursive
# A100 is sm_80 — critical difference vs H100/H200 (both sm_90)
export TORCH_CUDA_ARCH_LIST='8.0+PTX'
python3.12 -m pip install --no-build-isolation --quiet . 2>&1 | tail -5

# Step 4: lm-eval + deps
python3.12 -m pip install --quiet lm-eval==0.4.11 wonderwords nltk datasets scipy 2>&1 | tail -3
python3.12 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# Step 5: env file
cat > /root/.asym_env <<EOF
export HF_HOME=/runpod-volume/hf_cache/huggingface
export HF_TOKEN=$HF_TOKEN
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
EOF

# Step 6: verify
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
echo "A100 setup complete"
