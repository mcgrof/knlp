#!/bin/bash
# Q27 saturation pod setup — same as q27_setup.sh from the
# qwen-fragility experiment, plus the HybridAttentionMambaModelConfig
# tuple-API patch that was applied on the eval pod and is now
# documented in qwen36-27b-asym-20260425/README.md.
set -e
set -x
source /root/.q27_env

# Step 1: stock vLLM
python3.12 -m pip install --quiet vllm==0.19.0 ray setuptools_scm 2>&1 | tail -3
python3.12 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Step 2: asymmetric overlay
cd /root
[ -d vllm-asym ] || git clone --depth 50 --branch asymmetric-kv-plumbing \
  https://github.com/mcgrof/vllm.git vllm-asym
python3.12 -m pip uninstall -y vllm
cd vllm-asym
VLLM_USE_PRECOMPILED=1 python3.12 -m pip install --no-build-isolation -e . 2>&1 | tail -5

# Step 2b: hybrid-arch tuple-API patch (required for Qwen3.6-27B
# Qwen3_5ForConditionalGeneration; see qwen36-27b-asym-20260425/README.md)
python3.12 << 'PYEOF'
p = "/root/vllm-asym/vllm/model_executor/models/config.py"
src = open(p).read()
old = '        if cache_config.cache_dtype == "auto":\n            kv_cache_dtype = model_config.dtype\n        else:\n            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]'
new = '''        # Asymmetric (K_dtype, V_dtype) tuple from asymmetric-kv-plumbing
        # branch: linear-attention layers in hybrid models don't carry a
        # true KV cache the same way; fall back to the K dtype (typically
        # FP16) for these layers. The full-attention layers retain the
        # asymmetric treatment via the standard attention path.
        ct = cache_config.cache_dtype
        if isinstance(ct, tuple):
            ct = ct[0]
        if ct == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[ct]'''
if old in src:
    open(p, "w").write(src.replace(old, new))
    print("patched HybridAttentionMambaModelConfig")
elif "isinstance(ct, tuple)" in src:
    print("patch already applied")
else:
    print("ERROR: expected old block not found")
    raise SystemExit(1)
PYEOF

# Step 3: FlashInfer asym
python3.12 -m pip uninstall -y flashinfer-python flashinfer
cd /root
[ -d flashinfer-asym ] || git clone --depth 50 --branch asymmetric-kv-dtype \
  https://github.com/mcgrof/flashinfer.git flashinfer-asym
cd flashinfer-asym
git submodule update --init --recursive
export TORCH_CUDA_ARCH_LIST='9.0+PTX'
python3.12 -m pip install --no-build-isolation --quiet . 2>&1 | tail -5

# Step 4: download Qwen3.6-27B
python3.12 -c "
import os
os.environ['HF_HOME']='/runpod-volume/hf_cache/huggingface'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.6-27B',
  allow_patterns=['*.safetensors','*.json','*.py','tokenizer*','*.txt','*.model'])
print('downloaded Qwen3.6-27B')
"

mkdir -p /workspace/results
echo "Q27 sat setup complete"
