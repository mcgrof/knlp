#!/bin/bash
# Phase 2 MI300X pod setup — AITER + PTPC-FP8 on ROCm.
#
# Uses the rocm/vllm Docker image; vLLM is already installed
# inside the container.  This script configures ROCm env vars,
# pulls HF cache to the volume, logs in to HF, and downloads the
# four tolerant models used by Phase 1.
#
# Critical env vars:
#   VLLM_ROCM_USE_AITER=1           enable AITER attention backend
#   VLLM_ROCM_USE_AITER_FP4BMM=0    MI300X does not support FP4; MUST be 0
#                                   (default is 1, causes engine crash)
set -e
set -x

: "${HF_TOKEN:?HF_TOKEN required}"

# HF cache to the volume
mkdir -p /runpod-volume/hf_cache/huggingface
ln -sfn /runpod-volume/hf_cache/huggingface /root/.cache/huggingface

# Persist the ROCm env vars so every subprocess picks them up
cat > /etc/profile.d/rocm-vllm.sh <<'EOF'
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FP4BMM=0
EOF
# Also export for the current shell
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_FP4BMM=0

# Verify vLLM is present and the AITER flag is respected
python3 -c "
import os
os.environ.setdefault('VLLM_ROCM_USE_AITER', '1')
os.environ.setdefault('VLLM_ROCM_USE_AITER_FP4BMM', '0')
import vllm
print(f'vLLM: {vllm.__version__ if hasattr(vllm, \"__version__\") else vllm.__file__}')
"

# lm-eval + scipy + datasets
python3 -m pip install --quiet --no-deps lm-eval==0.4.11 wonderwords nltk datasets scipy 2>&1 | tail -2
python3 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)" 2>&1 | tail -3

# Login to HF
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Download the four tolerant models (same as Phase 1)
for M in \
  'meta-llama/Llama-3.1-8B-Instruct' \
  'mistralai/Mistral-7B-Instruct-v0.3' \
  'microsoft/phi-4' \
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
do
  python3 -c "
import os
os.environ['HF_HOME'] = '/runpod-volume/hf_cache/huggingface'
from huggingface_hub import snapshot_download
snapshot_download('$M', allow_patterns=['*.safetensors','*.json','*.py','tokenizer*','*.txt','*.model'])
print('downloaded: $M')
"
done

mkdir -p /workspace/results
echo "MI300X setup complete."
