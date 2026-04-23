#!/bin/bash
# One-shot H100 pod setup for Phase 1.
# Runs inside the pod.  Assumes /workspace/phase1/ contains:
#   - asym_patches/patch_all_asym.py  (master surgical patcher)
#   - asym_patches/fix_flashinfer_version.py
#   - asym_patches/fix_separator.py
#   - asym_patches/apply_asym_patches.py
# and the Phase 1 Python scripts.
#
set -e
set -x

: "${HF_TOKEN:?HF_TOKEN must be exported before running this script}"

# ---- Step 1: vLLM 0.19 base install ----
python3.12 -m pip install --quiet vllm==0.19.0 ray 2>&1 | tail -3

# ---- Step 2: FlashInfer 0.6.7 (stock, will be patched in Step 4) ----
python3.12 -m pip install --no-deps --quiet flashinfer-python==0.6.7

# ---- Step 3: lm-eval + calibration deps ----
python3.12 -m pip install --quiet lm-eval==0.4.11 wonderwords nltk datasets scipy 2>&1 | tail -3
python3.12 -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

# ---- Step 4: Apply the surgical FlashInfer + vLLM asymmetric patches ----
python3.12 /workspace/phase1/asym_patches/fix_flashinfer_version.py
python3.12 /workspace/phase1/asym_patches/patch_all_asym.py
python3.12 /workspace/phase1/asym_patches/fix_separator.py

# ---- Verify ----
python3.12 -c "
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
import inspect
sig = inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan)
if 'k_data_type' in sig.parameters and 'v_data_type' in sig.parameters:
    print('OK: asymmetric FlashInfer plumbing in place')
else:
    print('FAIL: asymmetric plumbing missing')
    raise SystemExit(1)
"

# ---- Pre-download model weights ----
export HF_HOME=/root/.cache/huggingface
mkdir -p $HF_HOME
for M in \
  'meta-llama/Llama-3.1-8B-Instruct' \
  'mistralai/Mistral-7B-Instruct-v0.3' \
  'microsoft/phi-4' \
  'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' \
  'Qwen/Qwen2.5-7B-Instruct'
do
  python3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download('$M', local_dir_use_symlinks=False,
                  allow_patterns=['*.safetensors','*.json','*.py','tokenizer*','*.txt','*.model'])
print('downloaded: $M')
"
done

mkdir -p /workspace/results

echo "Setup complete.  Run scripts from /workspace/phase1/."
