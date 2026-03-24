# knlp-rocm-bench: unified ROCm benchmark container for W7900
#
# Derived from rocm/vllm:latest which provides:
#   - vLLM with ROCm/HIP support
#   - PyTorch + ROCm 7.0
#   - FlashAttention for ROCm
#   - vLLM benchmark scripts at /app/vllm/benchmarks/
#
# This image adds the benchmark extras needed for the full
# evaluation stack: lm-eval, GuideLLM, long-context smoke
# deps for xKV and InfiniteBench, and helper scripts.
#
# The xKV and InfiniteBench repositories themselves remain
# bind-mounted from /data at runtime (they contain large
# model code and data that changes independently).  The
# Python libraries those repos import at eval time are
# baked into the image so no pip install is needed at
# container start.

FROM rocm/vllm:latest

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Enable experimental AOTriton attention paths for AMD GPUs.
# Without this, PyTorch silently falls back to slower SDPA.
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# ── Layer 1: core benchmark harnesses ─────────────────────
# lm-eval:  accuracy gate (MMLU, HellaSwag, ARC, etc.)
# guidellm: open-loop serving traffic generation
RUN pip install --no-cache-dir \
    lm-eval \
    guidellm

# ── Layer 2: xKV smoke dependencies ──────────────────────
# These are the libraries imported by xKV's evaluate/ code
# path (eval_acc.py, evaluator.py, data/dataset.py,
# data/metrics.py, data/utils.py).  We deliberately skip
# nemo_toolkit[all], pytorch-lightning, hydra-core, and
# other heavy deps from xKV's requirements.txt that are
# only needed for xKV's own training/patching paths, not
# for the evaluation smoke we run.
#
# The base image already provides: torch, transformers,
# numpy, datasets, accelerate, huggingface-hub, tqdm.
RUN pip install --no-cache-dir \
    termcolor \
    pandas \
    jieba \
    fuzzywuzzy \
    rouge \
    rouge_score \
    wonderwords \
    tabulate \
    tiktoken \
    einops \
    omegaconf \
    loguru \
    tenacity \
    protobuf \
    nltk \
    pyyaml

# ── Layer 3: InfiniteBench smoke dependencies ─────────────
# eval_hf_smoke.py needs: datasets, transformers, torch
# (already present).  eval_utils.py needs jieba and rouge
# (installed above).  The remaining lightweight deps:
RUN pip install --no-cache-dir \
    openai \
    xopen \
    python-dotenv

# Pre-download NLTK punkt tokenizer (used by rouge scoring)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ── Helper scripts ────────────────────────────────────────
COPY container/smoke-xkv.sh    /usr/local/bin/smoke-xkv
COPY container/smoke-ib.sh     /usr/local/bin/smoke-ib
RUN chmod +x /usr/local/bin/smoke-xkv /usr/local/bin/smoke-ib

# Create mount points for host data.
# These are documentation-only; actual mounts happen at
# docker run time via -v flags.
RUN mkdir -p /results /data

# vLLM benchmarks are already at /app/vllm/benchmarks/
# from the base image. Set VLLM_DIR for convenience.
ENV VLLM_DIR=/app/vllm

# Default working directory for benchmark runs
WORKDIR /workspace

# Default entrypoint: interactive bash shell.
# Override with specific benchmark commands at run time.
CMD ["/bin/bash"]
