# knlp-rocm-bench: unified ROCm benchmark container for W7900
#
# Derived from rocm/vllm:latest which provides:
#   - vLLM with ROCm/HIP support
#   - PyTorch + ROCm 7.0
#   - FlashAttention for ROCm
#   - vLLM benchmark scripts at /app/vllm/benchmarks/
#
# This image adds the benchmark extras needed for the full
# evaluation stack: lm-eval, GuideLLM, and helper deps.
# Long-context repos (xKV, InfiniteBench) are bind-mounted
# at runtime rather than vendored into the image.

FROM rocm/vllm:latest

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Enable experimental AOTriton attention paths for AMD GPUs.
# Without this, PyTorch silently falls back to slower SDPA.
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Install benchmark harness dependencies.
# lm-eval: accuracy gate (MMLU, HellaSwag, ARC, etc.)
# guidellm: open-loop serving traffic generation
RUN pip install --no-cache-dir \
    lm-eval \
    guidellm

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
