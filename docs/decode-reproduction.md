# Decode paper: build, paper, and reproducibility

**Use the `paper-memory-decode-v0.18` branches** (FlashInfer `2b532f7`, vLLM
`2315e62e2`) — the ONLY pair that serves asym e2e. The dev-tip branches
(`asym-prefill-refactor-stage` / `asymmetric-kv-plumbing`) do NOT: they fail on
`v_cache bf16 != fp8` (cross-repo dtype drift). Requires torch 2.10.0+cu128 and
setuptools_scm/build deps preinstalled (`--no-build-isolation` skips them). The
tested recipe (H100 SECURE pod, RunPod; mirrors
`knlp-key-results/tp-asym-validation-20260624/harness/pod_build.sh`):

```bash
# 0. torch 2.10 cu128 (pin it; vllm/flashinfer otherwise bump it to 2.12 and
#    break the _C.so ABI) + build deps for --no-build-isolation
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
pip install setuptools_scm setuptools wheel cmake ninja packaging

# 1. vLLM v0.18 (~40 min CUDA compile). use_existing_torch.py keeps the pin.
cd /root && git clone https://github.com/mcgrof/vllm.git vllm-v018
cd vllm-v018 && git checkout paper-memory-decode-v0.18 && git reset --hard 2315e62e2
python use_existing_torch.py
MAX_JOBS=64 NVCC_THREADS=2 pip install -e . --no-build-isolation

# 2. re-pin torch (the vLLM build drags it to 2.12) + drop mismatched vision/audio
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y torchvision torchaudio

# 3. FlashInfer v0.18 (--no-deps to protect torch; JIT-compiles on first use)
cd /root && git clone https://github.com/mcgrof/flashinfer.git flashinfer-v018
cd flashinfer-v018 && git checkout paper-memory-decode-v0.18 && git reset --hard 2b532f7
git submodule update --init --recursive
pip install -e . --no-deps --no-build-isolation
pip install transformers==4.57.6

# 4. Verify (import vllm._C, not vllm, to register _C_cache_ops)
python -c "import torch; import vllm._C; import flashinfer; print('stack OK')"
```

Serve asym K16/V8: pass `attention_config={"backend": "FLASHINFER"}` and
`kv_cache_dtype=("auto", "fp8_e4m3")` to `LLM()`; the `VLLM_ATTENTION_BACKEND`
env var is not honored, and drop `VLLM_BATCH_INVARIANT` (breaks engine init). For
TP>1 set `NCCL_NVLS_ENABLE=0` (the RunPod H100 image can't init NVLS multicast).

Tensor parallelism is validated at **TP=1, 2, and 4** (Qwen2.5-7B): K16/V8 is
lossless — prefill NLL bit-identical to FP16 and decode token-agreement 1.0 vs
FP16 — and composes with zero difference-in-differences interaction, even at
TP=4 where the 4 KV heads shard to one KV head per rank. Symmetric FP8 collapses
at every TP. Give a file entry point when using `spawn` (a heredoc/stdin script
makes the multiprocess worker fail at import, which can masquerade as an NCCL
init crash).

The asym K16/V8 production recipe in Python:

```python
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    kv_cache_dtype=("auto", "fp8_e4m3"),
    attention_config={"backend": "FLASHINFER"},
)
```

`VLLM_ATTENTION_BACKEND` env var is **not honored** in this vLLM
build; pass `attention_config={"backend": "FLASHINFER"}` to the LLM
constructor.  Auto-selection picks FlashAttention, which lacks the
asym tuple writer.

### Paper build

```bash
cd /home/mcgrof/devel/paper-memory-decode && make
```

Generates figures via Python scripts, then runs pdflatex (3 passes
for cross-refs).  Always verify the rendered PDF with:

```bash
pdftotext paper.pdf - | grep -nE '<pattern>'
```

Source-level grep misses issues in figure PDFs and broken LaTeX label
resolution (e.g., `Table V-C0c` from a `\label` inside `\begin{center}`
instead of `\begin{table}`).

## Reproducibility System (paper-memory-decode)

The knlp defconfig system is being extended with paper reproduction
profiles.  The planned targets:

```bash
make defconfig-decode       # Core asym claims (1×H100, 4-8h warm)
make defconfig-decode-sat   # Saturation model (1×H100, 18-36h)
make defconfig-decode-full  # Everything (multi-GPU, days)
```

After selecting a defconfig, `make` runs:

```text
decode-doctor → decode-fetch → decode-build →
decode-run → decode-report → decode-upload (optional)
```

The orchestrator lives under `tools/reproduce/paper_memory_decode/`.
Each stage writes results to `results/decode/<run_id>/stages/<stage>/`
with `DONE`, `metrics.jsonl`, `stdout.log`, `stderr.log`.  Rerunning
`make` resumes from the first missing `DONE`.

Telemetry: local JSONL is mandatory and canonical.  W&B and trackerio
are optional mirrors controlled by `.config` flags and env vars
(`WANDB_API_KEY`, `HF_TOKEN`).

The defconfigs pin exact git refs for vllm, flashinfer, lmcache, and
paper-memory-decode, and clone/fetch them into `../` (the parent
directory).

# Memory

I want you to remember most of our conversations about this project.

