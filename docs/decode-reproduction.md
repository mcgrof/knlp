# Decode paper: build, paper, and reproducibility

The vLLM asym branch requires torch >= 2.10, cmake >= 4.0, and the
FlashInfer cutlass submodule initialized.  The tested recipe (H100
SECURE pod, RunPod):

```bash
# 1. FlashInfer
cd /root && git clone --branch asym-prefill-refactor-stage \
    https://github.com/mcgrof/flashinfer.git flashinfer-src
cd flashinfer-src && git submodule update --init --recursive
pip install --no-build-isolation -e .

# 2. vLLM (pulls torch and rebuilds _C; ~60 min CUDA compile)
cd /root && git clone --branch asymmetric-kv-plumbing \
    https://github.com/mcgrof/vllm.git vllm-src
cd vllm-src && MAX_JOBS=32 NVCC_THREADS=2 \
    pip install --no-build-isolation -e .

# 3. Reinstall flashinfer editable (vllm pip overwrites with PyPI 0.6.6)
cd /root/flashinfer-src && pip install --no-build-isolation -e .

# 4. Verify
FLASHINFER_DISABLE_VERSION_CHECK=1 python -c "import vllm, flashinfer"
```

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

