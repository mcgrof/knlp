# Reproducing *Memory-Traffic Saturation in Autoregressive Transformer Decode*

This document tells a human or AI coding agent how to reproduce the
paper's findings using the `knlp` defconfig system. The paper itself
lives at <https://github.com/mcgrof/paper-memory-decode>; the companion
page is <https://knlp.io/decode>.

## Quick start

```bash
git clone https://github.com/mcgrof/knlp.git
cd knlp
make defconfig-decode
make
```

This selects the `decode` reproduction profile, then runs the pipeline:

```text
decode-doctor → decode-fetch → decode-build → decode-run → decode-report
```

`decode-doctor` checks GPU/CUDA/cmake/disk/HF-token before any clone or
build. `decode-fetch` clones the four companion repos into `..` (the
parent of the `knlp` checkout). `decode-build` installs FlashInfer,
vLLM, and LMCache as editable Python packages, in the order required
to dodge the "vLLM pulls flashinfer-python from PyPI and overwrites the
editable" pitfall. `decode-run` executes the configured stage list.
`decode-report` writes a Markdown + JSON summary.

If any stage fails, fix the underlying issue and rerun `make`. The
pipeline resumes from the first stage that does not have a `DONE`
marker.

## What gets cloned and built

| Repo | Branch | Path | What's modified |
|---|---|---|---|
| <https://github.com/mcgrof/vllm> | `asymmetric-kv-plumbing` | `../vllm` | Tuple K/V cache, FlashAttn writer patch, asym dtype plumbing |
| <https://github.com/mcgrof/flashinfer> | `asym-prefill-refactor-stage` | `../flashinfer` | FI-1..FI-5 CUDA template refactor for independent K/V dtypes |
| <https://github.com/mcgrof/LMCache> | `asymmetric-kv-codec` | `../lmcache` | K16/V8 codec, split-tier placement, serde, 74 CPU unit tests |
| <https://github.com/mcgrof/paper-memory-decode> | `main` | `../paper-memory-decode` | LaTeX source, figures, generate scripts |

You can override branches/URLs/paths by editing the corresponding
`CONFIG_KNLP_*` values in `.config` after running the defconfig.

## Profiles

| Profile | What it reproduces | Default hardware |
|---|---|---|
| `defconfig-decode` | Core full-stack quality battery (Qwen2.5-7B PPL@2K/8K, GSM8K n=200), FlashInfer standalone gates, vLLM writer gate, LMCache codec, split-tier microbench | 1× H100 80 GB |
| `defconfig-decode-sat` | Saturation model: H100 batch/context sweep, Hill fit, B vs B·T comparison | 1× H100 (8×H100 reduces wall time) |
| `defconfig-decode-full` | Everything possible from the paper, including saturation, NIAH long-context, static calibration, TurboQuant stress, speculative decoding, large models, full storage grid. Cross-GPU lanes auto-skip with structured skip reports when their hardware is missing. | Multi-GPU class (H100 + A100 + B200/H200 + W7900 + MI300X for true full reproduction) |

## Hardware requirements

Core (`defconfig-decode`):

- 1× NVIDIA H100 80 GB (or H200)
- ≥200 GB free disk (model weights, vLLM build, FlashInfer JIT cache)
- ≥64 GB system RAM
- CUDA-capable Linux (tested on Ubuntu 22.04 in the RunPod pytorch
  container)
- HF token in `HF_TOKEN` env or `~/.cache/huggingface/token` (Qwen2.5
  is open-access but rate-limited without one)

The `decode-doctor` stage checks all of the above before any clone or
build runs. If a check fails, you get a punch list, not a wasted
hour.

Saturation profile (`defconfig-decode-sat`) tolerates lower-end GPUs
but the H100 lane is what the paper's main fits use. `defconfig-decode-full`
issues a structured `skips.json` entry for any cross-GPU lane whose
hardware is absent rather than failing or silently skipping.

## Runtime estimates

Computed at `make decode-estimate` from the actual detected hardware.
Order-of-magnitude expectations:

| Profile | 1× H100 cold | 1× H100 warm | 8× H100 warm |
|---|---|---|---|
| `decode` | 8–14 h | 4–8 h | 1.5–3 h |
| `decode-sat` | 24–40 h | 18–36 h | 4–8 h |
| `decode-full` (H100 subset) | 3–7 days | 2–6 days | 18–36 h |

Cold = first run, includes repo clones, vLLM/FlashInfer CUDA build
(~60 min), Qwen2.5-7B model download (~38 GB), all stages.
Warm = repos and models already cached locally.

True cross-hardware reproduction of `defconfig-decode-full` requires
A100 + H100 + B200 (or H200) + AMD W7900 + AMD MI300X, and elapsed
time depends on provider availability (2–5 days realistic).

## Optional telemetry

Local JSONL is the canonical source of truth. W&B and trackerio are
optional mirrors; their failure never fails the run.

To enable W&B:

```bash
export WANDB_API_KEY=...
make defconfig-decode
sed -i 's/CONFIG_KNLP_ENABLE_WANDB=n/CONFIG_KNLP_ENABLE_WANDB=y/' .config
make
```

To enable trackerio:

```bash
export TRACKERIO_API_KEY=...
make defconfig-decode
sed -i 's/CONFIG_KNLP_ENABLE_TRACKERIO=n/CONFIG_KNLP_ENABLE_TRACKERIO=y/' .config
make
```

Both flags can also be flipped through `make menuconfig`.

## Output directory

```text
results/decode/<run_id>/
  manifest.json          # full run manifest (schema_version, git refs, hardware, models, datasets, telemetry config)
  metrics.jsonl          # canonical metrics stream (event log: run_start, stage_start, metric, artifact, stage_end, run_end)
  artifacts.json         # list of artifact paths logged across stages
  pip-freeze.txt         # captured at run start
  nvidia-smi-q.txt       # captured at run start
  torch-env.txt          # captured at run start
  report.md              # human-readable run summary (after `make decode-report`)
  report.json            # machine-readable run summary
  stages/
    00_doctor/
      DONE | SKIPPED     # exactly one of these signals stage status
      metrics.jsonl
      stdout.log
      stderr.log
    01_fetch_repos/...
    02_build_flashinfer/...
    ...
    11_report/...
```

Run IDs are timestamps: `decode-20260429T134500Z`. The orchestrator
picks up the most recent `<profile>-*` directory under `results_root`
when resuming, so `make` always continues the most recent run rather
than starting a new one.

## Resume

A failed stage leaves no `DONE` file. Rerun `make` and the pipeline
continues from the failed stage. To rerun a single stage:

```bash
make decode-run STAGE=07_qwen25_fullstack_quality
```

To force a fresh run, remove the run dir:

```bash
rm -rf results/decode/<run_id>
make
```

To wipe everything (results + cloned repos):

```bash
make decode-clobber
```

## Instructions for an AI coding agent

You are reproducing the paper *Memory-Traffic Saturation in
Autoregressive Transformer Decode*.

Do not improvise experiments. Use the selected defconfig.

1. Run `make defconfig-decode` and then `make decode-doctor`.
2. If doctor fails, fix only environment problems: missing HF token,
   insufficient disk, missing CUDA, missing Python deps, missing git
   refs. Do not invent new experiments to compensate.
3. Run `make`.
4. Do not edit experiment scripts unless a build or runtime error
   actually blocks reproduction.
5. If a stage fails: inspect `results/<profile>/<run_id>/stages/<stage>/stderr.log`,
   fix the narrow issue (missing dep, OOM, env variable), rerun
   `make`. Do not delete completed stage outputs unless they are
   corrupt.
6. Preserve all artifacts: `manifest.json`, `metrics.jsonl`, per-stage
   logs, the generated `report.md`, plots.
7. If W&B or trackerio is enabled, verify local JSONL still exists.
   Remote logging is a mirror, not the source of truth.
8. At the end, summarize: hardware, git refs, pass/fail per stage,
   headline Qwen2.5 results (PPL@2K, PPL@8K, GSM8K n=200 across
   FP16 / FP8-sym / asymmetric K16/V8), storage ratio, skipped
   stages and why.

## Troubleshooting

**HF token missing.** Set `HF_TOKEN` or write the token to
`~/.cache/huggingface/token`. The Qwen2.5 download won't proceed
without one on most provider images.

**`kconfiglib` not installed.** The `make defconfig-decode` step copies
the defconfig to `.config` and then runs `pyconf.py --olddefconfig` to
fill in defaults and validate the Kconfig tree.  That second step
requires `kconfiglib`.  Without it the raw defconfig is used as-is
(still functional, but unvalidated).  Install it with:

```bash
pip install kconfiglib
```

**`cmake: command not found` or version too old.** vLLM's editable
build wants cmake ≥ 4. Run `pip install --upgrade cmake`. The
`decode-doctor` stage checks for this and prints the exact fix.

**`ImportError: undefined symbol: c10::MessageLogger...`.** vLLM was
built against a different torch ABI than what's currently installed.
This typically happens when vLLM's pip install pulls a newer torch
after the vLLM `_C.abi3.so` was already compiled. Fix:

```bash
cd ../vllm && rm -rf build vllm/_C*.so /tmp/tmp*.build-temp
pip install --no-build-isolation --force-reinstall --no-deps -e .
```

**FlashInfer JIT cache stale after a refactor.** Clear it:

```bash
rm -rf ~/.cache/flashinfer
```

**`flashinfer-cubin version (0.6.6) does not match flashinfer version
(0.6.7)`.** Set `FLASHINFER_DISABLE_VERSION_CHECK=1` for the run.

**Asymmetric K16/V8 init crashes with `'tuple' object has no attribute
'startswith'` or similar.** vLLM auto-selected FlashAttention; force
FlashInfer explicitly:

```python
LLM(..., attention_config={"backend": "FLASHINFER"})
```

The `VLLM_ATTENTION_BACKEND` env var is *not honored* in this vLLM
build.

**TP > 1 fails with `_PyImport_Init: global import state already
initialized`.** Known multiprocess issue on certain provider images.
Try `VLLM_WORKER_MULTIPROC_METHOD=spawn` or `distributed_executor_backend="ray"`
(after `pip install ray`). The paper's TP=4 sanity attempt hit this and
is recorded as a limitation rather than a result.

**Disk full mid-build.** vLLM build artifacts + FlashInfer JIT cache +
Qwen2.5 weights (~38 GB) regularly push past 200 GB. Use `df -h` to
check, or run `make decode-doctor` to validate up front.

## Reproduction policy and pass/fail thresholds

The `decode` profile fails the run (returns non-zero) on any of:

- FlashInfer asymmetric decode rel err ≥ 0.10
- FlashInfer asymmetric prefill rel err ≥ 0.10
- vLLM writer gate fails K bit-exactness
- vLLM writer gate fails V tolerance (FP8 dequant noise threshold)
- Qwen2.5 asymmetric PPL differs from FP16 by more than 2%
- Qwen2.5 asymmetric GSM8K differs from FP16 by more than 1
  percentage point
- LMCache storage ratio not within ±2% of 0.7500
- LMCache K bit-exactness check fails

Throughput-only differences are noisy; they are recorded but do not
gate the run. W&B / trackerio upload failures are warnings, not
failures.

## Companion docs and code

- Paper companion site: <https://knlp.io/decode> (reads off the same
  result files)
- Paper repo: <https://github.com/mcgrof/paper-memory-decode>
- vLLM fork: <https://github.com/mcgrof/vllm/tree/asymmetric-kv-plumbing>
- FlashInfer fork: <https://github.com/mcgrof/flashinfer/tree/asym-prefill-refactor-stage>
- LMCache fork: <https://github.com/mcgrof/LMCache/tree/asymmetric-kv-codec>
- Defconfig sources: `defconfigs/decode`, `defconfigs/decode-sat`,
  `defconfigs/decode-full`
- Orchestrator: `tools/reproduce/paper_memory_decode/`
