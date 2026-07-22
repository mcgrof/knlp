# STILL runtime KV-cache compactor

This reproduces and evaluates STILL (arXiv:2606.07878), a *runtime* KV-cache
compactor, and compares it against full-context KV and the trained-cartridge
approach on a single GPU. STILL is a small per-layer Perceiver module that, after
a normal prefill, reads the full source KV cache and writes a much smaller
compact KV cache — for every new context at inference. The question this harness
answers is a systems one: does runtime compaction actually move the usable
per-request HBM frontier once construction, replacement, reuse, and concurrency
are all counted, and does a trained compactor preserve answer quality on unseen
contexts?

Unlike a trained cartridge (offline memory, one artifact per document, loaded
read-only at serving time), STILL does fresh compaction and HBM writes per new
context. Those costs are measured in separate phases here.

Everything is pure PyTorch + Triton + transformers with the base model frozen —
no external package to bootstrap. It was developed on an AMD Radeon Pro W7900
(gfx1100, ROCm 6.4, PyTorch 2.9.1, Triton 3.5.1); the fused kernel needs Triton
3.5.1 (a stray upstream Triton 3.6 breaks it on this stack).

## Quick start

```
make defconfig-still-kernel      # or -still-baselines / -still-ladder
# then on a GPU host:
PYTHON=/path/to/rocm-or-cuda/python OUT_DIR=out research/still_compactor/run.sh
```

`gen_config.py` turns `.config` into `config.json`; `run.sh` dispatches to the
selected experiment. Set `DEVICE` (default `cuda:0`) and, on a multi-GPU host,
`HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`.

## Experiments

- **kernel** — validate the fused streaming Triton compactor-attention kernel
  against the materialized reference (fp32 exact, bf16 as accurate as the
  reference), prove the quadratic `t x T` score workspace is eliminated, and time
  it against SDPA.
- **ledger** — a byte-accurate peak-HBM microbenchmark over context length and
  concurrency, contrasting post-prefill-replace vs chunked-streaming.
- **baselines** — peak HBM for eager full prefill vs a chunked full-KV baseline
  (no compaction) vs chunked STILL, so the base-attention workspace is separated
  from the retained-KV savings.
- **chunked** — the chunked-streaming construction on the real model, showing the
  construction peak stays bounded and nearly flat in context length.
- **concurrency** — fit and validate `peak = weights + C·compact + transient` and
  project the concurrent-request ceiling under serialized vs parallel compaction.
- **io** — HBM/PCIe/NVMe byte accounting, the query-reuse break-even, and the
  SSD-offload go/no-go.
- **ladder** — train the compactor with forward-KL distillation on thousands of
  fresh procedural contexts and evaluate held-out generalization.

## Defconfigs

- `still-kernel` — the fused-kernel validation. Fast; a good first check that the
  Triton stack works and the streaming kernel is correct.
- `still-baselines` — the peak-HBM comparison (eager / chunked-full-KV /
  chunked-STILL) that separates the workspace effects from the retained-KV win.
- `still-ladder` — the generalization training run (2000 fresh contexts, 512
  balanced held-out, 4x compression).

Scale knobs (`CONFIG_STILL_*`) are documented in the Kconfig help. Compression is
`CTX_TOKENS / T_COMPACT` for the ladder and `CHUNK / T_COMPACT` per chunk for the
systems experiments.

## What has been found (single-GPU, developed on a W7900)

The systems result, stated against the faithful baseline: STILL's retained state
stays roughly flat (~9–11 GiB) as context grows while a chunked full-KV baseline
grows linearly and runs out of memory by 64K tokens; the real win is the
retained-KV savings, which grows with context (about 5 GiB at 8K to 23 GiB at
32K). The frontier win requires the chunked-streaming construction schedule — a
naive prefill-then-compact-then-replace cannot lower the construction peak, since
the full source KV is resident before compaction. The fused kernel is what makes
that viable: it removes the compactor's own quadratic `t x T` workspace (flat
~2 MB regardless of source length) and beats SDPA, though it is slower than the
materialized path, which is fast only because it spends the gigabytes that make
it fail under concurrency and long context. Serialized compaction shares one
transient workspace, so concurrency is bounded by the small compact KV rather
than the workspace.

For quality, a trained compactor generalizes to unseen contexts on a synthetic
retrieval MCQ at 4x compression: held-out accuracy climbs epoch over epoch to
about 0.98, approaching the full-context ceiling, with a rising gold-answer
margin — the signature of a learned amortized compaction rule rather than
memorization. This holds, more slowly, at 8x. These are single-GPU,
short-context, synthetic-task results: the mechanism generalizes, but the
compression/length frontier and real-task quality at matched physical-KV budget
remain, and the identity initialization here is a faithful near-pass-through
(direction preserved; magnitude rescaled by RMSNorm), not the paper's exact
Appendix-C construction. The fused-kernel and memory-accounting results are
independent of any quality claim.
