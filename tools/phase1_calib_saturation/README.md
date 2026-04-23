# Phase 1 — calibration + saturation, H100

Scripts for the paper's calibration head-to-head and saturation model
refit on a single H100 pod.  Phase 2 (MI300X) reuses the same logic
with ROCm-specific env vars.

## Scripts

- `setup_h100_pod.sh` — one-shot pod bringup: installs vLLM 0.19,
  FlashInfer 0.6.7, patches in the asymmetric-kv-dtype branch,
  downloads model weights.
- `collect_kv_scales.py` — runs a forward pass over a WikiText-2
  calibration set and writes per-tensor and per-channel K absmax
  scales to JSON.  One JSON per model.
- `saturation_sweep.py` — per-model decode throughput sweep across
  `B x T` grid for all six configs.  Writes JSONL for Hill fitting.
- `calibration_head_to_head.py` — Llama-3.1-8B only, runs lm-eval
  across WikiText-2 PPL, GSM8K, MMLU, and NIAH multikey-3 under
  all six configs.  Writes JSONL.
- `qwen_sanity.py` — calibrated FP8 per-tensor on Qwen2.5-7B,
  WikiText-2 PPL at T=2048 only.  Five minutes.
- `fit_hill.py` — reads saturation sweep JSONL, fits scipy
  curve_fit to the Hill function, outputs per-config parameter
  table and plots.

## Configs

1. `fp16`            FP16 KV cache, unit scales (P0 baseline)
2. `fp8_uncalib`     symmetric FP8, unit scales (vLLM default)
3. `fp8_calib_pt`    symmetric FP8, per-tensor absmax scales
4. `fp8_calib_pc`    symmetric FP8, per-channel K absmax scales
5. `asym_uncalib`    asymmetric FP16-K/FP8-V, unit V scales
6. `asym_calib`      asymmetric FP16-K/FP8-V, calibrated V scales

## Models

- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `microsoft/phi-4` (Phi-4-14B)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

Qwen2.5-7B-Instruct is used only for the sanity check (Qwen is
excluded from the saturation fit because symmetric FP8 collapses
its quality).

## Operating grid

- `B in {2, 4, 8, 16, 32, 64}`
- `T in {1024, 4096, 16384}`

## Archive destination

`/data/knlp-key-results/phase1-calib-saturation-h100-<YYYYMMDD>/`
