# BPA v11 Final Report

## Model Selection

**Qwen/Qwen2.5-0.5B** (494M parameters)
- Architecture: Qwen2 with RoPE, GQA (14 heads, 2 KV heads)
- max_ctx: 32,768 tokens
- 24 layers, 896 hidden dim, 64 head dim
- KV cache: 12,288 bytes/token (fp16)
- Selected over Llama-3.2-1B due to stronger latency-vs-L
  scaling (2.3x from L=512 to L=8192), confirming bandwidth
  sensitivity.

## Hardware

- GPU: AMD Radeon Pro W7900 (48GB VRAM)
- ROCm: torch 2.6.0+rocm6.2.4, HIP 6.2
- All runs on single GPU, fp16 inference

## Headline Results — Batch=1 (Tuned Configs, 256 Steps)

### Tuning: Matched-Quality at tol=1%

All four context lengths PASS.

| L    | Dense PPL | BPA PPL | delta% | kept/L | KV ratio |
|------|-----------|---------|--------|--------|----------|
| 512  | 21.1      | 21.1    | -0.2%  | 354/512 | 0.49x   |
| 1024 | 15.1      | 15.1    | +0.2%  | 1013/1024 | 0.88x |
| 2048 | 10.9      | 11.0    | +0.9%  | 1973/2048 | 0.90x |
| 4096 | 16.3      | 16.3    | -0.1%  | 3894/4096 | 0.92x |

### Tuning: Matched-Quality at tol=3%

All four context lengths PASS.

| L    | Dense PPL | BPA PPL | delta% | kept/L | KV ratio |
|------|-----------|---------|--------|--------|----------|
| 512  | 21.1      | 21.4    | +1.4%  | 283/512 | 0.55x   |
| 1024 | 15.1      | 15.1    | +0.2%  | 1013/1024 | 0.88x |
| 2048 | 10.9      | 11.0    | +0.9%  | 1973/2048 | 0.90x |
| 4096 | 16.3      | 16.3    | -0.1%  | 3894/4096 | 0.92x |

### Decode Latency (batch=1, tuned tol=1% configs)

| L    | Method     | p50 ms | p95 ms | Gate% | tok/s |
|------|------------|--------|--------|-------|-------|
| 512  | dense      | 9.19   | 9.38   | 0.0%  | 108   |
| 512  | bpa_v11    | 8.71   | 8.84   | 3.2%  | 115   |
| 512  | static     | 9.01   | 9.07   | 0.0%  | 111   |
| 1024 | dense      | 10.07  | 10.18  | 0.0%  | 100   |
| 1024 | bpa_v11    | 9.77   | 9.82   | 3.0%  | 102   |
| 2048 | dense      | 11.47  | 11.67  | 0.0%  | 87    |
| 2048 | bpa_v11    | 11.21  | 11.28  | 2.7%  | 89    |
| 4096 | dense      | 14.44  | 14.62  | 0.0%  | 69    |
| 4096 | bpa_v11    | 13.91  | 14.04  | 2.9%  | 72    |

Latency improvement: **2.3-5.2%** across all L values.

### Bandwidth Sensitivity (batch=4, 128 steps)

| L    | Method  | p50 ms | GPU MB | tok/s |
|------|---------|--------|--------|-------|
| 512  | dense   | 9.36   | 2642   | 105   |
| 512  | bpa_v11 | 8.87   | 2625   | 112   |
| 1024 | dense   | 10.37  | 3347   | 96    |
| 1024 | bpa_v11 | 10.28  | 3283   | 97    |
| 2048 | dense   | 11.98  | 4687   | 83    |
| 2048 | bpa_v11 | 11.78  | 4659   | 85    |
| 4096 | dense   | 15.14  | 7188   | 66    |
| 4096 | bpa_v11 | 14.70  | 7163   | 68    |

Batch=4 latency gains: **0.9-5.2%**, comparable to batch=1.
GPU memory savings: 17-64MB (negligible at this model size).

## Adaptivity Analysis

### Far Budget Controller: FAIL

The far budget controller does not spike meaningfully on hard
spans. Across all stress tests:
- k_far_mean: 2.9-3.6 (baseline ~3.0)
- k_far_max: 4 (never reaches B_far_max=8)
- No difference between easy and hard spans

The entropy+residual pressure signal reflects overall prediction
difficulty, not retrieval need. It cannot distinguish tokens
that need far context from tokens that are locally difficult.

### W Duty Cycle

At tuned configs, W is nearly constant:
- L=512: W stays near W_min=102 (aggressive, 69% kept)
- L=1024: W stays near W_max=972 (conservative, 99% kept)
- L=2048: W stays near W_max=1945 (conservative, 96% kept)
- L=4096: W stays near W_max=3891 (conservative, 95% kept)

The controller is effectively static at these settings. The
adaptive machinery (PI governor, pressure signal) does not
produce meaningful W modulation.

### Static Sparse Comparison

Static sparse (fixed local window + random far) works at L=512
(PPL=21.2 vs dense 21.4) but fails catastrophically at L>=1024
(PPL in hundreds to thousands). This confirms that some form
of adaptive selection helps, but BPA's far budget selection
(random chunks from the non-local region) is barely better
than a larger local window.

## Key Finding

**BPA can match dense quality at all tested context lengths,
but only by keeping 88-99% of KV tokens at L>=1024.** The
real eviction opportunity is at short context (L=512, 45-55%
eviction) where the latency payoff is smallest.

The fundamental issue: well-trained language models with GQA
genuinely use most of their context. There is less redundancy
in the KV cache than the BPA hypothesis assumes, especially
at moderate context lengths (1K-4K).

## DONE Conditions Assessment

| Criterion | Status |
|-----------|--------|
| Model max_ctx >= 4096 | PASS (32768) |
| Matched-quality for >= 3 L values | PASS (all 4) |
| Far budget adaptation proven | FAIL |
| KV reduction without latency regression | PASS (2-5%) |

## Reproduction Commands

```bash
# Model probe
python scripts/bpa_v11_model_probe.py

# Dense baselines + BPA + static sparse (batch=1, 3 seeds)
python scripts/bpa_v11_bench.py bench --tuned \
  --L 512,1024,2048,4096 --steps 256 --seeds 1,2,3

# Matched-quality tuning
python scripts/bpa_v11_bench.py tune \
  --L 512,1024,2048,4096 --tol 1,3

# Stress tests
python scripts/bpa_v11_bench.py stress \
  --L 512,1024,2048 --steps 64 --seeds 1,2

# Bandwidth sensitivity (batch=4)
python scripts/bpa_v11_bench.py bench --tuned \
  --config-dir bpa_v11_results \
  --L 512,1024,2048,4096 --steps 128 --seeds 1,2 \
  --method dense,bpa_v11 --batch-size 4 \
  --output-dir bpa_v11_results_batch4
```
