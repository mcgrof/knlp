# Canonical Definitions for BPA v27

## Core Variables

**L** — Context length in tokens. The total number of tokens in the
KV cache at evaluation time (prefix + generated). Tested values:
L in {8192, 32768}.

**D** — Number of transformer layers (decoder blocks). Tested range:
D in {24, 28, 32}.

**epsilon (eps)** — Relative PPL tolerance. The maximum allowed
percentage increase in perplexity relative to the dense (fp16)
baseline. Two thresholds are used: eps=3% and eps=1%.

**PASS_eps** — A configuration passes at tolerance eps if and only if
max|delta_pct| <= eps across ALL tested (L, seed) pairs. Formally:

    PASS_eps iff max_{L in L_set, s in seeds} |delta_pct(L,s)| <= eps

where delta_pct = (ppl_compressed - ppl_dense) / ppl_dense * 100.

**k** — Number of layers assigned to INT8 (high-precision) protection
in a mixed INT4/INT8 KV quantization scheme. The remaining (D - k)
layers use INT4 with group size g.

**k*(eps)** — The minimal k such that the configuration passes at
tolerance eps across the full (L_set, seed) evaluation matrix. If
k=0 passes, then k*=0. If no k passes, k*=D (meaning the scheme
fails entirely).

**kv_ratio** — The effective KV bytes ratio, defined as total
compressed KV bytes per token divided by dense fp16 KV bytes per
token. Includes quantization payload AND scale metadata.

Dense bytes per token per layer:
    dense_layer = 2 * n_kv_heads * head_dim * 2  (K+V, fp16)

INT8 bytes per token per layer:
    int8_layer = 2 * n_kv_heads * head_dim * 1 + 2 * n_kv_heads * 2
               = payload + per-tensor scale

INT4 bytes per token per layer (group size g):
    n_groups = ceil(head_dim / g)
    int4_layer = 2 * n_kv_heads * head_dim * 0.5
               + 2 * n_kv_heads * n_groups * 2
               = payload + per-group scale

kv_ratio = (k * int8_layer + (D-k) * int4_layer) / (D * dense_layer)

## Quantization Scheme Variants

**g32 standard** — INT4 symmetric quantization with group_size=32
along head_dim. Scale overhead: ceil(head_dim/32) fp16 scales per
KV head per layer. This is the scheme used in all canonical results.

**Amortized-scale variants** (v22-v23) — Token-window amortization
of INT4 scale factors. Shares one scale across S consecutive tokens.
Reduces scale overhead but introduces approximation error. Tested
with g=8 S=8 (v22 Pareto-optimal) and g=4 S=2. Not used in the
headline claim because they introduce a second hyperparameter.

**INT8-all** — All layers at INT8. Used as a lossless baseline
(typically <0.5% delta). Not competitive on kv_ratio.

## Ranking Methods

**Oracle ranking** — Per-layer INT4 ablation: for each layer i,
quantize only layer i to INT4 (all others INT8), measure max
|delta_pct| across seeds. Rank layers by decreasing sensitivity.
This is the "best possible" ranking and requires D forward passes
per model. Used in all headline results.

**Transferred ranking** — Oracle ranking from one model applied to a
different model of the same architecture. Tested in v22 (0.5B
ranking applied to 1.5B): nearly passes (+3.42%) but does not match
oracle. Not used in headline claims.

**Proxy ranking** — Ranking derived from a forward-only signal
(e.g., residual_ratio from v19, adam v-hat from v18). None matches
oracle quality across models. Residual_ratio has rho=0.394 with
oracle (v19). Not used in headline claims.

## Derived Metrics

**k*/D** — Protected fraction. If k* is O(1) and D grows, k*/D
approaches zero. Observed: 0.083, 0.071, 0.071, 0.000.

**Tail fraction (C=m)** — The fraction of total per-layer
sensitivity in the tail (layers ranked below the top m). Formally:

    tail_frac(m) = sum(alpha[m:]) / sum(alpha)

where alpha is the sorted sensitivity vector. Low tail fraction
means a few layers dominate; high means sensitivity is spread.

**Empirical lower boundary (kv_ratio_min)** — The lowest
kv_ratio achieving PASS_eps across all tested (model, L, seed)
conditions, using the best available ranking and quantization
scheme. This is NOT an information-theoretic lower bound. It is
the best ratio achieved under our test protocol.

**Capacity multiplier** — Theoretical concurrent sequences vs
dense, assuming KV cache is the memory bottleneck:

    capacity_mult = 1 / kv_ratio

At kv_ratio=0.28, capacity_mult=3.57x.

## Evaluation Protocol

- Seeds: {0, 1, 2}. Each seed selects a different random wikitext
  passage for the prefix.
- Context lengths: L in {8192, 32768}.
- Oracle ablation: L=8192 only (faster, sufficient for ranking).
- k-sweep: L in {8192, 32768} with all seeds.
- Cache regions: W_sink=4 (always fp16), W_min=1024 (near region,
  always fp16), far region (compressed).
- Decoding: 64 continuation tokens, greedy.
- Dataset: wikitext-2-raw-v1 (HuggingFace).

## Model Specifications

| Model | D | n_kv_heads | head_dim | params | architecture |
|-------|---|------------|----------|--------|--------------|
| Qwen2.5-0.5B | 24 | 2 | 64 | 494M | Qwen2 (GQA) |
| Qwen2.5-1.5B | 28 | 2 | 128 | 1.5B | Qwen2 (GQA) |
| Qwen2.5-7B | 28 | 4 | 128 | 7.6B | Qwen2 (GQA) |
| Mistral-7B-v0.1 | 32 | 8 | 128 | 7.2B | Mistral (GQA) |

## Hardware

| GPU | VRAM | BPA versions | Role |
|-----|------|--------------|------|
| AMD W7900 | 48GB | v8-v24 | Initial development |
| NVIDIA H100 80GB | 85GB | v26 | Validation at scale |
