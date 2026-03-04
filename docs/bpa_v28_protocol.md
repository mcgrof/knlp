# BPA v28: Canonical Evaluation Protocol

## Purpose

This document freezes the evaluation protocol for all BPA headline
results. Every number in the canonical table must come from this
protocol, executed on the rqv H100.

## Dataset

- Source: `wikitext-103-raw-v1`, validation split
- Loading: `"\n\n".join(ds["text"])`, then `tokenizer.encode(text)`
- Token budget: first 500,000 tokens
- Text sampling: one contiguous passage per (L, seed) pair,
  starting at `RandomState(seed).randint(0, n_tokens - seq_len)`

## Evaluation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| DECODE_TOKENS | 64 | Sufficient for stable PPL estimate |
| SEEDS | [0, 1, 2] | Three seeds for variance |
| L_SET | [8192, 32768] | Short + long context |
| batch_size | 1 | Single sequence per eval |
| W_SINK | 4 | Attention sink protection |
| W_MIN | 1024 | Near-window protection |
| GROUP_SIZE | 32 | Standard INT4 group size |
| DTYPE | torch.float16 | Model and cache precision |

For models with max_position_embeddings < 32832 (32768+64), use
L_SET = [max_pos//2 - 32, max_pos - 64] instead.

## PPL Computation

1. Prefill: run model on tokens[0:L], collect cache and last logit
2. Decode: feed tokens[L:L+64] one at a time, collect all logits
3. Concatenate: all_logits = [prefill_last_logit] + decode_logits
4. Loss: cross_entropy(all_logits[:, :-1, :], continuation_tokens)
5. PPL: exp(min(loss, 20))

The shift (logits[:-1] predicts continuation) is standard causal LM
evaluation. The cap at exp(20) prevents overflow.

## Quality Criterion

- epsilon = 3% (relative PPL tolerance)
- delta = (ppl_quant - ppl_dense) / ppl_dense * 100
- PASS_3%: max(|delta|) <= 3.0 across all (L, seed) pairs
- k*(3%): smallest k such that protecting the top-k oracle layers
  achieves PASS_3%

## Quantization

INT4 g=32 symmetric: per-group absmax scaling, range [-8, 7].
INT8 symmetric: per-tensor absmax scaling, range [-128, 127].
Both simulate-quantize-dequantize in fp16 (no fused kernel).

Cache regions:
- Sink [0:W_SINK]: always fp16
- Far [W_SINK:cache_len-W_MIN]: quantized per layer_bits
- Near [cache_len-W_MIN:]: always fp16

## Oracle Ranking

For each layer i in [0, D):
1. Set layer i to INT4, all others to INT8
2. Measure max |delta| across all (L, seed) pairs
3. Rank layers by sensitivity (descending)

The top-k layers in this ranking are protected (kept at INT8).

## KV Ratio

kv_ratio = (k * bytes_int8 + (D-k) * bytes_int4) / (D * bytes_dense)

Where per-token-per-layer:
- dense = 2 * n_kv_heads * head_dim * 2 (K+V in fp16)
- int8 = 2 * n_kv_heads * head_dim + 2 * n_kv_heads * 2 (data + scales)
- int4 = 2 * n_kv_heads * head_dim * 0.5 + 2 * n_kv_heads * ceil(head_dim/32) * 2

## Hardware

- GPU: NVIDIA H100 80GB HBM3
- Host: rqv
- PyTorch: 2.10.0+cu126
- Transformers: 5.2.0
- CUDA: 12.6

## Reconciliation with Prior Versions

### v27 vs v26 discrepancies (now resolved)

1. **PPL computation**: v27 correctly shifts logits (predict next
   token). v26 used `compute_ppl(all_logits[:, :-1, :], continuation)`
   which is the same shift. No actual discrepancy — both shift.

2. **Text sampling**: v27 uses single contiguous passage per seed.
   v26 uses `get_text_batch()` which also samples single passages
   at batch_size=1. At batch_size=1, both are equivalent.

3. **DECODE_TOKENS**: v26 defaults to 256 for latency benchmarking
   phases but quality phases also use 64-step decode in the
   `run_single_eval` path. v28 freezes at 64 for all headline
   results.

4. **DATASET constant**: v27 had a stale `DATASET = "wikitext-2-raw-v1"`
   string but actually loaded wikitext-103. Fixed in v28 harness.

### Qwen-7B borderline result

v26 reported k*=2 with max_delta=1.05%. v27 verification found
max_delta=3.62% at L=32768 seed=2. The difference is due to
different random text offsets (v26 used `get_text_batch` RNG
which draws from a different sequence than v27's
`load_wikitext_passages`). v28 canonical re-run resolves this by
using the frozen protocol for all models.
