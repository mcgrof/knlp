# BPA v11: Model Selection

## Candidates Tested

| Model | Params | max_ctx | L=512 p50 | L=4096 p50 | L=8192 p50 | Peak GPU (8192) |
|-------|--------|---------|-----------|------------|------------|-----------------|
| Qwen2.5-0.5B | 494M | 32768 | 9.0ms | 14.1ms | 20.3ms | 3635MB |
| Llama-3.2-1B-Instruct | 1236M | 131072 | 13.0ms | 16.8ms | 21.2ms | 4910MB |

## Decision: Qwen/Qwen2.5-0.5B

Chosen for three reasons:

1. Stronger latency-vs-L scaling. p50 grows 2.3x from L=512 to L=8192,
   indicating the model transitions from compute-bound to bandwidth-bound
   as context grows. This is exactly the regime where KV eviction can
   deliver real latency wins. Llama-3.2-1B only scales 1.6x over the
   same range — its larger compute footprint masks bandwidth sensitivity.

2. All tested L values (512, 1024, 2048, 4096, 8192) are well within
   its 32768 max_ctx. No position interpolation needed. Results are
   purely in-range.

3. Lower peak memory leaves headroom for batch>1 experiments (Phase 4
   bandwidth stress). At L=8192 the model uses only 3.6GB of 48GB.

## Why Not Llama-3.2-1B?

Not a bad candidate — it works and has more capacity. But for BPA
validation the priority is showing bandwidth sensitivity, not model
quality. Qwen2.5-0.5B has better latency scaling and is faster to
iterate on.

## Architecture Details

- Type: Qwen2 (RoPE, GQA)
- Layers: 24
- Hidden: 896
- Attention heads: 14
- KV heads: 2 (GQA 7:1 ratio)
- Head dim: 64
- Vocab: 151936
- Context: 32768 (RoPE, no scaling)
- Dtype: float16

KV cache per token: 2 * 24 layers * 2 kv_heads * 64 head_dim * 2 bytes
= 12,288 bytes/token = 12 KB/token

At L=4096: 48 MB KV cache per sequence.
At L=8192: 96 MB KV cache per sequence.
