# Experiment Registry

Catalog of experiments conducted within knlp.

## Active Experiments

| Name | Purpose | Hardware | Status |
|------|---------|----------|--------|
| KV Bandwidth Scaling | Measure decode throughput vs batch size and context length | W7900, H100, B200 | Active |
| KV Cache Quantization | INT4 KV quantization with fused/non-fused pipelines | H100, B200 | Active |
| Selective Layer Protection | Find minimum INT8-protected layers (k*) for quality | W7900, H100 | Active |
| KV Tiering | Mixed-precision KV cache (INT4 + INT8) | W7900, H100 | Active |
| Extreme Context | Push context length to hardware limits | B200 | Active |
| Capability Benchmarks | HellaSwag, MMLU, GSM8K baseline and quantized | B200 | Active |
| BPA Prototypes | Bandwidth-constrained attention mechanisms | - | Prototype |

## Architecture Experiments

| Name | Purpose | Hardware | Status |
|------|---------|----------|--------|
| Reciprocal Attention | Learned Q@K.T / K@Q.T alternation | W7900, B200 | Active |
| KVSplice | FIM-guided KV cache compression on top of MLA | W7900, A100, H100, B200 | Active |
| Multi-head Latent Attention | DeepSeek-style cache compression | W7900, B200 | Active |
| Adam State-Based Pruning | bitter7 importance scoring via exp_avg_sq | W7900, B200 | Active |
| FIM-Guided Quantization | Diagonal Fisher for precision allocation | W7900 | Active |

## GNN Experiments

| Name | Purpose | Hardware | Status |
|------|---------|----------|--------|
| Page-Aware GNN Training | I/O-locality-aware mini-batch sampling | CPU | Active |
| FIM-Guided GNN Fraud | RA transfer from transformers to GNNs | CPU | Active |

## Deprecated Experiments

| Name | Reason | Version |
|------|--------|---------|
| BPA v1-v9 | Superseded by v10+ protocol improvements | v1-v9 |
| Random KV eviction | Catastrophic quality loss, sink token discovery | v12 |
| Frontier compression (MLA/splice) | Lossy random projections fail | v13 |
| V-only INT8 | K noise dominates, dead end | v20 |
| ScaleQuant (INT8 log-scale) | Failed to reduce overhead | v22 |
