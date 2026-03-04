# BPA v27: Paper Outline

## Working Title

"Bounded Protection in Mixed-Precision KV Caches: Empirical
Evidence for O(1) Layer Sensitivity Scaling"

## Target Length

8-10 pages (main), 4-6 pages (appendix)

## Structure

### 1. Introduction (1 page)

- KV cache is the memory bottleneck for long-context LLM inference
- Mixed-precision quantization (INT4/INT8) reduces cache size
- Open question: how many layers need protection?
- If the answer is O(1), the effective compression ratio approaches
  the quantization floor regardless of model depth
- We provide empirical evidence across 4 models, 2 architectures,
  D=24-32

### 2. Background and Related Work (1 page)

- KV cache quantization: KIVI, KVQuant, FlexGen, SmoothQuant
- Per-layer sensitivity: connection to pruning literature
- Attention sinks: StreamingLLM, Sink Token
- Mixed-precision approaches: GPTQ per-layer, AWQ
- What's new: systematic per-layer INT4 ablation to determine k*

### 3. Problem Formulation (0.5 pages)

- Definitions: D, k, k*(eps), kv_ratio, PASS_eps
- Evaluation protocol: wikitext-2, seeds, context lengths
- The question: is k*(eps) = O(1) over tested D?

### 4. Experimental Setup (1 page)

- Models: Qwen2.5-{0.5B, 1.5B, 7B}, Mistral-7B-v0.1
- Hardware: AMD W7900, NVIDIA H100
- Quantization: symmetric INT4 g=32 / INT8
- Cache structure: sink + far (compressed) + near (fp16)
- Oracle ranking procedure

### 5. Main Results (2 pages)

- Table 1: Canonical k* results (Fig. table_1_canonical_kstar)
- Fig 1: k* vs D, k*/D vs D
- Finding: k*(3%) = {2, 2, 2, 0} across D = {24, 28, 28, 32}
- kv_ratio at k*: 0.28-0.30

### 6. Mechanistic Analysis (1.5 pages)

- Fig 2: Oracle sensitivity distributions (4 models)
- Sink dominance (Qwen): layer 0 holds 67-99.99% of sensitivity
- Uniform robustness (Mistral): max per-layer delta <0.5%
- Fig 5: Sink dominance contrast
- Connection to attention sinks and RoPE
- Why both mechanisms yield bounded k*

### 7. Accumulation Theory (1 page)

- Error propagation through residual stream
- e_D = sum of amplified per-layer noise
- k(D,eps) lower bound derivation
- O(1) condition: bounded sink count + bounded tail
- Calibrated parameters from experiments

### 8. Practical Implications (0.5 pages)

- Table 2: Capacity multiplier (3.3-3.6x)
- kv_ratio approaching INT4-all floor (0.28125)
- What this means for serving systems
- Latency: requires fused kernels (not demonstrated)

### 9. Limitations (0.5 pages)

- Finite D range (24-32)
- Oracle ranking required
- No fused INT4 kernel
- Two architecture families only
- Wikitext evaluation only
- No downstream task evaluation

### 10. Conclusion (0.5 pages)

- k* is empirically O(1) over tested range
- Two distinct mechanisms support this
- The path to INT4-all floor is k* layers away
- Next: Llama architecture, fused kernels, larger D
