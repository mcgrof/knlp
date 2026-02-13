# BPA v12 Branch Tree

## Decision Points

### 1. Bandwidth-bound regime found?
YES — at L=8192->16384, batch=1: 1.61x latency scaling;
batch=4: 1.65x. Bandwidth dominates at long context on W7900.

### 2. Retrieval predictor AUC >= 0.75?
NO — best AUC = 0.6263 (cheap features only). Attention-derived
features scored AUC=0.40 (below random). Output-space features
carry marginal signal for predicting retrieval need.

### 3. Scaling exponent beta < 0.85?
NO — beta = 1.0435 (R^2=0.98). Scaling is linear. BPA keeps
82-99% of KV at matched quality. Local window dominates.

### 4. Layer-adaptive W reduces kept >=10% at same PPL?
NO — gradient profile reduces kept by 21% at L=4096 but PPL
degrades from 9.4 to 226.8. Quality cannot be maintained.

### 5. 32K context stable?
NO — BPA at L=32768 with conservative W=90-95% gives PPL=266
vs dense PPL=20. Even 9% eviction destroys coherence.

## Branch Assignment

**BRANCH C: Retrieval predictor weak (AUC < 0.7)**

The v12 evidence converges on Branch C with elements of B:
- Predictor AUC=0.63 < 0.70 threshold
- Beta ~1.0 confirms local window dominance (Branch B element)
- Available cheap features (entropy, logit margin, confidence)
  do not predict which tokens receive far attention mass
- Layer-adaptive W fails matched quality (Branch C/B overlap)

## Recommended Next Steps

1. **Richer signals needed.** The current feature set (output
   entropy, logit margin, top-k probability mass) is post-hoc —
   computed after attention already happened. A predictor would
   need pre-attention features: query projection norms, positional
   encoding magnitudes, or learned routing gates.

2. **Learned eviction.** Instead of random far chunk selection,
   train a lightweight scorer that ranks KV entries by importance.
   This is the approach taken by H2O, ScissorHands, and similar
   methods. The scorer would use attention accumulation from
   prior steps as signal.

3. **Architectural change for sublinear.** Beta ~1 means the
   model uses most of its context uniformly. Achieving sublinear
   scaling likely requires either (a) a different model architecture
   with sparse attention patterns (e.g., Mistral's sliding window),
   or (b) KV compression (quantization, MLA-style latent projection)
   rather than KV eviction.

4. **Abandon random eviction at long context.** The 32K stress
   test and quality collapse at L>=2048 with aggressive configs
   show random far chunk selection is too noisy. Any viable
   eviction strategy must be attention-informed.
