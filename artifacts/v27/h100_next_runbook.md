# H100 Next-Step Runbook

## Host

    ssh rqv

## Repo Path

    /mnt/tmpfs/knlp

## Environment

    source ~/bpa_env/bin/activate

## Priority 1: Llama Architecture Confirmation

### Goal
Add a third architecture data point to the O(1) table.

### Model
NousResearch/Llama-2-7b-hf (D=32, n_kv_heads=32, head_dim=128, MHA)

If gated Llama-3.1-8B access becomes available, prefer that instead
(D=32, n_kv_heads=8, head_dim=128, GQA).

### Command

    source ~/bpa_env/bin/activate
    cd /mnt/tmpfs/knlp
    python3 scripts/v27_confirmatory.py

The script automatically runs Llama-2-7b-hf as a third model after
verifying Qwen7B and Mistral7B.

For standalone Llama-only run (if verification already done):

    python3 -c "
    import sys; sys.path.insert(0, '.')
    from scripts.v27_confirmatory import run_new_model_quick, save_json
    result = run_new_model_quick(
        'llama2_7b', 'NousResearch/Llama-2-7b-hf', 32, 32, 128
    )
    save_json(result, 'results/v27/h100_confirmatory/k_star_llama2_7b.json')
    "

### Expected Runtime
- Dense baselines: ~30 min (6 configs: 2 L x 3 seeds)
- Oracle screening (1 seed, L=8K): ~2h (32 layers)
- k-sweep (3 seeds, 2 L, k=0..4): ~4h
- Total: ~6-7 hours

### Expected Outputs
- results/v27/h100_confirmatory/k_star_llama2_7b.json

### Acceptance Criteria
A) k*(3%) <= 3 confirms O(1) on third architecture
B) If k*(3%) > 4, document and explain

### Notes
Llama-2-7b uses MHA (32 KV heads = 32 attention heads), not GQA.
This is architecturally different from both Qwen (2-4 KV heads) and
Mistral (8 KV heads). With 32 KV heads, we expect uniform robustness
similar to or better than Mistral (8 KV heads). If so, this would
suggest that head diversity is the key determinant of per-layer
robustness.

## Priority 2: Larger-D Model

### Goal
Extend the D range beyond 32 to test O(1) scaling further.

### Candidates (in order of preference)
1. Qwen2.5-14B (if accessible, D likely 40-48)
2. NousResearch/Llama-2-13b-hf (D=40, MHA, freely accessible)
3. Mixtral-8x7B-v0.1 (D=32, MoE, different architecture class)

### Command (for Llama-2-13b)

    python3 -c "
    import sys; sys.path.insert(0, '.')
    from scripts.v27_confirmatory import run_new_model_quick, save_json
    result = run_new_model_quick(
        'llama2_13b', 'NousResearch/Llama-2-13b-hf', 40, 40, 128
    )
    save_json(result, 'results/v27/h100_confirmatory/k_star_llama2_13b.json')
    "

### Expected Runtime
- ~10h total (40 layers for oracle + k-sweep)

### Acceptance Criteria
A) k*(3%) <= 4 at D=40 extends O(1) range
B) kv_ratio <= 0.30 at k* config

### Notes
The H100 has 85GB VRAM. Llama-2-13b-hf in fp16 requires ~26GB for
weights + ~10GB for KV cache at L=32K. This should fit with room
for activations. If OOM, reduce to L=16K only.

## Priority 3: Bandwidth-Bound Profiling

### Goal
Determine if quantized decode is bandwidth-limited on H100 and
measure actual ms/token improvement.

### Prerequisites
- A fused INT4 attention kernel (FlashInfer, vLLM, or custom)
- Currently NOT available in bpa_env

### Approach (when available)

    # Install fused kernel
    pip install flash-attn-int4  # hypothetical

    # Profile decode with and without quantization
    python3 scripts/v27_latency_profile.py \
        --model Qwen/Qwen2.5-7B \
        --L 32768 \
        --batch-sizes 1 4 8 16 32 \
        --k 2 \
        --fused-kernel

### Expected Runtime
- ~2h per model

### Acceptance Criteria
C) >= 10% ms/token improvement in bandwidth-bound regime
D) If no improvement, document why (still compute-bound, etc.)

### Notes
Without a fused kernel, the simulation approach stores KV cache as
fp16 regardless of quantization. No memory or bandwidth savings are
realized. The decode step reads the same amount of fp16 data whether
or not it was "quantized."

A fused kernel would:
1. Store KV cache in INT4 format (4x less memory)
2. Dequantize on-the-fly during attention computation
3. Reduce memory bandwidth by ~4x for the far-region KV reads

This is the gap between "capacity improvement" (measured) and
"latency improvement" (not yet measured).

## Status Tracking

| Priority | Model | Status | k*(3%) | kv_ratio | Notes |
|----------|-------|--------|--------|----------|-------|
| 1 | Llama-2-7b-hf | DONE | 0 | 0.2812 | Uniform robustness, max_delta=1.01% |
| 2 | Llama-2-13b-hf | PENDING | — | — | Next priority for D range extension |
| 3 | Fused kernel | BLOCKED | — | — | No kernel available |
