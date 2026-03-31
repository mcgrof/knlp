# Triton fused routing kernel status — 2026-03-31

## Code landing
The fused routing kernel code is now copied into:
- `/data/knlp/routing/`

Primary files:
- `routing/fused_routed_attention.py`
- `routing/benchmark.py`
- `routing/tests/test_fused_routed_attention.py`
- `routing/README.md`

## Durable results
Durable root on prune:
- `/data/knlp-key-results/paper-router/triton-fused-routing-kernel-20260331T191500Z/`

This root contains:
- `routing/` source snapshot
- `routing_kernel_benchmark_20260331T190718Z.json`
- `routing_inference_retry_20260331T192630Z.json`

## What was proven
### Kernel correctness
- 79/79 committed tests passed
- 6/6 fresh independent tests passed
- correctness held across torch 2.6 and torch 2.10 after the vLLM install upgraded torch

### Strong performance results
#### A100 microbench vs Python-loop reference
At 4096 tokens:
- K=1: up to **96.5x** faster than Python-loop reference
- K=2: up to **89.0x** faster
- K=4: about **47.2x** faster
- K=8: about **24.6x** faster
- K=16: about **12.0x** faster

At the previously interesting operating points:
- BS=128, K=4: **26.6x** faster than Python-loop reference
- BS=256, K=8: **10.1x** faster than Python-loop reference

#### A100 routed-vs-dense latency
Per-layer decode, Marin-8B-like config:
- Dense (all blocks): `2.336 ms`
- Routed K=4 (fused): `0.039 ms` → **59.6x faster than dense**
- Routed K=4 (Python loop): `1.949 ms` → loop overhead nearly destroys the routing benefit

Full 32-layer decode:
- Dense: `74.724 ms`
- Routed K=4 (fused): `1.253 ms` → **59.6x faster than dense**
- Routed K=4 (Python loop): `63.630 ms`

This is the strongest systems result: the fused Triton kernel turns routing from an overhead-dominated toy into a bandwidth-dominated decode path.

## Accuracy status
Accuracy is still the limiting factor, not the kernel.

From the fused inference retry with routed priors in the current evaluation:
- BS=16, K=8: cosine about `0.17` → unusable
- BS=128, K=8: cosine about `0.51` → still poor
- BS=256, K=8: cosine about `0.71` → marginal at 50% KV reduction

Prior routing experiments suggested substantially better cosine once the priors are real/model-derived rather than synthetic or weak. The fused kernel removed implementation overhead; it did not solve prior quality.

## How to improve accuracy / routing prior quality
### 1. Replace weak or random priors with real model-derived priors
This is the most important next step. The current latency results prove the kernel. The remaining problem is choosing the right blocks. Use:
- query-dependent routing
- prefill-derived importance scores
- per-layer priors instead of one global heuristic

### 2. Use coarser blocks when accuracy is too brittle
The current data suggests larger blocks / coarser granularity can move us toward the first viable quality regime.

### 3. Make K adaptive by layer or query state
A fixed K everywhere is too blunt. The obvious next line is:
- small K where the prior is sharp
- larger K where uncertainty is high

### 4. Evaluate with real serving traces, not only synthetic random routing tests
The current quality caveat is partly an evaluation caveat. The next fair accuracy test should reuse model-derived routing priors from real prefill / decode traces.

## vLLM status
The new image fixed the old vLLM blocker:
- image: `runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204`
- `pip install vllm` now works cleanly
- final env on the routing pod reached `vllm 0.18.1`, `torch 2.10.0+cu128`, `triton 3.6.0`

Important nuance:
- the kernel is **not yet integrated into vLLM's serving path**
- so the current latency numbers are **decode-kernel / model-like benchmark numbers**, not end-to-end production TTFT

## TTFT vs baseline
We do **not** yet have a valid end-to-end TTFT claim from the new fused kernel lane.

What we do have:
- decisive routed-vs-dense decode latency wins
- decisive elimination of Python-loop overhead
- repaired vLLM environment on the new image

What remains to prove:
- end-to-end vLLM integration
- real serving-path TTFT and latency under model-derived routing priors

## Bottom line
### Strong positives
- kernel correctness: proven
- microbench latency: very strong
- routed decode vs dense: strongly positive
- new RunPod image: fixes the old vLLM blocker

### Remaining blocker
- routing prior quality / accuracy, not kernel speed

### Next phase
- integrate the kernel into the real serving path
- drive routing with model-derived priors
- measure true serving latency and TTFT, not just decode microbench time
