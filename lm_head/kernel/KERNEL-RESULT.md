# Certified-decode LM-head kernel — 25.3% bytes, lossless, 1.66x at batch-1

The shadow-bound certified decode was a Python replay: it proved that an idblock
LM head can certify the dense argmax while reading ~25% of the bytes
(argmax_match 1.0 at r=1280/int8 on Qwen2.5-7B), but it computed the full dense
logits as its oracle, so it measured the byte SAVINGS without realising the
DEPLOYMENT path. This is the kernel that realises it: a Triton shadow-bound GEMV
plus a greedy slab fetch that never materialises the dense logit vector, validated
lossless and timed against the dense head on a W7900.

## What the kernel does

Stage 1 (Triton). For a decode hidden state h, with the offline shadow head
aq[V,r] (int8, per-column scale) and delta[V], compute per token the valid upper
bound U_v = aq_v.(scale*q) + rho*delta_v + cst, where q = B^T h, rho = ||h - Bq||,
cst = aq_err_norm*||q||. This is an [V,r] int8 GEMV (r=1280 << d=3584, int8 < bf16),
reduced to idblock slab maxima U_b. It streams ~18% of the dense head's bytes.

Stage 2 (host loop + exact GEMV). Open idblock slabs in descending U_b, fetch the
TRUE logits for each opened slab from W_U, track ell* = max fetched logit, stop
(strict) once ell* exceeds the largest U_b among unopened slabs. The dense argmax
is then certified having touched only the opened rows.

## Result (Qwen2.5-7B, r=1280, int8 shadow, hidden_pca basis, 400 decode tokens)

| metric | value |
|---|---|
| kernel vs torch-ref bound (max abs) | 2.9e-06 |
| worst bound violation (must be <= 0) | -2.73 |
| argmax_match (lossless) | 1.0000 |
| fetched rows: mean / median / p95 | 7.4% / 2.0% / 48% |
| dense GEMV latency | 1.673 ms |
| shadow stage-1 (kernel) latency | 0.489 ms |
| full certified decode latency | 1.008 ms |
| speedup vs dense | 1.66x |
| bytes/token dense | 1090 MB |
| bytes/token shadow + fetch | 195 + 81 = 276 MB |
| byte ratio | 25.3% |

The bound is provably valid (worst violation -2.73 <= 0: every U_v dominates its
true logit), the kernel matches the reference to 1e-6, and the certified argmax
equals the dense argmax on every token. The byte ratio 25.3% matches the replay's
24.6% (the small gap is the kernel run's slightly higher fetched mean, 7.4 vs 6.7%,
on this token sample). It beats an int8 dense head (50%) and ties an int4 dense
head (25%) while being LOSSLESS, where int4 is not.

## Reading the latency

The kernel realises the deployable memory win in full -- 25.3% of the dense bytes,
lossless -- and is already 1.66x faster end to end at the single-token batch-1
decode that is the WORST case for it. Stage 1 alone (the shadow GEMV) is 3.4x
faster than the dense GEMV (0.489 vs 1.673 ms); the gap to the byte ratio is the
Stage-2 greedy fetch, a host-side Python loop over up to C=256 slabs with a small
GEMV each, which is launch-overhead-bound rather than memory-bound at batch 1. The
median token certifies after fetching ~2% of rows, so the loop usually exits in a
few iterations; the mean is pulled up by a ~48% p95 tail.

The clear next optimisation is to fuse Stage 2 -- batch the slab fetches and the
stop test into one or two kernel launches instead of a Python loop -- which should
pull the latency toward the 25.3% byte ratio on memory-bound decode (large batch
or the real serving regime, where bytes, not launch overhead, set the wall clock).
That, plus an int8xint8 Stage-1 variant (quantise q as well, at a little extra
slack), is the path to a contribution-ready vLLM logits path.

## Artifacts

`src/certdecode_kernel.py` (Triton kernel + certified decode), `src/certdecode_
bench.py` (validation + benchmark), `src/gen_artifact.py` (shadow artifact
builder), `artifact/` (qwen-7b aq/scale/delta/B/cof/H/gt/W_U), `bench_qwen7b_
r1280_int8.json`. Reproduce: gen_artifact.py then certdecode_bench.py on a Triton
GPU. Dev/validation ran on a W7900 (ROCm, Triton 3.5); the kernel is portable to
CUDA unchanged.
