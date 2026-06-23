# LM-head certified-decode latency on the W7900 (prune)

CUDA-event (HIP) timed, 20 warmup + 200 iters, p50. Qwen2.5-7B shapes (V=152064,
d=3584, r=1280, C=256), synthetic tensors (latency is shape/dtype-driven). Dense =
the bf16 head GEMV; stage-1 = the shadow-bound kernel; full = stage-1 + a
fixed-budget fetch proxy at the measured 7.4% fetched fraction.

## Batch-1 (single-stream decode -- the memory-saturated regime)

| path | p50 latency | vs dense |
|---|---|---|
| dense head GEMV | 1.68 ms | 1.0x |
| shadow stage-1 (kernel) | 0.47 ms | 3.6x |
| full certified decode | 1.14 ms | 1.47x |

At batch 1 the win is real and matches the byte story: the shadow head reads
195 MB (int8 [V,r]) vs the dense head's 1090 MB, and the stage-1 kernel hits
3.6x -- close to the 5.6x weight-traffic ratio, the rest lost to the fetch and
the projection. The full certified decode is 1.47x faster AND lossless at 25.3%
bytes. This is the regime the decode paper is about: one token per step, memory
bandwidth saturated, the weight read IS the latency.

## Batch sweep -- shadow stage-1 beats dense 3.2x FLAT across the batch

The dense head GEMV is FLAT across batch (1.68 -> 1.81 ms B=1..32): it reads W_U
once and reuses it for every token in the batch, so it is memory-bound on that
single 1090 MB read regardless of B. The shadow head reads only aq (int8 [V,r] =
195 MB, 5.6x less), so once it ALSO amortizes that read across the batch it is
flat and faster:

| B | dense p50 | shadow stage-1 | vs dense |
|---|---|---|---|
| 1 | 1.68 ms | 0.48 ms | 3.52x |
| 2 | 1.69 ms | 0.53 ms | 3.21x |
| 4 | 1.70 ms | 0.53 ms | 3.18x |
| 8 | 1.71 ms | 0.54 ms | 3.20x |
| 16 | 1.75 ms | 0.56 ms | 3.12x |
| 32 | 1.81 ms | 0.82 ms | 2.20x |

Getting here was a real bug, and the root cause is specific. Codex reviews
(threads 019eb0a1, 019eb0de) traced it: the original batched kernel wrote the
150k-wide U[Bt,V] (now fused to the tiny U_b[Bt,C] via a 2D-grid atomic slab-max),
and -- the dominant cost -- it did the score dot in fp32 with input_precision=
"ieee". RDNA3 (the W7900, gfx1100) has WMMA matrix units for f16/bf16/iu8/iu4 but
NOT fp32, so an fp32 dot falls back to the vector ALUs and runs ~32x over the
195 MB bandwidth floor (~7.5 ms). This was NOT a ROCm or hardware limit on the
idea; it was the wrong dot dtype.

The fix is BF16 WMMA: aq is exact in bf16 (int8 in [-127,127] fits bf16's mantissa),
the online operand scale*Q is bf16-rounded, and the dot runs on the WMMA unit with
fp32 accumulation. To keep the certified UPPER bound valid under the q-rounding, the
kernel adds the margin ||aq_v||_2 * ||scale*Q_b - bf16(scale*Q_b)||_2 (aq_l2[v] is a
precomputed property of the shadow head). Verified: the batched bound is ALWAYS >=
the exact per-token bound (min(batched - per-token) = 0.0 at Bt=2..16) -- never an
under-estimate, so losslessness is preserved -- with ~1.0 of conservative slack from
the margin (a small bump to the fetched fraction, the price of the bf16 dot).

So the shadow GEMV (stage 1) is now a clean 3.2x-over-dense win across batch 1..16,
not just at batch 1. The remaining batched cost is the stage-2 exact FETCH, which in
this harness is a per-token-looped proxy and so still scales with B; batching the
fetch (gather the union of opened slabs across the batch into one GEMV) is the next
step. But the part that smelled like a bug -- the batched shadow kernel -- is fixed
and is bandwidth-bound as it should be. Artifacts: latency_lmhead.json,
src/lmhead_latency.py, src/certdecode_kernel.py.

## Full-model decode-step dilution (the honest framing)

The 3.2x / 1.47x / 25.3%-bytes numbers above are the LM HEAD GEMV IN ISOLATION. A
decode step also reads all non-head weights (once per token) and the KV cache (grows
with context T, per sequence), so the FULL-step speedup is the head win diluted by
the head's share of total per-token decode traffic (Amdahl on bytes). Per-token
traffic (bf16): head = 2*V*d; non_head = 2*P - head; kv = B*L*2*n_kv*dh*T*2;
head_share = head/(non_head+head+kv); with certified head-table ratio q,
full_speedup = 1/(1 - head_share*(1-q)). Weights are reused across batch B, KV
scales with B, so head_share shrinks as B*T grows. With q_mean=0.253 (mean 7.4%
fetch) and q_p95=0.659 (the 48% fetch tail):

| model | head share @B1,T0 | full-step speedup (mean / p95) |
|---|---|---|
| Qwen2.5-7B | 7.2% | +5.6% / +2.5% |
| Llama-3.2-3B | 12.3% | +10.1% / +4.4% |
| Qwen2.5-1.5B | 15.2% | +12.8% / +5.4% |
| Qwen2.5-0.5B | 27.8% | +26.2% / +10.5% |
| 1B / 152k-vocab / d2048 | 31.1% | +30.3% / +11.9% |

So: at 7B the certified head is a low-single-digit full-step win; it becomes a real
full-model win (~20-30%) only for SMALL models with LARGE vocab, SHORT context, and
SMALL batch -- the regime where the head dominates per-token traffic. It dilutes
fast: Qwen-7B at B=16,T=32768 is only +1.8% (mean). Thresholds (mean q): >5% full
gain needs head_share>6.7%, >10% needs >13.4%, >20% needs >26.8%. Script:
amdahl_lmhead.py.

## Honest scope of every number here

- COMPONENT (solid, measured): shadow stage-1 is ~3.2x the dense head GEMV and
  streams ~17.9% of dense-head bytes; the certified lossless path is 25.3% of
  dense head-table bytes at MEAN fetch 7.4% (and ~65.9% at the p95 48% fetch tail
  -- the per-token byte ratio is a distribution, not a point).
- PROXY (not yet the real number): the "1.47x full" at batch-1 uses a FIXED-budget
  fetch (open round(0.074*C)=19 slabs, per-token-looped contiguous GEMV). The REAL
  certified greedy fetch is data-dependent (variable slabs/token, long p95 tail,
  non-contiguous slab IDs) and for batched decode must gather the UNION of opened
  slabs across the batch in waves. The real fetch LATENCY (distribution, batched
  union) is the outstanding measurement; the fetched-FRACTION distribution (mean
  7.4 / median 2.0 / p95 48%) is already measured in certdecode_bench.py.
- FULL-MODEL: per the table -- low single digits at 7B, up to ~20-30% for
  small-large-vocab-short-context, before scheduler/gather overhead and tail.
- Byte accounting note: the 25.3% is HEAD-TABLE bytes only; it excludes the online
  d*r basis projection (one extra read). rho should use
  sqrt(sum h^2 - sum q^2) (B orthonormal) to avoid a second basis read.

## MEASURED real certified-fetch latency (replaces the proxy) -- the latency win
## does NOT hold; the byte win does

Ran the REAL data-dependent certified greedy decode over 400 real Qwen2.5-7B hidden
states (not the fixed-budget proxy), CUDA-event timed per token:

| metric | dense head | certified real | 
|---|---|---|
| p50 latency | 1.673 ms | 1.232 ms (1.36x faster) |
| MEAN latency | 1.673 ms | 3.817 ms (0.44x = 2.3x SLOWER) |
| p95 latency | -- | 16.1 ms |
| p99 latency | -- | 20.3 ms |
| fetched frac mean/p50/p95/p99 | -- | 7.9% / 2.0% / 51.2% / 65.2% |

The proxy's 1.47x was optimistic. The MEDIAN token certifies fast (1.36x), but the
MEAN is 2.3x SLOWER than the dense head: the tail (p95 opens ~51% of slabs) runs
through a host-Python greedy loop doing one W_U[slab] @ h GEMV per opened slab, and
that per-slab launch overhead over ~130 slabs dominates the mean. So with the
current host-loop fetch, the certified decode is NOT an end-to-end latency win on
average -- only at the median.

Batched union makes it worse: different tokens open different slabs, so the batch
must fetch the UNION. Measured union fetched fraction: batch 2 -> 1.2%, batch 4 ->
2.7%, batch 8 -> 10.9%, batch 16 -> 17.6% (vs ~2.6% per-token). The byte advantage
erodes with batch.

What this means, precisely:
- BYTE reduction is REAL and lossless: mean head-table traffic 25.3% of dense (mean
  fetch 7.9%), and stage-1 (the shadow GEMV) is genuinely 3.2x faster. Those stand.
- LATENCY is NOT yet a win: the data-dependent stage-2 fetch, implemented as a host
  Python loop of per-slab GEMVs, has a tail that makes the MEAN decode latency 2.3x
  slower than dense. To turn the 25.3% byte reduction into real latency, stage-2
  must be a single FUSED gather-GEMV (gather the opened slab rows into one
  contiguous-ish GEMV per token, and the batch union in waves) -- not a per-slab
  Python loop. The shadow GEMV is not the bottleneck; the fetch loop is.
- So the honest end-to-end claim today: lossless 25.3% mean head-table bytes and a
  3.2x stage-1 GEMV, but the full certified-decode LATENCY needs a fused fetch
  kernel before it beats dense on the mean (it already wins at the median). The
  earlier "1.47x full" is withdrawn -- it was a fixed-budget proxy.
Artifacts: certfetch_latency.json, src/certfetch_latency.py.

## STAGE-2 FIXED: fused wave gather-GEMV + dense fallback -> real latency win

The per-slab host loop was the bottleneck (mean 2.3x SLOWER than dense). Two changes
fix it: (1) open slabs in WAVES of g and fetch each wave with ONE gathered GEMV
(launch count drops from ~#opened slabs to ~#opened/g); (2) a DENSE FALLBACK -- if a
token has not certified after fetching `fallback_frac` of the vocab, bail to the
exact dense GEMV (still lossless -- dense argmax is exact), which caps the heavy tail
(tokens that need most of the vocab cost ~dense instead of 20ms).

Measured (Qwen2.5-7B, 400 real hidden states, vs dense head 1.68 ms):

| config | mean ms | p50 | p95 | p99 | lossless | vs dense (mean/med) |
|---|---|---|---|---|---|---|
| per-slab loop (old) | 2.93 | 1.21 | 15.1 | 19.4 | 1.000 | 0.57x / 1.39x |
| wave-4, fallback 0.20 | 1.567 | 1.06 | 5.05 | 5.09 | 1.000 | 1.07x / 1.58x |
| wave-4, fallback 0.10 | 1.508 | 1.06 | 3.86 | 3.91 | 0.998* | 1.11x / 1.58x |

(* fallback uses the bf16 dense head, so 0.2% of tokens differ from the fp32 gt on
near-ties -- lossless w.r.t. the DEPLOYED bf16 head, which is the truth in
deployment; fallback 0.20 is also exactly fp32-lossless.)

So the fused stage-2 turns the certified head from a 2.3x regression into a real
~1.07x MEAN / ~1.58x MEDIAN latency win over the dense head, lossless, with the tail
bounded at ~5 ms. Larger waves over-fetch the cheap common case; wave-4 with fallback
~0.20 is the sweet spot. This is HEAD-component latency; the full-model dilution
(Amdahl section) still applies, so the full decode-step gain is +5% at 7B and larger
only for small-large-vocab models. Remaining headroom: a Triton fused gather-GEMV
(no W_U row-copy) would lift both the common waves and the fallback partial fetch,
pushing the mean further past dense. Artifacts: stage2_waves.json,
src/stage2_waves_bench.py, certified_decode_waves in src/certdecode_kernel.py.

## TAIL FIX: predictive gap-routing -> 1.21x mean, p99 5.0->3.0 ms (Codex 019eb249)

The wave+fallback path still had a 5 ms p99 tail (the ~12% of tokens that need most
of the vocab waste a partial fetch then hit the dense fallback). Predictive routing
fixes most of it: the shadow-bound shape predicts hardness BEFORE fetching, and hard
tokens go straight to dense (no wasted partial fetch). Predictor: gK = U_b_sorted[0]
- U_b_sorted[K-1] with K=52; corr(gK, fetch_fraction) = -0.597 (a real signal). Route
to dense if gK <= tau_gap. (A pre-stage-1 rho/||h|| veto was tried but rho corr is
only 0.249 -- too weak -- so it is off.)

Calibrated tau_gap=5.49 (routes ~13% of tokens, FN 2%, FP 3%). Measured (Qwen-7B, 400
real tokens, dense 1.68 ms):

| path | mean | p50 | p95 | p99 | lossless | vs dense |
|---|---|---|---|---|---|---|
| plain wave+fallback | 1.57 | 1.06 | 5.05 | 5.09 | 1.000 | 1.07x mean |
| + gap-routing (tau_gap 5.49) | 1.38 | 1.12 | 2.36 | 3.02 | 0.998* | 1.21x mean / 1.51x median |

(* gap-routed tokens use the bf16 dense head, so 0.2% differ from the fp32 gt on
near-ties -- lossless w.r.t. the deployed bf16 head.) Oracle (perfect routing)
ceiling is 1.13 ms (1.48x); the gap predictor reaches 1.38 ms. The residual p99 = 3 ms
is the ~2% false-negative hard tokens that slip into the certified path and hit the
5 ms fallback; a tighter tau_gap (lower FN, slightly higher mean) or Codex's richer
logistic predictor (g8/g16/g32 gaps + softmax entropy + rho) would push p99 toward
~2.1 ms. Net: the certified head is now a real ~1.21x mean / ~1.51x median latency win
with a bounded p99, lossless -- not just byte reduction. Full-model gain still
Amdahl-bounded (~+5% decode-step at 7B). Artifacts: route_calib_bench.json,
src/route_calib_bench.py, src/route_diag.py, certified_decode_routed in
src/certdecode_kernel.py.
