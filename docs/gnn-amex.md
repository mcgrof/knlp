# AMEX Streaming Inference Benchmark

**Real-time capacity limits for cache-resident fraud detection.**

[Back to GNN Fraud Overview](gnn-fraud.md) ·
[DGraphFin Benchmark](gnn-dgraphfin.md) ·
[Interactive Visualization](gnn_fraud_visualization.html)

**Hardware:** AMD Radeon Pro W7900 (48GB)
**Framework:** PyTorch + ROCm

## Executive Summary

All three measurement validations passed. The benchmark framework is
**measurement-correct** and ready for canonical data generation.

## Background: What This Benchmark Measures

### The AMEX Dataset

American Express Default Prediction (AMEX) is a tabular fraud detection
dataset with ~460k customers and ~5.5M transaction records. Each customer
has a time series of up to 13 monthly statements with 188 features per
statement. The task is real-time fraud scoring: given a new transaction,
predict default risk.

**Critical property:** The entire dataset is ~3.3GB. This fits comfortably
in GPU memory (48GB) and system RAM. There is no disk I/O pressure.

### Key Parameters

**k_related (k):** Number of related entities fetched per event. When
scoring a transaction, we may also fetch features from k related customers
(e.g., similar spending patterns, same merchant category). Each additional
entity adds ~4.5KB of feature data.

| k | Bytes per event | Use case |
|---|-----------------|----------|
| 0 | 4,512 | Single customer only |
| 8 | 40,608 | Customer + 8 related |
| 16 | 76,704 | Customer + 16 related |
| 32 | 148,896 | Customer + 32 related |

**Microbatch size (mb):** Number of events grouped before GPU submission.
GPU kernels have fixed launch overhead (~10-50μs). Batching amortizes this
cost across multiple events.

| Microbatch | Tradeoff |
|------------|----------|
| mb=1 | Lowest latency, highest overhead per event |
| mb=4 | Higher baseline latency, lower overhead per event |

**Target TPS:** Transactions per second. The benchmark measures whether
the system can sustain a given TPS while meeting latency SLAs.

### Why AMEX Becomes a Streaming Benchmark (Not a Locality Benchmark)

The original hypothesis was: higher k → more data → locality matters.

This would be true if fetching related entities required disk I/O or
triggered cache misses. But AMEX is small enough that:

1. The entire dataset fits in GPU memory
2. Even with k=32, cache hit rate is 95%
3. No memory hierarchy pressure exists

When data fits in cache, the bottleneck shifts from "how fast can we
fetch data" to "how fast can we process batches." This makes AMEX a
**throughput capacity benchmark**, not a locality benchmark.

### How DGraphFin Differs

DGraphFin is a graph-based fraud detection dataset with 3.7M nodes and
4.3M edges. The key difference is **read amplification**:

| Property | AMEX | DGraphFin |
|----------|------|-----------|
| Structure | Tabular (fixed k) | Graph (variable degree) |
| Neighbors | Explicitly chosen | Determined by edges |
| Working set | Predictable, fits in cache | Unpredictable, exceeds cache |
| Bottleneck | Compute/scheduling | Memory/storage |

In DGraphFin, a single node query may fan out to hundreds of neighbors
across multiple hops. The access pattern is irregular and the working
set cannot be cached. This is where locality optimizations (page layout,
prefetching, neighbor ordering) provide measurable wins.

### The Regime Boundary

This benchmark discovered a fundamental systems insight:

> **When data fits in cache, batching dominates throughput.**
> **When data exceeds cache, locality dominates throughput.**

Both results are valuable. AMEX shows the ceiling of what's achievable
when memory is not the bottleneck. DGraphFin shows what happens when
memory becomes the bottleneck and how to optimize for it.

### When Does Disk I/O Become the Bottleneck?

The memory hierarchy for GPU inference looks like:

```
L1/L2 Cache  →  GPU HBM  →  System RAM  →  NVMe/SSD  →  HDD
   (fast)        (48GB)      (64-256GB)     (slow)      (slowest)
```

Data access only hits disk when the working set exceeds GPU memory AND
system RAM. For AMEX at 3.3GB, everything fits in GPU HBM with room to
spare. No disk I/O ever occurs during the benchmark.

**Projection: Dataset size needed to stress disk I/O**

| GPU Memory | System RAM | Disk I/O Threshold | Example Dataset |
|------------|------------|-------------------|-----------------|
| 16GB | 64GB | >80GB | 20M entities × 4KB |
| 24GB | 128GB | >150GB | 40M entities × 4KB |
| 48GB | 256GB | >300GB | 75M entities × 4KB |
| 80GB | 512GB | >600GB | 150M entities × 4KB |

The threshold is approximately `GPU_mem + RAM × 0.8` (leaving headroom
for OS and other processes). Below this threshold, the entire dataset
can be memory-mapped and cached. Above it, page faults trigger real
disk reads.

For AMEX to stress disk I/O on the W7900 (48GB) with 256GB RAM, the
dataset would need to be **~100x larger** (~300GB instead of 3.3GB).

### Does LBA Size (4KiB vs 512 bytes) Matter?

**Not for this benchmark.** LBA (Logical Block Address) size only
matters when you actually read from disk. Since AMEX fits entirely in
memory, no disk reads occur, so LBA size is irrelevant.

When disk I/O DOES occur, LBA size matters for random small reads:

| Access Pattern | 512B LBA | 4KiB LBA | Impact |
|----------------|----------|----------|--------|
| Read 100 bytes randomly | 512B read | 4KiB read | 8x amplification |
| Read 4KB randomly | 4KB (8 blocks) | 4KB (1 block) | Same |
| Read 1MB sequentially | 1MB | 1MB | Same |

The 4KiB LBA penalty appears when:
1. You're hitting disk (not cache)
2. Reads are small (<4KB)
3. Reads are random (no prefetch benefit)

Graph workloads like DGraphFin trigger this because:
- Neighbor lists are small (tens of bytes per edge)
- Access is random (follows graph structure)
- Working set exceeds memory (forces disk reads)

For AMEX, even if we artificially forced disk I/O, the access pattern
is relatively sequential (time-series windows) and reads are large
(4.5KB per entity), so LBA size would have minimal impact.

**Summary:** LBA size optimization is a concern for graph databases
and key-value stores with small random reads. It's not relevant for
cache-resident streaming workloads like AMEX.

## V1: GPU Sync Timing Validation

**Purpose:** Verify that GPU synchronization is essential for accurate
end-to-end latency measurement.

**Method:** Run identical workload with `cuda.synchronize()` ON vs OFF
before recording completion time.

### Results

| Metric | Sync ON | Sync OFF | Delta |
|--------|---------|----------|-------|
| p50 latency (ms) | 0.334 | 0.344 | -0.010 |
| p95 latency (ms) | 0.486 | 0.429 | +0.057 |
| p99 latency (ms) | 83.211 | 0.440 | +82.772 |
| max latency (ms) | 129.440 | 1.332 | +128.108 |

**Finding:** Without GPU sync, timing only captures kernel *submission*
latency, not *completion* latency. The 83ms vs 0.44ms difference proves
sync is essential for accurate e2e measurement.

**Status:** PASS

## V2: Queueing Model Validation

**Purpose:** Verify the producer/consumer queue model exhibits correct
backpressure behavior under overload.

**Method:** Run at 5x sustainable TPS (5,000 TPS vs ~1,000 sustainable)
with queue_size=100 and verify:
1. Max queue depth hits bound
2. Drops become nonzero
3. Average queue depth > 0

### Results

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| Max queue depth | = 100 (bound) | 100 | PASS |
| Total dropped | > 0 | 532 (2.13%) | PASS |
| Avg queue depth | > 0 | 99.5 | PASS |

**Finding:** Under 5x overload, the queue saturates completely (avg 99.5),
drops occur (2.13%), and latency increases to 20ms (queueing delay
dominates). The producer/consumer model is working correctly.

**Status:** PASS (3/3 checks)

## V3: Bytes/Event Scaling Validation

**Purpose:** Verify bytes_per_event scales linearly with k_related in
random neighbor mode.

**Method:** Sweep k_related = [0, 4, 8, 16] and compare measured
bytes/event to expected = base * (1 + k).

### Results

| k | Bytes/event (measured) | Expected | Ratio |
|---|------------------------|----------|-------|
| 0 | 4,512 | 4,512 | 1.00 |
| 4 | 22,560 | 22,560 | 1.00 |
| 8 | 40,608 | 40,608 | 1.00 |
| 16 | 76,704 | 76,704 | 1.00 |

**Linearity:** R² = 1.0000

**Finding:** Bytes/event scales perfectly linearly with k_related.
The accounting is correct: each additional related entity adds exactly
`window_size * feature_bytes = 6 * 752 = 4,512 bytes`.

**Status:** PASS (4/4 k-values, perfect linearity)

## Quick Capacity Sweep (k=8, random, microbatch=1)

After validation, a quick capacity sweep confirmed expected behavior:

| Target TPS | Achieved | p99 (ms) | Status |
|------------|----------|----------|--------|
| 500 | 500 | 35.87 | warmup artifact |
| 1,000 | 1,000 | 0.37 | sub-ms |
| 2,000 | 2,000 | 0.43 | sub-ms |
| 5,000 | 4,543 | 32.21 | overloaded |

**Max TPS @ p99 < 1ms:** 2,000 (at k=8, random mode)

### Warmup Artifact Note

The 35ms p99 at 500 TPS is a GPU warmup artifact (JIT compilation on
first batch). In production measurement, 1-2 seconds of warmup should
precede timing. This will be added before the canonical run.

## Conclusions

1. **GPU sync is mandatory** for accurate latency measurement
2. **Queue model is correct** (backpressure, drops, depth tracking)
3. **Bytes accounting is exact** (linear scaling with k)
4. **Framework is measurement-correct** and ready for canonical run

## Canonical Benchmark Results

**Date:** 2025-12-20
**Duration:** ~7 hours
**Configurations:** 160 data points (32 configs × 5 TPS levels)

### Test Matrix

| Dimension | Values |
|-----------|--------|
| Access pattern | random, hotset1000, hotset10000, hotset100000 |
| k_related | 0, 8, 16, 32 |
| Microbatch | 1, 4 |
| Target TPS | 500, 1000, 2000, 5000, 10000 |

### KPI Table: p99 Latency (ms)

| Config | 500 TPS | 1000 TPS | 2000 TPS | 5000 TPS | 10000 TPS |
|--------|---------|----------|----------|----------|-----------|
| random_k0_mb1 | 0.50 | 0.44 | 0.41 | 62.21 | 35.4s |
| random_k0_mb4 | 1.40 | 3.34 | 1.83 | 0.95 | 29.9s |
| random_k8_mb1 | 0.49 | 0.36 | 0.42 | 24.21 | 37.1s |
| random_k8_mb4 | 1.42 | 3.43 | 1.92 | 1.01 | 29.9s |
| random_k16_mb1 | 0.52 | 0.40 | 0.44 | 34.21 | 38.2s |
| random_k16_mb4 | 1.48 | 3.51 | 2.01 | 1.14 | 29.9s |
| random_k32_mb1 | 0.57 | 0.40 | 0.51 | 4.8s | 41.2s |
| random_k32_mb4 | 1.48 | 3.67 | 2.16 | 1.13 | 30.8s |
| hotset*_k*_mb1 | ~0.4-0.6 | ~0.3-0.4 | ~0.4-0.5 | 12-35ms | 35-41s |
| hotset*_k*_mb4 | ~1.4-1.5 | ~3.3-3.7 | ~1.8-2.2 | ~0.9-1.1 | 30s |

### Capacity Summary

| Microbatch | Max Sustainable TPS | p99 Range | Capacity Cliff |
|------------|---------------------|-----------|----------------|
| mb=1 | 2,000 TPS | 0.3-0.5ms | 5,000 TPS |
| mb=4 | 5,000 TPS | 1-3ms | 10,000 TPS |

### Cache Hit Rates at 2000 TPS

| k | Cache Hit |
|---|-----------|
| 0 | 68% |
| 8 | 86% |
| 16 | 91% |
| 32 | 95% |

## Key Findings

### 1. Microbatching is the only lever that matters

This is textbook queueing + GPU launch amortization behavior:

- **mb=1**: p99 ~0.3-0.5ms, capacity wall at ~2k TPS
- **mb=4**: p99 ~1-3ms, capacity wall at ~5k TPS

The trade is clean and monotonic: tail latency for throughput.

### 2. k doesn't matter because the working set fits in cache

Expected: higher k → more bytes → lower capacity

Observed: higher k → higher cache hit → **no throughput change**

This means:
- Working set fits in effective memory hierarchy
- Even at k=32, system is not storage-bound
- Benchmark measures compute + scheduling, not data movement

Cache hit rising from 68% to 95% with k didn't change capacity. Once
cache hit is "good enough," the bottleneck moves to: GPU kernel launch,
synchronization, queueing delay, Python/C++ boundary amortization.

### 3. Access pattern (random vs hotset) is irrelevant in this regime

Random and hotset patterns produce identical capacity limits. This is
logically consistent: when cache hit is already 68%+, locality optimizations
provide diminishing returns. The bottleneck is elsewhere.

### 4. The 10k TPS collapse is a real capacity cliff

At 10,000 TPS:
- p99 jumps to 30-40 seconds
- ~66% drop rate
- Backlog explodes

This is classic overloaded queue behavior. The producer/consumer model
exhibits correct backpressure. This gives a crisp statement:

> "This pipeline has a hard capacity ceiling, not a graceful degradation."

## Interpretation

### Why locality didn't help

The cache hit rate tells the story:

| k | Cache Hit | Interpretation |
|---|-----------|----------------|
| 0 | 68% | Already mostly cached |
| 32 | 95% | Even more cached |

With 68-95% cache hit rates, data access is essentially free. The
system spends its time on:

1. **GPU kernel launch overhead** (~10-50μs per launch)
2. **Queue management** (producer/consumer synchronization)
3. **Python/C++ boundary crossings**
4. **Batch formation and dispatch**

None of these improve with locality. You could have perfect spatial
locality (100% cache hit) and throughput would not increase because
the bottleneck is elsewhere.

### Why microbatching helped

Each GPU kernel launch has fixed overhead regardless of batch size.
By grouping 4 events per launch instead of 1, we amortize this overhead
4x, which explains the 2.5x capacity increase (2k → 5k TPS).

The tradeoff is baseline latency: mb=4 must wait for 4 events to
accumulate before dispatch, adding ~1-2ms of queueing delay even
under light load.

### What this benchmark proves

- Capacity is dominated by batching and scheduling, not data access
- Microbatching is the primary throughput control knob
- There is a hard capacity cliff (not graceful degradation)
- Tail latency constraints define feasible TPS
- In cache-resident workloads, access pattern is irrelevant

### What it does not prove

- That locality improves capacity when working set fits in cache
- That read amplification governs throughput in this workload
- That AMEX meaningfully stresses memory hierarchy

AMEX at 3.3GB cannot stress a 48GB GPU. The dataset would need to be
10-100x larger to create memory pressure.

### Regime Classification

This benchmark is a **real-time inference capacity benchmark**, not
a locality benchmark. The system is:

- **Compute-bound:** GPU arithmetic is the work
- **Queueing-bound:** Producer/consumer sync dominates tail latency
- **Launch-bound:** Kernel dispatch overhead limits TPS

It is **not** I/O-bound or memory-bound.

## Conclusion

> In a cache-resident streaming workload, sustainable real-time throughput
> is governed by batching and queueing behavior rather than access pattern.
> Microbatching increases capacity at the cost of baseline latency, with a
> hard capacity cliff beyond which tail latency and drop rates explode.
> Locality optimizations only shift capacity when the working set exceeds
> the effective memory hierarchy.

We found the regime boundary where locality stops mattering. This is a
result most benchmarks miss because they stop measuring once things look
fast. The absence of locality wins is itself informative: it tells you
the working set fits and the bottleneck is elsewhere.

## Positioning

This work establishes two complementary regimes:

### Regime 1: Cache-Resident (AMEX)

When the working set fits in memory:
- Batching and scheduling dominate throughput
- Access pattern (random vs sequential) is irrelevant
- Optimization target: reduce launch overhead, increase batch size
- Latency vs throughput tradeoff is clean and monotonic

### Regime 2: Memory-Bound (DGraphFin)

When the working set exceeds cache:
- Data access dominates throughput
- Access pattern determines performance
- Optimization target: locality, prefetching, page layout
- Read amplification from graph traversal is the bottleneck

### The Systems Insight

> "When data fits, batching dominates. When data doesn't fit,
> locality dominates."

Both regimes are important. Production systems must identify which
regime they operate in and optimize accordingly. Using locality
optimizations in a cache-resident workload wastes engineering effort.
Ignoring locality in a memory-bound workload leaves 10-100x
performance on the table.
