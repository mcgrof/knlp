# GNN Fraud Detection Benchmarks

**Two regimes, two optimization strategies.**

[Interactive Visualization](gnn_fraud_visualization.html)

## The Systems Insight

> **When data fits in cache, batching dominates throughput.**
> **When data exceeds cache, locality dominates throughput.**

We benchmark both regimes to understand when each optimization matters.

## Two Benchmarks

| Benchmark | Dataset | Working Set | Bottleneck | Key Metric |
|-----------|---------|-------------|------------|------------|
| [AMEX](gnn-amex.md) | 3.3GB | Fits in GPU | Compute/scheduling | TPS capacity |
| [DGraphFin](gnn-dgraphfin.md) | Graph 3.7M nodes | Exceeds cache | Memory/storage | Read amplification |

### AMEX: Cache-Resident Streaming

The [AMEX benchmark](gnn-amex.md) measures real-time inference capacity for
fraud detection when the entire dataset fits in GPU memory.

**Key findings:**
- Microbatch size is the only lever (mb=1: 2k TPS, mb=4: 5k TPS)
- Access pattern (random vs sequential) is irrelevant
- Hard capacity cliff at 10k TPS with 66% drop rate
- Locality optimizations provide no benefit

**When to use:** Production systems where working set fits in memory.
Optimize for batch size and kernel launch overhead.

### DGraphFin: Memory-Bound Graph Training

The [DGraphFin benchmark](gnn-dgraphfin.md) measures I/O efficiency for
GNN training when neighbor sampling creates unpredictable access patterns.

**Key findings:**
- Page-aware batching achieves 4x fewer page reads (28.5x -> 6.8x RA)
- Zero quality loss compared to standard neighbor sampling
- METIS partitioning groups related nodes on same pages
- FIM-guided sampling adds no benefit over random boundary selection

**When to use:** Graph workloads where neighbor sampling fans out across
the dataset. Optimize for page locality and prefetching.

## Regime Classification

### Cache-Resident (AMEX)

- Dataset: 3.3GB (fits in 48GB GPU)
- Cache hit: 68-95%
- Bottleneck: GPU kernel launch, queueing
- Optimization: Increase microbatch size

### Memory-Bound (DGraphFin)

- Dataset: Graph with 3.7M nodes, 4.3M edges
- Cache hit: Variable (depends on traversal)
- Bottleneck: Page faults, random I/O
- Optimization: METIS layout, page-aware batching

## Production Guidance

1. **Measure your regime first**: Check cache hit rates and identify bottleneck
2. **Don't over-optimize**: Locality work in cache-resident regime is wasted
3. **Don't under-optimize**: Ignoring locality in memory-bound regime costs 4-10x
4. **Batch size matters**: Even in memory-bound regimes, batching amortizes overhead

## Files

- [docs/gnn-amex.md](gnn-amex.md) - AMEX streaming benchmark
- [docs/gnn-dgraphfin.md](gnn-dgraphfin.md) - DGraphFin page-aware training
- [docs/gnn_fraud_visualization.html](gnn_fraud_visualization.html) - Interactive visualization
- `gnn/benchmark_fim.py` - DGraphFin benchmark script
- `gnn/amex_benchmark.py` - AMEX streaming benchmark script
