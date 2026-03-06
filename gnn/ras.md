# Read Amplification Metrics: RA_fetch vs RA_signal

## The Core Problem

When training GNNs on large graphs, node features are stored on disk/SSD in pages
(typically 4KB). Accessing a single node's features may require reading an entire
page. **Read Amplification (RA)** measures this I/O inefficiency.

## Storage Model

```
PAGE_SIZE = 4096 bytes
FEATURE_BYTES = 68 bytes per node (17 float32 features)
NODES_PER_PAGE ≈ 60 nodes fit in one 4KB page
```

Nodes are stored contiguously by their storage order (determined by METIS
partitioning). Nodes on the same METIS partition tend to be on the same or
adjacent pages.

## Two Different Questions, Two Different Metrics

### RA_fetch: "How scattered are my memory accesses?"

```
RA_fetch = bytes_read / bytes_needed_for_loaded_nodes
         = (pages_touched × PAGE_SIZE) / (loaded_nodes × FEATURE_BYTES)
```

- **Numerator**: Actual bytes read from storage (whole pages)
- **Denominator**: Bytes for ALL nodes whose features were gathered
- **Answers**: "How good is my memory locality?"
- **Ideal value**: 1.0x (all loaded nodes packed perfectly in pages)
- **Worst case**: ~60x (each node on a different page)

### RA_signal: "How much I/O per training signal?"

```
RA_signal = bytes_read / bytes_for_supervised_nodes
          = (pages_touched × PAGE_SIZE) / (supervised_nodes × FEATURE_BYTES)
```

- **Numerator**: Same - actual bytes read from storage
- **Denominator**: Bytes for ONLY nodes contributing to loss (training nodes)
- **Answers**: "How much I/O did I pay per supervised gradient?"
- **Always ≥ RA_fetch** (supervised_nodes ≤ loaded_nodes)

## Visual Concept

Imagine a batch where:
- You load 1000 nodes (for message passing)
- Only 200 of those have training labels (contribute to loss)
- Those 1000 nodes touch 50 unique pages

```
RA_fetch  = (50 × 4096) / (1000 × 68) = 204,800 / 68,000 = 3.0x
RA_signal = (50 × 4096) / (200 × 68)  = 204,800 / 13,600 = 15.1x
```

RA_fetch says "3x overhead for locality" - reasonable.
RA_signal says "15x overhead per training signal" - includes neighbor overhead.

## The Three Samplers Compared

### 1. NeighborSampler (PyG baseline)

```
Batch creation:
  1. Sample 1024 seed nodes (training nodes)
  2. For each seed, sample 10 neighbors (hop 1)
  3. For each hop-1 neighbor, sample 5 more (hop 2)

Result:
  - loaded_nodes: varies based on graph structure
  - supervised_nodes = 1024 (only seeds)
  - Pages touched: nodes scattered across graph
```

**Measured RA values (gnn-fraud-fim-v6):**
- RA_fetch = 28.2x
- RA_signal = 83.7x

**Visualization idea**: Show 1024 seed nodes (red) exploding into many nodes
scattered across pages. Most pages have only a few nodes accessed.

### 2. Page-Batch Sampler (our approach)

```
Batch creation:
  1. Select 32 contiguous METIS pages
  2. All ~1920 nodes in those pages are "core" nodes
  3. Add boundary nodes (20% budget, 2-hop expansion) for message passing

Result:
  - Core nodes: ~1920 (densely packed)
  - Boundary nodes: ~400 (scattered across graph)
  - Pages touched: 32 core + many boundary pages
```

**Measured RA values (gnn-fraud-fim-v6):**
- RA_fetch = 6.4x
- RA_signal = 33.8x

The 2-hop boundary expansion pulls nodes from many additional pages beyond
the 32 core pages, inflating RA_fetch above the theoretical ~1.0x.

**Visualization idea**: Show 32 contiguous pages (blue blocks), densely filled
with nodes. Boundary nodes (yellow) scattered across many additional pages.

### 3. FIM-Importance Sampler (our enhancement)

```
Batch creation:
  1. Select 32 contiguous METIS pages (same as page-batch)
  2. Add boundary nodes, but prioritize HIGH-IMPORTANCE nodes
     (nodes with high Fisher Information from backward pass)

Result:
  - Same structure as page-batch
  - Boundary nodes selected by importance rather than randomly
```

**Measured RA values (gnn-fraud-fim-v6):**
- RA_fetch = 6.9x
- RA_signal = 34.5x

Similar to page-batch. FIM selection doesn't improve locality but may
improve fraud detection by selecting more informative boundary nodes.

**Visualization idea**: Same as page-batch, but boundary nodes (yellow) have
importance scores shown as heat/intensity. High-importance boundary nodes
(bright yellow/orange) are fraud-adjacent nodes identified by FIM.

## Summary Table (Measured - gnn-fraud-fim-v6)

| Metric | NeighborSampler | Page-Batch | FIM-Importance |
|--------|-----------------|------------|----------------|
| **RA_fetch** | **28.2x** | **6.4x** | **6.9x** |
| **RA_signal** | **83.7x** | **33.8x** | **34.5x** |
| Memory locality | Poor | 4x better | 4x better |
| Boundary quality | Random | Random | FIM-guided |

## Key Insight

- **RA_fetch** measures **sampler locality** (apples-to-apples comparison)
- **RA_signal** measures **training efficiency** (includes label sparsity)

Page-batch achieves ~4x better locality than NeighborSampler (RA_fetch: 6.4x vs 28.2x).
The 2-hop boundary expansion prevents achieving theoretical ~1.0x locality.
FIM-importance has the same locality but smarter boundary selection for fraud detection.
