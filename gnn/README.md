# GNN Fraud Detection

Page-Aware Mini-Batch Training for GNN-based fraud detection. Reduces
read amplification by 10x while achieving superior fraud detection
metrics compared to traditional neighbor sampling.

## Quick Start

```bash
# Build layout (one-time)
python scripts/build_graph_layout.py \
    --data-dir ../data \
    --method metis-bfs \
    --output layout_metis_bfs.npz

# Run comparison (1 hour each method)
python benchmark.py --time 3600

# Quick test (5 minutes)
python benchmark.py --time 300 --only-pageaware
```

## Results (DGraphFin)

| Metric | NeighborLoader | Page-Aware |
|--------|----------------|------------|
| Read Amp | 58.7x | 5.17x |
| Test F1 | 0.054 | 0.073 |
| PR-AUC | 0.025 | 0.033 |

See `../docs/gnn-fraud.md` for full documentation.

## Key Files

- `benchmark.py`: Main benchmark comparing methods
- `page_batch_sampler.py`: Page-aware mini-batch sampler
- `scripts/build_graph_layout.py`: METIS+BFS layout builder
- `scripts/transaction_motif_generator.py`: Synthetic graph generator
