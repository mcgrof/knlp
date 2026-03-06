# FIM-Guided Fraud Detection: State of Affairs

Last updated: 2025-12-17

## Latest Results (gnn-fraud-fim-v6)

W&B project: https://wandb.ai/mcgrof-citizen/gnn-fraud-fim-v6

Run command:
```bash
python benchmark_fim.py --time 300 --boundary-hops 2 --ablation --wandb-project gnn-fraud-fim-v6
```

### Ablation Results (300s, boundary-hops=2)

| Metric | neighbor_sampler | page_batch | fim_importance |
|--------|------------------|------------|----------------|
| **F1** | 0.0761 | **0.0784** | 0.0767 |
| **PR-AUC** | **0.0363** | 0.0347 | 0.0346 |
| **P@0.5%** | 0.0696 | 0.0686 | **0.0707** |
| **R@1%** | 0.0507 | **0.0542** | 0.0529 |
| **RA_fetch** | 28.2x | **6.4x** | 6.9x |
| **RA_signal** | 83.7x | **33.8x** | 34.5x |

W&B runs:
- neighbor_sampler: https://wandb.ai/mcgrof-citizen/gnn-fraud-fim-v6/runs/fk3b5bvh
- page_batch: https://wandb.ai/mcgrof-citizen/gnn-fraud-fim-v6/runs/sijnwvfq
- fim_importance: https://wandb.ai/mcgrof-citizen/gnn-fraud-fim-v6/runs/gqghsksb

### Key Findings

1. **4x locality improvement**: page_batch has ~4x better locality than
   neighbor_sampler (6.4x vs 28.2x RA_fetch).

2. **Fraud metrics similar**: All three samplers achieve comparable F1/PR-AUC.
   page_batch leads on F1 (0.078) and R@1% (0.054).

3. **RA_fetch higher than theoretical**: Expected ~1.0-1.2x for page-aware
   sampling, but measured 6.4x. The 2-hop boundary expansion pulls nodes
   from ~150-200 pages beyond the 32 core pages.

---

## Variants

| Variant | Description |
|---------|-------------|
| **neighbor_sampler** | PyG NeighborLoader [10, 5] - standard GNN baseline |
| **page_batch** | Page-aware batching + random boundary selection |
| **fim_importance** | Page-aware batching + FIM-guided boundary selection |

---

## DGraphFin Dataset

Binary fraud classification with background nodes for message passing:

| Class | Count | % | Role |
|-------|-------|---|------|
| 0 | 1,210,092 | 32.7% | Normal (LABELED) |
| 1 | 15,509 | 0.42% | Fraud (LABELED) |
| 2 | 1,620,851 | 43.8% | Background (UNLABELED) |
| 3 | 854,098 | 23.1% | Background (UNLABELED) |

- Only classes 0 and 1 appear in train/val/test masks
- Classes 2 & 3 participate in message passing but are NEVER evaluated
- Train split: 847,042 normal + 10,857 fraud (1.27% positive rate)
- Official DGraphFin baseline explicitly sets `nlabels = 2`

---

## Read Amplification (RA) Metrics

See `ras.md` for detailed explanation with visualizations.

**Constants:**
- `PAGE_SIZE = 4096` bytes
- `FEATURE_BYTES = 68` bytes per node (17 float32 features)
- `NODES_PER_PAGE ≈ 60` nodes per 4KB page

**RA_fetch (locality metric):**
```
RA_fetch = (pages_touched × PAGE_SIZE) / (loaded_nodes × FEATURE_BYTES)
```
- Answers: "How scattered were my feature accesses?"
- Perfect locality = ~1.0x, random scatter = ~60x

**RA_signal (training efficiency metric):**
```
RA_signal = (pages_touched × PAGE_SIZE) / (supervised_nodes × FEATURE_BYTES)
```
- Answers: "How much I/O per supervised gradient?"
- Inflated by label sparsity (only 23% of nodes in train mask)

### Measured vs Expected RA

| Sampler | Expected RA_fetch | Measured RA_fetch | Notes |
|---------|-------------------|-------------------|-------|
| neighbor_sampler | ~50-60x | 28.2x | Better than random due to graph locality |
| page_batch | ~1.0-1.2x | 6.4x | Boundary expansion adds page scatter |
| fim_importance | ~1.0-1.2x | 6.9x | Same as page_batch |

---

## Available Flags

```
--time N              Training time in seconds
--boundary-hops {1,2} Hops for boundary expansion (use 2 for 2-layer GNN)
--boundary-budget F   Fraction of boundary nodes to include (default 0.2)
--weighted-loss       Enable class-weighted loss for imbalance
--pos-weight-scale F  Additional scale for fraud class weight
--pos-oversample F    Probability of injecting positive-containing page (0-1)
--ablation            Run all three variants
--only {neighbor_sampler,page_batch,fim_importance}
--no-wandb            Disable W&B logging
```

---

## File Locations

- Main benchmark: `/data/knlp/gnn/benchmark_fim.py`
- FIM sampler: `/data/knlp/gnn/fim_sampler.py`
- FIM importance: `/data/knlp/gnn/fim_importance.py`
- Page batch sampler: `/data/knlp/gnn/page_batch_sampler.py`
- RA metrics explanation: `/data/knlp/gnn/ras.md`
- Data: `../dgraphfin.npz`
- Layout: `../layout_metis_bfs.npz`
- DGraphFin baseline: `/data/DGraphFin_baseline/`
