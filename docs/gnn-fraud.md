# Page-Aware GNN Training for Fraud Detection

**4× I/O efficiency with zero quality loss.**

[Interactive Visualization](gnn_fraud_visualization.html) ·
[W&B Results](https://wandb.ai/mcgrof-citizen/gnn-fraud-fim-v7)

## Results (1-Hour Training)

| Sampler | F1 | PR-AUC | P@0.5% | R@1% | RA_fetch | RA_signal |
|---------|------|--------|--------|------|----------|-----------|
| NeighborSampler | 0.0757 | 0.0349 | 0.0566 | 0.0499 | 28.5× | 85.1× |
| **Page-Batch** | 0.0763 | 0.0346 | **0.0773** | **0.0559** | **6.8×** | 34.3× |
| FIM-Importance | 0.0769 | **0.0351** | 0.0729 | 0.0490 | **6.8×** | 36.3× |

## Key Findings

1. **4× locality improvement**: Page-aware batching achieves 6.8× RA_fetch
   vs 28.5× for neighbor sampling — 4× fewer page reads.

2. **Zero quality loss**: All three samplers achieve identical fraud detection
   quality (F1 ~0.076, PR-AUC ~0.035).

3. **FIM-guided adds no benefit**: FIM-importance performs identically to
   page-batch with random boundary selection. Added complexity, no gain.

4. **Training saturates fast**: 5 minutes ≈ 1 hour results. The model learns
   what it can learn quickly.

## Recommendation

**Use page_batch for production.** Simpler, equally effective, 4× better I/O.

---

## Samplers

| Sampler | Description |
|---------|-------------|
| NeighborSampler | PyG NeighborLoader [10,5] — standard GNN baseline |
| Page-Batch | Page-aware batching + random boundary selection |
| FIM-Importance | Page-aware + FIM-guided boundary selection |

## DGraphFin Dataset

**Binary classification** (num_classes=2): normal vs fraud.

| Label | Count | Role |
|-------|-------|------|
| 0 | 1,210,092 (32.7%) | Normal |
| 1 | 15,509 (0.42%) | Fraud |
| 2 | 1,620,851 (43.8%) | Background (unlabeled) |
| 3 | 854,098 (23.1%) | Background (unlabeled) |

Labels 2/3 are background nodes for message passing only. They never appear
in train/val/test masks. The model outputs 2 logits, not 4.

Train split: 847,042 normal + 10,857 fraud (1.27% positive rate).

## Read Amplification

**RA_fetch** (locality metric):
```
RA_fetch = (pages_touched × PAGE_SIZE) / (loaded_nodes × FEATURE_BYTES)
```
Measures how scattered page accesses are. Use this to compare samplers.

**RA_signal** (training efficiency):
```
RA_signal = (pages_touched × PAGE_SIZE) / (supervised_nodes × FEATURE_BYTES)
```
Measures I/O cost per supervised gradient. Always ≥ RA_fetch.

### Why RA_signal has a ~4.3× floor

Even with perfect locality:
- Each page holds ~60 nodes
- Only ~23% of nodes are in train_mask
- So you supervise ~14 nodes per page
- Floor = 60/14 = **4.3×**

The measured 34.3× for page_batch includes boundary expansion overhead.
The 85.1× for neighbor_sampler combines poor locality with [10,5] fanout.

### Constants (DGraphFin)

- PAGE_SIZE = 4096 bytes
- FEATURE_BYTES = 68 bytes (17 float32 features)
- NODES_PER_PAGE = 60

## Usage

```bash
# Run ablation (all three samplers)
python gnn/benchmark_fim.py --time 3600 --boundary-hops 2 --ablation \
  --wandb-project gnn-fraud-fim

# Run single sampler
python gnn/benchmark_fim.py --time 3600 --boundary-hops 2 --only page_batch

# With class weighting
python gnn/benchmark_fim.py --time 3600 --boundary-hops 2 --weighted-loss
```

## Files

- `gnn/benchmark_fim.py` — Main benchmark script
- `gnn/fim_sampler.py` — FIM-guided sampler
- `gnn/fim_importance.py` — Backward hook importance tracking
- `gnn/page_batch_sampler.py` — C++ accelerated page-aware sampler
- `docs/gnn_fraud_visualization.html` — Interactive visualization
