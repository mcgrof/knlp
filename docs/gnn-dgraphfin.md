# DGraphFin: Page-Aware GNN Training

**4x I/O efficiency with zero quality loss.**

[Back to GNN Fraud Overview](gnn-fraud.md) ·
[AMEX Benchmark](gnn-amex.md) ·
[Interactive Visualization](gnn_fraud_visualization.html) ·
[W&B Results](https://wandb.ai/mcgrof-citizen/gnn-fraud-fim-v7)

## Results (1-Hour Training)

| Sampler | F1 | PR-AUC | P@0.5% | R@1% | RA_fetch | RA_signal |
|---------|------|--------|--------|------|----------|-----------|
| NeighborSampler | 0.0757 | 0.0349 | 0.0566 | 0.0499 | 28.5x | 85.1x |
| **Page-Batch** | 0.0763 | 0.0346 | **0.0773** | **0.0559** | **6.8x** | 34.3x |
| FIM-Importance | 0.0769 | **0.0351** | 0.0729 | 0.0490 | **6.8x** | 36.3x |

## Key Findings

1. **4x locality improvement**: Page-aware batching achieves 6.8x RA_fetch
   vs 28.5x for neighbor sampling - 4x fewer page reads.

2. **Zero quality loss**: All three samplers achieve identical fraud detection
   quality (F1 ~0.076, PR-AUC ~0.035).

3. **FIM-guided adds no benefit**: FIM-importance performs identically to
   page-batch with random boundary selection. Added complexity, no gain.

4. **Training saturates fast**: 5 minutes = 1 hour results. The model learns
   what it can learn quickly.

## Recommendation

**Use page_batch for production.** Simpler, equally effective, 4x better I/O.

---

## Samplers

| Sampler | Description |
|---------|-------------|
| NeighborSampler | PyG NeighborLoader [10,5] - standard GNN baseline |
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

### Why This is a Memory-Bound Workload

DGraphFin has 3.7M nodes and 4.3M edges. Key properties:

- **Graph structure**: Variable-degree fan-out during neighbor sampling
- **Read amplification**: A single node query fans out to 10+5 neighbors
- **Working set**: Unpredictable, cannot be fully cached
- **Access pattern**: Random, determined by graph edges

This is where locality optimizations matter. METIS partitioning groups
related nodes on the same pages, reducing random I/O by 4x.

## Read Amplification

**RA_fetch** (locality metric):
```
RA_fetch = (pages_touched x PAGE_SIZE) / (loaded_nodes x FEATURE_BYTES)
```
Measures how scattered page accesses are. Use this to compare samplers.

**RA_signal** (training efficiency):
```
RA_signal = (pages_touched x PAGE_SIZE) / (supervised_nodes x FEATURE_BYTES)
```
Measures I/O cost per supervised gradient. Always >= RA_fetch.

### Why RA_signal has a ~4.3x floor

Even with perfect locality:
- Each page holds ~60 nodes
- Only ~23% of nodes are in train_mask
- So you supervise ~14 nodes per page
- Floor = 60/14 = **4.3x**

The measured 34.3x for page_batch includes boundary expansion overhead.
The 85.1x for neighbor_sampler combines poor locality with [10,5] fanout.

### Constants (DGraphFin)

- PAGE_SIZE = 4096 bytes
- FEATURE_BYTES = 68 bytes (17 float32 features)
- NODES_PER_PAGE = 60

## Reproduce

The workload is wired into the Kconfig build, so a checkout reproduces it
without manual setup:

```bash
make defconfig-gnn-dgraphfin
make
```

`make` prepares the Python dependencies (torch_geometric, pymetis) and the
C++ page samplers, fetches `dgraphfin.npz` from HuggingFace if it is
missing, builds the metis page layout, then runs the NeighborLoader
baseline against the Page-Aware sampler. A GPU is optional: the code falls
back to CPU, the ~250 MB feature matrix fits in RAM, and the
read-amplification metric is computed CPU-side regardless. `make
gnn-dry-run` prints the plan without downloading or training, and `make
gnn-setup` only prepares dependencies.

Read amplification in this path is the in-RAM *modeled* metric: the sampler
counts which 4 KiB pages a batch would touch (`bytes_read = pages_touched x
4096`) while the features actually live in DRAM. That is enough to compare
samplers, but it issues no block-device reads. To make the reads real and
externally verifiable, use the force-ssd variant below.

You can still run the scripts directly — for example the FIM ablation
harness:

```bash
python gnn/benchmark_fim.py --time 3600 --boundary-hops 2 --ablation
```

## Verifying read amplification with real I/O

`make defconfig-gnn-dgraphfin-force-ssd` answers a fair objection to the
numbers above: the read amplification is computed, not measured. The
force-ssd harness writes the feature matrix to a raw file on an SSD, laid
out by the page layout, and reads each batch's 4 KiB pages back with real
I/O. With `O_DIRECT` every logical page read becomes a device read
regardless of dataset size, so an operator can attach their own tooling —
eBPF [`biosnoop`](https://github.com/iovisor/bcc)/`biolatency`, `blktrace`,
or plain `iostat` — and confirm the per-I/O read intent and the resulting
amplification independently. We deliberately do not run eBPF in the repo;
the harness only produces the reads.

```bash
make defconfig-gnn-dgraphfin-force-ssd
# point the store at the device you want to observe:
#   make menuconfig -> GNN -> Force-SSD -> On-disk feature store directory
make
```

For each layout the harness replays two access patterns and reports the
measured device read amplification:

| Access | What it shows |
|--------|---------------|
| neighbor | locality *value*: RA_fetch drops from `natural` to `metis` |
| page | the *gap*: RA_signal stays near the ~4.3x floor for every layout |

The neighbor pattern expands random training seeds to their sampled
neighborhood, so a locality-aware layout keeps those neighbors on a few
pages and the device serves far fewer reads. The page pattern sweeps whole
training pages; there RA_fetch is ~1x for any layout, but only a fraction
of nodes per page are supervised, so RA_signal — bytes read per supervised
node — cannot drop below the ~4.3x floor no matter how good the layout is.
That floor is the gap the layout work leaves open, and it is exactly what
an external observer should be able to confirm with their own counters.

Results land in `results/gnn/force_ssd_ra.json` next to the printed table.
To force the working set past page cache without O_DIRECT, set
`CONFIG_GNN_FRAUD_SSD_INFLATE_GB` (replicates the store) or
`CONFIG_GNN_FRAUD_SSD_DROP_CACHES`.

## Files

- `gnn/run_gnn.py` — config-driven entry point (what `make` runs)
- `gnn/benchmark.py` — NeighborLoader vs Page-Aware (in-RAM)
- `gnn/benchmark_ssd.py` — force-ssd real-I/O layout comparison
- `gnn/ssd_feature_store.py` — on-disk feature store (O_DIRECT reads)
- `gnn/benchmark_fim.py` — FIM ablation harness
- `gnn/scripts/build_graph_layout.py` — layout builder (natural/random/bfs/metis)
- `gnn/scripts/setup_gnn_deps.py` — dependency and C++ extension setup
- `gnn/page_batch_sampler.py` — C++ accelerated page-aware sampler
