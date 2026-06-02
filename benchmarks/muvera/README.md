# muvera-small — reproducible byte-floor benchmark for retrieval vectors

Measures the systems and quality envelope for sub-4096-byte retrieval
records. Built from Stage 5 of the REFRAG/MUVERA frontier work, hardened
into a standalone reproducible suite that anyone can run on a Linux box
with a single GPU.

An interactive walkthrough of the results — the cache byte floor, the NVMe
read-amplification story, the retrieval frontier, and how to reproduce it —
is at [`docs/muvera_visualization.html`](../../docs/muvera_visualization.html).

## What it measures

```
1. Cache-resident / mmap random reads improve materially below 4096 B.
2. Most random-access gain is captured around 256-512 B records.
3. GPU H2D transfer is overhead-bound below ~4096 B unless batch / fusion
   changes the result.
4. Cold NVMe / O_DIRECT reads do NOT preserve the sub-4096 advantage
   because the drive/page layer reads at block/page granularity.
   (Optional, gated by CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=y.)
5. Retrieval-quality side:
     - pooled int8 is a strong tiny-record baseline,
     - MUVERA/FDE with ColBERTv2 multi-vector queries beats pooled only
       at larger byte budgets,
     - int8 is effectively free in this setup,
     - PQ requires its own M/nbits/OPQ sweep and must not be assumed free.
```

## Strategic story

Sub-4096-byte retrieval records are real and useful for memory- /
cache-resident vector stores. But if the access path falls through to
raw NVMe random reads, the latency / read-amplification floor eats most
of the benefit. The storage stack needs packed layouts, batching,
caching, or near-storage filtering to actually exploit sub-page AI
retrieval records.

This is **not** a REFRAG warm-tier revival. It is a reproducible
infrastructure benchmark for retrieval vectors.

## Quick start

```
make defconfig-muvera-small
make muvera-small
make muvera-small-report
```

Outputs:

```
/data/knlp-key-results/muvera-small-YYYYMMDD-HHMM/
  config.json
  environment.txt
  git_commit.txt
  systems_byte_floor.csv
  retrieval_frontier.csv
  nvme_cold_byte_floor.csv      (only if CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=y)
  plots/
  report.md
```

The output directory path is configurable via
`CONFIG_MUVERA_OUTPUT_BASE`. Default is `/data/knlp-key-results` because
that's where Luis collects results, but `knlp` does **not** assume that
directory is a git tree. If you point `CONFIG_MUVERA_OUTPUT_BASE` at any
other writable directory, the benchmark works the same way.

## Optional NVMe cold benchmark

Disabled by default because it can consume large disk space and time.

```
# enable + point at a target NVMe mount
echo 'CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=y' >> .config
echo 'CONFIG_MUVERA_NVME_BENCH_PATH=/mnt/nvme/muvera_bench' >> .config
make muvera-small-nvme-cold
```

Demonstrates that sub-4096-byte logical records suffer ~4-32× read
amplification when the access path is raw NVMe random reads, regardless
of how cleverly the records were packed.

## Targets

```
make defconfig-muvera-small       # write .config from defconfig
make muvera-small-setup           # fetch dataset + encode embeddings
make muvera-small-systems         # systems byte-floor microbench (mmap, H2D, projection)
make muvera-small-nvme-cold       # OPTIONAL NVMe cold/O_DIRECT benchmark
make muvera-small-retrieval       # MUVERA FDE + pooled retrieval benchmark
make muvera-small-plots           # generate plots
make muvera-small-report          # generate the markdown report
make muvera-small                 # all of: setup → systems → retrieval → plots → report
```

## Validation gates

The benchmark is considered valid if:

  - all configured record sizes produce results in the systems CSV
  - all configured FDE points produce results in the retrieval CSV
  - chamfer oracle produces a valid recall@10/100 on scifact
  - pooled int8 / fp16 baselines run
  - all plots are written
  - environment is captured

If `CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=y`, additionally:

  - the NVMe cold CSV reports physical_read_bytes and read_amplification

## Dependencies

```
sentence_transformers
pylate                   # ColBERTv2 multi-vector encoder
faiss                    # PQ compression
datasets                 # BEIR/scifact loader
torch + torchvision      # ROCm or CUDA
matplotlib               # plots
numpy
psutil
```

`pylate` will pull in its own dependencies (transformers, sentencepiece,
etc.) on first install.

## Reading the report

The report emphasizes the contrast between memory/cache-resident
random-access (where sub-4096 wins) and raw NVMe random access (where
sub-4096 is erased by 4 KiB page reads). Both worlds are measured
explicitly when the NVMe benchmark is enabled.

The retrieval result emphasizes that pooled int8 at 128 B is a strong
small-byte baseline; FDE earns its complexity only above ~5 KiB on
scifact + ColBERTv2. The recipe for production retrieval is FDE for
fast top-100 candidate generation, chamfer for rerank.

## License

SPDX-License-Identifier: MIT (matches the rest of knlp).
