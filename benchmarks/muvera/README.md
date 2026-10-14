# muvera-small — reproducible byte-floor benchmark for retrieval vectors

This is the original memory-access benchmark and local fixed dimensional
encoding (FDE) approximation. It measures the relation between retrieval
record size and memory access cost. It grew from experiments motivated by
REFRAG and ColBERT. It is not Google's MUVERA implementation.

For the alternative that calls Google's public FDE code, run:

```bash
make defconfig-google-muvera-small && make google-muvera-small
```

The [Google-backed benchmark](../google_muvera/README.md) documents its
construction, metrics, dependencies, and next tests. The
[interactive page](../../docs/muvera_visualization.html) separates the two
implementations and labels the storage assumptions.

## Evidence and limits

The committed [sample](sample/sample_report.md) contains warm memory-mapped
reads, host-to-device transfers, and local-proxy retrieval results. The
cold-NVMe test was disabled. It does not establish observed NVMe read
amplification or a loss of the small-record advantage on an SSD.

The legacy FDE implementation sums both query and document buckets,
leaves empty buckets zero, and normalizes before scoring. Its quality
results do not validate the asymmetric construction in the MUVERA paper.
The sample does not show an FDE-over-BGE quality crossover: the best
listed FDE hit rate at 10 is 0.510, versus 0.857 for BGE. These are also
different embedding models, not just different encodings of one model.
The legacy CSV's `recall_at_10` and `recall_at_100` fields count whether any
relevant result is present. They therefore measure hit rate.

Quantized vectors are reconstructed as FP32 before scoring. Payload byte
estimates do not measure actual compressed-format search speed or process
memory. The historical CSVs and report remain unchanged for provenance;
the corrections here supersede their broader interpretation.

## Quick start

```
make defconfig-muvera-small
make muvera-small
make muvera-small-report
```

Outputs:

```
/data/results-archive/muvera-small-YYYYMMDD-HHMM/
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
`CONFIG_MUVERA_OUTPUT_BASE`. Default is `/data/results-archive` because
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

This legacy script assumes 4 KiB requests. It calculates a byte ratio
from that choice; it does not measure internal media reads. It also needs
a full-record read fix for records above 4 KiB. Revise it before using its
output to test the SSD hypothesis. A 4 KiB indirection unit can matter even
with 512-byte host logical blocks, but host traffic, device work, and
latency need separate measurements.

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
  - exact Chamfer produces valid hit-rate@10/100 results on SciFact
  - pooled int8 / fp16 baselines run
  - all plots are written
  - environment is captured

The optional NVMe CSV fields called `physical_read_bytes` and
`read_amplification` are based on the script's request-size assumption.
They are not a validation gate for measured device-level amplification.

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

Read the memory timings as measurements of the tested resident access
path. Treat the 4 KiB unit-to-record ratio as a model for an isolated
access, not an observed device result. A cold SSD experiment must verify
cache state, read complete records, and distinguish host bytes from media
bytes. The [next-test protocol](../google_muvera/README.md#tests-to-run-next)
sets out that experiment after a useful retrieval size has been found.

## License

SPDX-License-Identifier: MIT (matches the rest of knlp).
