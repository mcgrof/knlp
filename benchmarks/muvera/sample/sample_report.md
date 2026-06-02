# muvera-small benchmark report

Run directory: `/data/knlp-key-results/muvera-small-20260502-1408`

## A. Executive summary

Sub-4096-byte retrieval records are real and useful for
memory- / cache-resident vector stores. Raw NVMe random reads,
when measured cold or via O_DIRECT, suffer 4 KiB-page-read
amplification regardless of the logical record size, and the
sub-page advantage shrinks or disappears.

On the retrieval-quality side, pooled int8 is a strong tiny-
record baseline. MUVERA-style FDE compression of multi-vector
retrieval (ColBERTv2) earns its complexity above ~5 KiB byte
budgets on BEIR/scifact; below that, pooled int8 is hard to
beat. int8 quantization is effectively free across pooled and
FDE on this benchmark; PQ-256-8 has dim/dataset-dependent
behavior and is not a universal free lunch.

## B. Hardware and environment

```
hostname: prune
timestamp: 2026-05-02 14:08:31 PDT
uname: Linux prune 6.16.8+deb14-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.16.8-1 (2025-09-21) x86_64 GNU/Linux
cpu: AMD Ryzen 9 9950X3D 16-Core Processor
memory:
  total        used        free      shared  buff/cache   available
  Mem:            91Gi        36Gi        27Gi        48Mi        28Gi        55Gi
  Swap:           15Gi        10Gi       5.3Gi
torch: 2.9.1+rocm6.4
cuda available: True
  device 0 name: AMD Radeon Pro W7900
  sentence_transformers: 5.1.1
  pylate: 1.4.0
  faiss: 1.12.0
  datasets: 3.6.0
  transformers: 4.56.2
  numpy: 2.1.3
  matplotlib: 3.10.5
  psutil: 7.0.0
```

```
commit: 44ff10778ce29cdb6ed493195e8654d69014403e
branch: main
```

## C. Systems byte-floor result

Memory/cache-resident view:

| record bytes | mmap rand p50 (ns) | mmap rand p99 (ns) | H2D x64 p50 (μs) | proj x64 p50 (μs) |
|------:|---------:|---------:|--------:|--------:|
| 32 | 150 | 1100 | 27.4 | 30.3 |
| 64 | 140 | 1160 | 27.6 | 29.2 |
| 128 | 190 | 1370 | 28.8 | 31.4 |
| 256 | 630 | 1560 | 29.1 | 31.9 |
| 512 | 840 | 1730 | 21.2 | 32.0 |
| 1024 | 1000 | 1930 | 21.3 | 35.0 |
| 2048 | 1220 | 2130 | 29.6 | 35.7 |
| 4096 | 1400 | 2140 | 26.2 | 45.4 |
| 8192 | 1640 | 2420 | 42.9 | 70.7 |
| 16384 | 1970 | 3540 | 59.2 | 122.0 |

Plots in `plots/`:
- record_bytes_vs_mmap_random_p50.png
- record_bytes_vs_mmap_random_p95.png
- record_bytes_vs_h2d_batch64.png
- record_bytes_vs_total_serving_estimate.png

## D. Optional NVMe cold/O_DIRECT result

Not run (CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=n).

## E. Retrieval result

| variant | R | k_sim | FDE dim | precision | bytes/vec | recall@10 | recall@100 | lat μs/q |
|---------|--:|------:|--------:|----------|---------:|---------:|----------:|---------:|
| chamfer_oracle | - | - | - | fp32 | 120654 | 0.807 | 0.927 | 321050.0 |
| pooled-fp32 | - | - | 384 | fp32 | 1536 | 0.857 | 0.953 | 54.7 |
| pooled-fp16 | - | - | 384 | fp16 | 768 | 0.857 | 0.953 | 14.7 |
| pooled-int8 | - | - | 384 | int8 | 384 | 0.857 | 0.950 | 16.8 |
| FDE | 1 | 2 | 512 | fp32 | 2048 | 0.393 | 0.603 | 21.5 |
| FDE | 1 | 2 | 512 | fp16 | 1024 | 0.393 | 0.603 | 18.5 |
| FDE | 1 | 2 | 512 | int8 | 512 | 0.393 | 0.603 | 16.8 |
| FDE | 1 | 2 | 512 | pq | 256 | 0.387 | 0.607 | 41.5 |
| FDE | 1 | 3 | 1024 | fp32 | 4096 | 0.370 | 0.617 | 24.2 |
| FDE | 1 | 3 | 1024 | fp16 | 2048 | 0.370 | 0.617 | 21.8 |
| FDE | 1 | 3 | 1024 | int8 | 1024 | 0.367 | 0.620 | 20.7 |
| FDE | 1 | 3 | 1024 | pq | 256 | 0.350 | 0.610 | 36.5 |
| FDE | 1 | 4 | 2048 | fp32 | 8192 | 0.257 | 0.530 | 29.8 |
| FDE | 1 | 4 | 2048 | fp16 | 4096 | 0.257 | 0.530 | 25.4 |
| FDE | 1 | 4 | 2048 | int8 | 2048 | 0.263 | 0.530 | 25.8 |
| FDE | 1 | 4 | 2048 | pq | 256 | 0.213 | 0.447 | 70.3 |
| FDE | 5 | 2 | 2560 | fp32 | 10240 | 0.447 | 0.683 | 33.7 |
| FDE | 5 | 2 | 2560 | fp16 | 5120 | 0.447 | 0.683 | 27.6 |
| FDE | 5 | 2 | 2560 | int8 | 2560 | 0.447 | 0.683 | 28.0 |
| FDE | 5 | 2 | 2560 | pq | 256 | 0.337 | 0.603 | 78.3 |
| FDE | 5 | 3 | 5120 | fp32 | 20480 | 0.463 | 0.753 | 46.8 |
| FDE | 5 | 3 | 5120 | fp16 | 10240 | 0.463 | 0.753 | 42.0 |
| FDE | 5 | 3 | 5120 | int8 | 5120 | 0.460 | 0.753 | 40.0 |
| FDE | 5 | 3 | 5120 | pq | 256 | 0.330 | 0.610 | 61.2 |
| FDE | 5 | 4 | 10240 | fp32 | 40960 | 0.443 | 0.730 | 73.8 |
| FDE | 5 | 4 | 10240 | fp16 | 20480 | 0.443 | 0.730 | 62.7 |
| FDE | 5 | 4 | 10240 | int8 | 10240 | 0.447 | 0.730 | 63.3 |
| FDE | 5 | 4 | 10240 | pq | 256 | 0.250 | 0.543 | 272.5 |
| FDE | 10 | 2 | 5120 | fp32 | 20480 | 0.447 | 0.680 | 46.2 |
| FDE | 10 | 2 | 5120 | fp16 | 10240 | 0.447 | 0.680 | 41.5 |
| FDE | 10 | 2 | 5120 | int8 | 5120 | 0.447 | 0.677 | 40.4 |
| FDE | 10 | 2 | 5120 | pq | 256 | 0.273 | 0.547 | 60.1 |
| FDE | 10 | 3 | 10240 | fp32 | 40960 | 0.503 | 0.747 | 68.0 |
| FDE | 10 | 3 | 10240 | fp16 | 20480 | 0.503 | 0.747 | 63.8 |
| FDE | 10 | 3 | 10240 | int8 | 10240 | 0.503 | 0.743 | 63.7 |
| FDE | 10 | 3 | 10240 | pq | 256 | 0.253 | 0.563 | 78.5 |
| FDE | 10 | 4 | 20480 | fp32 | 81920 | 0.510 | 0.763 | 103.2 |
| FDE | 10 | 4 | 20480 | fp16 | 40960 | 0.510 | 0.763 | 107.2 |
| FDE | 10 | 4 | 20480 | int8 | 20480 | 0.510 | 0.763 | 106.3 |
| FDE | 10 | 4 | 20480 | pq | 256 | 0.263 | 0.530 | 144.9 |

Plots in `plots/`:
- bytes_vs_recall10.png
- bytes_vs_recall100.png
- latency_vs_recall10.png

## F. Interpretation

- Memory/cache-resident sub-4096 records help random access.
  Most of the win is captured by ~256-512 bytes; further
  shrinking buys little.
- CPU→GPU H2D transfer is overhead-bound below ~4 KiB total
  batch payload. Smaller records do not save H2D time below
  that threshold.
- GPU projection cost scales linearly with input dim. Smaller
  stored bytes save real GPU compute, but the constant matters.
- If the access path is raw NVMe random reads (cold / O_DIRECT),
  the 4 KiB page boundary erases the sub-4096 logical-record
  advantage. Sub-page records suffer read amplification.
- The storage stack must do something — packed layouts,
  batching, caching, near-storage filtering — to exploit
  sub-page records on NVMe.
- int8 retrieval vectors are a serious default. Quantization
  is free across pooled and FDE in this benchmark.
- MUVERA/FDE earns its complexity at multi-KiB byte budgets;
  pooled int8 is the tiny-byte baseline.
- PQ-256-8 has dim/dataset-dependent behavior and must not
  be assumed free; production deployment should sweep
  M, nbits, and consider OPQ.

## G. Reproduction

```
make defconfig-muvera-small
make muvera-small
make muvera-small-report
```

Optional NVMe cold benchmark:
```
# enable in .config
CONFIG_MUVERA_ENABLE_NVME_COLD_BENCH=y
CONFIG_MUVERA_NVME_BENCH_PATH=/path/to/nvme
make muvera-small-nvme-cold
```

## H. Next tests

- Run NVMe cold benchmark on different drives and filesystems
  to see how page granularity and queue depth affect the
  read-amplification floor.
- Test GPUDirect Storage if available; H2D may be much
  cheaper.
- Sweep PQ M, nbits, and OPQ on the FDE retrieval frontier.
- Run on larger BEIR datasets (NQ, HotpotQA, MS MARCO) to
  see how recall absolute numbers and the FDE-vs-pooled
  crossover scale.
