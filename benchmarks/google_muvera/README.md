# Google MUVERA small-corpus benchmark

This target measures how well Google's public fixed dimensional encoding
(FDE) implementation preserves ColBERTv2 retrieval results as its dimension
changes. It shares the SciFact dataset and embedding preparation with
[`muvera-small`](../muvera/README.md). It adds an independent target so the
original local approximation remains available for comparison.

```bash
make defconfig-google-muvera-small && make google-muvera-small
```

The new target does not yet have a recorded SciFact quality result. The
[interactive page](../../docs/muvera_visualization.html) labels the earlier
memory measurements and local-proxy quality results separately. They do
not validate this implementation or establish a dimension requirement.

## Contents

- [Build and first run](#build-and-first-run)
- [What the target measures](#what-the-target-measures)
- [Tests to run next](#tests-to-run-next)
- [Paper scope](#paper-scope)

## Build and first run

The FDE adapter compiles a pinned, unmodified subset of Google's
[graph-mining source](https://github.com/google/graph-mining/tree/main/sketching/point_cloud).
The source files, Apache-2.0 license, and integrity manifest are under
[`upstream/`](upstream/). The scoped build needs Python 3.10 or later,
CMake 3.24 or later, a C++20 compiler, a build tool such as Make, and network access for the
pinned Abseil, Protobuf, and Eigen dependencies. It builds under
`.cache/google-muvera`; it does not install system packages. The first
build downloads and compiles dependencies. Later builds reuse the cache.
The archive extractor validates paths and allows only regular files and
directories. It also applies Python’s data extraction filter when
available; it supports Python 3.10 releases without that filter.

```bash
make defconfig-google-muvera-small
make google-muvera-small-build
make google-muvera-small-test
make google-muvera-small-smoke
```

The tests and synthetic smoke run need NumPy, but no model
weights or GPU. They check the adapter and evaluation machinery. Passing
a synthetic fixture is not evidence of real retrieval quality.

The full run also needs PyTorch, PyLate, sentence-transformers, datasets,
NumPy, and the matplotlib/psutil dependencies of the shared systems
benchmark. Use a PyTorch build for the installed CUDA or ROCm stack.
One existing GPU is the starting point for embedding preparation and
host-to-device measurements; FDE construction itself runs on the CPU.

```bash
make google-muvera-small
```

The full target creates a new `google-muvera-small-*` directory under
`CONFIG_MUVERA_OUTPUT_BASE`, which can be any writable directory. It runs
setup, retrieval, memory measurements, and report generation. The default
path is `/data/results-archive`. The target does not run a cold-NVMe test.

Individual targets let a worker resume a named run without mixing results:

```bash
make google-muvera-small-setup GOOGLE_MUVERA_RUN_DIR=/path/to/run
make google-muvera-small-retrieval GOOGLE_MUVERA_RUN_DIR=/path/to/run
make google-muvera-small-systems GOOGLE_MUVERA_RUN_DIR=/path/to/run
make google-muvera-small-report GOOGLE_MUVERA_RUN_DIR=/path/to/run
```

To reuse an earlier embedding run, set
`CONFIG_GOOGLE_MUVERA_EMBEDDING_CACHE="/path/to/prior/run"` in `.config`.
Use the same ColBERT cache for exact Chamfer and every FDE setting. Check
model identity, document/query IDs, and cached token-vector counts before
comparing runs. A cache from a different model or corpus is not a matched
baseline.

## What the target measures

Google's query and document APIs perform different constructions. The
wrapper calls those APIs, keeps the document empty-partition option
explicit, and uses FDE inner products without cosine normalization. The
harness compares the candidates against exhaustive Chamfer scoring on the
same token vectors. It then reranks the FDE candidates from the precomputed
exact-score table. This checks ranking quality; it does not time an online
reranking implementation. The BGE pooled baseline uses a different model
and is reported separately.

The default sweep is deliberately small:

| Setting | Default | Meaning |
|---|---|---|
| `CONFIG_GOOGLE_MUVERA_FDE_CONFIGS` | `"5:4:16 20:4:8 20:5:16"` | Repetitions : hash bits : projected coordinates per bucket |
| `CONFIG_GOOGLE_MUVERA_SEEDS` | `"1"` | Base construction seeds; use nonoverlapping repetition ranges for the follow-up |
| `CONFIG_GOOGLE_MUVERA_FILL_EMPTY` | `y` | Fill empty document partitions with the official implementation |
| `CONFIG_GOOGLE_MUVERA_FINAL_DIM` | `0` | Optional final projection; zero disables it |
| `CONFIG_GOOGLE_MUVERA_FORMATS` | `"fp32 fp16 int8"` | Representation formats for the quality comparison |
| `CONFIG_GOOGLE_MUVERA_CANDIDATES` | `"10 100"` | Candidate counts before exact reranking |
| `CONFIG_GOOGLE_MUVERA_LENGTH_BINS` | `"32 64 128 256 512 1024"` | Boundaries in retained document token vectors |

Before a final projection, the dimension is `D = R * 2^k * p`. The default
settings produce 1,280, 2,560, and 10,240 coordinates. An FP32 payload uses
`4D` bytes. Original token vectors remain necessary for exact reranking;
the FDE payload alone is not the total index cost.

The outputs include `metrics.csv`, `metrics.json`,
`length-bucket_fidelity.csv`, and `report.md`, along with run provenance.
The current length table groups queries by the retained vector count of
each query’s exact top-1 document, then averages top-10 candidate recovery
within each group. It keeps retrieval over the full corpus. This is a
first diagnostic; it does not assign a required dimension to each document.
Keep these files with the recorded source revision and configuration.
Neighbor recovery against exact Chamfer is a fidelity metric. Relevance
recall, hit rate, and nDCG answer a different question. In particular,
hit rate is the fraction of queries with any relevant result; it is not
recall when a query has multiple relevant documents.

FP16 and INT8 quality tests reconstruct vectors for scoring. Their byte
counts describe the stored payload, including scale overhead where
reported. Their search times do not measure native compressed-format
kernels. Product quantization (PQ) is a follow-up, not part of this default
sweep. Exhaustive FDE scoring avoids approximate-index recall as a
confounder, but is not the paper's complete DiskANN pipeline and cannot
reproduce its search-latency claims.

## Tests to run next

First run the build, test, and synthetic smoke commands above. A failure
blocks interpretation of quality results. Then run the full default
SciFact configuration once to check the dataset path, output schema,
finite scores, ID alignment, and candidate/rerank consistency. Keep the
same immutable embedding cache for the next sweep.

For the first fidelity sweep, set
`CONFIG_GOOGLE_MUVERA_SEEDS="1 1001 2001"`. Google's implementation seeds
each repetition with the base seed plus its repetition number. Adjacent
base seeds therefore reuse most repetition seeds: with 20 repetitions,
base seeds 1 and 2 share 19 of 20. Keep base seeds separated by at least
the largest repetition count in the sweep. The example spacing of 1,000
is valid for sweeps with at most 1,000 repetitions. It avoids overlapping
repetition seeds without changing the upstream algorithm.

Hold the candidate budget at 100 and sweep repetitions, hash bits, and
bucket projection dimensions independently. Equal total dimensions can
have different quality, so record the full parameter tuple. A concrete
initial acceptance target is at least 0.95 recovery of exact top-10
Chamfer neighbors in the 100 candidates for every seed. This target is a
proposed test criterion, not a claim from the paper. Report all seed
results and the reranked relevance metrics, even when the criterion fails.

To answer document length versus dimensions, first inspect the current
query groups. Then extend aggregation to stratify exact-neighbor recovery
by the retained token-vector count of each target document, so all ten
exact neighbors contribute to their own length groups. Keep
retrieval over the same full corpus; do not silently make the task easier
by removing documents outside a length bin. Report the number of target
neighbors and contributing queries in every bin. Use query-level
resampling for uncertainty. Record encoder truncation and query lengths.
A long source document truncated by the encoder is not a long multivector
representation. Empty bins cannot support a sizing claim.

Choose parameters on a separate tuning split before the final test. For
each populated bin, report the smallest *tested* dimension that passes
at the fixed candidate budget and all seeds. If none passes, report that
fact and extend the sweep. Do not infer that all larger dimensions pass,
or fit a general sizing law from SciFact alone. A longer corpus with
verified untruncated embeddings is the next dataset when the small run
passes. Split long source documents into retrieval chunks only under an
explicit policy and count every chunk's storage.

After the FP32 fidelity sweep passes, measure FP16, INT8, and PQ against
the same targets. Train PQ on a separate sample, record its codebooks and
parameters, and measure serialized bytes including metadata. Report
payload, codebooks, approximate-index overhead, and the retained reranking
vectors separately. Measure actual resident memory and any compressed
search kernel separately from reconstruction-based quality tests.

The SSD test comes after a usable size/quality point exists. Run a verified
cold workload larger than available cache. Sweep record size, packing,
locality, batch size, and queue depth. Read each complete record, including
records above 4 KiB. Record the namespace's logical block format and the
drive's known indirection-unit (IU) size. A 512-byte namespace can still
have a 4 KiB IU. Count requested and completed host bytes; call bytes
physical or media bytes only when device telemetry measures that layer.
A modeled `4096 / record_bytes` ratio is not such a measurement. The legacy
optional NVMe script must be revised before it can support this protocol.

## Paper scope

[Google's June 25, 2025 announcement](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
released FDE construction code. The public implementation supplies the
construction; knlp supplies dataset preparation and evaluation.

[MUVERA v2](https://arxiv.org/abs/2405.19504v2) corrects a proof and weakens
a sufficient worst-case dimension bound. It does not prove that every
shorter encoding fails, nor give a numerical document-length calculator.
The practical dimension frontier is an empirical result to collect.

[REFRAG v2](https://arxiv.org/html/2509.01092v2) motivates a related systems
question about compact representations, but its trained chunk embeddings
serve generation. Retrieval fidelity here does not establish REFRAG
answer quality.
