# RGSA: Retrieval-Gated Sparse Attention

RGSA adds retrieval-based sparse attention to GPT-2. Context is
chunked into blocks, each compressed to a routing embedding via
mean-pooling + projection. A learned retrieval gate selects the
top-B most relevant chunks per query position. Attention runs on
the local window plus retrieved chunks.

## Dynamic Chunking (L2M-inspired)

By default RGSA uses a fixed `chunk_size` (typically 64). Dynamic
chunking makes chunk granularity depend on sequence length:

    chunk_size_eff = clamp(round(seq_len^alpha), min, max)

With alpha=0.5 (default), chunk size grows as the square root of
sequence length. This is configurable via RGSAConfig fields:

| Field | Default | Description |
|---|---|---|
| `dynamic_chunking` | False | Enable dynamic chunk sizing |
| `chunk_size_alpha` | 0.5 | Exponent for power schedule |
| `chunk_size_min` | 32 | Minimum chunk size |
| `chunk_size_max` | 256 | Maximum chunk size |
| `chunk_size_schedule` | "power" | "power" or "piecewise" |
| `chunk_size_piecewise` | "" | Threshold spec, e.g. "512:32,2048:64" |
| `chunk_size_rounding` | "pow2" | "pow2", "multiple_of_8", "nearest" |

Dynamic chunking does not change parameter count. It only affects
how tokens are partitioned into chunks at each forward pass.

### Expected chunk sizes (alpha=0.5, pow2 rounding)

| seq_len | chunk_size_eff |
|---------|---------------|
| 256 | 32 |
| 512 | 32 |
| 1024 | 32 |
| 2048 | 32 |
| 4096 | 64 |

## Ablation Modes

Three ablation flags isolate what matters:

- **static** (default): Fixed chunk_size, learned routing
- **dense_mode**: Router params exist but routing is skipped
- **random_routing**: Chunks selected uniformly at random

These compose with dynamic chunking (dynamic+dense, dynamic+random).

## Running the Ablation Matrix

### TinyStories smoke test

```bash
./scripts/gpt2/run_compare_rgsa.sh --track tinystories --seeds "42" --dry-run
```

### FineWebEdu full comparison

```bash
# Baseline + static + dynamic (alpha 0.4/0.5/0.6) + piecewise
./scripts/gpt2/run_compare_rgsa.sh --track finewebedu --seeds "1 2 3"

# Add dense/random ablations
./scripts/gpt2/run_compare_rgsa.sh --track finewebedu --seeds "1 2 3" --ablations
```

### Analysis

```bash
python scripts/gpt2/analyze_rgsa_compare.py --dir rgsa_compare_finewebedu_<timestamp>
```

The analyzer produces:
- `compare_val_ppl.png`: Mean +/- std validation perplexity
- `compare_rgsa_diagnostics.png`: Routing entropy and load balance
- `report.md`: Summary table with dynamic chunking statistics

## Available Defconfigs

### TinyStories (smoke tests)
- `gpt2-tinystories-baseline`
- `gpt2-tinystories-rgsa` (static)
- `gpt2-tinystories-rgsa-dynamic`
- `gpt2-tinystories-rgsa-dense`
- `gpt2-tinystories-rgsa-random`

### FineWebEdu (primary comparisons)
- `gpt2-finewebedu-baseline`
- `gpt2-finewebedu-rgsa-static`
- `gpt2-finewebedu-rgsa-dynamic-a04` (alpha=0.4)
- `gpt2-finewebedu-rgsa-dynamic-a05` (alpha=0.5)
- `gpt2-finewebedu-rgsa-dynamic-a06` (alpha=0.6)
- `gpt2-finewebedu-rgsa-dynamic-piecewise`

## Tests

```bash
python tests/test_rgsa_dynamic_chunking.py
```

14 tests covering bounds, rounding, schedules, parameter count
parity, forward passes, and ablation combos.
