# Context cartridges: generated prefix key/value memory

This package tests whether a small trained writer can turn an unseen context
into a reusable prefix key/value cache in one forward pass, so that a frozen
reader answers later questions from that artifact after the context itself is
removed. The writer never sees the questions. The first target is Qwen3-0.6B,
a 1,024-token source, and 64 memory positions: the raw BF16 key/value payload
of the source is 112 MiB and the artifact is 7 MiB, a 16-fold reduction of the
source payload (not of total model memory or latency).

The code is a standalone research harness. It uses PyTorch, Transformers and
PEFT directly, ordinary files, and the typed `prefix_kv` artifact schema from
[tools/reproduce/context_artifacts](../../tools/reproduce/context_artifacts/README.md).
It does not touch vLLM, LMCache or any serving stack.

## What the experiment measures

A **world** is a synthetic staff ledger: eight live staff members with a team,
a locker number, a badge code and a mentor (32 facts), four later corrections,
four archived members as distractors, and unrelated filler lines. Each source
is exactly 1,024 tokens under the pinned tokenizer. Eight questions per world
cover four families: a direct fact, a corrected fact (the latest value wins),
a linked fact through the mentor relation (two records), and an exact badge
code. Answers are verified from the structured facts before rendering.

The **writer** is the frozen base model with rank-16 adapters on the attention
projections and 64 trainable input embeddings appended after the source. The
per-layer keys and values at those 64 positions are the cartridge. The
**reader** is a second, untouched instance of the same checkpoint. It receives
only the cartridge rows as its initial attention cache, then the instruction
and the question. The reader's answer loss trains the writer; the reader's
weights never change.

Positions follow one contract: memory rows carry logical positions 1,024 to
1,087 and the first reader token has logical position 1,088 while its physical
cache index is 64. Keys are stored after rotary embedding. A dense masked
reference implementation checks the cache path in FP32, values and gradients,
in `tests/context_cart`.

Arms compared through the same reader, prompt and greedy decoding:

| arm | what precedes the question |
| --- | --- |
| full | the whole ledger as text, prefilled once per world |
| empty | nothing |
| tail | the last 64 source tokens as text |
| summary | a query-blind text summary the frozen reader wrote first |
| untrained | the writer at initialization |
| learned | the trained writer |
| wrong | another world's learned cartridge |

A **cold-start replay** test exports cartridges, answers the questions with
the rows still in memory, then starts two fresh reader-only processes that
load only the base model, the artifacts and the question text, install the
rows as the initial cache, and answer again. Token sequences and logits are
compared against repeated unmodified-reader variability. A **bounded
direct-learning comparison** fits a residual MLP prefix per world (100 updates
from a source-independent base, and from the generated prefix) and compares
held-out answers against the generated-only cartridge.

## Running it

Generate the dataset on a CPU host (about three minutes):

```sh
python -m research.context_cart.cli prepare --out /path/dataset
```

Qualify the reader on the calibration pool, profile, and train:

```sh
python -m research.context_cart.cli evaluate --dataset /path/dataset \
    --pool calibration --run-dir /path/run --text-arms full,empty,tail,summary
python -m research.context_cart.cli train --set run_dir='"/path/run/profile"' \
    --set dataset_dir='"/path/dataset"' --set mode='"profile"'
python -m research.context_cart.cli train --set run_dir='"/path/run/train_s61031"' \
    --set dataset_dir='"/path/dataset"' --set seed=61031
```

Export cartridges from a checkpoint, run the replay protocol, evaluate the
sealed test from saved artifacts only, fit the diagnostic prefixes and report:

```sh
python -m research.context_cart.cli export --dataset /path/dataset --pool calibration \
    --checkpoint /path/run/train_s61031/ckpt_1000.pt --out /path/run/replay/producer \
    --namespace run --worlds 8 --replay-reference
python -m research.context_cart.cli replay --artifacts /path/run/replay/producer/cartridges \
    --questions /path/dataset/questions_calibration.jsonl --out /path/run/replay/consumer_a --range 0:4
python -m research.context_cart.cli compare-replay --producer /path/run/replay/producer \
    --consumers /path/run/replay/consumer_a /path/run/replay/consumer_b \
    --out /path/run/replay/replay_acceptance.json
python -m research.context_cart.cli evaluate --dataset /path/dataset --pool test_ordinary \
    --run-dir /path/run --text-arms full,empty,tail,summary \
    --artifact-arm learned_s61031=/path/run/cartridges/learned_s61031/test_ordinary/cartridges \
    --wrong-arm wrong_s61031=/path/run/cartridges/learned_s61031/test_ordinary/cartridges
python -m research.context_cart.cli fit --dataset /path/dataset \
    --checkpoint /path/run/train_s61031/ckpt_1000.pt --run-dir /path/run
python -m research.context_cart.cli report --run-dir /path/run
```

`smoke.sh` runs every stage on a tiny random model with the real tokenizer on
a CPU in about two minutes. The unit tests run with
`python -m pytest tests/context_cart -q`.

## Limits

A pass on this synthetic task shows the mechanism: a small artifact that
survives a process restart and supports unseen questions without the source.
It does not show general conversation memory, recursive compaction,
competitive serving speed, or transfer to other model weights. Result files
and measured numbers live outside this repository.
