# fdec: fight decode

fdec is the knlp method for combining decode optimizations and proving that
the combination is safe. A decode step on a GPU is limited by the bytes it
moves, not by arithmetic, and those bytes sit in separate pools: the model
weights, the KV cache and the output head. Each pool has its own best
reducer. No single technique wins decode, so the useful result is a set of
reducers that work together, with a measurement that shows they do not
damage each other.

We call each reducer a *brick*. A brick attacks one byte pool and has a
solo result of its own. fdec adds one rule on top: a brick counts only
after it has been measured together with every other brick, on text that no
calibration step has seen. `make defconfig-fdec && make` runs that
measurement and fails if any combination does not compose.

Measured on Qwen2.5-7B, Llama-3.2-3B and Qwen3-8B, the three dense-model
bricks compose in all eight combinations. The full stack removes
40% to 46% of the bytes a decode step reads at a 32,768-token context,
at a perplexity within 0.5% of the unmodified model. A fourth brick,
the trained cartridge, composes with int8 weights and with the
asymmetric KV cache at a combined loss cost of 0.1%.

## Contents

- [Why decode needs several bricks](#why-decode-needs-several-bricks)
- [The bricks](#the-bricks)
- [What "composes" means](#what-composes-means)
- [The measured matrix](#the-measured-matrix)
- [Cartridges as a brick](#cartridges-as-a-brick)
- [What the matrix corrected](#what-the-matrix-corrected)
- [Running it](#running-it)
- [Adding a brick](#adding-a-brick)
- [Limits and open cells](#limits-and-open-cells)

## Why decode needs several bricks

Autoregressive decode at low batch is memory-bound. For each generated token
the GPU reads every weight in the decoder blocks once, reads the whole KV
cache, and multiplies the hidden state against the output head. The
arithmetic is small next to the memory traffic, so latency follows bytes
read divided by memory bandwidth. This is the result of the
[memory-traffic saturation study](memory_traffic_saturation_in_autoregressive_decode.md).

The pools do not have fixed shares. For Qwen2.5-7B at a 32,768-token
context the decoder blocks are about 81% of the bytes, the KV cache 12%
and the output head 7%. The KV share grows linearly with context and with
batch, and the head share grows with vocabulary size. A reducer for one
pool therefore has a ceiling set by that pool's share, and the stack needs
one reducer per pool.

Two kinds of optimization exist. *Byte movement* bricks reduce what a step
reads: quantized weights, a quantized or shorter KV cache, a partial read of
the head. *Arithmetic intensity* bricks do more useful work per byte read,
for example by moving a mixed-precision attention kernel onto Tensor Cores.
The matrix on this page measures the byte movement bricks. Kernel speed is
measured in the serving stack and is listed under open cells.

## The bricks

| brick | pool | what it does | solo result |
|---|---|---|---|
| int8 weights | decoder blocks | group-wise int8 for every linear layer, no calibration | 49% fewer block bytes, near-lossless ([weight quantization study](fim_quantization.html)) |
| asymmetric KV cache | KV cache | keys stay 16-bit, values go to FP8 | 25% fewer KV bytes ([asymmetric quantization](asymmetric-quantization.html)) |
| certified LM-head decode | output head | int8 screen of the head, exact fetch of the rows that can still win | exact argmax from 51% to 57% of the head ([certified decode](lm-head-decode.md)) |
| trained cartridge | KV cache length | a trained KV prefix replaces a long document | a few hundred cache positions in place of the document ([cartridges](cartridges.html)) |

The weight brick is int8 and not int4. The Fisher-sensitivity study that
ranked which weights need more bits found int8 to be the point where
quality holds without calibration. Int4 with activation-aware or
second-order error compensation looked good on a six-prompt proxy and lost
quality on a held-out evaluation, so it is not in the stack.

The KV brick keeps keys at 16 bits because keys are what break. Symmetric
FP8 keys collapse on models with large key bias, and the damage follows the
bias magnitude ([FP8 failure atlas](fp8-kv-failure-atlas.html)).

Linear-attention models (Gated DeltaNet, Mamba2, RWKV) remove the KV pool
instead of shrinking it. They are a model class and not a brick: the KV and
cartridge bricks do not apply to them, and the weight and head bricks have
not been measured on them.

## What "composes" means

A cell is one model with one subset of bricks enabled. The harness measures
every subset, so three bricks give eight cells. A cell composes when three
conditions hold.

1. It runs. A shape, dtype or code-path conflict is recorded as a failed
   cell, not as a crash.
2. Every exactness claim still holds. The LM-head certificate must return
   the dense argmax of that cell's own model at every evaluated position.
   It is checked on hidden states that the other bricks have already
   perturbed.
3. Quality costs add and do not multiply. The increase in log-perplexity
   of the cell must be within 0.01 of the sum of the increases of its
   bricks measured alone.

Quality is measured against the unmodified 16-bit model on WikiText-2 test:
perplexity, top-1 agreement of the predicted token, and mean KL divergence.
Calibration, where a brick needs it, uses WikiText-2 train. The harness
also reports the bytes each cell reads per token from an analytical model
of the three pools. The input embedding is left out of that model because a
decode step reads one row of it.

## The measured matrix

Each run uses 8,192 evaluation tokens and 65,536 calibration tokens. "Bytes
saved" is the reduction in bytes read per decode step at a 32,768-token
context. "Head read" is the fraction of the dense output head that the
certificate reads, shadow included.

### Qwen2.5-7B

Baseline perplexity 9.844. Bytes read per token: decoder blocks 13051 MB (81%), KV cache 1879 MB (12%), output head 1090 MB (7%).

| cell | perplexity | top-1 agreement | KL | certificate exact | head read | bytes saved |
|---|---:|---:|---:|---|---:|---:|
| none | 9.844 | 1.000 | 0.0000 | n/a | n/a | 0.0% |
| int8 weights | 9.804 | 0.917 | 0.0564 | n/a | n/a | 40.1% |
| asymmetric KV | 9.855 | 0.964 | 0.0084 | n/a | n/a | 2.9% |
| certified head | 9.844 | 1.000 | 0.0000 | yes | 56.7% | 3.0% |
| int8 weights + asymmetric KV | 9.798 | 0.917 | 0.0566 | n/a | n/a | 43.0% |
| int8 weights + certified head | 9.804 | 0.917 | 0.0564 | yes | 56.7% | 43.0% |
| asymmetric KV + certified head | 9.855 | 0.964 | 0.0084 | yes | 56.7% | 5.9% |
| all three | 9.798 | 0.917 | 0.0566 | yes | 56.8% | 46.0% |

### Llama-3.2-3B

Baseline perplexity 11.807. Bytes read per token: decoder blocks 5637 MB (55%), KV cache 3758 MB (37%), output head 788 MB (8%).

| cell | perplexity | top-1 agreement | KL | certificate exact | head read | bytes saved |
|---|---:|---:|---:|---|---:|---:|
| none | 11.807 | 1.000 | 0.0000 | n/a | n/a | 0.0% |
| int8 weights | 11.812 | 0.981 | 0.0014 | n/a | n/a | 27.2% |
| asymmetric KV | 11.807 | 0.983 | 0.0011 | n/a | n/a | 9.2% |
| certified head | 11.807 | 1.000 | 0.0000 | yes | 51.4% | 3.8% |
| int8 weights + asymmetric KV | 11.804 | 0.979 | 0.0018 | n/a | n/a | 36.5% |
| int8 weights + certified head | 11.812 | 0.981 | 0.0014 | yes | 51.4% | 31.0% |
| asymmetric KV + certified head | 11.807 | 0.983 | 0.0011 | yes | 51.4% | 13.0% |
| all three | 11.804 | 0.979 | 0.0018 | yes | 51.4% | 40.2% |

### Qwen3-8B

Baseline perplexity 15.048. Bytes read per token: decoder blocks 13892 MB (70%), KV cache 4832 MB (24%), output head 1245 MB (6%).

| cell | perplexity | top-1 agreement | KL | certificate exact | head read | bytes saved |
|---|---:|---:|---:|---|---:|---:|
| none | 15.048 | 1.000 | 0.0000 | n/a | n/a | 0.0% |
| int8 weights | 15.046 | 0.979 | 0.0021 | n/a | n/a | 34.2% |
| asymmetric KV | 15.055 | 0.982 | 0.0017 | n/a | n/a | 6.0% |
| certified head | 15.048 | 1.000 | 0.0000 | yes | 54.1% | 2.9% |
| int8 weights + asymmetric KV | 15.054 | 0.978 | 0.0025 | n/a | n/a | 40.3% |
| int8 weights + certified head | 15.046 | 0.979 | 0.0021 | yes | 54.1% | 37.1% |
| asymmetric KV + certified head | 15.055 | 0.982 | 0.0017 | yes | 54.1% | 8.9% |
| all three | 15.054 | 0.978 | 0.0025 | yes | 54.1% | 43.1% |

All 24 cells compose. The certificate is exact in every cell that has it,
including the cells where int8 weights and FP8 values have changed the
hidden states it certifies. A shadow built from the unmodified model reads
the same fraction of the head as one rebuilt for the quantized model, so
the artifact does not need rebuilding when the other bricks change.

The interaction term is at most 0.002 in log-perplexity, inside the
noise of an 8,192-token evaluation. Top-1 agreement is the more sensitive
measure. On Qwen2.5-7B int8 weights change 8% of the predicted tokens while
perplexity stays level, so the changed predictions are near-ties. That
model has unusually large activations, and it is the one model here where
calibration-free int8 should be checked on the target task before use.

## Cartridges as a brick

A cartridge is a KV prefix trained by self-study so that a frozen model
answers questions about a document without reading it. It shortens the KV
cache, so it is a byte movement brick for the KV pool. It was trained
against the full-precision model, which raises two composition questions:
does it survive a quantized reader, and does it survive being stored
quantized.

The test scores two Qwen3-8B cartridges, trained on two LongHealth patient
records, on 256 self-study conversations each. The score is the trainer's
own distillation loss against the full-document teacher. The same cartridge
is read four ways, so the comparison is paired and carries no
training-to-training variance.

| reader weights | cartridge storage | loss, patient 1 | loss, patient 2 |
|---|---|---:|---:|
| 16-bit | 16-bit keys and values | 0.015102 | 0.016170 |
| int8 | 16-bit keys and values | 0.015111 (+0.06%) | 0.016181 (+0.07%) |
| 16-bit | 16-bit keys, FP8 values | 0.015106 (+0.03%) | 0.016177 (+0.04%) |
| int8 | 16-bit keys, FP8 values | 0.015116 (+0.09%) | 0.016188 (+0.11%) |
| int8 | FP8 keys and values | 0.015307 (+1.36%) | 0.016357 (+1.16%) |
| 16-bit, no cartridge | none | 0.043067 | 0.045057 |

The cost of both bricks together equals the sum of the two costs alone, so
they compose. The loss without a cartridge is 2.8 times higher with either
reader, so the cartridge keeps its value under int8 weights. Symmetric FP8
keys still cost more than 1%, the same penalty the
[cartridge quantization study](cartridge-asymmetric-quantization.html)
found with a full-precision reader.

These cartridges saw the scored conversations during training, so the loss
level is in-sample. The paired differences between readings do not depend
on that. The certified head is exact for any hidden state, so it needs no
separate cartridge test.

## What the matrix corrected

The first held-out run of the matrix found that the certified LM-head
decode, as first published, saved no bytes. The certificate bounds every
logit with a low-rank shadow of the head plus a residual term for the part
of the hidden state outside the shadow basis. The earlier replay built a
rank-1280 basis from the same 600 to 1,500 hidden states it then scored. A
basis fitted to so few states contains them, the residual was near zero,
and the certificate read about a quarter of the head. With the basis built
from 65,408 train positions and scored on 8,176 test positions, the same
certificate opened every vocabulary block and read 118% of the dense head.

The token was exact in both cases. Only the byte count was wrong, and only
a held-out measurement could show it. A rank sweep then found that the
saving returns near full rank, where the method becomes an int8 screen of
the whole head followed by an exact fetch of 1% to 6% of the rows. That
configuration reads 51% to 57% of the head on unseen text, and it is the
one the matrix now uses. The [certified decode page](lm-head-decode.md)
carries the full sweep.

This is the reason for the rule that a brick counts only after the matrix
has measured it held-out and in combination.

## Running it

```
make defconfig-fdec
make
```

`defconfig-fdec` selects Qwen2.5-7B and Llama-3.2-3B with all three dense
bricks. `make` runs [`scripts/fdec_compose.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_compose.py)
once per model, writes one JSON result per model and a `REPORT.md` table
under `results/fdec`, and exits non-zero if any cell fails to compose. A
model with a result file is skipped, so an interrupted run resumes. The
7B model needs a GPU with 24 GiB; each model takes two to five minutes.

`make defconfig-fdec-smoke && make` runs the same eight cells on
Qwen2.5-0.5B on the CPU in under three minutes. It checks the harness, not
the numbers: its calibration set is small.

The options live in [`Kconfig.fdec`](https://github.com/mcgrof/knlp/blob/main/Kconfig.fdec). `CONFIG_FDEC_MODELS`
takes a space-separated list of Hugging Face model ids. Each brick has its
own switch, and the harness measures every subset of the enabled bricks.

Two more tools cover what the matrix does not.
[`scripts/fdec_cert_heldout.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cert_heldout.py) sweeps the
certificate rank on held-out hidden states. The cartridge cells use
[`cas_cart_loss.py`](https://github.com/mcgrof/knlp/blob/main/research/cartridges_cas/scripts/cas_cart_loss.py)
with `W8=1` for the int8 reader and
[`cas_cart_quantize.py`](https://github.com/mcgrof/knlp/blob/main/research/cartridges_cas/scripts/cas_cart_quantize.py)
for the stored cartridge; they need a trained cartridge and its self-study
conversations.

## Adding a brick

A new brick enters the stack in four steps. State the byte pool it attacks
and its solo result on a held-out evaluation. Add it to
`scripts/fdec_compose.py` as either a transform that changes the model or
its attention, or a measurement that checks an exactness claim on the
perturbed model. Add its factor to the byte model. Then run the matrix: the
brick is in when every cell that contains it composes on at least two model
families.

A brick that needs calibration must take it from the train split. A brick
with an exactness claim must have that claim checked in every cell, not
only alone.

## Limits and open cells

The matrix measures quality and bytes with the bricks emulated in PyTorch:
weights are rounded in place, values pass through an FP8 cast, and the
certificate is replayed against dense logits. It shows that the bricks do
not conflict in what they compute. It does not measure speed.

The bytes-saved column is a model of bytes read, not a latency. Turning it
into latency needs each brick in a real kernel in one serving stack. The
asymmetric KV cache has that today in the modified vLLM and FlashInfer
branches described in [decode reproduction](decode-reproduction.md). Int8
weights and the full-rank certified head have not been served together with
it, and the latency of the full-rank certificate has not been measured.

Evaluation is WikiText-2 only. The weight study also used a cross-domain
calibration check; the matrix does not yet repeat it, and it has no
task-accuracy benchmark.

Linear-attention models are not in the matrix. The cartridge cells cover
one model and two documents.
