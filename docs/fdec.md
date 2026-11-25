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
at a perplexity within 0.3% of the unmodified model. On GSM8K and MMLU
the full stack answers as many questions as the 16-bit model within the
resolution of the test: the 95% interval of the
[difference in accuracy](#task-accuracy) contains zero for all three
models on both tasks. A fourth brick, the trained cartridge, composes
with int8 weights and with the asymmetric KV cache at a combined loss
cost of 0.1%.

## Contents

- [Why decode needs several bricks](#why-decode-needs-several-bricks)
- [The bricks](#the-bricks)
- [What "composes" means](#what-composes-means)
- [The measured matrix](#the-measured-matrix)
- [Task accuracy](#task-accuracy)
- [Cartridges as a brick](#cartridges-as-a-brick)
- [What the measurements corrected](#what-the-measurements-corrected)
- [Speed so far](#speed-so-far)
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
The matrix on this page measures the byte movement bricks.
[Speed](#speed-so-far) is a separate measurement.

## The bricks

| brick | pool | what it does | solo result |
|---|---|---|---|
| int8 weights | decoder blocks | group-wise int8 for every linear layer, no calibration | 49% fewer block bytes, near-lossless ([weight quantization study](fim_quantization.html)) |
| asymmetric KV cache | KV cache | keys stay 16-bit, values go to FP8 by a direct cast | 25% fewer KV bytes ([asymmetric quantization](asymmetric-quantization.html)) |
| certified LM-head decode | output head | int8 screen of the head, exact fetch of the rows that can still win | exact argmax from 51% to 56% of the head ([certified decode](lm-head-decode.md)) |
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
   the token that the full output head selects, computed in float32, at
   every evaluated position of that cell's own model. It is checked on
   hidden states that the other bricks have already perturbed.
3. Quality costs add and do not multiply. The increase in log-perplexity
   of the cell must not exceed the sum of the increases of its bricks
   measured alone by more than 0.01. The difference is the interaction
   term, and the harness reports it with a 95% interval.

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

The three models were measured in bfloat16 on one Radeon Pro W7900 with
PyTorch 2.9.1 and transformers 4.57.1, at knlp commit
[`0d5fe72b6`](https://github.com/mcgrof/knlp/commit/0d5fe72b6) with the
direct FP8 cast selected. That cast is the default since commit
[`4aa007a2c`](https://github.com/mcgrof/knlp/commit/4aa007a2c).

### Qwen2.5-7B

Baseline perplexity 8.443. Bytes read per token: decoder blocks 13051 MB (81%), KV cache 1879 MB (12%), output head 1090 MB (7%).

| cell | perplexity | top-1 agreement | KL | certificate exact | head read | bytes saved |
|---|---:|---:|---:|---|---:|---:|
| none | 8.443 | 1.000 | 0.0000 | n/a | n/a | 0.0% |
| int8 weights | 8.449 | 0.976 | 0.0018 | n/a | n/a | 40.1% |
| asymmetric KV | 8.452 | 0.980 | 0.0018 | n/a | n/a | 2.9% |
| certified head | 8.443 | 1.000 | 0.0000 | yes | 55.9% | 3.0% |
| int8 weights + asymmetric KV | 8.465 | 0.973 | 0.0026 | n/a | n/a | 43.0% |
| int8 weights + certified head | 8.449 | 0.976 | 0.0018 | yes | 56.0% | 43.1% |
| asymmetric KV + certified head | 8.452 | 0.980 | 0.0018 | yes | 56.0% | 5.9% |
| all three | 8.465 | 0.973 | 0.0026 | yes | 56.0% | 46.0% |

### Llama-3.2-3B

Baseline perplexity 11.811. Bytes read per token: decoder blocks 5637 MB (55%), KV cache 3758 MB (37%), output head 788 MB (8%).

| cell | perplexity | top-1 agreement | KL | certificate exact | head read | bytes saved |
|---|---:|---:|---:|---|---:|---:|
| none | 11.811 | 1.000 | 0.0000 | n/a | n/a | 0.0% |
| int8 weights | 11.812 | 0.983 | 0.0013 | n/a | n/a | 27.2% |
| asymmetric KV | 11.822 | 0.981 | 0.0011 | n/a | n/a | 9.2% |
| certified head | 11.811 | 1.000 | 0.0000 | yes | 51.4% | 3.8% |
| int8 weights + asymmetric KV | 11.828 | 0.980 | 0.0018 | n/a | n/a | 36.5% |
| int8 weights + certified head | 11.812 | 0.983 | 0.0013 | yes | 51.4% | 31.0% |
| asymmetric KV + certified head | 11.822 | 0.981 | 0.0011 | yes | 51.4% | 13.0% |
| all three | 11.828 | 0.980 | 0.0018 | yes | 51.4% | 40.2% |

### Qwen3-8B

Baseline perplexity 15.024. Bytes read per token: decoder blocks 13892 MB (70%), KV cache 4832 MB (24%), output head 1245 MB (6%).

| cell | perplexity | top-1 agreement | KL | certificate exact | head read | bytes saved |
|---|---:|---:|---:|---|---:|---:|
| none | 15.024 | 1.000 | 0.0000 | n/a | n/a | 0.0% |
| int8 weights | 15.048 | 0.980 | 0.0017 | n/a | n/a | 34.2% |
| asymmetric KV | 15.036 | 0.983 | 0.0013 | n/a | n/a | 6.0% |
| certified head | 15.024 | 1.000 | 0.0000 | yes | 54.1% | 2.9% |
| int8 weights + asymmetric KV | 15.059 | 0.978 | 0.0021 | n/a | n/a | 40.3% |
| int8 weights + certified head | 15.048 | 0.980 | 0.0017 | yes | 54.1% | 37.1% |
| asymmetric KV + certified head | 15.036 | 0.983 | 0.0013 | yes | 54.1% | 8.9% |
| all three | 15.059 | 0.978 | 0.0021 | yes | 54.1% | 43.1% |

All 24 cells compose. The certificate is exact in every cell that has it,
including the cells where int8 weights and FP8 values have changed the
hidden states it certifies. A shadow built from the unmodified model reads
the same fraction of the head as one rebuilt for the quantized model, so
the artifact does not need rebuilding when the other bricks change.

The interaction term is at most 0.0008 in log-perplexity, inside the
noise of an 8,192-token evaluation. Top-1 agreement is the more sensitive
measure. Each brick alone changes about 2% of the predicted tokens, and
the full stack changes 2% to 3%, while perplexity moves by at most 0.26%.
The changed predictions are therefore near-ties. The next section checks
what that does to answers.

### The FP8 cast of the values

The asymmetric cache casts each value to FP8 as it is. That is the default
of the vLLM FP8 cache, so the cell computes what a serving engine computes.
The first version of these tables divided each value tensor by one scale
before the cast, which uses the whole FP8 range. With that scale the cells
that contain the cache have a log-perplexity 0.0001 to 0.0012 lower and a
top-1 agreement within 0.001, every cell composes, and the full stack is
within 0.14% of the 16-bit perplexity. The cells without the cache are
identical in the two runs. `--kv-scale tensor` selects this variant. Both
variants were also measured on one RTX 4090 with PyTorch 2.10, for
Qwen2.5-7B and Llama-3.2-3B. Every cell composes there, and the two GPUs
agree within 0.008 in perplexity and 0.004 in top-1 agreement in every
cell. With the direct cast, the full stack on the RTX 4090 is within 0.17%
and 0.10% of the 16-bit perplexity of the two models.

## Task accuracy

On three models, 26 of 30 paired comparisons show no difference in task
accuracy between a cell with bricks and the 16-bit model: the 95% interval
of the difference contains zero. With all three bricks, MMLU accuracy is
56.77% against 56.57% on Llama-3.2-3B, 74.17% against 74.25% on Qwen2.5-7B
and 75.05% against 74.96% on Qwen3-8B. GSM8K accuracy with all three bricks
is 29.3% against 29.4% on Llama-3.2-3B, 81.0% against 81.1% on Qwen2.5-7B
and 89.4% against 88.9% on Qwen3-8B. The interval of each of these six
differences contains zero.

Four of the 30 intervals exclude zero: two losses and two gains. On
Qwen2.5-7B, int8 weights with the asymmetric cache lose 0.21 points of MMLU
accuracy, and the certified head alone loses 1.29 points of GSM8K accuracy.
On Qwen3-8B, int8 weights alone gain 1.36 points of GSM8K accuracy and the
certified head alone gains 0.83 points. None of the four is significant
after a correction for the 30 comparisons. A test of each table as a whole
finds that the cells of the two GSM8K tables of the Qwen models differ among
themselves: the lowest and the highest cell are 2.35 and 1.82 points apart,
and the 16-bit cell is between them.

Control runs test these differences, each with a rule set before the run.
The three controls for Qwen2.5-7B are done, and none reproduces its
difference. The controls for Qwen3-8B are in the queue. The table gives the
state of each control.

| Difference in the task tables | Control | Result of the control | Outcome by the rule |
|---|---|---|---|
| Qwen2.5-7B, MMLU: int8 weights with the asymmetric cache are 0.21 points below the 16-bit model | The same cell with two other int8 group sizes and with another FP8 cast of the values | Mean difference of 0.00 points from the 16-bit model, with a 95% interval from 0.14 points below to 0.14 points above | Not reproduced |
| Qwen2.5-7B, GSM8K: the certified head is 1.29 points below the 16-bit model | The same cell on 3,957 other problems | 0.28 points above, with a 95% interval from 0.16 points below to 0.72 points above | Not reproduced |
| Qwen2.5-7B, GSM8K: three cells span 2.35 points | The same cells on 3,957 other problems | The cells span 0.45 points in the reverse order, and the test finds no difference among them (p 0.29) | Not reproduced |
| Qwen3-8B, GSM8K: int8 weights are 1.36 points above the 16-bit model | The same cell on 2,638 other problems, and with two other int8 group sizes on the test problems | In the queue | Open |
| Qwen3-8B, GSM8K: the certified head is 0.83 points above the 16-bit model | The same cell on 2,638 other problems | In the queue | Open |
| Qwen3-8B, GSM8K: four cells span 1.82 points | The same cells on 2,638 other problems | In the queue | Open |

The first column gives a difference whose 95% interval excludes zero, or a
table whose cells differ in the test of the table as a whole. The
subsections below give each rule and the data of each control.

Perplexity and top-1 agreement are measured on running text. The task check
asks what a user asks: does a model with the bricks answer as many questions
correctly as the 16-bit model?
[`scripts/fdec_task_acc.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_task_acc.py)
scores every cell on the same items, so each comparison is paired.

GSM8K is 1,319 grade-school math problems. The prompt holds eight worked
examples, decoding is greedy, and an answer is correct when the first number
after the answer marker equals the reference. A completion with no answer
marker counts as wrong. MMLU is 14,042 four-choice questions with five
examples per subject; the answer is the choice letter with the largest
logit. A certified-head cell computes the logits of the last position in
float32, which is the token the certificate returns. The other cells take
the largest 16-bit logit, so the two can differ where the two largest logits
tie within the 16-bit rounding step. Every table has the certified head
alone and in the full stack. The asymmetric KV cells cast the values to FP8
directly, as the [matrix](#the-fp8-cast-of-the-values) does.

All runs are in bfloat16. Llama-3.2-3B and Qwen2.5-7B ran at knlp commit
[`4aa007a2c`](https://github.com/mcgrof/knlp/commit/4aa007a2c) on one RTX
4090 with PyTorch 2.10 and transformers 5.3.0, in batches of up to 16 and 8
prompts. Qwen3-8B ran at the same commit on one Radeon Pro W7900 with
PyTorch 2.9.1 and transformers 4.57.1, in batches of 32 prompts for GSM8K
and up to 16 for MMLU. An earlier GSM8K run of Qwen2.5-7B serves as a
control below. It ran on the W7900 at
[`dd3a68306`](https://github.com/mcgrof/knlp/commit/dd3a68306), with the
certified-head cell at
[`0d5fe72b6`](https://github.com/mcgrof/knlp/commit/0d5fe72b6), in batches
of 32 prompts and with one FP8 scale per value tensor.

The 16-bit references are close to the published scores of the models. The
MMLU accuracy of Llama-3.2-3B averaged over the 57 subjects is 57.95%, and
its [model card](https://huggingface.co/meta-llama/Llama-3.2-3B) reports 58
with five examples. Qwen2.5-7B answers 74.25% of the MMLU items, or 75.46%
averaged over the subjects, and the [Qwen2.5
report](https://arxiv.org/abs/2412.15115) gives 74.2 with five examples. It
solves 81.1% of GSM8K with the eight examples used here; the report gives
85.4% with four examples of its own. Qwen3-8B answers 74.96% of the MMLU
items, or 76.88% averaged over the subjects, and the [Qwen3
report](https://arxiv.org/abs/2505.09388) gives 76.89 for its base model. It
solves 88.9% of GSM8K; the report gives 89.84 for the base model with four
examples.

### Qwen2.5-7B

| task | cell | accuracy | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|
| GSM8K, 1,319 items | 16-bit model | 81.12% | reference | | | | |
| GSM8K, 1,319 items | int8 weights | 80.97% | -0.15 | -1.26 to +0.96 | 29 | 27 | 0.89 |
| GSM8K, 1,319 items | asymmetric KV | 82.18% | +1.06 | -0.13 to +2.25 | 25 | 39 | 0.10 |
| GSM8K, 1,319 items | certified head | 79.83% | -1.29 | -2.31 to -0.27 | 32 | 15 | 0.02 |
| GSM8K, 1,319 items | int8 weights + asymmetric KV | 81.35% | +0.23 | -1.14 to +1.60 | 41 | 44 | 0.83 |
| GSM8K, 1,319 items | all three | 80.97% | -0.15 | -1.53 to +1.23 | 44 | 42 | 0.91 |
| MMLU, 14,042 items | 16-bit model | 74.25% | reference | | | | |
| MMLU, 14,042 items | int8 weights | 74.12% | -0.13 | -0.31 to +0.05 | 95 | 77 | 0.19 |
| MMLU, 14,042 items | asymmetric KV | 74.18% | -0.06 | -0.23 to +0.11 | 79 | 70 | 0.51 |
| MMLU, 14,042 items | certified head | 74.28% | +0.04 | -0.14 to +0.21 | 75 | 80 | 0.75 |
| MMLU, 14,042 items | int8 weights + asymmetric KV | 74.04% | -0.21 | -0.39 to -0.03 | 99 | 70 | 0.03 |
| MMLU, 14,042 items | all three | 74.17% | -0.08 | -0.27 to +0.11 | 100 | 89 | 0.47 |

### Llama-3.2-3B

| task | cell | accuracy | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|
| GSM8K, 1,319 items | 16-bit model | 29.42% | reference | | | | |
| GSM8K, 1,319 items | int8 weights | 29.72% | +0.30 | -0.97 to +1.58 | 35 | 39 | 0.73 |
| GSM8K, 1,319 items | asymmetric KV | 30.17% | +0.76 | -0.52 to +2.04 | 32 | 42 | 0.30 |
| GSM8K, 1,319 items | certified head | 29.64% | +0.23 | -1.16 to +1.61 | 42 | 45 | 0.83 |
| GSM8K, 1,319 items | int8 weights + asymmetric KV | 29.26% | -0.15 | -1.48 to +1.18 | 41 | 39 | 0.91 |
| GSM8K, 1,319 items | all three | 29.26% | -0.15 | -1.74 to +1.43 | 58 | 56 | 0.93 |
| MMLU, 14,042 items | 16-bit model | 56.57% | reference | | | | |
| MMLU, 14,042 items | int8 weights | 56.57% | +0.01 | -0.25 to +0.27 | 172 | 173 | 1.00 |
| MMLU, 14,042 items | asymmetric KV | 56.49% | -0.08 | -0.35 to +0.19 | 188 | 177 | 0.60 |
| MMLU, 14,042 items | certified head | 56.71% | +0.14 | -0.14 to +0.42 | 190 | 210 | 0.34 |
| MMLU, 14,042 items | int8 weights + asymmetric KV | 56.66% | +0.09 | -0.19 to +0.38 | 202 | 215 | 0.56 |
| MMLU, 14,042 items | all three | 56.77% | +0.20 | -0.12 to +0.52 | 251 | 279 | 0.24 |

### Qwen3-8B

| task | cell | accuracy | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|
| GSM8K, 1,319 items | 16-bit model | 88.93% | reference | | | | |
| GSM8K, 1,319 items | int8 weights | 90.30% | +1.36 | +0.48 to +2.25 | 9 | 27 | 0.004 |
| GSM8K, 1,319 items | asymmetric KV | 88.48% | -0.45 | -1.24 to +0.33 | 17 | 11 | 0.34 |
| GSM8K, 1,319 items | certified head | 89.76% | +0.83 | +0.22 to +1.44 | 3 | 14 | 0.01 |
| GSM8K, 1,319 items | int8 weights + asymmetric KV | 89.46% | +0.53 | -0.35 to +1.41 | 14 | 21 | 0.31 |
| GSM8K, 1,319 items | all three | 89.39% | +0.45 | -0.41 to +1.32 | 14 | 20 | 0.39 |
| MMLU, 14,042 items | 16-bit model | 74.96% | reference | | | | |
| MMLU, 14,042 items | int8 weights | 74.94% | -0.02 | -0.18 to +0.13 | 63 | 60 | 0.86 |
| MMLU, 14,042 items | asymmetric KV | 74.98% | +0.01 | -0.14 to +0.17 | 59 | 61 | 0.93 |
| MMLU, 14,042 items | certified head | 74.98% | +0.02 | -0.07 to +0.11 | 18 | 21 | 0.75 |
| MMLU, 14,042 items | int8 weights + asymmetric KV | 75.03% | +0.07 | -0.10 to +0.25 | 74 | 84 | 0.47 |
| MMLU, 14,042 items | all three | 75.05% | +0.09 | -0.09 to +0.26 | 73 | 85 | 0.38 |

"Lost" counts the items that the 16-bit model answers correctly and the cell
does not; "gained" counts the reverse. The interval is the 95% interval of
the paired difference in accuracy points. The p-value is the exact McNemar
test on the lost and gained counts. Llama-3.2-3B and Qwen2.5-7B are the base
models, not the instruction-tuned ones, so their accuracy level is lower
than chat-model reports. Qwen3-8B is the released chat model, given the same
plain few-shot prompt.

The bricks do change single answers. An item flips when one of two cells
answers it correctly and the other does not. Against the 16-bit model, 2.5%
to 3.8% of the MMLU questions flip on Llama-3.2-3B, 1.1% to 1.3% on
Qwen2.5-7B and 0.3% to 1.1% on Qwen3-8B. On GSM8K, 3.6% to 8.6% of the
problems flip on the first two models and 1.3% to 2.7% on Qwen3-8B. In 26 of
the 30 cells the counts of lost and gained items are consistent with an even
split, which is what a small perturbation does to items that the model
answers by a narrow margin.

The test resolves about 0.1 to 0.3 points on MMLU and 0.6 to 1.6 points on
GSM8K; a smaller effect would not show. On Qwen2.5-7B every cell with int8
weights or the asymmetric cache is below the 16-bit model on MMLU, by 0.06
to 0.21 points, and the interval of one of these cells excludes zero. With
two other int8 group sizes, the two cells with int8 weights are from 0.05
points below to 0.12 points above; the table of roundings below gives them.
On GSM8K the same cells have both signs, from 0.15 points below to 1.06
points above. For the full stack the test cannot exclude a loss of 0.27
points on MMLU and 1.5 points on GSM8K.

On Llama-3.2-3B the differences have both signs, and for the full stack the
test cannot exclude a loss of 0.12 points on MMLU and 1.7 points on GSM8K.

On Qwen3-8B the five MMLU cells are within 0.09 points of the 16-bit model,
and for the full stack the test cannot exclude a loss of 0.09 points. On
GSM8K the full stack is 0.45 points above the 16-bit model, and the test
cannot exclude a loss of 0.41 points. The asymmetric cache alone is the only
cell of that table below the 16-bit model, by 0.45 points, and the test
cannot exclude a loss of 1.2 points.

### A test of each table as a whole

The comparisons of one table share the 16-bit cell, so they are not
independent. One test per table does not have this problem. It asks if all
cells of the table have the same accuracy, and it makes no assumption on how
the cells depend on each other.

| table | cells | lowest cell | highest cell | range, points | p, equal accuracy |
|---|---:|---|---|---:|---:|
| Qwen2.5-7B, GSM8K | 6 | certified head, 79.83% | asymmetric KV, 82.18% | 2.35 | 0.007 |
| Qwen2.5-7B, MMLU | 6 | int8 weights + asymmetric KV, 74.04% | certified head, 74.28% | 0.24 | 0.20 |
| Llama-3.2-3B, GSM8K | 6 | int8 weights + asymmetric KV, 29.26% | asymmetric KV, 30.17% | 0.91 | 0.80 |
| Llama-3.2-3B, MMLU | 6 | asymmetric KV, 56.49% | all three, 56.77% | 0.28 | 0.67 |
| Qwen3-8B, GSM8K | 6 | asymmetric KV, 88.48% | int8 weights, 90.30% | 1.82 | 0.003 |
| Qwen3-8B, MMLU | 6 | int8 weights, 74.94% | all three, 75.05% | 0.11 | 0.87 |

The test is the Wald test of equal accuracy of the cells on paired items
(Bhapkar's test); its degrees of freedom are the number of cells minus one.
The range is the accuracy of the highest cell minus the accuracy of the
lowest cell.

Four tables show no difference among their cells. The two GSM8K tables of
the Qwen models do, with p 0.007 and p 0.003; after a correction for the six
tables these are 0.04 and 0.02. The value for Qwen2.5-7B is at the limit:
Cochran's Q test, which assumes that all pairs of cells depend on each other
equally, gives 0.011 for it, or 0.06 after the correction, and 0.0005 for
Qwen3-8B. In both tables the largest difference is between two cells with
bricks, and the 16-bit cell is between them. On Qwen2.5-7B the asymmetric
cache alone solves 2.35 points more than the certified head alone: of the 65
problems that only one of the two cells solves, it solves 48. On Qwen3-8B
int8 weights alone solve 1.82 points more than the asymmetric cache alone:
of the 42 problems that only one of the two solves, they solve 33. The
asymmetric cache is the highest cell of one table and the lowest cell of the
other.

At these p-values the sample of problems is an unlikely cause of the two
ranges. The test does not exclude that cause, and it does not tell what the
cause is. A cell is one set of rounded numbers on one GPU. The test is
evidence that two such sets differ in GSM8K accuracy on the test problems;
it does not show that a brick is the cause. On MMLU the control of the first
loss below gives the scale for one cell: four roundings of the same pair of
bricks span 0.33 points, from 74.04% to 74.37%, which is more than the 0.24
points between the lowest and the highest cell of that table. The intervals
in the tables cover the sample of items and not the choice of rounding. On
GSM8K, a difference of 1 to 2 points between two cells of a Qwen model is
therefore not, alone, evidence about a brick.

A control for these two tables runs the cells in question on other GSM8K
problems, which are training problems that are not prompt examples: the
16-bit cell, the certified head and the asymmetric cache on 3,957 problems
for Qwen2.5-7B, and the same cells with int8 weights on 2,638 problems for
Qwen3-8B. The rule for the test was published before the cells with the
asymmetric cache ran. If the test rejects equal accuracy on the other
problems and the same cells are the lowest and the highest, the differences
belong to the cells. If it rejects equal accuracy with other cells at the
two ends, the cells differ and their order depends on the set of problems.
If the test finds no difference there, the differences of the test tables
are not reproduced.

| model | GSM8K problems | accuracy of each cell | lowest cell | highest cell | range, points | p, equal accuracy |
|---|---|---|---|---|---:|---:|
| Qwen2.5-7B | 1,319 test problems, the run in the table | 16-bit model 81.12%; asymmetric KV 82.18%; certified head 79.83% | certified head | asymmetric KV | 2.35 | <0.001 |
| Qwen2.5-7B | 1,319 test problems, earlier run on the W7900 | 16-bit model 81.43%; asymmetric KV 80.82%; certified head 80.52% | certified head | 16-bit model | 0.91 | 0.27 |
| Qwen2.5-7B | 3,957 other problems | 16-bit model 87.54%; asymmetric KV 87.36%; certified head 87.82% | asymmetric KV | certified head | 0.45 | 0.29 |

The test of equal accuracy over the cells that are measured on the other
problems. The first row of a model gives these cells in the run of the task
table. The cells were selected because two of them are the lowest and the
highest cell of that table, so the p-value of the first row overstates the
evidence. The other rows have no such selection. The earlier run is the
W7900 run of the controls below: another GPU, batches of 32 prompts and one
FP8 scale per value tensor.

The control of Qwen2.5-7B is done, and it does not reproduce the differences
of that table. On the 3,957 other problems the test finds no difference
among the three cells (p 0.29). The cells are within 0.45 points, and their
order is reversed: the certified head is the highest cell and the asymmetric
cache is the lowest. The asymmetric cache changes the text of 1,316 of the
3,957 completions and the outcome of 151 problems, with 79 lost and 72
gained. Against the certified head it is 0.45 points below, with a 95%
interval from 1.09 points below to 0.18 points above. On the test problems
it is 2.35 points above, which is far outside that interval. By the rule,
the differences among the cells of the Qwen2.5-7B table are not reproduced.
The control of Qwen3-8B is in the queue.

An earlier run gives a second view, which was not planned as a control. It
scored the same three cells on the test problems with other arithmetic. The
test finds no difference among the cells there either (p 0.27). The
asymmetric cache is 0.61 points below the 16-bit cell in that run, with an
interval that contains zero. Its high result in the task table repeats
neither on the same problems with other arithmetic nor on other problems.
The certified head is below the 16-bit cell in both runs of the test
problems, by 1.29 and 0.91 points, and 0.28 points above it on the other
problems. The other problems are training problems, and the model may have
seen them. For the certified head the evidence therefore cannot tell the
sample of test problems from a loss that exists only on problems the model
has not seen. The next subsection gives that control and the rules for the
single cells whose interval against the 16-bit model excludes zero.

### The four intervals that exclude zero

With 30 comparisons at the 95% level, one or two intervals are expected to
exclude zero when no cell differs from the 16-bit model. Four do: two losses
and two gains. If the comparisons were independent, four or more would occur
in about one set of 16. They are not independent: the cells of one table
share the 16-bit cell, and the two gains are in one table. With a Bonferroni
correction over the 30 comparisons none of the four is significant: the
smallest p-value becomes 0.12. Three of the four are in the two GSM8K tables
whose cells differ among themselves in the test above. Each of the four has
a control run. The controls of the two losses are done, and neither
reproduces its loss. The controls of the two gains are in the queue.

The two gains are in the GSM8K table of Qwen3-8B. int8 weights solve 90.30%
of the problems against 88.93% for the 16-bit model, with 27 problems gained
and 9 lost (p 0.004). The certified head alone solves 89.76%, with 14 gained
and 3 lost (p 0.013). A second run of the 16-bit cell on the same GPU gives
the same outcome on all 1,319 problems, so the gains are not a variation of
the reference from run to run.

The two cells are not independent evidence. They share the 16-bit cell, and
9 of the 14 problems that the certified head gains are among the 27 that
int8 weights gain; about 3 would be common if the two sets were independent.
The gains do not add. The full stack has both bricks and solves 89.39%, with
20 problems gained and 14 lost. It is 1.29 points below the sum of the
differences of its three bricks alone, with a 95% interval from 2.59 points
below to 0.01 points above. In this table the outcome of 65 problems differs
among the six cells. The 16-bit cell solves 25 of the 65, and the five cells
with bricks solve 19, 31, 32, 36 and 43.

Two readings fit these counts. In the first, a difference belongs to one set
of rounded numbers: this set of int8 weights solves more of the problems
that the model solves by a narrow margin, this FP8 cast of the values solves
fewer, and another set would land elsewhere. In the second, a brick raises
GSM8K accuracy on this model. Rounding the weights is not expected to raise
accuracy, and the other models do not show it on this task: int8 weights are
0.3 points above the 16-bit model on Llama-3.2-3B and 0.15 points below it
on Qwen2.5-7B. On MMLU the same weights are level with the 16-bit model on
Qwen3-8B: 74.94% against 74.96%, with 63 questions lost and 60 gained. For
the certified head a gain has a possible cause, because the float32 logits
are the more exact ones. But the same head is level with the 16-bit model on
the MMLU questions of this model, 74.98% against 74.96% with 18 questions
lost and 21 gained, and it is 1.29 points below on the GSM8K test problems
of Qwen2.5-7B. The evidence does not yet tell the two readings apart.

Three controls are queued. The first runs the 16-bit cell, the certified
head, the asymmetric cache and int8 weights on the 2,638 other GSM8K
problems. It runs on the RTX 4090, so it changes the problems and the GPU.
The second runs the 16-bit cell, the certified head and int8 weights on the
test problems on that GPU. If the first control does not reproduce a gain,
the second tells if the gain belongs to the test problems or to the
arithmetic of the W7900. The third runs int8 weights with two other group
sizes on the W7900, which round the same weights to other grids. The rules
are set before the runs. A gain is not reproduced if the interval on the
other problems contains zero. A gain of the certified head that returns on
the other problems, with an interval that excludes zero, is a property of
the head on this model. A gain of int8 weights is a property of the brick on
this model only if it returns on the other problems and with both other
group sizes. If it returns on the other problems and not with the other
group sizes, it belongs to one set of int8 weights.

The first loss: int8 weights with the asymmetric cache on Qwen2.5-7B answer
74.04% of the MMLU questions against 74.25%, with 99 questions lost and 70
gained (p 0.03). Other roundings of the same cell do not reproduce the loss.
The control was defined before it ran. It measures the same cell with int8
group sizes 64 and 256, which round the same weights to other grids. The
rule: if every rounding is below the 16-bit model and the mean is near 0.2
points, the pair of bricks has a real cost; if the signs are mixed or the
mean is near zero, the 0.21 points belong to one rounding.

The table below gives four roundings of the cell: the run of the task table,
the same weights with one FP8 scale per value tensor, and the two other
group sizes. Three of the four are below the 16-bit model, by 0.21, 0.07 and
0.05 points, and the cell with group size 64 is 0.12 points above it. The
signs are mixed. The mean of the four is 0.05 points below the 16-bit model,
with a 95% interval from 0.19 points below to 0.09 points above. The run of
the task table was selected for its result and the three others were not;
their mean is 0.00 points, with an interval from 0.14 points below to 0.14
points above. The loss does not follow the coarseness of the grid: group
size 256 has one scale for twice as many weights as group size 128, and it
is 0.05 points below the 16-bit model. Fewer than half of the 99 lost
questions are wrong again with any other rounding: 36 to 48 of them. int8
weights alone give the same picture: 0.13 points below with group size 128,
0.01 points below with 64 and 0.01 points above with 256.

By the rule, the 0.21 points belong to one rounding and are not a cost of
the pair of bricks. The control does not prove a cost of zero: it cannot
exclude a mean loss of 0.14 points. The correction over the five cells of
this model and task, which is the rule that `make fdec-report` applies, gave
the same answer for the run of the table (p 0.15).

| model and task | cell | rounding | accuracy | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B, MMLU | int8 weights | group size 128, the run in the table | 74.12% | -0.13 | -0.31 to +0.05 | 95 | 77 | 0.19 |
| Qwen2.5-7B, MMLU | int8 weights | group size 64 | 74.24% | -0.01 | -0.18 to +0.16 | 75 | 74 | 1.00 |
| Qwen2.5-7B, MMLU | int8 weights | group size 256 | 74.26% | +0.01 | -0.17 to +0.19 | 83 | 84 | 1.00 |
| Qwen2.5-7B, MMLU | int8 weights | mean of the three roundings | | -0.04 | -0.19 to +0.11 | | | |
| Qwen2.5-7B, MMLU | int8 weights | mean of the two without the run in the table | | +0.00 | -0.16 to +0.16 | | | |
| Qwen2.5-7B, MMLU | int8 weights + asymmetric KV | group size 128, the run in the table | 74.04% | -0.21 | -0.39 to -0.03 | 99 | 70 | 0.03 |
| Qwen2.5-7B, MMLU | int8 weights + asymmetric KV | group size 128, one FP8 scale per tensor | 74.18% | -0.07 | -0.25 to +0.11 | 91 | 81 | 0.49 |
| Qwen2.5-7B, MMLU | int8 weights + asymmetric KV | group size 64 | 74.37% | +0.12 | -0.06 to +0.30 | 75 | 92 | 0.22 |
| Qwen2.5-7B, MMLU | int8 weights + asymmetric KV | group size 256 | 74.20% | -0.05 | -0.23 to +0.13 | 86 | 79 | 0.64 |
| Qwen2.5-7B, MMLU | int8 weights + asymmetric KV | mean of the four roundings | | -0.05 | -0.19 to +0.09 | | | |
| Qwen2.5-7B, MMLU | int8 weights + asymmetric KV | mean of the three without the run in the table | | +0.00 | -0.14 to +0.14 | | | |

Each row is one cell of a run of the 14,042 MMLU questions on the same RTX
4090. The group size is the number of weights that share one int8 scale. The
row with one FP8 scale per tensor has the int8 weights of the run in the
table and another cast of the cached values. The 16-bit cell answers every
question the same way in the four runs. A mean row gives the mean of the
paired differences over the roundings of that cell.

The second loss: the certified head alone on Qwen2.5-7B solves 79.83% of
GSM8K against 81.12%, with 32 problems lost and 15 gained (p 0.019). This
cell changes no weight and no cached value. The certified head returns the
token with the largest float32 logit, and the 16-bit head returns the token
with the largest 16-bit logit. The two heads differ only where the largest
16-bit logits tie; the 16-bit head then returns the token with the lower
index. A check on Qwen2.5-0.5B agrees: the two heads select different tokens
at 59 of 6,583 prompt positions, and each of the 59 is such a tie. A real
effect would therefore not be a loss of information, because the float32
logits are the more exact ones. It would mean that the token with the lower
index is the better choice at a tie on this model and task.

The evidence is mixed. An earlier run of the same two cells on the W7900 has
the certified head 0.91 points below the 16-bit model (34 lost, 22 gained, p
0.14); 18 problems flip in both runs. In the run of the table, the full
stack differs from the cell with int8 weights and the asymmetric cache only
by the head, and it is 0.38 points below that cell (31 lost, 26 gained). On
MMLU the certified head is 0.04 points above the 16-bit model on Qwen2.5-7B,
and on Llama-3.2-3B it is 0.14 and 0.23 points above on the two tasks. On
Qwen3-8B it is 0.02 points above on MMLU and 0.83 points above on GSM8K,
which is the second gain above. With a Bonferroni correction over the five
cells of this model and task, the loss is not significant (p 0.09).

The control does not reproduce this loss. It ran the same two cells on 3,957
other GSM8K problems, on the same GPU with the same batch size. The other
problems are the first 3,957 training problems of GSM8K after the eight
prompt examples. The rule was set before the run: a loss with an interval
that excludes zero is a property of the head on this model, and an interval
that contains zero means that the loss is not reproduced. On the other
problems the certified head solves 87.82% against 87.54% for the 16-bit
model, with 34 problems lost and 45 gained (p 0.26). It is 0.28 points
above, with an interval from 0.16 points below to 0.72 points above. The
interval contains zero, and it excludes a loss of the size seen on the test
problems.

| model | cell | GSM8K problems | 16-bit model | cell | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | certified head | 1,319 test problems, the run in the table | 81.12% | 79.83% | -1.29 | -2.31 to -0.27 | 32 | 15 | 0.02 |
| Qwen2.5-7B | certified head | 3,957 other problems | 87.54% | 87.82% | +0.28 | -0.16 to +0.72 | 34 | 45 | 0.26 |
| Qwen2.5-7B | asymmetric KV | 1,319 test problems, the run in the table | 81.12% | 82.18% | +1.06 | -0.13 to +2.25 | 25 | 39 | 0.10 |
| Qwen2.5-7B | asymmetric KV | 3,957 other problems | 87.54% | 87.36% | -0.18 | -0.79 to +0.43 | 79 | 72 | 0.63 |

Each cell is compared with the 16-bit cell of the same run, first on the
test problems of the table above and then on the other problems. Both runs
used one RTX 4090 and batches of 8 prompts. The rows of the asymmetric cache
belong to the control of the test per table above. The run on the other
problems is at knlp commit
[`913f411b5`](https://github.com/mcgrof/knlp/commit/913f411b5), which adds
the choice of the problem set to the tool.

The completions of this run show what the head changes. The two cells write
different text for 864 of the 3,957 problems (21.8%). At the first token
that differs, the token of the 16-bit head has the lower index in all 864
problems, which is what a tie of the largest 16-bit logits gives. Both cells
solve 688 of the 864 problems and neither solves 97; 34 are lost and 45 are
gained. A tie changes the wording of a solution far more often than its
answer, and on these problems neither choice at a tie is better within the
resolution of the control, which is 0.44 points.

The control has one limit. The other problems are training problems of the
dataset, and the model may have seen them: the 16-bit model solves 87.54% of
them against 81.12% of the test problems, and the head changes the outcome
of 2.0% of them against 3.6% of the test problems. A loss that exists only
on problems the model has not seen would not show in this control. The test
problems alone do not establish such a loss. The comparison was selected
among all comparisons of this section because its interval excludes zero,
and it is not significant after a correction.

### Controls: how much does a rounding change do?

Four controls give the scale for these differences. None of them changes a
weight or the storage format of the cache.

| what changes | model | task | items that flip | accuracy change, points |
|---|---|---|---:|---:|
| Second run of the 16-bit cell and of the int8 cell | Llama-3.2-3B | GSM8K | 0 of 1,319 in each cell | 0.00 |
| Second run of the 16-bit cell and of the int8 cell | Llama-3.2-3B | MMLU | 0 of 14,042 in each cell | 0.00 |
| Second run of the 16-bit cell and of the int8 cell | Qwen2.5-7B | MMLU | 0 of 14,042 in each cell | 0.00 |
| Second run of the 16-bit cell | Qwen3-8B | GSM8K | 0 of 1,319 | 0.00 |
| 16-bit cell and int8 cell on another GPU, PyTorch version and batch size | Qwen2.5-7B | GSM8K | 58 and 56 of 1,319 (4.4% and 4.2%) | -0.30 and +0.45 |
| float32 logits in place of 16-bit logits (certified head) | Qwen2.5-7B | GSM8K | 47 of 1,319 (3.6%) | -1.29 |
| float32 logits in place of 16-bit logits (certified head) | Qwen2.5-7B | MMLU | 155 of 14,042 (1.1%) | +0.04 |
| float32 logits in place of 16-bit logits (certified head) | Llama-3.2-3B | GSM8K | 87 of 1,319 (6.6%) | +0.23 |
| float32 logits in place of 16-bit logits (certified head) | Llama-3.2-3B | MMLU | 400 of 14,042 (2.8%) | +0.14 |
| float32 logits in place of 16-bit logits (certified head) | Qwen3-8B | GSM8K | 17 of 1,319 (1.3%) | +0.83 |
| float32 logits in place of 16-bit logits (certified head) | Qwen3-8B | MMLU | 39 of 14,042 (0.3%) | +0.02 |
| Direct FP8 cast of the values in place of one scale per tensor | Llama-3.2-3B | GSM8K | 83 of 1,319 (6.3%) | +1.29 |
| Direct FP8 cast of the values in place of one scale per tensor | Llama-3.2-3B | MMLU | 390 of 14,042 (2.8%) | -0.03 |
| Direct FP8 cast of the values in place of one scale per tensor | Qwen2.5-7B | MMLU | 157 of 14,042 (1.1%) | +0.08 |

The accuracy change is the second condition minus the first. The second run
used a later commit of the tool on the same GPU with the same batch size.
The second run of Qwen3-8B is on the W7900 and has the 16-bit cell; its int8
cell is in the queue. The other second runs are on the RTX 4090. The run on
another GPU is the earlier W7900 run of Qwen2.5-7B against the RTX 4090 run
of the table; the row gives the 16-bit cell and the int8 cell. The certified
head is compared with the 16-bit cell of the same run. The two FP8 casts are
compared on the asymmetric KV cell.

A second run of the same cell on the same GPU gives the same outcome on
every item, on the RTX 4090 and on the W7900, so a flip comes from changed
arithmetic and not from the run. A change of GPU is enough to change the
arithmetic. The 16-bit Qwen2.5-7B, with no brick, answers 58 of the 1,319
GSM8K problems differently on the second GPU (4.4%), and its accuracy moves
by 0.3 points. The difference of a cell from the 16-bit cell moves too: int8
weights are 0.91 points below the 16-bit model in the W7900 run and 0.15
points below in the RTX 4090 run.

The certified head changes no weight and no cached value; it replaces the
16-bit logits of the last position with float32 logits, and it still flips
1.3% to 6.6% of the GSM8K problems and 0.3% to 2.8% of the MMLU questions.
The two FP8 casts of the values differ by 0.1% in perplexity, and they
disagree on 6.3% of the GSM8K problems: on Llama-3.2-3B the asymmetric cache
solves 28.9% with one cast and 30.2% with the other. One changed token can
change the rest of a greedy completion. On MMLU the two casts disagree on
1.1% to 2.8% of the questions.

On Llama-3.2-3B and Qwen2.5-7B, int8 weights and the asymmetric cache, alone
or together, flip 4.2% to 6.4% of the GSM8K problems and move accuracy by
1.1 points or less. A change of GPU or of rounding does as much. The MMLU
counts agree: these cells flip 1.1% to 1.2% of the questions on Qwen2.5-7B
and 2.5% to 3.0% on Llama-3.2-3B, and a rounding change flips 1.1% and 2.8%.
On these two models, the count of flips cannot tell the effect of a brick
from the effect of a rounding change. Only the balance of lost and gained
items can. It differs from an even split in the four cells of the subsection
above. In two of the four the imbalance does not repeat: in one with other
roundings of the same weights, and in the other on other problems.

### Do the bricks compose on task accuracy?

They do, within the resolution of the test, with one case at the limit. The
table gives the accuracy change of each combined cell minus the sum of the
changes of its bricks alone. Every interval contains zero. The interval of
the full stack of Qwen3-8B on GSM8K ends 0.01 points above zero: that cell
is 1.29 points below the sum of its bricks alone. The sum holds the two
single-brick gains of that table, and the full stack does not repeat them.
It is still 0.45 points above the 16-bit model.

| model | task | cell | cell minus sum of its bricks alone, points | 95% interval |
|---|---|---|---:|---|
| Qwen2.5-7B | GSM8K | int8 weights + asymmetric KV | -0.68 | -2.23 to +0.87 |
| Qwen2.5-7B | GSM8K | all three | +0.23 | -1.77 to +2.23 |
| Qwen2.5-7B | MMLU | int8 weights + asymmetric KV | -0.01 | -0.25 to +0.22 |
| Qwen2.5-7B | MMLU | all three | +0.08 | -0.25 to +0.40 |
| Llama-3.2-3B | GSM8K | int8 weights + asymmetric KV | -1.21 | -3.01 to +0.58 |
| Llama-3.2-3B | GSM8K | all three | -1.44 | -3.86 to +0.98 |
| Llama-3.2-3B | MMLU | int8 weights + asymmetric KV | +0.16 | -0.20 to +0.53 |
| Llama-3.2-3B | MMLU | all three | +0.13 | -0.35 to +0.61 |
| Qwen3-8B | GSM8K | int8 weights + asymmetric KV | -0.38 | -1.46 to +0.70 |
| Qwen3-8B | GSM8K | all three | -1.29 | -2.59 to +0.01 |
| Qwen3-8B | MMLU | int8 weights + asymmetric KV | +0.08 | -0.13 to +0.29 |
| Qwen3-8B | MMLU | all three | +0.07 | -0.16 to +0.30 |

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
training-to-training variance. The stored values are cast to FP8 directly,
as the [matrix](#the-fp8-cast-of-the-values) does.

| reader weights | cartridge storage | loss, patient 1 | loss, patient 2 |
|---|---|---:|---:|
| 16-bit | 16-bit keys and values | 0.015102 | 0.016170 |
| int8 | 16-bit keys and values | 0.015111 (+0.06%) | 0.016181 (+0.07%) |
| 16-bit | 16-bit keys, FP8 values | 0.015106 (+0.03%) | 0.016177 (+0.04%) |
| int8 | 16-bit keys, FP8 values | 0.015118 (+0.11%) | 0.016191 (+0.13%) |
| int8 | FP8 keys and values | 0.015310 (+1.38%) | 0.016385 (+1.33%) |
| 16-bit, no cartridge | none | 0.043067 | 0.045057 |

With both bricks the loss is 0.11% and 0.13% above the 16-bit reading, with
95% intervals from 0.08% to 0.15% and from 0.09% to 0.16%. The cost of both
bricks minus the sum of the two costs alone is 0.02% of the loss for each
document, with intervals from -0.02% to +0.06% and from -0.01% to +0.05%.
Both intervals contain zero, so the two bricks compose within the resolution
of the test. [`scripts/fdec_cart_cells.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cart_cells.py)
computes the intervals with a bootstrap over the packed sequences of the
test, 110 for patient 1 and 91 for patient 2. With one FP8 scale per value
tensor, which this page used before, the cost of both bricks is 0.10% and
0.11%. A second scoring of the 16-bit cartridge returns the same loss to
every digit with both readers, so the differences in the table do not come
from variation between runs.

The loss without a cartridge is 2.8 to 2.9 times the loss with it, with
either reader, so the cartridge keeps its value under int8 weights.
Symmetric FP8 keys still cost more than 1%, the same penalty the
[cartridge quantization study](cartridge-asymmetric-quantization.html)
found with a full-precision reader.

These cartridges saw the scored conversations during training, so the loss
level is in-sample. The paired differences between readings do not depend
on that. The certified head is exact for any hidden state, so it needs no
separate cartridge test.

## What the measurements corrected

Each stricter measurement found an error in an earlier result. Both
corrections are recorded here, because they are the reason for the rules of
the method.

### A published byte saving was in-sample

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
configuration reads 51% to 56% of the head on unseen text, and it is the
one the matrix now uses. The [certified decode page](lm-head-decode.md)
carries the full sweep.

This is the reason for the rule that a brick counts only after the matrix
has measured it held-out and in combination.

### The first matrix rounded attention scores to 16 bits

The first version of the matrix gave the 16-bit Qwen2.5-7B model a
perplexity of 9.84. The correct value is 8.44. The task-accuracy run
exposed the error: the same 16-bit model solved only 18% of the GSM8K
problems.

The harness replaces the attention function of the model, because the
asymmetric cache changes how values are read. For the unmodified model it
called the plain attention function of the transformers library. That
function rounds the query-key scores to the model data type before the
softmax. bfloat16 keeps 8 significant bits, so a score near 700 moves in
steps of 4. Qwen2.5-7B has a mean absolute score of 713 in its last layer,
and the rounding moves a score there by 1.4 on average. A fused attention
kernel accumulates the scores in float32 and does not have this error. The
fused kernel is the default path of the library and of a serving engine.

| attention arithmetic, 16-bit Qwen2.5-7B | perplexity | GSM8K problems solved, of 32 |
|---|---:|---:|
| fused kernel, the library default | 7.462 | 24 |
| plain function, scores rounded to 16 bits | 8.843 | 7 |
| float32 scores and value sum, 16-bit output | 7.463 | 25 |

The perplexity is on WikiText-2 test, 8 sequences of 512 tokens, and the
problems are the first 32 of the GSM8K test set, on one RTX 4090. The same
check on one Radeon Pro W7900 gives 7.459, 8.801 and 7.459, and 25, 7 and
25 problems.
Llama-3.2-3B and Qwen3-8B have attention scores below 50, and the three
paths agree within 0.03 in perplexity on both.

Every cell now computes attention with float32 scores and a float32 value
sum, the 16-bit model included. The asymmetric cache cell therefore differs
from its reference only in the precision of the values.

All three models were measured again. Llama-3.2-3B and Qwen3-8B moved by
less than 0.03 in perplexity. Qwen2.5-7B changed in two ways. Its 16-bit
perplexity is 8.44 and not 9.84. And it is no longer the weakest model: the
first matrix showed int8 weights changing 8.3% of its predicted tokens, and
with correct attention they change 2.4%, close to the 1.7% and 2.0% of the
other two models. With scores rounded to 16 bits, a small change to the
model moves many scores across a rounding step, which is the likely source
of the 8.3%.
[`scripts/fdec_attn_precision.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_attn_precision.py)
runs this check on any model.

## Speed so far

The matrix shows that the bricks agree in what they compute. Speed needs
each brick in a real kernel, and two measurements exist.

Two bricks have run together in one serving engine. Packed int4 weights and
the K16/V8 cache ran in one vLLM and FlashInfer process on Qwen3-8B, with
8,192 input tokens at batch 32 on one H100. Request throughput was 3.45%
higher than the same weights with a 16-bit cache, against 3.55% with dense
weights, so the cache brick keeps its gain when the weights are packed
([asymmetric KV results](asymmetric-results.html#throughput)). That
workload is mostly prefill and its weights are int4, so it does not test
the decode byte model here.

The certified head has Triton kernels but is not in a serving engine. On
hidden states it has not seen it runs at 0.99 to 1.28 times the speed of
the dense head in the mean, and on three of the four models its
95th-percentile token takes about 1.8 times as long.
[`scripts/fdec_cert_latency.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cert_latency.py)
builds the full-rank shadow from WikiText-2 train hidden states and times
1,000 WikiText-2 test positions, one token at a time, against the dense
16-bit head on the same GPU.

| model | GPU | dense head, mean | certified head, mean | speed, mean | speed, median | p95, dense → certified | certified | head read |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | RTX 4090 | 1.166 ms | 1.087 ms | 1.07× | 1.23× | 1.17 → 2.11 ms | 88.0% | 73.1% |
| Llama-3.2-3B | RTX 4090 | 0.867 ms | 0.678 ms | 1.28× | 1.30× | 0.87 → 0.67 ms | 98.6% | 57.6% |
| Qwen3-8B | Radeon Pro W7900 | 2.002 ms | 1.978 ms | 1.01× | 1.16× | 2.01 → 3.68 ms | 87.5% | 73.5% |
| Qwen2.5-0.5B | RTX 4090 | 0.311 ms | 0.315 ms | 0.99× | 1.04× | 0.32 → 0.58 ms | 95.0% | 67.1% |

The timed variant makes one fused fetch of the vocabulary blocks with the
highest bounds and checks the certificate once. A token that does not
certify falls back to the dense head, and "head read" counts the whole head
for it. The variant that reads the fewest bytes is not the fastest: opening
one block at a time reads 51% to 56% of the head, the figure in the matrix,
but runs at 0.72 to 1.16 times the dense head because each block costs a
kernel launch. A dense head at batch 1 takes 0.3 to 2 ms, so there is
little time to win.

The earlier figures of 1.97 and 1.85 times came from the in-sample shadow.
The same in-sample setup on these GPUs gives 1.56 to 1.78 times on the
three larger models, so the difference is the shadow and not the kernel.
The [certified decode page](lm-head-decode.md#latency-on-held-out-hidden-states)
has the details.

## Running it

```
make defconfig-fdec
make
```

`defconfig-fdec` selects Qwen2.5-7B and Llama-3.2-3B with all three dense
bricks. `make` runs [`scripts/fdec_compose.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_compose.py)
once per model, writes one JSON result per model and a `REPORT.md` table
under `results/fdec`, and exits non-zero if any cell fails to compose. The
report gives the interaction term of each model with its 95% interval. A
model with a result file is skipped, so an interrupted run resumes. The
7B model needs a GPU with 24 GiB; each model takes two to five minutes.

`make defconfig-fdec-smoke && make` runs the same eight cells on
Qwen2.5-0.5B on the CPU in under three minutes. It checks the harness, not
the numbers: its calibration set is small.

`make fdec-tasks` adds the task check. It runs
[`scripts/fdec_task_acc.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_task_acc.py)
once per model: GSM8K and MMLU for the 16-bit model, each enabled brick
alone and the stack, on the same items. It is not part of the default
target, because it takes about four hours for the 3B model on one RTX 4090.
`make fdec-report` then adds the task tables to `REPORT.md` and fails if a
cell is significantly below the 16-bit model. The test is the exact McNemar
test on the items that changed, with a Bonferroni correction over the cells
of one model and task, at 0.05. When a cell differs from the 16-bit model
on the test split, you can check it on other problems:
`scripts/fdec_task_acc.py --gsm-split train` scores the 7,465 GSM8K
training problems that are not prompt examples.

The options live in [`Kconfig.fdec`](https://github.com/mcgrof/knlp/blob/main/Kconfig.fdec). `CONFIG_FDEC_MODELS`
takes a space-separated list of Hugging Face model ids. Each brick has its
own switch, and the harness measures every subset of the enabled bricks.

Two more tools cover what the matrix does not.
[`scripts/fdec_cert_heldout.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cert_heldout.py) sweeps the
certificate rank on held-out hidden states. The cartridge cells use
[`cas_cart_loss.py`](https://github.com/mcgrof/knlp/blob/main/research/cartridges_cas/scripts/cas_cart_loss.py)
with `W8=1` for the int8 reader and
[`cas_cart_quantize.py`](https://github.com/mcgrof/knlp/blob/main/research/cartridges_cas/scripts/cas_cart_quantize.py)
with `SCALE=none` for the stored cartridge; they need a trained cartridge
and its self-study conversations.
[`scripts/fdec_cart_cells.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_cart_cells.py)
turns the two result files of a document into the cell table with its
intervals.

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
weights are rounded in place, values pass through a direct FP8 cast, attention
runs with float32 arithmetic, and the certificate is replayed against
float32 logits. It shows that the bricks do
not conflict in what they compute. It does not measure speed.

The bytes-saved column is a model of bytes read, not a latency. Turning it
into latency needs each brick in a real kernel in one serving stack. The
asymmetric KV cache has that today in the modified vLLM and FlashInfer
branches described in [decode reproduction](decode-reproduction.md). No
serving engine runs all the bricks together, and no measurement yet
compares the decode latency of the full stack with the byte model.

The perplexity evaluation is WikiText-2 only. The weight study also used a
cross-domain calibration check; the matrix does not yet repeat it. The
task tables are complete. The control runs for two of the four
comparisons whose interval excludes zero are not finished.

Linear-attention models are not in the matrix. The cartridge cells cover
one model and two documents.
