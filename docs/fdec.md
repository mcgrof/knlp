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
at a perplexity within 0.3% of the unmodified model. A fourth brick,
the trained cartridge, composes with int8 weights and with the
asymmetric KV cache at a combined loss cost of 0.1%.

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
identical in the two runs. The per-tensor variant was measured on the W7900
and on one RTX 4090 with PyTorch 2.10, and the two GPUs agreed within 0.008
in perplexity and 0.002 in top-1 agreement in every cell. `--kv-scale
tensor` selects it.

## Task accuracy

On Llama-3.2-3B and Qwen2.5-7B, int8 weights, the asymmetric cache and the
two together leave task accuracy within the noise of the test. On
Llama-3.2-3B, MMLU accuracy is 56.57% for the 16-bit model and 56.63% with
both bricks, and GSM8K accuracy is 29.4% and 30.3%. On Qwen2.5-7B, MMLU
accuracy is 74.25% and 74.18%, and GSM8K accuracy is 81.4% and 80.5%. Every
paired interval contains zero. The Qwen3-8B runs are not finished.

Perplexity and top-1 agreement are measured on running text. The task check
asks what a user asks: does a model with int8 weights, the asymmetric
cache, or both, answer as many questions correctly as the 16-bit model?
[`scripts/fdec_task_acc.py`](https://github.com/mcgrof/knlp/blob/main/scripts/fdec_task_acc.py)
scores every cell on the same items, so each comparison is paired.

GSM8K is 1,319 grade-school math problems. The prompt holds eight worked
examples, decoding is greedy, and an answer is correct when the first
number after the answer marker equals the reference. A completion with no
answer marker counts as wrong. MMLU is 14,042 four-choice questions with
five examples per subject; the answer is the choice letter with the largest
logit. The certified head is not a cell here. It returns the largest
float32 logit, and these runs take the largest 16-bit logit, so the two can
differ only where the two largest logits tie within the 16-bit rounding
step. The asymmetric KV cells in these tables scale each value tensor
before the FP8 cast; the runs with the direct cast are not finished. The
[matrix](#the-fp8-cast-of-the-values) shows that the two differ by at most
0.001 in top-1 agreement.

All runs are in bfloat16 at knlp commit
[`dd3a68306`](https://github.com/mcgrof/knlp/commit/dd3a68306).
Llama-3.2-3B and the MMLU run of Qwen2.5-7B ran on one RTX 4090 with
PyTorch 2.10 and transformers 5.3.0. The GSM8K run of Qwen2.5-7B used one
Radeon Pro W7900 with PyTorch 2.9.1 and transformers 4.57.1.

The 16-bit references are close to the published scores of the models. The
MMLU accuracy of Llama-3.2-3B averaged over the 57 subjects is 57.95%, and
its [model card](https://huggingface.co/meta-llama/Llama-3.2-3B) reports 58
with five examples. Qwen2.5-7B answers 74.25% of the MMLU items, or 75.46%
averaged over the subjects, and the
[Qwen2.5 report](https://arxiv.org/abs/2412.15115) gives 74.2 with five
examples. It solves 81.4% of GSM8K with the eight examples used here; the
report gives 85.4% with four examples of its own.

### Qwen2.5-7B

| task | cell | accuracy | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|
| GSM8K, 1,319 items | 16-bit model | 81.43% | reference | | | | |
| GSM8K, 1,319 items | int8 weights | 80.52% | -0.91 | -2.13 to +0.31 | 40 | 28 | 0.18 |
| GSM8K, 1,319 items | asymmetric KV | 80.82% | -0.61 | -1.92 to +0.71 | 43 | 35 | 0.43 |
| GSM8K, 1,319 items | int8 weights + asymmetric KV | 80.52% | -0.91 | -2.22 to +0.40 | 45 | 33 | 0.21 |
| MMLU, 14,042 items | 16-bit model | 74.25% | reference | | | | |
| MMLU, 14,042 items | int8 weights | 74.12% | -0.13 | -0.31 to +0.05 | 95 | 77 | 0.20 |
| MMLU, 14,042 items | asymmetric KV | 74.11% | -0.14 | -0.31 to +0.03 | 83 | 63 | 0.12 |
| MMLU, 14,042 items | int8 weights + asymmetric KV | 74.18% | -0.07 | -0.25 to +0.11 | 91 | 81 | 0.49 |

### Llama-3.2-3B

| task | cell | accuracy | difference, points | 95% interval | lost | gained | p |
|---|---|---|---|---|---|---|---|
| GSM8K, 1,319 items | 16-bit model | 29.42% | reference | | | | |
| GSM8K, 1,319 items | int8 weights | 29.72% | +0.30 | -0.97 to +1.58 | 35 | 39 | 0.73 |
| GSM8K, 1,319 items | asymmetric KV | 28.89% | -0.53 | -1.82 to +0.76 | 41 | 34 | 0.49 |
| GSM8K, 1,319 items | int8 weights + asymmetric KV | 30.33% | +0.91 | -0.47 to +2.29 | 37 | 49 | 0.24 |
| MMLU, 14,042 items | 16-bit model | 56.57% | reference | | | | |
| MMLU, 14,042 items | int8 weights | 56.57% | +0.01 | -0.25 to +0.27 | 172 | 173 | 1.00 |
| MMLU, 14,042 items | asymmetric KV | 56.52% | -0.05 | -0.31 to +0.21 | 174 | 167 | 0.75 |
| MMLU, 14,042 items | int8 weights + asymmetric KV | 56.63% | +0.06 | -0.22 to +0.35 | 204 | 213 | 0.70 |

"Lost" counts the items that the 16-bit model answers correctly and the
cell does not; "gained" counts the reverse. The interval is the 95%
interval of the paired difference in accuracy points. The p-value is the
exact McNemar test on the lost and gained counts. The models are the base
models, not the instruction-tuned ones, so the accuracy level is lower than
chat-model reports.

The bricks do change single answers. Against the 16-bit model, 2.4% to 3.0%
of the MMLU items flip on Llama-3.2-3B and 1.0% to 1.2% on Qwen2.5-7B, and
5.2% to 6.5% of the GSM8K problems flip. The flips go both ways in almost
equal numbers, which is what a small perturbation does to items that the
model answers by a narrow margin.

The test resolves about 0.2 to 0.3 points on MMLU and 1.4 points on GSM8K;
a smaller effect would not show. On Qwen2.5-7B all six differences are
negative: 0.07 to 0.14 points on MMLU and 0.6 to 0.9 points on GSM8K. None
is significant, and for the two bricks together the test cannot exclude a
loss of 0.25 points on MMLU and 2.2 points on GSM8K. On Llama-3.2-3B the
differences have both signs.

The bricks also compose on task accuracy. The table gives the change of the
pair minus the sum of the two single changes. Every interval contains zero.

| model | task | pair minus sum of singles, points | 95% interval |
|---|---|---:|---|
| Llama-3.2-3B | MMLU | +0.11 | -0.26 to +0.47 |
| Llama-3.2-3B | GSM8K | +1.14 | -0.61 to +2.89 |
| Qwen2.5-7B | MMLU | +0.20 | -0.04 to +0.44 |
| Qwen2.5-7B | GSM8K | +0.61 | -1.09 to +2.30 |

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
training-to-training variance. The stored values use FP8 with one scale per
tensor.

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
other two models. With scores rounded to 16 bits, a small change to the model moves
many scores across a rounding step, which is the likely source of the 8.3%.
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
under `results/fdec`, and exits non-zero if any cell fails to compose. A
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
of one model and task, at 0.05.

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
task check covers Llama-3.2-3B and Qwen2.5-7B so far, and it does not
include the certified head.

Linear-attention models are not in the matrix. The cartridge cells cover
one model and two documents.
