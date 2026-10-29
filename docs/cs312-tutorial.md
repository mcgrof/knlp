# CS312 tutorial lane: empirical machine learning on a small decoder

This page describes the `cs312` lane in knlp. It runs the sweeps from
Stanford's CS312 "Deep Learning Alchemy" assignments on the course's small
Llama-style decoder, through knlp's usual `make defconfig-* && make`
workflow. The lane exists for two reasons. First, the assignments are a
compact, well-designed course in running experiments properly: one change
at a time, log-spaced grids, noise floors before claims, and scaling ladders
that are checked rather than trusted. Second, the course model family is a
good organism for ideas knlp cares about. A 35M-parameter decoder trains in
minutes, has a built-in depth ladder from d4 to d18, and is standard enough
that later experiments on learned prefixes, cartridges or KV layout can be
tried here before they cost real compute.

Table of contents: [the model organism](#the-model-organism) ·
[what knlp adds](#what-knlp-adds) · [problem 1(a)](#problem-1a-one-hyperparameter-at-a-time) ·
[run it on Modal](#run-it-on-modal) · [run it on a local GPU](#run-it-on-a-local-gpu) ·
[outputs](#outputs-and-the-report) · [extending the lane](#extending-the-lane) ·
[limits](#limits)

## The model organism

The default model, d8, is a decoder-only Transformer with 8 pre-norm
blocks, hidden size 512, 8 heads of dimension 64, rotary position
embeddings, RMSNorm, query-key normalization, a SwiGLU MLP, a 4096-token
vocabulary, a 1024-token context and untied embeddings, about 35M
parameters in total. The course recipe trains it on 600,000 sequences of
1024 tokens (614.4M tokens) with AdamW, betas 0.9 and 0.95, peak learning
rate 0.003, one percent linear warmup, linear decay to zero, batch size 64
sequences, weight decay 0.1 excluding embeddings, norms and biases, and
gradient clipping at norm 1.0. One run takes roughly ten to fifteen minutes
on an H100. Changing the depth changes every other dimension with it:
hidden size is 64 times depth and the MLP is 3.5 times hidden.

## What knlp adds

The course code lives in its own repository and carries no license file,
so knlp does not copy it. The lane pins a commit of that repository, clones
it under a cache directory, builds its uv environment, and drives its own
launchers. knlp contributes the sweep planner, a small entry script that
runs inside the course environment, receipts and a report. The Kconfig
menu "CS312 tutorial" holds every knob: the pinned commit, the problem, the
grid base and size, depth and training horizon, the backend, and the
Weights & Biases project.

Two backends exist. The Modal backend uses the course's launcher
unchanged, so every configuration becomes its own detached function on the
chosen GPU, with the course data read from a Modal Volume. The local
backend runs the same plan on one visible GPU, one process per
configuration, writing a receipt per run so an interrupted sweep resumes
where it stopped. Both backends tag their W&B runs with the plan digest and
a suffix, so the same plan executed twice can be compared run by run.

## Problem 1(a): one hyperparameter at a time

The first assignment asks for the shape of the loss when warmup, learning
rate, batch size and weight decay each move alone along a log axis around
the d8 defaults, while the training horizon stays fixed so every run is
compute matched. The planner builds that grid. With the defaults, learning
rate, weight decay and warmup use base 3 and five points, giving for
example learning rates of 0.000333, 0.001, 0.003, 0.009 and 0.027. Batch
size uses base 2, giving 16, 32, 64, 128 and 256, because batch sizes are
integers and the course launcher expects gradient accumulation above 256.
The default point is shared, so four axes produce seventeen unique runs.
The planner orders them inner ring first: the baseline, then the eight
runs one step away, then the eight runs two steps away, so a sequential
sweep gives a complete coarse picture early.

## Run it on Modal

You need a Modal account, a W&B account, and `uv`. Create a W&B secret
named `dl-alchemy-wandb` holding `WANDB_API_KEY` in the Modal environment
you will use, then:

```sh
make defconfig-cs312-problem-1
make cs312-fetch     # clone the pinned course commit and build its environment
make cs312-data      # download the course dataset into a Modal Volume (once)
make                 # plan and launch the seventeen runs
make cs312-collect   # later: pull losses and timings from W&B
make cs312-report    # write results/cs312-problem-1/report.md
```

`make cs312-data` runs the course's own download module inside a small
CPU container that writes straight into the Volume the course launcher
mounts, so the 20 GB dataset never passes through your machine. Plain
`make` returns as soon as the runs are spawned; the launch receipt holds
the Modal call ids. A W&B entity can be set with
`CONFIG_KNLP_CS312_WANDB_ENTITY`; empty uses the account default.

## Run it on a local GPU

The second defconfig executes the identical plan on the local machine:

```sh
make defconfig-cs312-problem-1-local
make cs312-fetch
make cs312-data      # downloads the dataset under the scratch root
make                 # runs all seventeen configurations in sequence, then reports
```

The course pins CUDA wheels of PyTorch 2.6. On a machine where those do
not fit, for example an AMD GPU with a ROCm build of PyTorch, point
`CONFIG_KNLP_CS312_PYTHON` at an interpreter that already has torch, numpy,
wandb, tqdm, huggingface-hub, sentencepiece, matplotlib and requests, and
`make cs312-fetch` skips the uv step. `CONFIG_KNLP_CS312_SCRATCH_ROOT`
selects where data and checkpoints go. Every Kconfig value can also be
overridden for one invocation with a `CS312_` environment variable, for
example `CS312_NUM_TRAIN_SEQUENCES=4096 CS312_NUM_EVALS=2
CS312_OUTPUT=results/cs312-smoke make` for a quick rehearsal (the course
evaluates 100 times per run, which on a short run means every step); the plan digest changes with the horizon, so a
rehearsal never mixes with the real sweep.

## Outputs and the report

Everything lands under the configured output directory, by default
`results/cs312-problem-1`. `plan.json` is the grid with one `train_config`
per run, exactly the keyword arguments passed to the course's
`TrainConfig`. `launch.json` is the launch receipt: Modal call ids and app
name, or the local device, interpreter and session times. Local runs add
`runs/<run name>/receipt.json` and `train.log`. `results.json` joins the
plan with W&B summaries, receipts and Modal call results, and `report.md`
turns it into one table per axis with the final validation loss, its
difference from the baseline, and minutes per run. `make cs312-report`
accepts `--against` through the runner to place two output directories
side by side, which is how a Modal sweep and a local sweep of the same
plan are compared.

## Extending the lane

Later problems of the assignment, pairs of co-varying hyperparameters,
scheduler interactions, the d4 to d9 scaling ladder, seed variance and
single-token perturbations, fit the same shape: a planner function that
returns runs with `overrides`, a Kconfig choice entry, and nothing else.
The entry script already accepts any `TrainConfig` field in a run's
`train_config`. Keep each problem's plan digest stable by deriving it only
from the overrides, depth and horizon, so results from different backends
and dates remain comparable.

## Limits

The lane reproduces the course's recipe; it does not change it. Numbers
from the d8 organism are specific to a 35M-parameter language model on a
particular tokenized corpus, and the course itself warns that they often
disagree with published results at scale. A validation loss difference is
meaningful only once the seed-to-seed spread is known, which is the subject
of the assignment's later problems, not of this sweep. The local backend
has been written for one GPU; multi-GPU data parallelism is left to the
course code's own behaviour.
