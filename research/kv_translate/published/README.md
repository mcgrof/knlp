# KV-Lingo independent reproduction

This directory preserves the experimental Qwen3-4B/8B cache translator and
the controls used for the KNLP closeout. The tested recipe did not qualify for
a service in which either model may start a conversation. Checkpoint search is
closed. A fixed schedule that starts with 8B remains an unconfirmed development
observation.

This is an independent implementation of
[KV-Lingo](https://arxiv.org/abs/2609.32610), not the authors' code or an exact
reproduction of their data protocol. It is opt-in research code. No import,
help command, Kconfig target, or default build downloads a model, rents a GPU,
or starts training.

## Released and withheld artifacts

The repository releases:

- translator fitting, distillation, retained-cache inference, native controls,
  scoring, audit, migration comparison, and checkpoint-selection code;
- CPU fixtures for shapes, cache ownership, scoring, gates, resume contracts,
  checkpoint comparison, and receipt handling;
- the 16-comparison aggregate at
  [`docs/data/kv-lingo-results.json`](../../../docs/data/kv-lingo-results.json);
  and
- [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json), which maps the clean public
  import to the executed source snapshot.

The four trained translator checkpoints, raw conversations, raw model outputs,
and full frozen token stream are not released. The aggregate can reproduce the
published decisions and figures. It cannot independently replay model
inference or training without equivalent inputs and newly trained maps.

## Recorded configuration

The executed study used Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c` and Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`. The final provider-local inference
environment recorded Python 3.12.12, PyTorch 2.11.0+cu128, Transformers 5.14.1,
NumPy 2.5.1, safetensors 0.8.0, tokenizers 0.22.2, and huggingface-hub 1.26.0.
Inference used one NVIDIA H100 80 GB GPU, bfloat16 language models, FP32
translator parameters, SDPA attention, greedy decoding, and reasoning disabled.
Weights & Biases was not used.

Stage 1 used 400 conversations and FP64 sufficient statistics to initialize
separate token-wise linear key and value maps for corresponding layers. Keys
were captured before key normalization. Values were captured after `v_proj`.
The receiver applies its own key normalization and RoPE after translation.

Stage 2 kept both language models frozen and minimized forward KL from native
target logits to translated-cache logits. It used AdamW, global batch 8, peak
learning rate 3e-5, 250 warmup steps, cosine decay through 5,000 updates, zero
weight decay, and gradient clipping at 1.0. The recorded run used one seed.

The frozen training stream was an independent construction from
`nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` revision
`1a9454ed054b8544503ab8d8c0a519d141a44c5b`. It mixed natural and packed
prefixes and separated reasoning-on and reasoning-off records. The paper's
exact sample identities and loader were not available.

## CPU verification

Install NumPy, pytest, and a CPU-capable PyTorch build, then run:

```sh
make defconfig-kv-lingo
make CONFIG_KV_LINGO_PYTHON=/path/to/python
```

This runs only offline CPU tests and verifies the released aggregate SHA-256.
The direct aggregate check is:

```sh
python -m research.kv_translate.published.verify_public_results \
  docs/data/kv-lingo-results.json \
  --expect-sha256 \
  2051e9abdb85ad49c38872d7caf15d8488d4e3a3795115d363e5355d238d44b1
```

## Full data preparation

The following steps are expensive and require separately staged data and model
snapshots. The modules use `local_files_only=True` for model and tokenizer
loads. Prepare those snapshots before starting a run.

Set explicit locations; there are no host-specific defaults:

```sh
export KV_RUN=/path/to/kv-lingo-run
export KV_DATASET=/path/to/nemotron-jsonl
export KV_TOKENIZER=/path/to/qwen3-4b-tokenizer
mkdir -p "$KV_RUN"
```

`$KV_DATASET` must contain `reasoning_off.jsonl` and `reasoning_on.jsonl` from
the pinned dataset revision. Freeze the version-2 stream:

```sh
python -m research.kv_translate.published.freeze_kv_lingo \
  --dataset-dir "$KV_DATASET" \
  --tokenizer "$KV_TOKENIZER" \
  --out "$KV_RUN/frozen" \
  --stream-version 2 \
  --seed 42 \
  --train-samples 40000 \
  --validation-samples 64 \
  --stage1-samples 400
```

To build a fresh CoQA evaluation cohort, supply the official CoQA development
JSON and a directory whose text files identify any conversations already
examined. An empty history directory produces a new independent split; it does
not reconstruct the unreleased KNLP cohort.

```sh
python -m research.kv_translate.published.freeze_kv_lingo_coqa \
  --data /path/to/coqa-dev-v1.0.json \
  --history-root /path/to/exposure-audit-root \
  --tokenizer "$KV_TOKENIZER" \
  --out "$KV_RUN/coqa" \
  --development 32 \
  --reserve 256
```

Do not inspect the reserve while selecting a checkpoint.

## Stage 1: closed-form initialization

This command is GPU-expensive because it loads both frozen models and captures
their full cache representations:

```sh
python -m research.kv_translate.published.kv_lingo_train stage1 \
  --rows "$KV_RUN/frozen/train.jsonl" \
  --tokens "$KV_RUN/frozen/train.tokens.u32" \
  --out "$KV_RUN/stage1" \
  --stage1-samples 400 \
  --stage1-tolerance 1e-8
```

It emits one Stage-1 map for each direction and a receipt with input hashes.

## Stage 2: output-distribution distillation

These are long, GPU-expensive runs. Four or eight GPUs preserve global batch 8
with synchronous gradients. The same commands can run on one GPU by replacing
`torchrun ... -m` with `python -m`.

```sh
export KV_SOURCE_SHA="$(git rev-parse HEAD)"

torchrun --standalone --nproc-per-node=4 -m \
  research.kv_translate.published.kv_lingo_train train \
  --direction 4b-to-8b \
  --rows "$KV_RUN/frozen/train.jsonl" \
  --tokens "$KV_RUN/frozen/train.tokens.u32" \
  --validation-rows "$KV_RUN/frozen/validation.jsonl" \
  --validation-tokens "$KV_RUN/frozen/validation.tokens.u32" \
  --stage1 "$KV_RUN/stage1/stage1-4b-to-8b.pt" \
  --out "$KV_RUN/forward" \
  --max-steps 5000 \
  --checkpoint-steps 1000,5000 \
  --source-commit "$KV_SOURCE_SHA"

torchrun --standalone --nproc-per-node=4 -m \
  research.kv_translate.published.kv_lingo_train train \
  --direction 8b-to-4b \
  --rows "$KV_RUN/frozen/train.jsonl" \
  --tokens "$KV_RUN/frozen/train.tokens.u32" \
  --validation-rows "$KV_RUN/frozen/validation.jsonl" \
  --validation-tokens "$KV_RUN/frozen/validation.tokens.u32" \
  --stage1 "$KV_RUN/stage1/stage1-8b-to-4b.pt" \
  --out "$KV_RUN/reverse" \
  --max-steps 5000 \
  --checkpoint-steps 1000,5000 \
  --source-commit "$KV_SOURCE_SHA"
```

The resume contract binds model revisions, data hashes, the Stage-1 artifact,
capture cut, map architecture, loss, schedule, optimizer cursor, topology, and
random state. `compare_kv_lingo_checkpoints.py` checks topology-equivalent
outputs. `make_kv_lingo_resume_contract.py` exists only for legacy checkpoints
that predate embedded contracts.

## Retained multi-turn evaluation and native controls

The retained protocol keeps one cache line for each model. Each line preserves
its native earlier spans. When the other model receives a turn, only missing
spans are translated into that receiver's line. This is not one universal
cache, and it does not translate the full prefix back and forth on every turn.

Evaluate one forward/reverse checkpoint pair:

```sh
python -m research.kv_translate.published.kv_lingo_coqa_runner \
  --conversations "$KV_RUN/coqa/DEVELOPMENT_CONVERSATIONS.json" \
  --forward "$KV_RUN/forward/stage2-4b-to-8b-step5000.pt" \
  --reverse "$KV_RUN/reverse/stage2-8b-to-4b-step5000.pt" \
  --out "$KV_RUN/eval-forward5000-reverse5000" \
  --turns 10

python -m research.kv_translate.published.kv_lingo_eval \
  --raw "$KV_RUN/eval-forward5000-reverse5000/RAW.jsonl" \
  --native "$KV_RUN/eval-forward5000-reverse5000/NATIVE_TRAJECTORY.jsonl" \
  --out "$KV_RUN/eval-forward5000-reverse5000/SUMMARY.json"
```

The runner records two controls. `same_transcript_native` re-prefills the
receiver on the translated run's exact generated history. The alternating
native trajectory evolves independently with native re-prefill at each turn.
Neither control is an always-8B service.

Run all four forward/reverse pairs before selection. The analysis and selector
accept descriptive pair names such as `forward_5000_reverse_1000`; use
`--help` for the exact `RAW.jsonl,NATIVE_TRAJECTORY.jsonl` and summary syntax.
The frozen acceptance rule requires every starting model and both controls to
pass: at most 3 F1 points overall, at most 5 in each domain, at most 5 on pooled
turns 6-10, and at most 2 percentage points unhealthy-rate excess.

Use the terminal receipt wrapper around long stages when a scheduler or
deadline may terminate them:

```sh
python -m research.kv_translate.published.run_with_receipt \
  --receipt "$KV_RUN/stage.json" \
  --deadline-seconds 7200 \
  -- your-command --with its-arguments
```

The wrapper always replaces `RUNNING` with `COMPLETE`, `FAILED`, or
`TIMED_OUT`, including command-start errors. This corrects the shell-state bug
found during the final inference study without changing inference or scoring.

## Interpretation boundary

The released aggregate contains one seed, 32 exposed development
conversations across five domains, ten turns, and no untouched confirmation.
It does not establish arbitrary routing, cross-family translation, a learned
Cartridge transfer, long-context behavior, latency savings, or one-cache
storage savings. See the [closeout page](../../../docs/kv-lingo.html) for the
complete result and current limits.
