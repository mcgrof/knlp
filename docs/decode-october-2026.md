# October 2026 asymmetric serving configuration

This is a separate research recipe. The older defconfigs retain their
historical defaults. One H100 SXM 80 GB, TP1, FP16 execution, Python
3.12.11, Torch 2.11.0+cu129 and CUDA 12.9 were used for the qualified
Qwen3 comparisons. Do not substitute an upstream package for these forks.

| Component | Dated public branch | Exact measured commit |
| --- | --- | --- |
| FlashInfer | [20261016-asym-k16v8-serving](https://github.com/mcgrof/flashinfer/tree/20261016-asym-k16v8-serving) | [800a550](https://github.com/mcgrof/flashinfer/commit/800a550f8b736e49f8cfa136b82bfaf6d9d32711) |
| vLLM compact | [20261016-k16v8-compact-prefill](https://github.com/mcgrof/vllm/tree/20261016-k16v8-compact-prefill) | [26cd429](https://github.com/mcgrof/vllm/commit/26cd429bac01d07909ed91e353f564d7aaa8d38f) |
| Preceding mixed control | Same branch ancestry | [1a4dcc0](https://github.com/mcgrof/vllm/commit/1a4dcc01ab09cd337499c0138c732eb741218b57) |
| Native control | Same branch ancestry | [c21c087](https://github.com/mcgrof/vllm/commit/c21c0878210695db7b108cdc285aaf0cf890190a) |

FlashInfer adds one output-scale fusion commit to July 6 base
`cab13ce297dcdde4cbcbca03fd2d23e45ad942ad`. vLLM adds the single writer
and bounded compact prefill to July 6 base `c21c087`. These extend the
full mixed-cache serving line, separately from the later pre-bias line.
The published tips are the measured commits, with no runtime changes.

## Install from public source

Run from a clean knlp checkout, on an H100 with CUDA 12.9 and `uv`:

```sh
uv run --python 3.12.11 --no-project -m tools.kv.native.october_setup "$PWD/stack"
export PATH="$PWD/stack/env/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/stack/vllm:$PWD/stack/flashinfer"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8 NVCC_THREADS=2 TOKENIZERS_PARALLELISM=false
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export FLASHINFER_ASYM_FP8_PV=0 KNLP_NATIVE_OBSERVER=off
export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22
```

The measured jobs used 22 CPU threads. Check the cgroup CPU quota before
using that setting; a smaller quota defines a different replication.
The installer verifies an upstream wheel SHA256 and identical compiled
source paths before reusing its binaries. It installs the dated editable
Python sources and JIT FlashInfer headers. It removes optional prebuilt
FlashInfer caches. First-use JIT is outside measured intervals. The full
constraints and resolved lock are retained. Record driver, loaded package
paths, every loaded `.so` hash and source trees with the run; a successful
install alone is not serving qualification.

## Prepare and calibrate a new replication

The commands below prepare the measured 36-layer Qwen3 recipes. The
engine and qualification probe also accept explicitly frozen layer/head
inventories; this does not extend the backend's compact-prefill guards.
The preparation commands generate disjoint Wikipedia calibration/evaluation documents
and the full 1,319-question GSM8K split. This creates a new replication;
it does not distribute private campaign payloads or promise identical
confirmation timing. Keep input hashes, chat template and parser fixed.
For a new template that opens a reasoning channel, freeze
`retrieval_final_marker` and `retrieval_max_output_tokens` in the manifest
before any scoring. A declared final delimiter is required; a missing
delimiter scores wrong. The default retains the original text parser.
Install `datasets` in a separate CPU preparation environment with
`transformers==5.5.3`; record that environment's resolved lock.

```sh
python -m tools.kv.native.october_prepare prepare \
  --model Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dataset-revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
  --gsm-revision 740312add88f781978c0658806c59bc2815b9866 \
  --output inputs
export VLLM_COMPACT_PREFILL=0 VLLM_ASYM_SINGLE_WRITER=0 FLASHINFER_FUSE_V_SCALE=1
git -C stack/vllm checkout --detach c21c0878210695db7b108cdc285aaf0cf890190a
python -m tools.kv.native.october_calibrate --inputs inputs \
  --output calibration --cache-bytes 34359738368
cp calibration/scales.json inputs/scales.json
python -m tools.kv.native.october_prepare freeze --output inputs \
  --install stack --format fp16 --cache-bytes 34359738368
```

For AWQ, use `Qwen/Qwen3-8B-AWQ`, revision
`4da05a8edb55c6046cce958586c33b61da07bb79`, `--format awq` and
51,539,607,552 cache bytes (48 GiB) throughout. Its packed INT4 weights
stay packed, with FP16 activations. Dense FP16 uses 32 GiB. Calibration
observes real native writes and 128 generated steps per calibration
prompt, then rounds each max-abs/448 scale to FP32 once. Evaluation
documents never enter calibration. Never tune scales on scored answers.
Scored serving rejects native-only placeholder scales. When reusing prior
calibration, match the scale hash to the effective protocol used by the
accepted quality run. A matching copy in a local preparation folder is
insufficient evidence; that folder can still contain a calibration placeholder.

Commit the frozen inputs/protocol in your results repository before GSM
scoring. Create `SCORING-UNLOCKED.json` in each variant input directory,
containing `{"protocol_commit":"<full committed protocol SHA>"}`. This is
an explicit provenance receipt, not a bypass for missing quality work.

## Qualify, then time fresh processes

For each variant, select its exact vLLM commit between processes and set
the flags below. Keep FlashInfer at `800a550`. Do not switch a checkout
while an engine is alive.

| Variant | vLLM | single writer | fused V scale | compact prefill |
| --- | --- | --- | --- | --- |
| native | c21c087 | 0 | 1 | 0 |
| mixed | 1a4dcc0 | 1 | 1 | 0 |
| compact | 26cd429 | 1 | 1 | 1 |

For example, the compact arm is:

```sh
git -C stack/vllm checkout --detach 26cd429bac01d07909ed91e353f564d7aaa8d38f
export VLLM_ASYM_SINGLE_WRITER=1 FLASHINFER_FUSE_V_SCALE=1 VLLM_COMPACT_PREFILL=1
python -m tools.kv.native.october geometry --variant compact \
  --inputs inputs/compact --output geometry-compact --cache-bytes 34359738368
cp geometry-compact/geometry.json inputs/compact/geometry.json
python -m tools.kv.native.october nll --variant compact --inputs inputs/compact \
  --output nll-compact --batch 8 --length 8192 --cache-bytes 34359738368
python -m tools.kv.native.october retrieval --variant compact --inputs inputs/compact \
  --output retrieval-compact --batch 8 --length 8192 --cache-bytes 34359738368
python -m tools.kv.native.october gsm8k --variant compact --inputs inputs/compact \
  --output gsm-compact --batch 32 --cache-bytes 34359738368
```

Repeat NLL/retrieval at 2,048 tokens, and run the same gates for native
and mixed. Use the production writer, typed-read and graph tests in the
dated dependencies too: `tests/kernels/test_mixed_reshape_and_cache.py`
and `tests/attention/test_mixed_output_scale.py`. Profile separately to
attest dispatch, genuine FP8 storage, graph replay and absence of a full
per-layer FP16 V shadow. Deduplicate storage aliases when counting memory.

Analyze all frozen cases with `tools.kv.native.statistics` and
`tools.kv.prequantized.contract`: NLL upper bound at most `ln(1.01)`,
retrieval gross-loss upper bound at most 0.05, GSM8K gross-loss upper bound
at most 0.02, and useful native accuracy (GSM at least 0.50; retrieval at
least 0.80). Errors and truncations are wrong; flexible extraction is
diagnostic. Compare gross paired losses, not only net accuracy. The timing
entrypoint requires a result `qualification.json` with `passed`,
`production`, `nll`, `retrieval`, `gsm8k` all true, the exact admitted
protocol hashes in `protocol_sha256`, and admitted `workloads`, for example
`[{"input_tokens":8192,"batch":8}]`. Derive it from the complete raw
tests; do not create a success receipt merely to enable timing.

```sh
python -m tools.kv.native.october timing --variant compact --inputs inputs/compact \
  --output timing-compact-r0 --batch 8 --length 8192 \
  --cache-bytes 34359738368 --qualification qualification.json
```

Timing excludes one fixed warmup and synchronizes CUDA at both boundaries.
Freeze `engine.disable_log_stats=false` in both arm protocols. vLLM's offline
default disables request metrics; the timing entrypoint rejects that setting
before warmup because TTFT and request latency require those clocks. Apply the
same statistics setting to both arms and keep its protocol hash with the run.
It measures 128 forced output tokens,
complete-request output tokens/s, completion-linked full-active decode,
TTFT and preemption. The cohort decode span runs from the first
prefill-produced token to the last request's final decode token; it can
include staggered admission and mixed prefill/decode. The separate
completion-linked metric includes only sustained full-active intervals.
APC and chunking are off, capture is
`FULL_DECODE_ONLY`, capture sizes are 1/8/32/44/45/59/60/96, and the token
budget is 65,536. Every process reinstalls frozen scales before capture.
Use seven fresh three-arm process blocks, frozen counterbalanced orders,
and within-block median ratios for the October B8 replication. Do not
count repeated requests as independent process blocks. The separate B32
study needs its own frozen protocol and workload-specific qualification.

The accepted compact route is H100, Q32/KV8/D128, FP16, NHD, page16,
whole-prompt causal prefill within 4,096 gathered pages, with no prefix
reuse, chunking, window, softcap or sinks. Other routes use the mixed
fallback. In particular, 8K/B32 can mix admissions and decode and does
not inherit the B8 result. The BF16 port is separate and quality-negative;
this recipe does not enable it or the closed symmetric AWQ recipe.

## Matched B32 retest configuration

The original quality phase used knlp
`c4bb3827078c979fd8529db7a518a58101cb5927`. Phi-4 and AWQ timing use
`a6016292d39734c006cfc272eab4c7f131d6bc4d`. Corrected dense Qwen3 work uses
`720bc0261a97ea11d129c6dfcb6cf8c2d4ca5055`. Use the last immutable checkout
for a new replication; public main continues to move.
The dependency commits in the source table are unchanged.

Freeze `engine.disable_log_stats=false` in both protocols. The offline
vLLM default omits request metrics, which made the initial timing attempts
invalid. The corrected source rejects disabled statistics before warmup.
No valid timing is pooled across sources. Phi-4 reuses its full original
quality panel after verifying unchanged scoring/model math, inputs,
scales and routes except statistics bookkeeping, plus fresh actual
production profiles. Other valid negative primary cells stay closed.

An audit found a native-only
unit-scale placeholder in the first dense Qwen3 fixtures, so those attempts
are excluded. Dense-only reruns restore the exact previously accepted
calibration, SHA256 `267dc6420febfb8fd6201ba60dd0eeae6b16315e93090337a82005fd9bf4f656`,
verified against actual prior quality provenance. The corrected source
rejects placeholders before serving. No new calibration or scale tuning
is selected from the scored answers.

Primary timing uses FP16 weights and activations, 32 unique pretokenized
1024-token prompts and 128 forced outputs, with EOS ignored only for
timing. Native uses `FLASH_ATTN` with `flash_attn_version=3`; asymmetric
uses FlashInfer with its supported compact or mixed route. This includes
backend differences and does not isolate a single kernel. Both arms use
the same explicit byte pool within each model. The following metadata,
revisions, routes and native floors were frozen before model scoring.
Accepting metadata is not proof that a backend supports that geometry:
the production reference and actual graph/route checks remain mandatory.

| Model | Revision | L / Q / KV / D | Asymmetric route | Primary pool GiB | Native retrieval / GSM floors |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | 36 / 32 / 8 / 128 | compact | 32 | 0.80 / 0.50 |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 28 / 28 / 4 / 128 | mixed | 32 | 0.50 / 0.50 |
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 24 / 14 / 2 / 64 | mixed | 8 | 0.50 / 0.15 |
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 28 / 12 / 2 / 128 | mixed | 8 | 0.50 / 0.25 |
| mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 32 / 32 / 8 / 128 | compact | 32 | 0.50 / 0.50 |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | 32 / 32 / 8 / 128 | compact | 32 | 0.50 / 0.50 |
| microsoft/Phi-4 | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | 40 / 40 / 10 / 128 | mixed | 24 | 0.50 / 0.50 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | 28 / 28 / 4 / 128 | mixed | 32 | 0.50 / 0.50 |

For a new primary replication, use the preparation command above with
`--lengths 1024` and the chosen immutable model revision. Before native
calibration, record its actual layer/head inventory in the input
manifest's `geometry`. Before committing the protocol, record the same
inventory in every variant's `protocol.json`, with `packed_execution`
set to FP16, the actual attention-layer count, four linear modules per
layer, and one embedding module for tied weights or two otherwise. Set
the native primary `attention_config` to the FlashAttention 3 configuration
above. Keep the asymmetric configuration as FlashInfer. The default
freeze helper remains the separate Qwen3/FlashInfer B8 recipe; these
explicit protocol choices must precede scoring and its committed receipt.

Freeze the model's native floors with the input metadata. For the R1
pinned template, retrieval uses `retrieval_final_marker="</think>"` and
`retrieval_max_output_tokens=2048`; missing delimiters and truncations are
wrong. No other model uses that delimiter. GSM keeps its strict parser
and 2048-token cap. No absolute perplexity ceiling is added. Never change
these fields or calibration scales after reading scored model answers.

Run the public geometry, NLL, retrieval and GSM entrypoints for each
required arm, using `--batch 32 --length 1024` for the primary NLL and
retrieval stages. Comparative timing requires the complete workload-bound
qualification receipt. Each eligible primary model receives ten fresh
paired processes, with five native-first and five asymmetric-first blocks
in an order frozen before scoring. Require all quality/production gates,
a median ratio of at least 1.01 and the positive sign test after Holm
correction over the original 24 comparisons. Missing historical arms
retain p=1. Preserve every paired ratio and invalid-run record.

The 8192-token bridge uses native FlashInfer, not the primary FA3 native
control. It has three fresh pairs per cell, separate from the primary
family. Dense B8 uses 32 GiB and dense B32 uses 40 GiB in both arms; native
B32 live payload alone needs 36.5625 GiB. AWQ uses 48 GiB at both batches,
with model revision `4da05a8edb55c6046cce958586c33b61da07bb79`.
Existing accepted quality is reused only after exact input, scale, parser,
source, scheduler and math/route equivalence checks, plus a fresh actual
profile. The inherited full GSM receipts contain actual B32 work; B8-only
qualification is insufficient. Dense B32 receives a fresh full panel.

The pinned Mistral tokenizer has a historical Hub metadata size mismatch
for `tokenizer.model`: the Git LFS pointer is 130 bytes, while its verified
payload is 587404 bytes, SHA256
`37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89`.
The study restored exact Git/LFS bytes into the standard pinned Hub cache
before model execution; it did not change tokenizer or runtime code.
Record any equivalent cache repair and its hashes in a replication.

The page's derived data retains all primary models and bridge cells,
including failed and ineligible outcomes. These are new workloads; do not
infer old-to-new deltas from the historical April table. Input generation
for a replication remains a new frozen evaluation, not a distribution of
private campaign prompts or a promise of identical timings.
