# DroidSpeak mechanism baseline

This directory implements the central DroidSpeak mechanism from the NSDI 2026
paper: reuse a sender model's context KV cache outside one contiguous layer
interval, inject the sender's layer-input embedding at the interval boundary,
and recompute that interval with the receiver model.

The current code is an in-process Transformers baseline for decoder-only
models that expose the standard Transformers decoder-layer stack. It is
qualified on Qwen2 and Mistral. It tests the algorithm before a vLLM and
LMCache runtime patch. It does not claim the paper's distributed-system
speedups.

## Paper contract

The baseline preserves these parts of the paper:

- The sender and receiver must have the same architecture and token vocabulary.
- The sender records KV for the whole input prompt, except the final input
  token needed to obtain the receiver's first output logits, plus the
  layer-input embedding `E` at possible transition layers.
- The receiver recomputes one contiguous half-open interval `[start, end)`.
- The receiver uses its new KV inside that interval and sender KV elsewhere.
- The offline profiler searches every contiguous interval on two-layer
  boundaries. This is `O(L^2)` as in the paper.
- The selected Pareto point recomputes the fewest layers while its mean profile
  F1 is no more than 5% below full receiver prefill. Measured TTFT breaks ties.
- The profile and evaluation samples are disjoint.

The qualification run differs from the paper in these ways:

- It uses the paper-named Mistral-7B `conllpp` sender and `glue_sst2`
  receiver adapters, but the paper does not publish exact checkpoint revisions.
- It runs both models in one process and clones tensors in GPU memory. It does
  not include remote cache transfer or overlap in its timing.
- It bounds the passage at 2,048 tokens, while the paper reports much longer
  HotpotQA inputs. The instruction and question are retained.
- The paper does not identify its exact 50 HotpotQA profile records. This code
  makes a seeded split and records the seed and record identifiers.
- It evaluates a disjoint HotpotQA held-out set. This is not the paper's
  cross-dataset evaluation.

The full receiver interval is a hard correctness audit. Its KV tensors must be
exactly equal to an ordinary receiver prefill. A mismatch stops the run before
quality results are accepted.

## Run the baseline

Install PyTorch for the target CUDA runtime, then install the pinned Python
dependencies:

```sh
python -m pip install -r research/droidspeak/requirements.txt
```

A bounded mechanism smoke test is:

```sh
python -m research.droidspeak.run_baseline \
  --dataset data/longbench/hotpotqa.jsonl \
  --output results/droidspeak-smoke \
  --profile-samples 2 \
  --eval-samples 4 \
  --max-context-tokens 2048 \
  --max-new-tokens 16
```

One named model pair from the paper matches two public Predibase LoRA adapters
on the same Mistral-7B base. The paper does not publish repository revisions,
so this is the closest identifiable artifact pair, not proven byte identity
with the authors' checkpoints:

```sh
python -m research.droidspeak.run_baseline \
  --sender-model mistralai/Mistral-7B-v0.1 \
  --receiver-model mistralai/Mistral-7B-v0.1 \
  --sender-adapter predibase/conllpp \
  --receiver-adapter predibase/glue_sst2 \
  --dataset data/longbench/hotpotqa.jsonl \
  --output results/droidspeak-mistral-paper-pair
```

The downloader fetches only `hotpotqa.jsonl` from the pinned LongBench
revision. Omit the sample and context overrides for the 50-record profile
contract. That full profile is much more expensive because it evaluates every
two-layer contiguous interval on every profile record.

Use a new output directory for every run. The runner refuses a nonempty one so
that a retry cannot mix records from different executions.

Important outputs are:

- `manifest.json`: exact model commits, dataset hash, code commit, hardware,
  and run parameters.
- `full_recompute_audit.json`: receiver-native KV parity.
- `profile_samples.jsonl`: every sample and interval result.
- `profile_summary.json`: all points, Pareto frontier, quality floor, and the
  selected interval.
- `eval_samples.jsonl`: held-out selected, direct-reuse, full-receiver, and
  sender controls.
- `summary.json`: held-out quality and TTFT aggregates.

F1 values are in `[0, 1]`. TTFT includes receiver recomputation, local hybrid
cache materialization, the final input token, and the first generated token.
Sender cache production is excluded because DroidSpeak assumes that cache
already exists. Network transfer is also excluded and is stated in every run
manifest.

## LMCache boundary

The user-linked LMCache cross-instance guide is for deprecated in-process
mode. Its centralized example can be rendered for compatibility with:

```sh
python -m research.droidspeak.lmcache_plan \
  --legacy-centralized-yaml results/lmcache_config.yaml \
  --json results/lmcache_capabilities.json
```

New work should use LMCache multiprocess (MP) mode. The included control starts
one current MP cache server and two same-model vLLM servers on two GPUs:

```sh
OUTPUT_DIR=results/lmcache-mp-smoke \
  bash research/droidspeak/lmcache_mp_smoke.sh
```

For a functional one-GPU control with two small-model engines, set
`GPU_A=0 GPU_B=0 GPU_MEMORY_UTILIZATION=0.35`. The official guide uses two
GPUs. Treat the one-GPU form as a transport check, not a performance result.

The control fails unless the second vLLM or LMCache server log contains
positive cache-retrieval evidence.
It proves shared KV transport only. It does not prove cross-model correctness.

Stock LMCache has useful pieces: shared KV transport, legacy layerwise
transfer, and a hidden-state store. The documented MP SDK can retrieve and
replace complete KV tensors, but it only exposes query intermediates, not an
arbitrary transition `E` cache. A production DroidSpeak path still needs:

- a model-pair-aware namespace and compatibility check;
- remote transfer of the selected transition `E` cache;
- a vLLM receiver hook that runs only the selected contiguous interval;
- assembly of receiver and sender KV layers; and
- asynchronous layer loading overlapped with recomputation.

See the paper and current LMCache documentation:

- <https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf>
- <https://docs.lmcache.ai/getting_started/quickstart/share_kv_cache.html>
- <https://docs.lmcache.ai/mp/index.html>
- <https://docs.lmcache.ai/mp/p2p.html>
- <https://docs.lmcache.ai/non_kv_cache/hidden_states.html>
- <https://docs.lmcache.ai/kv_cache_optimizations/layerwise.html>
- <https://docs.lmcache.ai/mp/sdk.html>
