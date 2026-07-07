# Content trace → vLLM/LMCache replay + KV-corpus capture

This note describes the `kvio-lmsys` reproduction profile: replaying
**real** datasets — LMSYS-Chat-1M multi-turn conversations or LongBench
long-context questions — through vLLM + LMCache, measuring timing /
prefix-cache reuse / KV-offload behaviour, and — this is the point of the
stage — **capturing the LMCache KV corpus plus the kvio semantic trace** (the
identity join) so the KV geometry / reuse / storage behaviour can be studied
offline on CPU, with no GPU.

It is the content-bearing counterpart of the Mooncake stage
(`kvio-mooncake`). Mooncake carries no tokens, so it *synthesises*
token IDs that preserve a hash structure; here we tokenize the **real dataset
content** with the target model's tokenizer, so the prefixes and their KV
bytes are the ones the model actually produces.

## Two datasets, a smoke gate, and two-phase capture/replay

**LMSYS-Chat-1M and LongBench are two different datasets** (different sources,
record shapes, and repeated-prefix structure — LMSYS grows a per-turn prefix,
LongBench shares a document prefix; LMSYS is HF-gated, LongBench is not). Each
has its own clearly-named defconfig:

- `make defconfig-kvio-lmsys` — LMSYS-Chat-1M
- `make defconfig-kvio-longbench` — LongBench
- `make defconfig-kvio-smoke` — offline CPU-only pipeline smoke (no GPU / fetch
  / build); `make kvio-replay` runs the build/tokenize/schedule + smoke gate
  and records the offline JSON. This is how you validate the pipeline
  separately from a real capture.

**Two phases.** The recorded set lives in `CONFIG_KNLP_KVIO_RECORD_DIR`
(defaults to `RESULTS_ROOT/record`; override the path on the make line with
`KVIO_PATH`, e.g. `make defconfig-kvio-lmsys KVIO_PATH=/data/set`). The first
`make` on a GPU box **captures** a recorded set there (the raw_block L2 corpus
image, the `lmcache_kvio_trace.jsonl` semantic trace, and a `kvio_record.json`
manifest). A later `make kvio-replay` finds that recorded set and **replays it
GPU-free** through the configured backend (`CONFIG_KNLP_KVIO_BACKEND`, default
LMCache raw_block via `CONFIG_KNLP_KVIO_REPLAY_DRIVER`) instead of re-capturing.

## Dataset format (normalized)

Loaders live in
`tools/reproduce/kvio/content_datasets/` (named
`content_datasets` — **not** `datasets` — so it never shadows the pip
`datasets` module the loaders import). Each returns a normalized record list:

| dataset   | normalized record                                            |
| --------- | ------------------------------------------------------------ |
| LMSYS     | `{"conversation_id", "turns": [{"role","content"}, ...]}`     |
| LongBench | `{"doc_id", "document", "questions": [str, ...]}`            |

The pure normalizers (`normalize_lmsys`, `normalize_longbench`,
`group_longbench`) need no network and are unit-tested on tiny fixtures.
LMSYS-Chat-1M is **HF-gated** (needs an accepted licence + auth token); the
loader **gracefully skips** (returns `[]` + a reason) when the dataset is not
accessible, and the stage then falls back to a tiny built-in fixture so the
offline smoke gate still runs. No gated download happens in CI/tests.

## Real tokenization + the repeated-prefix builder

`content_trace.py` builds requests whose prompts share a genuine token prefix
(this is what produces real prefix-cache hits):

- **LMSYS (growing prefix).** For one conversation with turns `t1..tn`,
  request `k`'s `prompt_token_ids` is `tokens(t1) ++ … ++ tokens(tk)`. So
  within a conversation request `k+1`'s prompt has request `k`'s as a **strict
  prefix**. To make that invariant *exact* — independent of BPE merges across
  a turn boundary — each turn is tokenized independently and the token-ID
  lists are concatenated. KV blocks are per-token, so this is the faithful
  realization of "prompt = tokens of turns `1..k`" and it guarantees the
  strict-prefix property the cache relies on. Distinct conversations are
  independent prefix chains.
- **LongBench (shared document prefix).** Each question about a document is
  `tokens(document) ++ tokens(question)`; every question about that document
  shares the long document token prefix.

Tokenization uses the model's HF tokenizer with `add_special_tokens=False`
(raw content tokens, no BOS/EOS injected). Requests carry raw
`prompt_token_ids` and are submitted via the **Completions API** (no chat
template, no detokenize/re-tokenize drift). Decode length is forced with
`max_tokens = output_length` + `ignore_eos`.

On a CPU box without `transformers`, the stage and tests fall back to a
deterministic `HashTokenizer` (stable word→ID hash) so the build / prefix /
arrival logic still runs — that fallback is not a real tokenizer and is
flagged in the manifest (`real_tokenizer=false`).

## Arrival synthesis

These datasets have no timestamps, so arrival times are **synthesised**
(fixed-rate or seeded Poisson) and fed to the **same** open-loop scheduler
used by Mooncake (`serving_replay.schedule_arrivals`, shared via
`serving_replay.py`). The manifest records `synthesised_arrivals=true`, the
mode, and the rate.

## Capture discipline (why these flags)

The captured KV byte geometry only matches real content if content-dependent
and cache-shortcut paths are OFF. The stage bakes this into the LMCache/vLLM
config it sets:

- **vLLM automatic prefix caching OFF** — otherwise vLLM's in-GPU APC
  satisfies a hit and LMCache L2 never records the store/load you want to
  capture.
- **KV dtype BF16/FP16, never FP8**; **no serde / compression / blending**;
  `save_unfull_chunk=false`; `save_decode_cache` explicit (default off,
  `CONFIG_KNLP_KVIO_CONTENT_SAVE_DECODE_CACHE`).
- **No MLA / full-attention model; TP=1.**
- **L2 = `raw_block` backend on a local pre-sized file** (Phase-1; swapping to
  an NVMe `/dev/ng` device is a later change), with **`LMCACHE_KVIO_TRACE`**
  exported so the run captures the KV-object semantic trace (the identity
  join) alongside the corpus.
- **L2 eviction off / capacity large** — otherwise captured KV objects are
  evicted and lost.

Every run writes a full reproducibility manifest
(`build_content_manifest`): model + tokenizer, vLLM/LMCache versions,
KV dtype / chunk size, all the flags above, the L2 path/backend/capacity, the
dataset + revision, the seed, and the kvio semantic-trace path.

### Known integration item (raw_block extension)

The kvio-capable LMCache branch's `rust/raw_block` extension needs
`maturin develop`, which the build stage's `pip install -e` does **not** run.
Wiring that up is a deliberate GPU-run integration step, tracked here and in
the stage note — it is **not** done by this stage. Handle it when standing up
the real GPU capture run.

### Phase-2 Q-probe hook (not implemented)

`content_trace.q_probe_hook` is a documented **stub**. A future extension would
capture, for a *sampled* set of decode positions, the post-RoPE query vectors,
the query-head→kv-head (GQA) mapping, and the attention softmax scale, so
attention against the captured KV corpus can be reconstructed offline on CPU.
Wiring it needs a model-forward hook on a GPU run; it is intentionally out of
scope for the capture-by-default KV-corpus + kvio-trace path implemented now.

## Running it

```bash
make defconfig-kvio-lmsys
cat >> .config << 'EOF'
CONFIG_KNLP_KVIO_CONTENT_MODEL="Qwen/Qwen2.5-7B-Instruct"
CONFIG_KNLP_KVIO_CONTENT_MAX_REQUESTS=200
EOF
make
```

Switch dataset by selecting `CONFIG_KNLP_KVIO_CONTENT_DATASET_LONGBENCH=y` (or
`..._LMSYS=y`) before `make`. Point LMCache at a feature branch (e.g. the
kvio-capable branch) with either variable — `LMCACHE_BRANCH` is an accepted
alias for `LMCACHE_REF`:

```bash
make defconfig-kvio-lmsys LMCACHE_BRANCH=my-kvio-branch
```

The defconfig deliberately does **not** pin `CONFIG_KNLP_LMCACHE_REF`
(`pyconf --olddefconfig` would keep a pinned value and shadow the CLI
override); Kconfig's default-chain supplies both the canonical pin and the CLI
variable. If both are passed, `LMCACHE_BRANCH` wins.

**Smoke first, then scale.** Cap `CONFIG_KNLP_KVIO_CONTENT_MAX_REQUESTS` to 50–200
(or leave the dataset gated so the built-in fixture is used) and verify on CPU
that the growing/shared prefixes are present before running the full replay.

The tokenize / build / schedule path is pure and CPU-testable. If no GPU,
vLLM, or LMCache is present the stage **skips gracefully**, still writing the
capture manifest and the offline smoke report to
`results/kvio-lmsys/content/content_trace_replay.json`.

## CPU unit tests

```bash
python -m pytest tests/test_content_trace.py -v
```

covers dataset normalization, the growing-prefix (LMSYS) and shared-document
(LongBench) properties, tokenization determinism (real tokenizer gated with
`importorskip`), arrival synthesis (fixed-rate ordering, ms→s, speedup,
Poisson determinism under seed), reuse of the shared open-loop scheduler on
content requests, and the capture-discipline manifest.
