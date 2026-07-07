# Mooncake trace → vLLM/LMCache replay + timing/hit-rate

This note describes the `kvio-mooncake` reproduction profile: replaying a
real production request trace (Mooncake, FAST25 release) through vLLM +
LMCache to measure how long the trace takes, TTFT/e2e latency, prefix-cache
reuse, and KV-offload storage behaviour. It is the real-trace counterpart of
a synthetic shared-prefix put/get workload.

## What the trace contains

The public Mooncake traces (`conversation_trace.jsonl`,
`toolagent_trace.jsonl`, `synthetic_trace.jsonl`) are JSONL with, per record:

| field           | meaning                                              |
| --------------- | ---------------------------------------------------- |
| `timestamp`     | relative arrival time in **milliseconds**            |
| `input_length`  | number of prompt (prefill) tokens                    |
| `output_length` | number of decode tokens                              |
| `hash_ids`      | list of remapped **512-token prefix block** hashes   |

Two records that share leading `hash_ids` share that prefix; an identical
`hash_id` denotes an identical, reusable 512-token KV block. Sanity check:
`len(hash_ids) ≈ ceil(input_length / 512)`.

There are **no tokens or text** in the trace (privacy). Only timing,
lengths, and block hashes.

## Token-ID synthesis (how we run it on a real model)

Because the trace carries no tokens, `mooncake_trace.py` synthesises token-ID
sequences that **preserve the hash structure** so real prefix-cache hits
occur:

- Each `hash_id` maps deterministically to a **fixed block of 512 token
  IDs**, via a **stable** hash (`hashlib` SHA256 keyed by `hash_id`, seeding
  `numpy.default_rng`). This is stable across processes and machines —
  Python's built-in `hash()` is not and must not be used.
- Token IDs are bounded to a configurable vocab size and drawn from
  `[reserved_ids, vocab_size)` so no **special-token IDs** (BOS/EOS/PAD/…)
  are ever emitted.
- A request's prefill token IDs are the concatenation of its `hash_ids`'
  512-token blocks. If that is **longer** than `input_length` it is trimmed;
  if **shorter**, the remainder is filled with **request-unique deterministic
  noise** (keyed by request index) — never a shared pad token. Shared padding
  would manufacture fake cache hits.
- Requests carry raw `prompt_token_ids` and are submitted via the
  **Completions API** (never a text string) to avoid detokenize/re-tokenize
  drift that would break hash→block identity.
- Decode length is forced with `max_tokens = output_length` + `ignore_eos`.

The **content is synthetic**; the **reuse / length / arrival structure is
faithful**. So the run is valid for exact prefix-reuse, KV byte pressure,
store/load traffic, TTFT/throughput, and reproducing a KV-offload dataset —
**not** for semantic quality.

## Measurement caveats (read before trusting numbers)

- **vLLM APC vs LMCache L2.** vLLM's own automatic prefix cache can satisfy a
  hit in-GPU so LMCache L2 never records the store/load you wanted. To force
  the LMCache L2 traffic (the kvio dataset), **disable vLLM automatic prefix
  caching**.
- **Storage-faithfulness.** KV byte geometry depends on shapes, not token
  values — a length+reuse-faithful synthetic trace is storage-faithful **iff**
  content-dependent paths are OFF: disable LMCache serde/compression/blending,
  set `save_decode_cache=false` and `save_unfull_chunk=false`, and force
  `output_length` (no EOS shortening). LMCache default `chunk_size=256`, so a
  512-token block is 2 chunks.
- **`local_cpu` default-true** can mask disk L2 — watch it when measuring
  disk traffic.
- Every run records a full **config manifest** (model, tokenizer, trace path +
  SHA256, seed, vocab/block size, speedup, max-in-flight, and the
  vLLM/LMCache versions + flags above) alongside results.

## Running it

```bash
make defconfig-kvio-mooncake
cat >> .config << 'EOF'
CONFIG_KNLP_KVIO_MOONCAKE_TRACE_PATH="/data/mooncake/conversation_trace.jsonl"
CONFIG_KNLP_KVIO_MOONCAKE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CONFIG_KNLP_KVIO_MOONCAKE_MAX_REQUESTS=200
EOF
make
```

**Smoke first, then scale.** Never start with the full 23.6k-request
`toolagent_trace.jsonl`. Cap `CONFIG_KNLP_KVIO_MOONCAKE_MAX_REQUESTS` to 50–200 (or
leave the trace path empty to use the built-in 3-record fixture with a known
shared-prefix pair) and verify:

1. synthesised token IDs match for shared hash prefixes (offline gate — runs
   on CPU),
2. vLLM shows prefix hits,
3. LMCache logs the store/load events,

then run `conversation`, then `toolagent`.

The parse / token-synth / schedule path is pure and CPU-testable. If no GPU,
vLLM, or LMCache is present the stage **skips gracefully**, still writing the
config manifest and the offline smoke report to
`results/kvio-mooncake/mooncake/mooncake_trace_replay.json`.

## CPU unit tests

```bash
python -m pytest tests/test_mooncake_trace.py -v
```

covers parser schema/block-count sanity, token-synth determinism (same
`hash_id` → identical block across processes; shared leading hashes → shared
prefix; distinct hashes → distinct blocks), the no-shared-pad tail guarantee,
length correctness after trim/pad, and the arrival scheduler
(timestamp ordering, ms→s multiplier, speedup, max-in-flight cap).
