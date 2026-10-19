# Measuring asymmetric KV through HTTP

This is an experimental measurement path. Its CPU recorder tests pass;
GPU worker qualification and serving comparisons are separate gates.
The existing [installation](asymmetric-install.md) and
[measured results](paper_memory_decode.html) retain their original scope.

The server imports `tools.kv.native.serving_worker.ServingWorker` in the
actual worker process. It installs the frozen model scales before warmup,
checks them immediately before graph capture, and saves loaded source,
weight and physical-cache receipts. It requires the pinned source trees
from the current installation. Select the exact native or candidate
vLLM checkout between server processes.

On a dedicated H100 host, after installing the current recipe:

```sh
python -m tools.kv.native.serving_pilot \
  --install ./asymmetric-stack --output ./awq-http-pilot \
  --format awq --arm candidate --length 8192
```

The server binds to loopback. Its development RPC endpoint controls
qualification, profiling and measurement intervals. Heavy numerical
oracles and profiles run outside timing. Pilot output does not establish
a performance result or unlock confirmation by itself.

`serving_trace` generates an explicit arrival trace from the frozen prompt
panel. `serving_client` replays that trace, preserving intended arrival,
dispatch, admission, send, headers, every SSE event and completion or
failure. Open-loop runs have no hidden concurrency cap. Closed-loop
concurrency counts HTTP requests, not GPU batch size. All intervals use
the client monotonic clock. No server clock is subtracted from it.

TTFT starts at the first event with output token IDs, including invisible
tokens. TPOT is the per-request post-first-token interval divided by
output tokens minus one. Multi-token events remain visible in the raw
record; they do not establish individual token-gap timing. Percentiles
are over requests. Finite-trace throughput includes drain, and failures
remain in the offered-request accounting. No production SLO is assumed.

Verify a raw client record independently:

```sh
python -m tools.kv.native.serving_verify ./awq-http-pilot/closed-32-client.json
python -m pytest -q tests/test_native_serving.py
```

`serving_quality` captures the frozen NLL, retrieval and full GSM8K tasks
through the actual HTTP worker. Its teacher-forced NLL records raw losses
before selecting the frozen target IDs. Natural scored completions retain
EOS, truncation, response bodies and generated token IDs. Compare serial
HTTP outputs with `serving_reference --sequential-reference` separately
from batch-order sensitivity.

After all three quality arms and their serial references complete, audit
the retained records without a GPU:

```sh
python -m tools.kv.native.serving_quality_audit \
  --root ./http-study --format awq --output ./awq-quality-audit.json
```

The reader verifies raw response bodies, teacher registration, every target
and cached-decode step, fixture identities and parser decisions. It computes
paired preservation bounds against both native controls. An aggregate score
gain cannot cancel correct-to-wrong transitions in the frozen loss test.
Quality and controlled parity remain separate from performance qualification.

After independent qualification, `serving_trial` starts a fresh server for
one arm of one frozen block. It verifies the plan hash, input trace hashes,
qualified scope and serving implementation hashes before launch. The plan
fixes warmup, cell order, request traces and settling intervals. It rejects
retries. The caller must freeze the complete counterbalanced block order,
primary comparisons and analysis family before confirmation; the runner
never selects loads or adds blocks from observed results.

The `mixed_transfer` variant and `transfer_smoke` entry point are experimental
qualification tools for additional head geometries. They retain the original
Qwen3 default and accept explicit conventional-attention geometry for Phi-4.
Their synthetic checks cover page remapping, K/V gathers, output placement,
stale maps and shifted outputs. Scored quality and fresh performance tests
remain required before any model or hardware is advertised as supported.

`serving_statistics` computes paired effects across independent run blocks.
It reports an exact sign test, a distribution-free median interval with its
actual coverage, and Holm correction across the full declared primary family.
A practical median-effect screen is distinct from an interval-bound claim.
A comparison with incomplete or failed blocks cannot pass its superiority gate.

For a new 1K HTTP study, select the frozen 1K document panel explicitly.
The trace, serial reference and quality capture use that same panel; they
do not truncate the 8K panel. Pass `--length 1024 --cache-bytes 34359738368`
to the pilot, quality capture and reference commands when the frozen plan
uses a 32 GiB pool. The independent quality reader accepts the same length
and cache budget and rejects mismatched captures. `--prefix` selects an
artifact-name prefix when retaining multiple context-length studies.
These options do not qualify 1K performance or reuse 8K quality by themselves.
