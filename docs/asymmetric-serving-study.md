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
