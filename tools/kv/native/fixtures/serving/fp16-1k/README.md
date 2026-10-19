# Exact dense Qwen3-8B 1K HTTP arrival traces

These 25 traces preserve the original request IDs, nanosecond offsets, prompt
IDs and forced output lengths. They use the 32 documents in the 1,024-token
panel of the [current FP16 fixture](../../current/README.md), with its original
model, tokenizer, scales, dataset attribution and licenses.

Restore all 75,021,321 original bytes on CPU into a new directory:

```sh
python -m tools.kv.native.serving_trace_bundle \
  --catalog tools/kv/native/fixtures/serving/fp16-1k/catalog.json.gz \
  --manifest tools/kv/native/fixtures/current/fp16/qwen.json.gz \
  --output restored-http-1k-traces
```

`--verify-only` checks every reconstructed hash without writing the traces.
The measured harness is `9a531755ae4da676bd6965d68b9c1af70f6966b2`.
The exact `DESIGN.json` SHA-256 is
`2b27314f55b3b247e4b1cdc62636365473a6a3b8917fa651cc7025868e371add`.

| Cell | Traces | Requests per trace | Policy |
| --- | ---: | ---: | --- |
| Closed loop | 10 | 768 | Concurrency 32 |
| Open loop, near load | 10 | 768 | 14.9 offered requests/s |
| Closed loop, descriptive | 1 | 128 | Concurrency 8 |
| Open loop, descriptive | 2 | 128 | 3.7 and 8.1 offered requests/s |
| Isolated requests | 1 | 8 | Five-second offsets |
| Warmup prompt inventory | 1 | 32 | One batched API call |

Every timing request has 1,024 input tokens and 128 forced outputs. The
persistent KV budget is 32 GiB. Counts and absolute rates come only from
excluded native pilots. Ten fixed blocks supply the complete four-hypothesis
family; the other cells are descriptive. A failed request remains a failure.
The trace inventory is not itself a performance result.

The [8K replication procedure](../REPRODUCE.md) also describes the same
qualification and replay tools used here. Keep a current tools checkout that
contains this 1K directory; do not switch it to the earlier CPU-tools pin
shown in the 8K example. Use a separate worktree for the measured source.
For this 1K design, use its measured
harness above, `--length 1024`, `--cache-bytes 34359738368`, this catalog and
design, and the `fp16-1k` capture/output prefix throughout. Requalify all three
arms on the replication machine. The CPU plan builder rejects mismatched
source, context, budget, protocol or quality evidence. The historical batched
1K study is a different experiment and is not repeated by this procedure.
