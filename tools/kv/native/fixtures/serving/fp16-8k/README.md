# Exact dense Qwen3-8B 8K HTTP arrival traces

These 25 traces preserve the measured request IDs, nanosecond arrival
offsets, prompt token IDs and output lengths. The compact catalog refers
to the 32 shared 8K prompts in the [current FP16 fixture](../../current/README.md).
It does not regenerate random arrivals. The prompt source, tokenizer,
attribution and license remain those of that fixture.

From the repository root, restore the original 394,966,354 bytes with a
CPU-only command. The output directory must not already exist:

```sh
python -m tools.kv.native.serving_trace_bundle \
  --catalog ${KNLP_KV_FIXTURES}/serving/fp16-8k/catalog.json.gz \
  --manifest ${KNLP_KV_FIXTURES}/current/fp16/qwen.json.gz \
  --output restored-http-traces
```

Use `--verify-only` instead of `--output` to check every reconstructed
SHA-256 without writing traces. `manifest.json` records the compressed
catalog hash and each original trace hash. The tool rejects a changed
prompt manifest or changed restored bytes.

The measured harness is `0e34f048a82420a3a7b7a5d6d27254d4cd7692b8`.
Its frozen design SHA-256 is
`f2d5258bd05d6903427e62dda9ce543461369f8d33ecc0e913a3dfc3156371b5`.
This export helper is a later CPU utility, not a replacement measured
harness. All three GPU arms receive identical traces within each block.

| Cell | Traces | Requests per trace | Policy |
| --- | ---: | ---: | --- |
| Closed loop | 10 | 512 | Concurrency 32 |
| Open loop, near load | 10 | 512 | 2.9 offered requests/s |
| Closed loop, descriptive | 1 | 128 | Concurrency 8 |
| Open loop, descriptive | 2 | 128 | 0.8 and 1.6 offered requests/s |
| Isolated requests | 1 | 8 | Five-second offsets |
| Warmup prompt inventory | 1 | 32 | One batched API call; one repetition |

The warmup file supplies the 32 prompt token lists. The trial sends them
together in one non-streaming `/v1/completions` call with 128 forced output
tokens per prompt. It does not replay the warmup file as 32 separate HTTP
requests. Its stored concurrency field does not control this batched call.
The measured closed-loop cells do send separate streaming HTTP requests.

Every request has exactly 8,192 input tokens and 128 forced output tokens.
Request and drain timeouts are 300 seconds, with no retries. Timing,
quality and production checks have separate records; a trace alone does
not establish a successful run or a performance benefit. Open-loop
throughput includes the final drain. No operational latency objective or
maximum sustainable arrival rate is defined by these finite traces.
