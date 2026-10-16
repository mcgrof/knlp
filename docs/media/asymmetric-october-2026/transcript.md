# October 2026: optimized asymmetric KV serving
Published 2026-10-16. Stock synthetic narration (Aria).

## Smaller KV. Two qualified B8 gains.
This October 2026 update reports two qualified batch eight serving gains. Packed AWQ Qwen 3 eight B improves complete request throughput by 2.28 percent over native FP 16 KV. Dense Qwen 3 eight B, executing in FP 16, improves by 2.49 percent. K 16 V 8 uses 25 percent less persistent KV payload at equal token slots. That saving is not total GPU memory.

## The workload defines the result
The workload is one H 100 SXM 80, tensor parallelism one,8192 input tokens, batch eight, and 128 generated tokens. Prefix caching and chunked prefill are off. Each headline is a median paired ratio from seven fresh three arm process blocks, separate from its pilot. Each experiment keeps the same weight checkpoint across cache arms. AWQ weights remain packed. These are workload specific results.

## Remove enough prefill overhead
Earlier writer, scale, and graph work improved decode, but long prompt prefill held back complete requests. Compact gathers stored keys and already quantized values into one bounded shared FP 16 workspace, then uses native FA 3 prefill. Casting does not recover lost precision. There is no full value shadow per layer. Decode still reads compressed values. AWQ compact is 6.19 percent faster than the preceding mixed stack.

## Count persistent and temporary memory
Persistent KV and temporary workspace are separate. Compact owns about 296.14 mebibytes of constant GPU workspace, including 256 mebibytes of shared payload, plus variable metadata and 8 mebibytes of pinned CPU storage. Count aliases once. Fixed timing pools are 48 gibibytes for AWQ and 32 for dense FP 16. More token slots need not reduce reserved memory.

## Capacity is a separate measurement
A separate automatic pool test confirms 58 resident requests for compact,59 for preceding mixed, and 44 for native. Compact admits 31.82 percent more than native after workspace costs. This measures capacity, not throughput or queueing. Do not multiply the capacity increase by the batch eight speed ratio.

## Keep the boundaries visible
The existing AWQ eight K, batch 32 result is about 1.44 percent slower than native, with mixed prefill and decode fallback. The BF 16 transfer has a small speed gain, but fails frozen GSM gross loss preservation despite a higher net score. It remains experimental. A separate symmetric AWQ recipe fails quality and has no comparative timing.

## Next question: matched batch32
Next, the matched batch 32 study will keep all eight historical models visible. It will recover old workload details where possible and otherwise label a new standardized comparison. It will separate complete requests, completed decode, prefill overhead, and capacity pressure. Positive claims require quality and timing gates. Regressions, unsupported routes, and missing cells will be reported. Source and reproduction settings are linked on the decode page.
