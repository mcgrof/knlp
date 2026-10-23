# Phi-4: quality transfer and interrupted throughput studies

Phi-4 passes the frozen NLL, retrieval and full GSM8K preservation checks
against native FlashInfer and actual FlashAttention 3 (FA3). The physical
K16/V8 route also passes production qualification at 1K/batch 32 and
8K/batch 16 on H100. These results establish the tested model transfer. Both fixed throughput
studies have infrastructure failures and support no confirmed speed claim.
HTTP serving is outside this Phi-4 study.

All paths use the same pinned Phi-4 checkpoint, FP16 weights and compute,
original model-specific scales, and a 32 GiB persistent KV budget. The
candidate stores FP16 keys and E4M3 FP8 values. Native controls store FP16
keys and values. The geometry has 40 attention layers, 40 query heads,
ten KV heads and head dimension 128.

| Quality check | K16/V8 | Native FlashInfer | Native FA3 |
| --- | ---: | ---: | ---: |
| Full GSM8K strict correct / 1,319 | 1,253 | 1,253 | 1,252 |
| Native-correct → candidate-wrong | — | 4 | 3 |
| Native-wrong → candidate-correct | — | 4 | 4 |
| GSM8K loss-probability upper bound | — | 0.693% | 0.587% |
| 1K retrieval correct / 100 | 100 | 100 | 100 |
| 8K retrieval correct / 100 | 99 | 100 | 100 |

Bounds are one-sided 95% exact bounds on the loss-transition probability,
using all paired questions as the denominator. GSM8K requires an upper
bound below 2%; retrieval requires below 5%. At 8K, one retrieval loss
has an upper bound of 4.656%, so it passes this rule without establishing
zero loss. The full GSM8K task uses natural question lengths and EOS;
it is reused in both context summaries, not counted as two experiments.
The frozen strict extractor's known terminal-dot limitation remains.

NLL uses 32 documents per context and 128 cached target steps per document;
prefill step zero is excluded. The 1K upper bounds on paired delta NLL
are −0.000308 and −0.000238 against FlashInfer and FA3. The 8K bounds are
+0.000510 and +0.000606. All are below the frozen `ln(1.01)` margin.
The archived result
includes paired accuracy intervals, all task counts, source identities and
raw evidence hashes. Passing these tasks is not a universal accuracy claim.

Production checks verify calibrated scales inside the engine before graph
capture, FP16 weight dispatch, actual FA3 dispatch, physical KV storage,
output placement and independent candidate routing checks. Initial native
8K rotation checks failed. A separate, prospectively frozen check required
exact original-order repetition and retained rotation as diagnostic. All
three paths passed that check; the earlier failures remain failed.

The allocated candidate pool has 223,696 token slots per layer, versus
167,760 for either native control. This is approximately one third more
slots at the same byte budget, with page-rounding differences. The
candidate also needs 320 MiB of staging. The measured pools do not prove
a maximum request-residency gain or a 25% reduction in total GPU memory.
The stated 74 GiB allocator and 78 GiB driver envelopes are larger than
the persistent KV budget; production-profile peaks are labeled separately
from startup-through-measurement telemetry.

The separately archived inputs, described in the
[input documentation](https://github.com/mcgrof/knlp/tree/main/tools/kv/native/fixtures/phi4), include the original scales, both context panels, complete GSM8K task,
all six measured protocols, byte hashes and dataset attribution. This
experimental transfer has its own geometry port and qualification; the
[current Qwen installer](asymmetric-install.md) retains its stated scope.


## Fixed throughput studies

Both contexts planned ten fresh, counterbalanced blocks against the same
native FlashInfer and native FA3 controls. The metric is complete-request
output tokens divided by CUDA-synchronized host duration. It includes
prefill, scheduling and decode. Each arm excludes two warmup cohorts,
then measures ten cohorts at 1K/batch 32 or four at 8K/batch 16. All requests
force 128 output tokens. The same 32 GiB KV pool and 22 CPU threads apply.

**Neither fixed schedule completed successfully.** At 1K, a source-host
DNS failure prevented all three arms of block 4 from launching. At 8K,
an operator mistake enabled offline model loading after an incomplete
preflight. The loader's earlier full-snapshot check required ancillary
files absent from the cache, so ten subsequent launches failed before
GPU model loading. The preceding stalled online launch recovered and
completed unchanged. These are preparation failures, not measured
quantization failures. Every failed attempt remains in the result.

| Workload | Valid arm captures / planned | Complete blocks / planned | Observed paired median vs FlashInfer | Observed paired median vs FA3 |
| --- | ---: | ---: | ---: | ---: |
| 1K / batch 32 | 27 / 30 | 9 / 10 | −0.224% | −2.058% |
| 8K / batch 16 | 20 / 30 | 6 / 10 | +1.246% | +1.411% |

These effects describe only complete observed blocks. **No primary tests
or confidence intervals are computed on either reduced set.** No failed
block was replaced and no extra block was added. The frozen rule requires
the full schedule, a 1% median gain screen and Holm correction over both
native comparisons. A positive descriptive median does not pass that rule;
an incomplete schedule does not prove absence of a benefit either.
The full performance record
retains each valid arm, all failures, frozen designs and source hashes.
The [independent CPU reader](https://github.com/mcgrof/knlp/blob/main/tools/kv/native/cohort_confirmation.py)
checks the raw completions, actual active decode batches and memory before
any statistical inference.

## Matched diagnostic profiles

Separate qualified smoke profiles cover one complete cohort per arm and
context. They are excluded from confirmation. At 1K, matrix operations
account for about 79% of summed GPU kernel duration; attention and merge
kernels total 434.4 ms for K16/V8, 444.6 ms for FlashInfer and 425.4 ms for
FA3. The candidate also gathers values and writes the compressed cache.
At 8K, matrix operations account for about 69–70%; attention and merge
kernels total 1,973.1, 2,107.8 and 2,178.1 ms respectively. Actual fresh
8K admissions are 8/7/1, followed by batch-16 decode.

Kernel names classify components, and CUDA launch correlations join them
to actual scheduler steps. These sums can overlap and include profiling
overhead; they are not an additive wall-time explanation or an independent
speed confirmation. The profiles show workload-dependent component costs.
They identify no single established code defect that justifies claiming a
repaired performance gain.
