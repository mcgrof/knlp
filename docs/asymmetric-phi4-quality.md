# Phi-4: quality and physical-storage transfer

Phi-4 passes the frozen NLL, retrieval and full GSM8K preservation checks
against native FlashInfer and actual FlashAttention 3 (FA3). The physical
K16/V8 route also passes production qualification at 1K/batch 32 and
8K/batch 16 on H100. These results establish the tested model transfer;
performance and HTTP serving require separate evidence.

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
The [machine-readable result](data/asymmetric-phi4-quality-20261019.json)
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

The [exact public inputs](https://github.com/mcgrof/knlp/tree/main/tools/kv/native/fixtures/phi4)
include the original scales, both context panels, complete GSM8K task,
all six measured protocols, byte hashes and dataset attribution. This
experimental transfer has its own geometry port and qualification; the
[current Qwen installer](asymmetric-install.md) retains its stated scope.
