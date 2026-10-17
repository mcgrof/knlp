# October 16 repair checks: neither candidate qualified

Two separate bounded H100 checks followed the completed B32 study. They
produced no fresh quality or timing result. The previous measured source
refs and published results retain their original scope.

The mixed-prefill source candidate rebases fresh-prefill query boundaries,
page indptr and source page IDs together inside a mixed scheduler batch.
Decode, the writer, compiled vLLM code and the 256 MiB workspace are unchanged.
Seventeen focused CPU tests exercise metadata, output placement and lifetime
paths. The candidate starts at accepted vLLM `26cd429`; FlashInfer stays
`800a550`. A separate D64 producer/consumer candidate starts at `800a550`.
Neither new dependency candidate is a qualified public release.

The AWQ check freezes Qwen/Qwen3-8B-AWQ revision
`4da05a8edb55c6046cce958586c33b61da07bb79`, accepted scales, FP16,
8K/B32/128, TP1, page16/NHD, a 48 GiB pool, 22 threads, 96 maximum sequences,
65,536 batched tokens, APC/chunking off and FULL_DECODE_ONLY graphs.
Native `c21c087` and accepted compact `26cd429` remain separate controls
with FlashInfer `800a550`. Prior control quality reuse was bound to actual
model/tokenizer, scale, prompt, B32 and packed-weight provenance.

The synthetic D128 production fixture passes, including remapped cache
writes, decode and prefill. CPU reconstruction gives maximum errors below
0.00003 against the unchanged 0.005 maximum / 0.001 mean limits.
The model loads all 36 attention layers and 144 packed Machete modules,
attests genuine shared K16/V8 storage and captures decode graphs.
Observed allocator peak reserved memory is 63.72852 GiB, within 68 GiB.

The first admission contains eight fresh prompts and no decode requests.
All 34 completed layer comparisons are bit exact against the existing
mixed-cache fallback for full output and LSE. The changed mixed-batch route
has not been exercised when a fixed sampled reference check stops the run.
At layer 33, request ordinal 3, final query position 8191, Q head 0,
dimension 81, the output is 2.45703125. The independent FP32 calculation,
with the prescribed output rounding and scale order, gives 2.4609375.
Their difference, 0.00390625, exceeds the frozen absolute-plus-relative
bound of 0.0034609376453. This is one violation among 417,792 sampled
output elements; sampled LSE checks pass. Full operands are retained.

CPU replay reproduces the violation. A diagnostic that rounds normalized
attention probabilities to FP16 returns the observed value exactly.
The unchanged kernel converts softmax intermediates to its computation
type before PV multiplication. This supports a rounding explanation;
the diagnostic does not emulate the full tiled kernel or replace the
frozen reference. The failed gate does not establish a routing regression.
No fresh quality, pilot, confirmation or B8 canary process was unlocked.

For D64, CPU tests cover 108 concrete producer cases using actual C++
producer/helper/permuter code. Four CUDA layout instantiations compile.
The numerical JIT then cannot find `ninja`: it is installed in the virtual
environment, but the launcher's PATH omits that environment's bin directory.
No original D64 numerical output or repaired-kernel result exists from this
check. The earlier Qwen0.5 failure row is unchanged. Launchers must expose
the environment's JIT tools; compilation alone is not numerical evidence.

Reusable public knlp changes are separate: legacy attribution `70c6d2be`,
failure capture `29a33de4`, D64 regression driver `abab1c56`, and optional
stored-value model checks / candidate identity `c802ffe8`. The original
0.005 maximum / 0.001 mean geometry limits and 1e-3 absolute / relative
model comparison limits were not relaxed. Twenty-eight focused knlp CPU
tests pass. The inherited vLLM typecheck failures match the control.

Raw evidence, source candidates and failed gates remain in the private
research record. Every stopped-pod export is SHA-verified and losslessly
restored; independent CPU replay checks the tensors and residuals. Both pods
are deleted, deadlines retired, and neither protocol was extended or retried.
Total use is 0.334486151 GPU-hours. The provisional all-in estimate is
$1.351324 including conservative disk allowance; invoices are unsettled.
