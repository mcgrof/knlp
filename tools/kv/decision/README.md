# Precision placement and scale-policy reference

This harness compares native K/V with asymmetric K16/V8, reverse K8/V16
and symmetric K8/V8. Each compressed format has fixed per-layer/component
scales and dynamic per-token/head scales. Q and attention arithmetic remain
native. Actual E4M3FN tensors store compressed components; each attention
layer temporarily reconstructs them. These timings do not measure a fused
serving backend.

The scoped checkpoints are Qwen3-8B FP16 at
`b968826d9c46dd6066d109eabc6255188de91218` and Qwen2.5-7B-Instruct BF16 at
`a09a35458c702b33eeacc393d103063234e8bc28`. Other architectures are rejected.
The earlier `tools/kv/recent` implementation and results are unchanged.

Use Python 3.12.11, Torch 2.11.0 with CUDA 12.9, Transformers 5.10.4,
Accelerate 1.10.1, NumPy 1.26.4, SciPy 1.17.1 and pytest 8.4.2.
Input preparation additionally uses datasets 4.8.4. Install the CUDA Torch
wheel from the official PyTorch index and the remaining packages from PyPI.

```sh
python -m pytest -q tests/test_decision_cache.py tests/test_decision_statistics.py
python -m tools.kv.decision.prepare --model qwen3 --out inputs/qwen3
python -m tools.kv.decision.prepare --model qwen25 --out inputs/qwen25
python -m tools.kv.decision.freeze --out source.tar.gz
```

Run from an extracted source archive, which includes `SOURCE.json` and
checks every source file before execution. Download the exact checkpoint
revision with `huggingface_hub.snapshot_download`. Create an official weight
manifest from `HfApi.model_info(..., files_metadata=True)` with `model`,
`revision` and `files` mapping safetensor names to `bytes` and LFS `sha256`.
The runner verifies every downloaded weight file against that manifest.
Freeze source, weight, input and protocol identities before model outputs.

```sh
python -m tools.kv.decision.run --inputs inputs.json.gz --model-path model \
  --weights official-weights.json --out calibration --stage calibration
python -m tools.kv.decision.oracle --calibration calibration
python -m tools.kv.decision.run --inputs inputs.json.gz --model-path model \
  --weights official-weights.json --out development --stage development \
  --scales calibration/scales.json --batch 8
python -m tools.kv.decision.run --inputs inputs.json.gz --model-path model \
  --weights official-weights.json --out confirmation --stage confirmation \
  --scales calibration/scales.json --development development --batch 8
```

Calibration observes native cache-write operands after normalization/RoPE
on 32 independent documents at 1K and 8K, including 128 forced greedy decode
steps. Each fixed component scale is FP32 `max(maxabs, 1e-12)/448` over all
those operands. A second native replay audits clipping and error for both
scale policies on every calibration operand. Saved activation fragments
also undergo independent E4M3FN codebook and reconstruction comparisons
on CUDA when available (otherwise CPU). The recorded device is explicit.
Dynamic scales apply the same equation independently to each fresh
head/token vector. Stored codes are never requantized. Conversion clips
to ±448, uses nearest/even E4M3FN rounding and reconstructs by FP32
multiplication followed by the native dtype cast.

Input preparation excludes the first 128 eligible Wikipedia articles used
by the earlier reference, then draws 32 calibration, 16 development and
64 confirmation documents from a seeded new pool. Document IDs are disjoint
between stages. Within each stage the same documents supply the two context
lengths; inference remains per document within each named endpoint.
Retrieval instances have independent generators and unique cluster IDs.
Development uses 32 retrieval instances/context and 64 GSM8K training items.
Confirmation uses 200 retrieval instances/context and all 1,319 GSM8K test
items. GSM8K is an exposed standard benchmark. The previous reference's
answer-label parser and 2,048-token limit remain unchanged; saved truncation
and parser-failure fields are retained with that scoring rule.

Each candidate receives two separate certificates: answer preservation and
aggregate-math noninferiority. The two families each have alpha .025 across
12 candidate/model combinations. Every certificate requires native accuracy
at least 50%, retrieval gross-loss upper bounds at most 5% at both lengths,
and relative mean NLL upper bounds at most 1% at both lengths. Preservation
also requires GSM gross-loss upper at most 2%. Aggregate-math instead requires
a paired net GSM accuracy lower bound at least −1 percentage point.
Gross bounds are exact binomial. Net bounds invert Tango's efficient score
with approximate coverage. NLL uses 100,000 paired document bootstrap draws,
seed 20261020 and linear quantiles. Report absolute NLL deltas separately.

Direct A-minus-R and A-minus-S tests form eight contrasts per named endpoint
family across the two models and policies. Binary outcomes use exact McNemar
p-values and descriptive paired score intervals; NLL uses a centered paired
bootstrap with descriptive and Bonferroni-eight intervals. Holm adjustment
is applied across all eight contrasts per endpoint, without pooling different
endpoints into a general superiority claim. Failed or incomplete panels
cannot be replaced or enlarged to obtain a pass.

CPU analysis expects `inputs/{qwen3,qwen25}/inputs.json.gz` and
`{qwen3,qwen25}/artifact-mirror/confirmation/` below the supplied study root. A `PLAN.json` `executions` mapping may
name retained attempt directories explicitly; no latest-result selection occurs:

```sh
python -m tools.kv.decision.analyze --study study --out analysis.json
```

A quality-reference pass alone does not qualify production support, memory
capacity under a serving allocator or useful serving throughput. Those need
actual-worker tests and a separate frozen quality/serving protocol.
