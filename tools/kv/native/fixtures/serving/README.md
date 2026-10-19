# HTTP serving evidence tools

The [dense 8K catalog](fp16-8k/README.md) reconstructs exact measured
arrival traces from the shared public prompt fixture. The tools below
read captures produced by the public HTTP harness. They run on CPU;
production, quality and performance remain separate checks.

```sh
# Actual worker, calibrated scales, graph changes, physical storage,
# routing, weight dispatch and native FA3 dispatch:
python -m tools.kv.native.serving_production_audit \
  --root captures --format fp16 --prefix fp16 \
  --tag qualification-refresh --output production-audit.json

# Recompute all frozen tasks and both serial-reference orders:
python -m tools.kv.native.serving_quality_audit \
  --root captures --format fp16 --prefix fp16 --length 8192 \
  --cache-bytes 42949672960 --output quality-audit.json

# Read all ten frozen blocks plus the separate descriptive cells:
python -m tools.kv.native.serving_confirmation \
  --root captures --plan frozen-plan --prefix fp16-8k \
  --output confirmation-audit.json
```

Use the normal Python interpreter, with assertions enabled. The quality
reader also uses the dependencies of the paired statistics modules.
Production checks are specific to the tested Qwen3-8B geometry and
dispatch. Confirmation checks require the dense FP16 H100 protocol,
either 1,024 or 8,192 input tokens and 128 forced outputs. A 1K capture
uses its own prefix, frozen plan and 32 GiB budget.

The confirmation reader verifies raw request events independently of the
recorder's summaries. It includes every offered request in reliability
accounting. It reports latency conditional on valid completion, finite
throughput including drain, actual worker admission, per-cell preemption,
client backlog and sampled server gauges. Stream gaps are gaps between
token-bearing events; a chunk with multiple tokens does not reveal its
internal token times.

The statistical unit is a fresh matched block. The reader retains the
whole predeclared comparison family and applies Holm correction. The
median effect screen is separate from its uncertainty interval. A failed
request prevents an unqualified superiority result. Missing or malformed
engine captures reject a complete audit; successful blocks must not be
silently substituted for them. No extra blocks or retries are implied.

Torch memory values are peaks since the worker's measurement reset.
Driver values are explicitly labeled snapshots; they are not a continuous
startup-to-shutdown driver peak. Equal persistent cache bytes do not mean
equal token capacity or 25% less total GPU memory.

The frozen plan directory must contain `DESIGN.json`, each arm/block
plan, its original qualification receipt and all referenced traces.
These are the records of a particular execution. A new replication must
derive its qualification from its own valid production and quality
evidence; copying a historical success receipt does not qualify a changed
model, scale file, implementation or engine policy.
