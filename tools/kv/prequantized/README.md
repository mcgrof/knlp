# Cached KV precision with packed weights

This runner compares four key/value (KV) precision arms without replacing
the checkpoint's shipped AWQ or MXFP4 weights. It measures quality during
cached decode. Stored entries are rounded once to scaled E4M3FN, then held
in 16-bit tensors. The measurement is `fake_quant_cached_decode`; it does
not establish compressed-cache capacity or serving throughput.

AWQ is activation-aware four-bit weight quantization. MXFP4 stores
four-bit floating-point weights with shared block scales. Their packed
weight formats are independent of the cache format. E4M3FN is an eight-bit
floating-point format with finite range up to 448. Each layer uses separate
K and V scales, frozen from the maximum absolute calibration value divided
by 448. Rounding uses nearest with ties to even and FP32 scale arithmetic.

The six stages are `prepare`, `doctor`, `calibrate`, `quality`, `summarize`
and `native`. Run the script from its own directory. This ensures that
Hugging Face's `kernels` loader is imported before the unrelated package
with that name in the repository:

The qualified environment uses Python 3.12, PyTorch 2.8.0 with CUDA 12.8,
and Triton 3.4.0. Install the PyTorch CUDA wheel first, then the pinned
dependencies. A working C compiler is required by Triton's driver probe.

```bash
cd tools/kv/prequantized
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python runner.py prepare --matrix /path/matrix.json \
  --model Qwen/Qwen3-8B-AWQ --revision FULL_MODEL_COMMIT \
  --dataset wikimedia/wikipedia --dataset-revision FULL_DATA_COMMIT \
  --manifest /path/qwen.json.gz --output /path/output
python runner.py doctor --manifest /path/qwen.json.gz --output /path/output
python runner.py calibrate --manifest /path/qwen.json.gz \
  --scales /path/scales.json --output /path/output
for arm in k16_v16 k8_v16 k16_v8 k8_v8; do
  python runner.py quality --manifest /path/qwen.json.gz \
    --scales /path/scales.json --output /path/output --arm "$arm"
done
python runner.py summarize --manifest /path/qwen.json.gz \
  --scales /path/scales.json --output /path/output
python runner.py native --manifest /path/qwen.json.gz --output /path/output
```

`prepare` accepts matrix schema version 1; start with
[matrix.example.json](matrix.example.json). Required numerical fields are
validated explicitly. It pins checkpoint, tokenizer, dataset, generator
source, templates and exact token IDs. The default corpus is WikiText-103
raw. If its test split cannot supply 32 distinct long articles, preparation
fails before evaluation. The explicit Wikipedia alternative collects the
first 128 eligible English articles from the pinned `20231101.en` stream,
then freezes a seeded disjoint 16/64 calibration/evaluation split. This
bounded sample supports an exploratory result on those documents, not a
population-wide claim. The last 32 evaluation documents and retrieval
cases 100–255 form a frozen reserve; they are not evaluated by default.

`doctor` records packed module buffers and kernel dispatch, bypass error,
the quantizer reference, actual attention reads, zeroed-past logits,
architecture boundary checks and ten native chat fixtures. GPT-oss uses
the original eager attention with sink logits and sliding-window masks.
Its external Triton kernel revision must be supplied to `prepare` through
`--kernel-revision`. Whole-model MXFP4 expansion fails closed.

The Qwen loader supplies the removed `PytorchGELUTanh` import alias needed
by AutoAWQ 0.2.9 with Transformers 4.57.6. That class must never occur in
the loaded model; the qualified checkpoint uses SiLU. Packed GEMM buffers
and kernel dispatch remain subject to T00. AutoAWQ is a deprecated kernel
dependency, so this receipt qualifies that pinned stack only.
The existing Triton GEMM runs with `split_k_iters=1` for cached steps.
The default split-K kernel uses FP16 atomic accumulation and can vary
between repeated baselines. The single-partition setting keeps the same
packed buffers and quantization recipe while removing that atomic order
variation. T00 records the setting, and all arms use it. Prefill retains
the backend's temporary per-linear weight unpacking and matmul heuristic.

The architecture comparator is the unmodified cached backend at the same
positions and appends. Bypass cache tensors must match exactly, including
sliding-window eviction. Full-history replay is reported separately as a
diagnostic, including any discrepancy in the original backend. Its cause
is not established by this gate. A passing bypass gate does not prove
full-history numerical equivalence. Both computations retain the selected
packed-weight backend and kernel settings.

All arms use the same frozen per-layer K and V scales. Manifests name the
complete cache recipe. `fresh_current_qdq_v1` leaves new prefill, chunk and
decode tokens fresh for their own attention call; past entries are rounded.
`write_before_attention_qdq_v1` rounds new entries before attention,
including the current decode token. Both keep physical tensors in the
activation dtype. Positions and eviction remain controlled by DynamicCache.
An attention wrapper verifies that its inputs are the exact returned
cache tensors. Choose a serving backend's actual recipe before claiming
that a QDQ experiment models it; neither recipe alone qualifies a backend.

`doctor` requires the model-specific packed decode kernel, finite logits,
and exact native repeat and instrumentation bypass results. A generic
matrix multiplication event cannot pass its packed-dispatch gate.
`quality` and `summarize` validate the complete current manifest, source,
gates, calibration sample IDs, scale hash and every result's metadata.
Missing, failed, duplicate or foreign samples fail closed. Resume also
validates existing rows. Supply a new output directory after a source fix.
The NLL
stage measures negative log likelihood: the mean negative log probability
of the next ground-truth token. It scores 128 predictions made by real
one-token cached decode calls,
excluding the prefill prediction. Summaries bootstrap whole documents;
retrieval loss uses an exact upper bound, including nonzero uncertainty
when no paired failures occur. Run all four arms before summarizing.
Direct accuracy differences use conservative exact bounds on the paired
correct-to-wrong and wrong-to-correct transition probabilities. Generation
agreement and repetition counts are sensitivity diagnostics, not accuracy.

`native` records an unsupported capability receipt. This implementation
does not supply a packed mixed-cache serving kernel. Its 16-bit QDQ cache
must never be reported as native cache bytes or performance. A future
backend needs its own correctness and per-arm numerical bridge before
making such claims.

## Deferred queries and replay diagnostics

A deferred query stores a catalogue before the model sees which key to
retrieve. Every record uses the same random key and value distributions.
The generator selects the requested record after building the catalogue.
It tokenizes the complete chat prompt once, then freezes its token IDs
and catalogue/query split. The two cache calls therefore preserve the
same prompt and positions as concatenated execution.

`followup.py prepare` reuses frozen document windows and calibration IDs
from an earlier manifest while generating new independent retrieval
catalogues. Development uses seed 20261016 and evaluation 20261017.
The full logical prompts contain exactly 2048 or 8192 tokens. The new
matrix omits unlabelled generation diagnostics. Prepare only after
committing all source, and use a new output directory:

```sh
python followup.py prepare --old-manifest /path/original.json.gz \
  --manifest /path/deferred.json.gz --output /path/new-output \
  --recipe write_before_attention_qdq_v1
python followup.py drift --manifest /path/deferred.json.gz \
  --output /path/new-output
python runner.py doctor --manifest /path/deferred.json.gz \
  --output /path/new-output
python followup.py sensitivity --manifest /path/deferred.json.gz \
  --output /path/new-output
```

Sensitivity controls query three different keys from independent copies
of each native catalogue cache. They also zero catalogue contents while
preserving positions, change only the requested value in a reprefilled
catalogue of the same length, and compare concatenated execution.
The manifest freezes the acceptance criterion before any quantized
comparison. A failed sensitivity control closes the quality gate; it
cannot become evidence that quantization preserves retrieval.
Then run calibration, all four quality arms and summary as above.
The fresh-current recipe remains available for quality-only experiments.

The drift diagnostic compares full replay, one-token cached appends and
four-token chunks at identical positions. It records next-token loss,
top-1 changes and distribution error on eight fixed documents per length.
GPT-oss also covers its six sliding-window boundary lengths. Captured
residuals, projections and router outputs locate early divergence.
A captured head is replayed in FP32 with absolute causal and sliding
masks, post-RoPE Q/K, grouped-query head mapping and the original sink
logit. This diagnostic does not relax the exact same-path bypass gate
or automatically establish the cause of a discrepancy.
