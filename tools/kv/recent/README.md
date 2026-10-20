# Recent-model physical KV-cache reference

This experiment asks whether keeping keys at BF16 preserves quality
better than compressing both keys and values. It compares three real
persistent-cache layouts with the same official model implementation:

| Arm | Keys | Values | Scale storage |
| --- | --- | --- | --- |
| `native` | BF16 | BF16 | None |
| `k16v8` | BF16 | E4M3 FP8 | FP32 per token and value head |
| `k8v8` | E4M3 FP8 | E4M3 FP8 | Separate FP32 per token and key/value head |

Each quantized vector uses `max(max(abs(x)), 1e-12) / 448` as its scale.
Only fresh tokens are quantized. Older codes are retained without
requantization. Attention reconstructs its current layer's inputs to
BF16 temporarily. It does not retain a reconstructed full-cache shadow.
Independent Qwen text prediction-loss requests supply explicit token
positions. This prevents the multimodal wrapper from reusing model-level
position offsets left by an earlier generation batch. The official
model implementation is unchanged.

Recurrent states keep the official implementation and precision.
Sliding layers discard old backing storage outside their window.

This is a quality and storage reference. The quantizer, runtime and
transient reconstruction differ from the optimized FlashInfer/vLLM
recipe. Its timings do not establish production serving performance.

## Checkpoints and fixed inputs

The checkpoints were selected using current public adoption data. Those
counts are selection proxies, not evidence of model quality.

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |

The preparation program pins Wikipedia revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa` and GSM8K revision
`740312add88f781978c0658806c59bc2815b9866`. It freezes token IDs and
selection metadata before candidate outputs are observed. Thinking is
disabled through the official chat template. Calibration IDs are
retained as provenance but no calibration is used.

The measured input archives have these SHA-256 identities:

```text
qwen38 73545fd2ce9703ee400d0d9c2067cc8935b46a18d8aaadd23ce3aaa52c65d985
gemma4 de4af07a5db69d5c175c7df6082ff45bc132bb349f5ffb73e076f229e6bc18e8
```

A different compressed-file hash is not automatically the same input
panel. Compare the frozen token IDs, metadata and source versions before
combining or claiming a reproduction of measured results.

## Run the reference

Use Python 3.12.11, a CUDA 12.9 environment and sufficient GPU memory
for the original BF16 weights, persistent states and temporary attention
inputs. The Qwen allocation is advertised with 288 GB GPU memory; the
Gemma allocation has 96 GB. The execution is one GPU per model; all
three arms for a model share that GPU. Access to a checkpoint remains
subject to its publisher's license and access rules.

```bash
uv venv --python 3.12.11 .venv-recent
uv pip install --python .venv-recent/bin/python \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install --python .venv-recent/bin/python \
  transformers==5.10.4 accelerate==1.10.1 scipy==1.16.3 \
  numpy==1.26.4 flash-linear-attention==0.5.2 pytest==8.4.2 \
  pillow==11.3.0 datasets==4.8.4
```

Run from the repository root. Create a source manifest for the exact
tracked checkout. This includes transitive imports required by the
runner. The run verifies every file before it loads the model.

```bash
.venv-recent/bin/python - <<'PY'
import hashlib, json, pathlib, subprocess
files = [
    'tools/kv/recent/__init__.py', 'tools/kv/recent/cache.py',
    'tools/kv/recent/prepare.py', 'tools/kv/recent/run.py',
    'tools/kv/native/gsm8k.py', 'tools/kv/prequantized/fixtures.py',
    'tools/kv/prequantized/contract.py',
    'tests/test_recent_reference_cache.py',
    'tools/kv/fp8_failure/statistics.py',
]
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
manifest = {'commit': commit, 'files': {}}
for name in files:
    content = pathlib.Path(name).read_bytes()
    tracked = subprocess.check_output(['git', 'show', f'{commit}:{name}'])
    assert content == tracked, f'Uncommitted runtime source: {name}'
    manifest['files'][name] = hashlib.sha256(content).hexdigest()
pathlib.Path('SOURCE.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv-recent/bin/python -m tools.kv.recent.run --help
.venv-recent/bin/python -m pytest tests/test_recent_reference_cache.py -q
.venv-recent/bin/python -m tools.kv.recent.prepare \
  --model qwen38 --out recent-inputs
.venv-recent/bin/python -m tools.kv.recent.run \
  --inputs recent-inputs/inputs.json.gz --out recent-development \
  --stage development --batch 8
```

Use `--model gemma4` in preparation for the other checkpoint. Keep output
directories separate; the runner refuses to overwrite a stage. It can
use `--model-path` for a previously downloaded, hash-verified checkpoint.

Before the quality stage, require `NATIVE-PARITY.json` to report `pass`,
`DONE.json` to exist, and at least eight of the 16 native development
answers to be correct. Parity compares exact generated token IDs on four
training prompts for 64 tokens, using explicit official native cache
layers and the reference native cache. Tiny model checks also compare
native logits and exercise quantized cached updates. Candidate errors
remain evidence; they do not authorize changing inputs or rules.

```bash
.venv-recent/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path('recent-development')
assert (p / 'DONE.json').is_file()
assert json.loads((p / 'NATIVE-PARITY.json').read_text())['pass']
r = json.loads((p / 'native-development.json').read_text())['outputs']
assert len(r) == 16 and sum(x['correct'] for x in r) >= 8
PY
.venv-recent/bin/python -m tools.kv.recent.run \
  --inputs recent-inputs/inputs.json.gz --out recent-quality \
  --stage quality --batch 8
```

The quality stage runs every arm. Qwen uses native, K16/V8, K8/V8 order;
Gemma reverses the two quantized arms. Preserve all generated token IDs,
text, truncations, errors and incomplete panels. Do not replace a failed
panel or infer a complete-study result from a completed subset.

## Frozen quality rules

Per model and arm, the quality panel contains:

- All 1,319 GSM8K test questions, with natural question lengths and up to
  2,048 output tokens. The fixed final-answer parser takes the last
  numeric answer introduced by the requested answer label. Formatting
  failures count as failures.
- 100 retrieval prompts at each of 1,024 and 8,192 input tokens, with up
  to 128 output tokens.
- 32 Wikipedia documents at each context length. After prefill, the model
  consumes 128 teacher-forced continuation tokens and predicts the next
  128 tokens. The first scored target is position `context + 1` using
  zero-based token positions.

Generation is greedy and ends naturally at an official end token.
GSM8K and short retrieval use batch eight; long retrieval uses batch two;
prediction loss uses batch one. Every arm uses identical frozen inputs.

The family has two models and two quantized candidates. Each candidate
uses one-sided alpha 0.0125, giving a Bonferroni family limit of 0.05.
Every endpoint must pass:

- Native accuracy is at least 50% on GSM8K and on each retrieval panel.
- The one-sided Clopper-Pearson upper bound on the fraction of all items
  that change from native-correct to candidate-wrong is at most 2% on
  GSM8K and 5% on each retrieval panel. Gains are reported separately.
- At each context, the upper bound on relative mean prediction-loss
  change is at most 1%. The paired document bootstrap uses 100,000
  resamples, seed 20261019, a linear quantile and the native mean in each
  resample's denominator. This is a relative negative-log-likelihood
  margin, not a 1% perplexity margin.

A candidate passing and another failing these preservation rules does
not establish a difference between candidates. Their direct GSM8K
comparison uses an exact two-sided McNemar test, with Holm correction
across the two models. Report its direction and net accuracy difference.
The two quantized arms have different memory budgets. This comparison
does not establish that retaining keys is better than retaining values
at an equal budget; it does not include a K8/V16 control.

## Memory accounting

Fixed-shape memory is measured after the prefill and 128-token
continuation: 1,152 or 8,320 consumed tokens. It includes FP32 scales and
unchanged recurrent state. Sliding layers retain only the final
`window - 1` entries, while global attention retains the full context.
Both logical tensor bytes and unique allocated backing bytes are saved.

For equal-sized attention key/value vectors of dimension D, the payload
plus scales is `4D` bytes for native, `3D + 4` for K16/V8 and `2D + 8`
for K8/V8. Recurrent state and model weights reduce the relative saving
in total inference memory. The familiar 25% K16/V8 payload reduction
excludes scales and is not a 25% whole-GPU-memory claim.

Generation also records Torch allocated, reserved and peak allocated
bytes. Those observations depend on the completed output lengths and
allocator state. They are diagnostic, not a matched serving-capacity
comparison. This reference does not qualify beam search, speculative
decoding, prefix-cache sharing or a production fused attention kernel.
