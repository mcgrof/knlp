# Frozen Phi-4 transfer inputs

These are the exact numerical inputs for the Phi-4 model-transfer study.
The checkpoint and tokenizer are `microsoft/Phi-4` revision
`2db69c1c3e91a05d2c64a3185acfbaf36f744e25`. The geometry is 40 attention
layers, 40 query heads, ten KV heads and head dimension 128. Weights and
compute use FP16. The K16/V8 candidate uses model-specific frozen scales.

Keep `qwen.json.gz` compressed; its historical filename does not identify
the model. Decompress the other three `.json.gz` files once. Each entry in
`manifest.json` binds the downloaded bytes and the restored input bytes.
All four restored files are byte-identical to the measured inputs. No
scale provenance was removed and no new calibration was performed.

The panel contains 16 original 1K calibration documents and 32 primary
NLL documents at each of 1K and 8K. The 8K extension excludes the original
calibration articles. Each context has 100 primary synthetic retrieval
records; reserve and development records are explicitly marked. The
GSM8K adaptation contains all 1,319 test questions. Its instruction,
strict extraction rule and natural-completion token cap remain frozen.
These task definitions are not an unmodified leaderboard evaluation.

The six `measured-*-protocol.json` files preserve the actual source pins,
input hashes, engine settings and memory envelopes for the three arms at
each context length. The 1K batch is 32; the 8K batch is 16 on H100. All
use a 32 GiB persistent KV pool and the stated 74 GiB allocator and 78 GiB
driver envelopes. At 8K, original-order repetition must match exactly;
rotation is retained as a separate diagnostic. These are measurement
records, not a replacement for the Qwen-only `current` installer.

For retained complete-cohort captures and their frozen plan directory,
the independent CPU reader checks input and source hashes, qualification,
physical-memory envelopes, exact prompts and completions, scheduler
activity, arm order, and the full two-control statistical family:

```sh
python -m tools.kv.native.cohort_confirmation \
  --root CAPTURES --plan FROZEN_PLAN \
  --prefix phi4-8k-confirmation --output confirmation.json
```

The plan directory supplies `PLAN.json`, the exact recorder and wrapper,
and original job declarations. The capture directory supplies each arm's
raw outputs, telemetry, command and completion receipts. A missing
attempt remains pending; a failed attempt remains failed. The reader
does not replace blocks or compute primary inference on a reduced set.
Python assertions must remain enabled. These are in-process complete
requests; the reader does not infer HTTP or per-step GPU latency.

The manifest's historical `fixture_generation_source_sha` identifies the
public fixture helper. The protocol identifies the executed harness;
the complete file hashes identify the exact inputs. Do not substitute
one of these identities for another.

Prompt source: [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configuration
`20231101.en`, under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
and [GFDL 1.3](https://www.gnu.org/licenses/fdl-1.3.html), as recorded in the
[pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md).
The [article attribution](ATTRIBUTION.md) lists titles and IDs. The inputs
contain tokenized excerpts and synthetic retrieval records.
GSM8K source: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k),
revision `740312add88f781978c0658806c59bc2815b9866`, under the
[MIT license](LICENSE-GSM8K.txt). Model weights remain separate downloads
under the publisher's license.
