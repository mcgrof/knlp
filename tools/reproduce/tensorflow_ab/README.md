# TensorFlow build-system reproduction

This harness builds one pinned TensorFlow target with its reference builder,
an externally supplied alternative builder, or both. The public repository
contains no alternative implementation. Both adapters receive the same
canonical request hash and write into isolated output roots.

## Select a mode

Reference builder only:

```bash
make defconfig-tensorflow-gemma2b-default
make TF_SOURCE=/data/tensorflow
```

Alternative builder only:

```bash
make defconfig-tensorflow-gemma2b-alternative
make TF_SOURCE=/data/tensorflow \
  TF_ALT_BUILDER=/private/path/tensorflow-builder
```

A/B:

```bash
make defconfig-tensorflow-gemma2b-ab
make TF_SOURCE=/data/tensorflow \
  TF_ALT_BUILDER=/private/path/tensorflow-builder
```

Useful overrides are `TF_RESULTS`, `TF_TARGET`, `TF_BUILD_ARGS`, `TF_JOBS`,
`TF_BAZEL`, and `TF_ALLOW_DIRTY=1`.

Plain `make` runs:

1. `tensorflow-doctor`
2. `tensorflow-stage`
3. `tensorflow-build`
4. `tensorflow-report`

The stage step installs the managed `//knlp_tensorflow` package into the
TensorFlow checkout. `make tensorflow-clean` removes only that managed package;
it never removes build results.

## Alternative builder contract

An adapter is invoked as:

```text
adapter --request REQUEST.json --result RESULT.json
```

The request contains:

- `common_request_sha256`
- pinned source directory, Git commit, status, and `.tf_configure.bazelrc` hash
- staged workload hashes
- target, artifact-relative path, jobs, common arguments, and profile policy
- a variant-specific output directory

The adapter must always write a result, including on failure:

```json
{
  "schema_version": 1,
  "variant": "candidate",
  "common_request_sha256": "...",
  "status": 0,
  "engine": {"name": "private-name", "version": "..."},
  "wall_seconds": 123.4,
  "command": ["..."],
  "log": "/path/build.log",
  "profile": "/path/build.profile",
  "artifacts": [
    {
      "path": "/path/gemma_savedmodel_runner",
      "sha256": "...",
      "size_bytes": 123
    }
  ]
}
```

The implementation may add fields, but it must preserve the request hash and
the fields above.

## Prepare the Gemma workload

Create an isolated model-export environment on the GPU host. KerasHub's
Hugging Face loader imports `sentencepiece` and `safetensors` as optional
dependencies, so install them explicitly:

```bash
python3 -m venv .venv-tensorflow-model
. .venv-tensorflow-model/bin/activate
pip install "tensorflow[and-cuda]==2.20.0" keras-hub huggingface-hub \
  sentencepiece safetensors
```

With model access configured:

```bash
make tensorflow-export TF_EXPORT_ARGS='
  --preset gemma_2b_en
  --model-output /workspace/gemma_savedmodel
  --feeds-output /workspace/gemma_feeds'
```

Generate an independent TensorFlow reference:

```bash
make tensorflow-reference TF_REFERENCE_ARGS='
  --model /workspace/gemma_savedmodel
  --feeds /workspace/gemma_feeds
  --output-dir /workspace/results/reference'
```

Run a balanced four-trial comparison:

```bash
make tensorflow-runtime TF_RUNTIME_ARGS='
  --binary reference=/workspace/bin/reference_runner
  --binary candidate=/workspace/bin/candidate_runner
  --model /workspace/gemma_savedmodel
  --feeds /workspace/gemma_feeds
  --results /workspace/results/runtime'
```

The runtime driver alternates process order, rotates cases, records every
status and log, reports median and MAD latency, computes the paired geometric
mean, and performs a chunked full-logit correctness comparison.

Cloud provisioning and teardown intentionally remain outside this harness.
Run it only after an authorized GPU host exists, then collect the results
before terminating that host.
