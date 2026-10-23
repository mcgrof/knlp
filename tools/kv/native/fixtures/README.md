# External experiment inputs

Captured prompts, calibrated scales, measured protocols and arrival catalogs
are stored outside the source repository. This directory contains format,
licensing and reproduction documentation only.

For exact historical replay, supply the archived input bundle and set:

```sh
export KNLP_KV_FIXTURES=/path/to/experiment-inputs
```

The external directory contains `current/awq`, `current/fp16`, `phi4` and
`serving` subdirectories as needed by the chosen recipe. Keep `manifest.json`,
measured protocols and compressed payloads together. The reproduction tools
verify recorded archive and decompressed hashes; they do not synthesize
replacement calibration data or silently download captured measurements.

`tools.kv.native.current` and the HTTP launchers require this external root.
Other runners already take explicit `--inputs`, `--manifest` or `--catalog`
arguments. Source configurations and model-weight download commands remain
public. Exact measured replay requires the separately supplied evidence.

For optional CPU integration tests against retained captures:

```sh
export KNLP_CAPTURE_ROOT=/path/to/archived-public-artifacts
python -m pytest tests/test_serving_trace_bundle.py tests/kmesh_forks/test_evidence.py
```

`KNLP_CAPTURE_ROOT` contains the archived `docs/data` paths. Normal unit tests
use small synthetic inputs and do not require experiment captures.
