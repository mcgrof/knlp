# Modular build-system A/B reproduction

This workflow builds pinned public Modular targets through a reference
builder, an externally supplied alternative builder, or both.
Build-time is always per target; runtime currently supports exactly one target
per campaign for apples-to-apples execution comparisons.

The default pinned workload is:

```text
repository: https://github.com/modular/modular.git
commit: cc7155d854a049dce99aefc8d2570311e1e0b15f
target: //mojo/examples/gpu-block-and-warp:tiled_matmul
artifact: mojo/examples/gpu-block-and-warp/tiled_matmul
```

## Prepare the source

```bash
git clone https://github.com/modular/modular.git /data/modular
git -C /data/modular checkout cc7155d854a049dce99aefc8d2570311e1e0b15f
```

The source commit and clean-worktree requirement are checked before a build or
runtime campaign. Use `MODULAR_SOURCE=/path/to/modular` to place the checkout
elsewhere. Use `MODULAR_ALLOW_DIRTY=1` only for a deliberately recorded source
change.

## Reference build

```bash
make defconfig-modular-tiled-matmul-default
make modular-repro
```

To build multiple targets, define `MODULAR_TARGETS` and (optionally)
`MODULAR_ARTIFACT_RELPATHS`:

```bash
make defconfig-modular-tiled-matmul-default
MODULAR_TARGETS='//max/examples/capi:example //max/examples/capi:graph_capture' \
  MODULAR_ARTIFACT_RELPATHS='max/examples/capi/example max/examples/capi/graph_capture' \
  make modular-build
```

The reference adapter invokes the checkout's `./bazelw`, gives the run a fresh
Bazel output base, records the command and artifact hash, and emits a compressed
Bazel profile.

## A/B build and GPU runtime campaign

The alternative implementation is intentionally outside this public tree.
Point the harness at an adapter implementing the contract below:

```bash
make defconfig-modular-tiled-matmul-ab
MODULAR_ALT_BUILDER=/absolute/path/to/adapter make modular-campaign
```

To avoid long shell-quoted target lists, pass a targets file:

```bash
MODULAR_TARGETS_FILE=tools/reproduce/modular_ab/targets/curated-non-test-strict-190.txt \
  make modular-build
```

Each line in the file should be a Bazel target label.

`modular-campaign` performs one clean build through each adapter, one warm-up
per artifact, and ten measured runs per artifact. Measured pairs alternate
reference-first and alternative-first order. The build stage fails on first
build error; it does not abort remaining targets for that variant. The campaign
fails if runtime checks fail: an executable returns nonzero, omits `Validation
PASSED`, or produces stdout that is not byte-identical to the other runs.
executable returns nonzero, omits `Validation PASSED`, or produces stdout that
is not byte-identical to the other runs.

Builds and runtime can also be split across hosts:

```bash
MODULAR_ALT_BUILDER=/absolute/path/to/adapter make modular-repro
make modular-runtime MODULAR_RUNTIME_OPTIONS='\
  --reference-artifact /path/on/gpu/reference/tiled_matmul \
  --alternative-artifact /path/on/gpu/alternative/tiled_matmul'
make modular-report
```

Copy each executable and any libraries it requires to the GPU host before the
runtime step. Without overrides, runtime uses the artifact paths recorded by
the local build results.

## Alternative builder contract

The orchestrator invokes an adapter as:

```text
adapter --request REQUEST.json --result RESULT.json
```

The request contains a variant name, a canonical common-request hash, an output
directory, and the common request. The common request records the exact source
commit, target, artifact relative path, build arguments, job count, profile
requirement, runtime settings, and relevant compiler environment.

The adapter must perform a clean build and write a result with these fields:

```json
{
  "schema_version": 1,
  "variant": "alternative",
  "common_request_sha256": "hash copied from the request",
  "status": "succeeded",
  "returncode": 0,
  "wall_seconds": 12.34,
  "command": ["builder", "build", "target"],
  "build_log": "/absolute/path/to/build.log",
  "profile": "/absolute/path/to/profile",
  "artifacts": [
    {
      "path": "/absolute/path/to/tiled_matmul",
      "sha256": "artifact sha256",
      "size_bytes": 1234
    }
  ]
}
```

The result also includes a `build` object that copies the selected target and its
artifact-relative path for traceability:

```json
"build": {
  "target": "//mojo/examples/gpu-block-and-warp:tiled_matmul",
  "artifact_relpath": "mojo/examples/gpu-block-and-warp/tiled_matmul"
}
```

The adapter may add fields, but it must preserve the common request hash and
return exactly one executable as the first artifact.

## Results

The default results directory is `results/modular-tiled-matmul` and contains:

```text
doctor.json
common-request.json
build/reference/{request.json,result.json,adapter.log,build.log}
build/alternative/{request.json,result.json,adapter.log,...}
  # multi-target example layout:
  build/reference/001-//max-examples-capi-example/{request.json,result.json,...}
  build/reference/002-//max-examples-capi-graph-capture/{request.json,result.json,...}
runtime-summary.json
runtime/*.stdout
runtime/*.stderr
summary.json
```

Open the Bazel `build.profile.gz` in Perfetto. If a browser cannot decompress it
directly, use `gzip -dc build.profile.gz > build.profile.json` first.

Remove only this workflow's marked result directory with:

```bash
make modular-clean
```
