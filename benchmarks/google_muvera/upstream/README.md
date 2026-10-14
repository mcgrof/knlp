# Official Google FDE source

The files under `google/graph-mining` are unmodified copies from Google's
[graph-mining repository](https://github.com/google/graph-mining/tree/621ddb4e4c3b3ba3dc7d15ad34f1e0bb39ef25d5/sketching/point_cloud),
commit `621ddb4e4c3b3ba3dc7d15ad34f1e0bb39ef25d5`. The source uses the Apache
License 2.0; its complete license is included in `google/graph-mining/LICENSE`.
`source-lock.json` records SHA256 hashes for these files. Do not format or
edit the copied source. The build verifies the hashes before compiling.

`google_fde_cli.cc` and `CMakeLists.txt` are knlp adapters under the MIT
license. The CLI supplies binary input and output. It calls the official
generator without replacing its algorithm. The build is scoped to that
generator and its dependencies. It does not claim to be Google's original
retrieval system or its full benchmark.

`dependencies-lock.json` pins the source archives by URL and SHA256. Abseil
20250512.0, Protobuf 30.1, and the Eigen archive match the dependencies in
Google's later Bzlmod build. Their source trees and license files are
downloaded to the build cache, not copied into the knlp repository. The
build requires Python, CMake 3.24 or later, a C++20 compiler, and Make or
Ninja. It builds Protobuf's compiler locally, so a system `protoc` is not
needed. It does not install system packages.

From the repository root:

```sh
python benchmarks/google_muvera/scripts/build_google_fde.py
python -m unittest discover -s benchmarks/google_muvera/tests -p test_google_fde.py -v
```

The default output is `.cache/google-muvera/google_fde`. Set `--build-dir`
to choose another cache. Set `GOOGLE_FDE_BINARY` for the adapter or tests
when using a different path. `provenance.json` records source and dependency
pins, the binary hash, compiler version, and adapter hashes. A first build
needs network access and compiles the dependencies; later builds use the
cache. No GPU is needed for FDE construction or the synthetic fixtures.

The query and document generators share a seed and configuration. Queries
sum token vectors; documents average them. The adapter enables document
empty-bucket filling by default. Optional inner and final projections call
the corresponding upstream code. Returned vectors have no global L2
normalization. Compare their raw inner products for ranking; divide by the
repetition count when comparing unprojected scores to Chamfer similarity.

The CLI protocol stores little-endian float32 values and uint32 counts.
Its input is the eight-byte magic `GFDEv1\0\0`, cloud count, embedding
dimension, then each cloud's token count and row-major coordinates. Its
output is `GFDEo1\0\0`, cloud count, output dimension, then row-major FDE
coordinates. Python handles this framing. Empty or nonfinite point clouds
are rejected before calling upstream code. The protocol is an internal
batch transport, not a durable retrieval-index format.
