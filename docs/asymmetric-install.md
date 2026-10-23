# Install the asymmetric Qwen3 recipe

This entry point selects the mixed fresh-prefill route measured at 8K/batch 32.
It keeps model-weight precision separate from K16/V8 cache storage. The older
[October replication](https://github.com/mcgrof/knlp/blob/main/docs/decode-october-2026.md) and `make defconfig-decode`
retain their historical sources. Do not use them as the current quick start.

**Validation status:** a fresh H100 installation passed on October 19, 2026.
AWQ and dense FP16 ran sequentially through the commands below at installer
`5a3d9d74edc87ce763e741859dfc32bd01d452d4`. Both passed 360 numerical routing
checks, physical storage and calibrated-scale checks, changing decode graphs,
clean memory caps, actual weight dispatch and finite deterministic generation.
The existing statistical quality and throughput results remain separate.

Use Linux x86-64, one H100 SXM 80 GB (SM90), at least 100 GB system RAM,
100 GB free disk, a CUDA 12.9 toolkit with `nvcc`, a compatible NVIDIA driver,
Git, a C/C++ compiler, and `uv` 0.8.17. Python 3.12.11 is installed by `uv`.
The checked platform is Ubuntu 22.04.5 with NVIDIA driver 580.126.09.
Its CUDA development image is pinned below.
For an Ubuntu CUDA development image, install the host tools first:

```sh
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates
curl -LsSf https://astral.sh/uv/0.8.17/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8 NVCC_THREADS=2 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22

git clone https://github.com/mcgrof/knlp.git
cd knlp
git switch main
export FLASHINFER_WORKSPACE_BASE="$PWD/stack/jit"
uv run --python 3.12.11 --no-project -m tools.kv.native.october_setup \
  "$PWD/stack" --recipe current
export KNLP_KV_FIXTURES=/path/to/experiment-inputs
stack/env/bin/python -m tools.kv.native.current download \
  --format awq --output awq-download.json
stack/env/bin/python -m tools.kv.native.current smoke \
  --install "$PWD/stack" --format awq --output awq-smoke
```

Check the cgroup CPU quota before selecting 22 threads. A smaller allocation
can run the check with smaller thread pools, but is a different replication
of performance. The Qwen checkpoints used here are public; no token is
required. For authenticated downloads, use `hf auth login` from the installed
environment. Never put a token in a command or tracked file.

After the AWQ process exits, validate dense FP16 through the same interface:

```sh
stack/env/bin/python -m tools.kv.native.current download \
  --format fp16 --output fp16-download.json
stack/env/bin/python -m tools.kv.native.current smoke \
  --install "$PWD/stack" --format fp16 --output fp16-smoke
```

Each invocation builds one engine and exits. The smoke checks physical FP16 K
and E4M3 FP8 V storage, actual packed or dense weights, calibrated scales
before capture, all 36 layers of mixed fresh-prefill routing, output placement,
changing decode graph inputs and a clean memory window. It also generates a
short answer twice, requiring finite logits and identical output. The output
includes `receipt.json`, full source and binary identities, and the line:

```text
PASS: physical K16/V8; calibrated scales; mixed fresh-prefill route; changing decode graphs; clean memory; finite deterministic generation
```

This is an installation and route check, not a new throughput or statistical
quality result. On the checked host, setup took 3.5 minutes and the two complete checks
took 18.2 and 12.0 minutes; see the stage breakdown below.
A failed check retains its output; use a fresh output directory after correcting
the cause. Unsupported inherited settings fail before model loading.

## Verified installation

The installation began with empty package, model and FlashInfer caches.
All sources and inputs came from public repositories. Dense reused the
same installation and compiled kernels after the AWQ process exited;
this was one cold installation, with two sequential model selections.
Both short generations produced `2 + 2 equals 4.` twice, with identical
token IDs and finite logits. No manual source edit was required.

Setup took 209.9 seconds, including the first Python bootstrap. Model
commands below include process startup and the public checkpoint download.

| Format | Model-download command | Geometry and first-use JIT | Serving validation |
| --- | ---: | ---: | ---: |
| AWQ | 10.6 s | 143.5 s | 950.4 s |
| Dense FP16 | 16.2 s | 11.7 s | 710.7 s |

The four recorded FlashInfer Ninja build/link spans total 293.6 seconds.
They occur inside the check commands, so do not add them to the table.
Serving validation includes engine setup, remaining first-use compilation,
numerical instrumentation and profiling. These elapsed times describe this
installation; they are not throughput benchmarks or portable runtime promises.

The compact install receipt records
source and binary hashes, package-lock identity, storage, workspace buffers,
clean memory, both generations and stage timings. The full performance and
quality evidence is in the [methods](asymmetric-methods.md).

## Exact recipe

| Component | Pin |
| --- | --- |
| vLLM | `276305b29e0d11f95ada503ba76323ab24992e59` |
| FlashInfer | `800a550f8b736e49f8cfa136b82bfaf6d9d32711` |
| Installer / smoke harness | `5a3d9d74edc87ce763e741859dfc32bd01d452d4` |
| Validation image | `nvcr.io/nvidia/cuda@sha256:38804006c937a83f28f63a959abcee688042072319c8614ad57b350958a30bd3` |
| Torch | `2.11.0+cu129` |
| Transformers | `5.5.3` |
| Qwen3-8B-AWQ | `4da05a8edb55c6046cce958586c33b61da07bb79` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |

The installer checks the upstream vLLM wheel SHA256 and compiled-source
compatibility before installing editable Python. It uninstalls prebuilt
FlashInfer cache packages, keeps the [package constraints](https://github.com/mcgrof/knlp/blob/main/tools/kv/native/october-constraints.txt),
and saves the resolved lock. Use a fresh installation directory; its dedicated
JIT cache avoids reusing kernels compiled from another checkout. The smoke
records actually loaded libraries and
retains their bytes. An installed package's version string alone is insufficient.

The complete engine configurations are in the
AWQ protocol
and dense protocol.
Both use FP16 compute, TP1, NHD layout, page size 16, Q32/KV8/D128,
96 maximum sequences, 65,536 scheduled tokens, a 10,240-token model limit,
and disabled prefix caching and chunked prefill. Compilation mode is 0 with
`FULL_DECODE_ONLY` graphs at sizes 1/8/32/44/45/59/60/96. AWQ has a 48 GiB
KV pool; dense has 40 GiB. The original 68 GiB Torch and 72 GiB driver caps
apply to the clean observation. This does not measure maximum resident batch.

The entry point sets and checks `VLLM_ASYM_SINGLE_WRITER=1`,
`FLASHINFER_FUSE_V_SCALE=1`, `FLASHINFER_ASYM_FP8_PV=0`, and
`VLLM_COMPACT_PREFILL=1`, plus the required runner/layout settings. It calls the
existing calibrated-scale loader before warmup and graph capture. It rejects
placeholder scales and verifies every captured layer. Fresh whole prompts of
at most 8,192 tokens and 4,096 referenced pages use bounded compact staging;
serving retains its guarded fallback for unsupported calls. This smoke requires
the eligible mixed calls to take the optimized route.

## Try versus reproduce

The quick start requires separately supplied frozen inputs and effective
scales. Set `KNLP_KV_FIXTURES` to their external directory before running it.
The [input documentation](https://github.com/mcgrof/knlp/tree/main/tools/kv/native/fixtures)
describes the layout. Archived manifests bind both export and original hashes.
It does not produce the published performance table. Full replay requires the
original measured harness pins, controls, quality gates and fresh counterbalanced
blocks described in the [methods and reproduction reference](asymmetric-methods.md). The quick start records
its new source identity separately. Generating new prompts or calibrating new
scales creates a new replication that needs its own qualification.
