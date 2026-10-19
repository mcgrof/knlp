# Install the asymmetric Qwen3 recipe

This entry point selects the mixed fresh-prefill route measured at 8K/batch 32.
It keeps model-weight precision separate from K16/V8 cache storage. The older
[October replication](decode-october-2026.md) and `make defconfig-decode`
retain their historical sources. Do not use them as the current quick start.

**Validation status:** the current installer is awaiting a cold H100 check.
The existing quality and performance results qualify the pinned serving
sources; they do not yet certify this new packaging path.

Use Linux x86-64, one H100 SXM 80 GB (SM90), at least 100 GB system RAM,
100 GB free disk, a CUDA 12.9 toolkit with `nvcc`, a compatible NVIDIA driver,
Git, a C/C++ compiler, and `uv` 0.8.17. Python 3.12.11 is installed by `uv`.
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
uv run --python 3.12.11 --no-project -m tools.kv.native.october_setup \
  "$PWD/stack" --recipe current
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
quality result. First-use compilation and validation can take several minutes.
A failed check retains its output; use a fresh output directory after correcting
the cause. Unsupported inherited settings fail before model loading.

## Exact recipe

| Component | Pin |
| --- | --- |
| vLLM | `276305b29e0d11f95ada503ba76323ab24992e59` |
| FlashInfer | `800a550f8b736e49f8cfa136b82bfaf6d9d32711` |
| Torch | `2.11.0+cu129` |
| Transformers | `5.5.3` |
| Qwen3-8B-AWQ | `4da05a8edb55c6046cce958586c33b61da07bb79` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |

The installer checks the upstream vLLM wheel SHA256 and compiled-source
compatibility before installing editable Python. It removes prebuilt
FlashInfer caches, keeps the [package constraints](https://github.com/mcgrof/knlp/blob/main/tools/kv/native/october-constraints.txt),
and saves the resolved lock. The smoke records actually loaded libraries and
retains their bytes. An installed package's version string alone is insufficient.

The complete engine configurations are in the
[AWQ protocol](https://github.com/mcgrof/knlp/blob/main/tools/kv/native/fixtures/current/awq/measured-protocol.json)
and [dense protocol](https://github.com/mcgrof/knlp/blob/main/tools/kv/native/fixtures/current/fp16/measured-protocol.json).
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

The quick start uses the [actual frozen inputs](https://github.com/mcgrof/knlp/tree/main/tools/kv/native/fixtures/current)
and effective scales, so it needs no private repository, local cache or manual
source edit. The bundled manifests disclose export hashes and original hashes.
It does not produce the published performance table. Full replay requires the
original measured harness pins, controls, quality gates and fresh counterbalanced
blocks described in the [reproduction reference](decode-october-2026.md). The quick start records
its new source identity separately. Generating new prompts or calibrating new
scales creates a new replication that needs its own qualification.
