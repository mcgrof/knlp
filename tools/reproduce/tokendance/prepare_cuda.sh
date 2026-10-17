#!/bin/bash
# SPDX-License-Identifier: MIT
# Prepare the committed CUDA profile inside its pinned container image.
set -euo pipefail

if [[ $# != 1 || $1 != /* || -e $1 ]]; then
    echo "usage: $0 /absolute/path/to/a/new/runtime-directory" >&2
    exit 2
fi
for tool in uv git nvcc c++ python3; do
    command -v "$tool" >/dev/null
done
input_dir=$(cd -- "$(dirname -- "$0")" && pwd)
runtime_dir=$1
mkdir -p -- "$runtime_dir"
uv venv --python 3.13.15 "$runtime_dir/venv"
uv pip install --python "$runtime_dir/venv/bin/python" \
    -r "$input_dir/requirements-cuda.lock" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match

python3 - "$input_dir" "$runtime_dir" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

inputs, runtime = map(Path, sys.argv[1:])
profile = json.loads((inputs / "profile.json").read_text())
fork = json.loads((inputs.parent / "kmesh_forks/profile-current.json").read_text())
for name, repo in (("vllm", "vllm"), ("lmcache", "LMCache")):
    checkout = runtime / name
    subprocess.run(["git", "clone", "--no-checkout", "--depth=1", "--single-branch",
                    "--branch", "2026-10-16-feature-kmesh-current-validation",
                    "https://github.com/mcgrof/" + repo + ".git", str(checkout)], check=True)
    commit = profile["source_pairs"]["current"][name]["commit"]
    subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", commit], check=True)
    tree = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True).strip()
    if tree != profile["source_pairs"]["current"][name]["tree"]:
        raise RuntimeError("source tree differs: " + name)
subprocess.run(["git", "-C", str(runtime / "vllm"), "fetch", "--depth=1", "origin",
                fork["vllm"]["commit"]], check=True)
subprocess.run(["git", "-C", str(runtime / "vllm"), "worktree", "add", "--detach",
                str(runtime / "vllm-serving"), fork["vllm"]["commit"]], check=True)
PY

runtime_python="$runtime_dir/venv/bin/python"
export PATH="$runtime_dir/venv/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CPATH="$runtime_dir/venv/lib/python3.13/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
(
    cd "$runtime_dir/lmcache"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
        MAX_JOBS="${MAX_JOBS:-4}" NVCC_THREADS="${NVCC_THREADS:-1}" \
        SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
        "$runtime_python" setup.py build_ext --inplace
)
"$runtime_python" - "$runtime_dir" <<'PY'
from pathlib import Path
import sys
import sysconfig
from huggingface_hub import snapshot_download

runtime = Path(sys.argv[1])
wheel = Path(sysconfig.get_paths()["purelib"]) / "vllm"
for name in ("vllm", "vllm-serving"):
    for binary in wheel.rglob("*.so"):
        destination = runtime / name / "vllm" / binary.relative_to(wheel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(binary)
revision = "7ae557604adf67be50417f59c2c2f167def9a775"
model = Path("/scratch/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots") / revision
if not model.exists():
    snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision=revision, local_dir=str(model))
PY

model_dir=/scratch/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
wheel_dir="$runtime_dir/venv/lib/python3.13/site-packages/vllm"
{
    printf 'export TOKENDANCE_PY=%q\n' "$runtime_python"
    printf 'export TOKENDANCE_VLLM_SOURCE=%q\n' "$runtime_dir/vllm"
    printf 'export TOKENDANCE_LMCACHE_SOURCE=%q\n' "$runtime_dir/lmcache"
    printf 'export TOKENDANCE_MODEL_DIR=%q\n' "$model_dir"
    printf 'export TOKENDANCE_WHEEL_PACKAGE=%q\n' "$wheel_dir"
    printf 'export KMESH_FORK_PY=%q\n' "$runtime_python"
    printf 'export KMESH_FORK_PYTHON=%q\n' "$runtime_python"
    printf 'export KMESH_FORK_VLLM_SOURCE=%q\n' "$runtime_dir/vllm-serving"
    printf 'export KMESH_FORK_WHEEL_PACKAGE=%q\n' "$wheel_dir"
    printf 'export KMESH_SERVING_VLLM_SOURCE=%q\n' "$runtime_dir/vllm-serving"
    printf 'export KMESH_SERVING_MODEL_DIR=%q\n' "$model_dir"
    printf 'export KMESH_SERVING_WHEEL_PACKAGE=%q\n' "$wheel_dir"
    python3 - "$input_dir/profile.json" <<'PY'
import json, shlex, sys
print("export KMESH_IMAGE_DIGEST=" + shlex.quote(json.load(open(sys.argv[1]))["current_cuda"]["image"]))
PY
} > "$runtime_dir/run.env"
echo "Prepared runtime. Source $runtime_dir/run.env before make."
