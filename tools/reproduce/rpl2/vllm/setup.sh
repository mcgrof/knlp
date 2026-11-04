#!/bin/bash
# Build LMCache with its CUDA extensions for the vLLM profile, without
# installing anything into the given Python environment.
#
#   setup.sh
#
# Settings from the environment (the Makefile sets them):
#   RPL2P_ROOT       task root; the tree goes to $RPL2P_ROOT/src/lmcache-cuda
#   PY               Python with vLLM and CUDA PyTorch
#   LMCACHE_REPO LMCACHE_REF LMCACHE_COMMIT
#   CUDA_HOME        CUDA toolkit for nvcc (default /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST  optional; default: the first GPU's architecture
#                         as nvidia-smi reports it
#
# LMCache dependencies missing from the environment are installed with
# --no-deps into $RPL2P_ROOT/vllm-deps, which run.sh puts on PYTHONPATH.
set -euo pipefail
R=${RPL2P_ROOT:?}
PY=${PY:?Python with vLLM}
SRC=$R/src/lmcache-cuda
DEPS=$R/vllm-deps
: "${LMCACHE_REPO:?}" "${LMCACHE_REF:?}" "${LMCACHE_COMMIT:?}"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
"$PY" -c 'import vllm, torch; assert torch.version.cuda, "PyTorch without CUDA"; print("vLLM", vllm.__version__, "torch", torch.__version__)'
mkdir -p "$R/src" "$DEPS"
if [ ! -d "$SRC/.git" ]; then
	git clone --quiet --branch "$LMCACHE_REF" "$LMCACHE_REPO" "$SRC"
fi
git -C "$SRC" fetch --quiet origin "$LMCACHE_REF" || true
git -C "$SRC" checkout --quiet --detach "$LMCACHE_COMMIT"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$LMCACHE_COMMIT" ] || { echo "LMCache is not at $LMCACHE_COMMIT" >&2; exit 1; }
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
	TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
fi
if ! ls "$SRC"/lmcache/cuda_ops*.so "$SRC"/lmcache/lmcache_fs*.so >/dev/null 2>&1; then
	(cd "$SRC" && env CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" \
		TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="${MAX_JOBS:-8}" \
		"$PY" setup.py build_ext --inplace >"$R/lmcache-cuda-build.log" 2>&1) ||
		{ tail -30 "$R/lmcache-cuda-build.log"; exit 1; }
fi
# Import names of LMCache's runtime dependencies that a vLLM environment
# often lacks, and the distribution that provides each. The MP server's
# CUDA cache context needs CuPy built for PyTorch's CUDA major version.
cuda_major=$("$PY" -c 'import torch; print(torch.version.cuda.split(".")[0])')
missing=()
for pair in sortedcontainers:sortedcontainers aiofile:aiofile caio:caio \
	aiofiles:aiofiles redis:redis fastrlock:fastrlock \
	cupy:cupy-cuda${cuda_major}x \
	opentelemetry.exporter.prometheus:'opentelemetry-exporter-prometheus>=0.50b0,<=0.61b0'; do
	mod=${pair%%:*}
	PYTHONPATH="$SRC:$DEPS" "$PY" -c "import $mod" 2>/dev/null || missing+=("${pair#*:}")
done
if [ ${#missing[@]} -gt 0 ]; then
	echo "installing into $DEPS: ${missing[*]}"
	uv=$(command -v uv || ls "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" 2>/dev/null | head -1 || true)
	if "$PY" -m pip --version >/dev/null 2>&1; then
		"$PY" -m pip install --quiet --no-deps --target "$DEPS" "${missing[@]}"
	elif [ -n "$uv" ]; then
		"$uv" pip install --quiet --python "$PY" --no-deps --target "$DEPS" "${missing[@]}"
	else
		echo "neither pip in $PY nor uv found; install these into $DEPS with --no-deps:" >&2
		printf '  %s\n' "${missing[@]}" >&2
		exit 1
	fi
fi
CUDA_VISIBLE_DEVICES= PYTHONPATH="$SRC:$DEPS" "$PY" -c '
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import FSNativeL2AdapterConfig
from lmcache.cli.main import main
import inspect, cupy, lmcache.lmcache_fs, lmcache.cuda_ops
params = inspect.signature(FSNativeL2AdapterConfig.__init__).parameters
assert "held_fd_capacity" in params, "this LMCache build has no held_fd_capacity"
print("LMCache MP server and fs_native with held_fd_capacity: ok")'
echo "$SRC"
