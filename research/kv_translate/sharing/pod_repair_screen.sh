#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# Prepare a rented accelerator and run the bounded sparse-replay comparison.
# The driver is resumable; completed contexts are skipped on a later call.
#
# usage: pod_repair_screen.sh <code-root> <config.json> <timeline.jsonl>

set -euo pipefail

ROOT=$1
CONFIG=$2
TIMELINE=$3

VENV=${VENV:-/workspace/venv}
# Model weights are large sequential files and belong on the pod's local
# container disk. A provider volume may be network storage and is slower.
export HF_HOME=${HF_HOME:-/workspace/hf}
export TOKENIZERS_PARALLELISM=false

if [ -z "${OMP_NUM_THREADS:-}" ] && [ -r /sys/fs/cgroup/cpu.max ]; then
	read -r quota period </sys/fs/cgroup/cpu.max
	if [ "$quota" != "max" ]; then
		OMP_NUM_THREADS=$((quota / period))
		[ "$OMP_NUM_THREADS" -lt 1 ] && OMP_NUM_THREADS=1
		[ "$OMP_NUM_THREADS" -gt 16 ] && OMP_NUM_THREADS=16
		export OMP_NUM_THREADS
		export MKL_NUM_THREADS=$OMP_NUM_THREADS
	fi
fi

mkdir -p "$HF_HOME" "$(dirname "$TIMELINE")"

mark() {
	printf '{"phase": "%s", "seconds": %s, "utc": "%s"}\n' \
		"$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$TIMELINE"
}

timed() {
	local phase=$1
	shift
	local t0
	t0=$(date +%s.%N)
	"$@"
	mark "$phase" "$(echo "$(date +%s.%N) - $t0" | bc -l | xargs printf '%.1f')"
}

install_packages() {
	if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c \
		'import torch, transformers, safetensors; assert torch.cuda.is_available(); assert transformers.__version__ == "4.55.0"' \
		2>/dev/null; then
		return
	fi
	python3 -m pip --version >/dev/null 2>&1 ||
		curl -sS https://bootstrap.pypa.io/get-pip.py | python3
	python3 -m pip install -q uv
	python3 -m uv venv --python 3.11 "$VENV"
	python3 -m uv pip install -q --python "$VENV/bin/python" \
		torch==2.14.0 transformers==4.55.0 accelerate safetensors numpy \
		huggingface_hub
	if ! "$VENV/bin/python" -c \
		'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
		python3 -m uv pip install -q --python "$VENV/bin/python" \
			--index-url https://download.pytorch.org/whl/cu126 \
			--reinstall torch==2.14.0
	fi
	"$VENV/bin/python" - <<-'PY'
		import torch, transformers
		assert torch.cuda.is_available(), "the card is not visible to torch"
		assert transformers.__version__ == "4.55.0"
		x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
		assert torch.isfinite(x @ x).all()
		print("packages ok", torch.__version__, torch.version.cuda,
		      transformers.__version__)
	PY
}

fetch_weights() {
	"$VENV/bin/python" - "$CONFIG" <<-'PY'
		import json, sys
		from huggingface_hub import snapshot_download
		cfg = json.load(open(sys.argv[1]))
		for which in ("source", "target"):
		    snapshot_download(cfg[which], revision=cfg[f"{which}_revision"])
	PY
}

run_workload() {
	"$VENV/bin/python" -m research.kv_translate.sharing.run_repair_screen \
		--config "$CONFIG"
}

cd "$ROOT"
command -v bc >/dev/null || (apt-get update -qq && apt-get install -y -qq bc)
mark start 0
timed install_packages install_packages
timed fetch_weights fetch_weights
if [ -n "${PREPARE_ONLY:-}" ]; then
	echo "POD_PREPARED"
	exit 0
fi
timed run_workload run_workload
echo "REPAIR_SCREEN_POD_COMPLETE"
