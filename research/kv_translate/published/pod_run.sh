#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# Run the paired comparison on a rented pod, keeping a timeline of what the
# card was billed for while it was not doing the work.
#
# A pod is billed from the moment its container starts, so installing
# packages and downloading weights are paid at the card's rate. Each of those
# phases is timed separately here, because they are exactly what differs
# between a pod and a platform that bills only while a function runs.
#
# usage: pod_run.sh <code-root> <work-dir> <config.json> [stop-after]
#
# <code-root> holds research/kv_translate/published. The run is resumable:
# start it again with the same arguments and finished stages are skipped.

set -euo pipefail

ROOT=$1
WORK=$2
CONFIG=$3
STOP=${4:-}

VENV=${VENV:-/workspace/venv}
# weights go on the attached volume when there is one; the container's own
# disk is too small for them and its mount point differs between providers
if [ -z "${HF_HOME:-}" ]; then
	if [ -d /runpod-volume ]; then
		HF_HOME=/runpod-volume/hf
	else
		HF_HOME=/workspace/hf
	fi
fi
export HF_HOME
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

# a pod shows every processor of its host and enforces a quota far below
# that; thread pools sized from the visible count thrash against the quota
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

mkdir -p "$WORK" "$HF_HOME"
TIMELINE=$WORK/POD_TIMELINE.jsonl

mark() {
	# phase, seconds, free-form note
	printf '{"phase": "%s", "seconds": %s, "utc": "%s", "note": "%s"}\n' \
		"$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${3:-}" >>"$TIMELINE"
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
		'import torch, lm_eval, transformers; assert torch.cuda.is_available()' \
		2>/dev/null; then
		return
	fi
	command -v bc >/dev/null || (apt-get update -qq && apt-get install -y -qq bc)
	python3 -m pip --version >/dev/null 2>&1 ||
		curl -sS https://bootstrap.pypa.io/get-pip.py | python3
	python3 -m pip install -q uv
	python3 -m uv venv --python 3.11 "$VENV"
	# the default wheel needs a recent driver; fall back to an older runtime
	# when the host's driver cannot run it
	python3 -m uv pip install -q --python "$VENV/bin/python" \
		torch==2.14.0 transformers==4.55.0 accelerate datasets==3.6.0 \
		lm_eval==0.4.9.1 safetensors numpy huggingface_hub
	if ! "$VENV/bin/python" -c \
		'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
		python3 -m uv pip install -q --python "$VENV/bin/python" \
			--index-url https://download.pytorch.org/whl/cu126 \
			--reinstall torch==2.14.0
	fi
	# a check of the capability, not of having reached this line
	"$VENV/bin/python" - <<-'EOF'
		import torch
		assert torch.cuda.is_available(), "the card is not visible to torch"
		x = torch.randn(512, 512, device="cuda", dtype=torch.float64)
		torch.linalg.cholesky(x @ x.T + torch.eye(512, device="cuda", dtype=torch.float64))
		import lm_eval, transformers  # noqa: F401
		print("packages ok", torch.__version__, torch.version.cuda)
	EOF
}

fetch_weights() {
	"$VENV/bin/python" - "$CONFIG" <<-'EOF'
		import json, sys
		from huggingface_hub import snapshot_download
		sys.path.insert(0, ".")
		from research.kv_translate.published.run_pair import default_config
		cfg = default_config()
		cfg.update(json.load(open(sys.argv[1])))
		for which in ("source", "target"):
		    snapshot_download(cfg[which], revision=cfg[f"{which}_revision"])
	EOF
}

run_stages() {
	local extra=()
	[ -n "$STOP" ] && extra=(--stop-after "$STOP")
	"$VENV/bin/python" -m research.kv_translate.published.run_pair \
		--work "$WORK" --config "$CONFIG" "${extra[@]}"
}

describe() {
	"$VENV/bin/python" - "$WORK" <<-'EOF'
		import json, os, subprocess, sys, torch
		props = torch.cuda.get_device_properties(0)
		out = {
		    "provider": os.environ.get("PROVIDER", "pod"),
		    "card": torch.cuda.get_device_name(0),
		    "card_memory_gib": round(props.total_memory / 2**30, 1),
		    "driver": subprocess.run(
		        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
		        capture_output=True, text=True).stdout.strip(),
		    "torch": torch.__version__,
		    "processors_visible": os.cpu_count(),
		    "timeline": [json.loads(l) for l in open(
		        os.path.join(sys.argv[1], "POD_TIMELINE.jsonl")) if l.strip()],
		}
		json.dump(out, open(os.path.join(sys.argv[1], "PLATFORM.json"), "w"),
		          indent=2, sort_keys=True)
	EOF
}

cd "$ROOT"
command -v bc >/dev/null || (apt-get update -qq && apt-get install -y -qq bc)
mark start 0 "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
timed install_packages install_packages
timed fetch_weights fetch_weights
timed run_stages run_stages
describe
echo "POD_RUN_COMPLETE"
