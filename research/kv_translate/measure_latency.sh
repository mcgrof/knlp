#!/bin/bash
# The measured run, as one auditable file rather than something typed at a
# prompt. The driver it sources stops at the first failing stage, and the
# qualification below is a genuine prerequisite of the timing: the runner
# refuses to time an operator whose receipt is absent, malformed, from
# another contract, or recording its own failure.
#
# Expects: W (workspace), O (output dir), T0 (unix seconds at which the
# allocation was requested, so the deadline is measured against occupied
# allocation time rather than against this process).
set -u
W="${W:-/home/ubuntu/ws/kvt}"
export O="${O:-$W/out}"
T0="${T0:-0}"
DEADLINE="${DEADLINE:-900}"
PROBE_SHA="${PROBE_SHA:-f0650f8f13841fbcfaa3e830cddc4d75f92a67d5f2d84d74fda311a11d8b97ff}"
EXPERIMENT="${EXPERIMENT:-kv-translate-latency-closure}"
ATTEMPT="${ATTEMPT:-attempt-1}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$W"
mkdir -p "$O"
source research/kv_translate/drive_latency.sh
A="$W/artifacts"
stamp () { echo "$1 $(date -u +%s.%N)" >> "$O/stages.txt"; }

stamp qualify_start
stage qualify python3 research/kv_translate/qualify_operator.py \
    --artifact "$A/train154_lin.pt" --fixtures "$A/fixtures.pt" \
    --out "$O/operator_a100.json"
stamp qualify_done

stamp timing_start
stage timing python3 research/kv_translate/time_first_next_token.py \
    --artifact "$A/train154_lin.pt" \
    --require-artifact-sha256 "$PROBE_SHA" \
    --fixtures "$A/timing_fixtures.pt" \
    --qualification "$O/operator_a100.json" \
    --experiment-id "$EXPERIMENT" --attempt-id "$ATTEMPT" \
    --started-at "$T0" --deadline-seconds "$DEADLINE" --extra-block \
    --lengths 512,2048,4096 --reps 20 --warmups 3 \
    --out "$O/timing_a100.json"
stamp timing_done

echo "ALL_OK" | tee -a "$O/drive.log"
touch "$O/DONE"
