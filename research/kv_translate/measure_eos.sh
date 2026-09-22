#!/bin/bash
# The corrected terminal-supervision pair, trained and scored in one run.
#
# Two arms, identical in every input the contract freezes, differing only in
# whether the terminal term contributes to the answer loss. The denominator is
# the same in both by construction, which is the thing the earlier arm got
# wrong. Both are qualified as serving artifacts before either is scored, and
# both are scored against native, an empty cache, their own wrong-prompt cache
# and the anchor.
#
# Expects: W (workspace), O (output dir), T0 (allocation request, unix).
set -u
W="${W:-/home/ubuntu/ws/kvt}"
export O="${O:-$W/out}"
T0="${T0:-0}"
SRC_REV="${SRC_REV:-989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
TGT_REV="${TGT_REV:-a09a35458c702b33eeacc393d103063234e8bc28}"
EXPERIMENT="${EXPERIMENT:-kv-translate-eos-repair}"
ATTEMPT="${ATTEMPT:-attempt-1}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$W"
mkdir -p "$O"
source research/kv_translate/drive_latency.sh
A="$W/artifacts"
DEADLINE="${DEADLINE:-6600}"   # stage E ceiling minus teardown headroom

over_cap () {
    [ "$T0" = "0" ] && return 1
    local now; now=$(date +%s)
    [ $(( now - ${T0%.*} )) -gt "$DEADLINE" ]
}
stamp () { echo "$1 $(date -u +%s.%N)" >> "$O/stages.txt"; }

train_arm () {
    local arm="$1"
    stamp "train_${arm}_start"
    stage "train_${arm}" python3 research/kv_translate/run_probe.py \
        --manifests "$A/manifests.json" --role train154 \
        --affine "$A/unw_k16.pt" --init "$A/probe_init" \
        --objective-mix 0.5 --forbid-codes "$A/dev64_gold.json" \
        --steps 800 --lr 3e-4 --seed 0 \
        --expect-train-documents 154 --expect-examples 286 \
        --expect-example-documents 143 \
        --source-revision "$SRC_REV" --target-revision "$TGT_REV" \
        --eos-arm "$arm" --require-init \
        --out-dir "$O/eos_${arm}"
    stamp "train_${arm}_done"
}

stage verify_contract python3 research/kv_translate/verify_contract.py \
    --contract "$A/CONTRACT.json" --staged "$A" --out "$O/contract_verified.json"

train_arm on
if over_cap; then
    echo "STOP: stage cap reached after the on arm" | tee -a "$O/drive.log"
    printf 'cap_after_on_arm\n' > "$O/FAILED"; exit 1
fi
train_arm off

# Both trained artifacts are qualified before either is scored. An artifact
# that has not been shown to compute its own definition is not a candidate,
# whatever it scores.
for arm in on off; do
    stamp "qualify_${arm}_start"
    stage "qualify_${arm}" python3 research/kv_translate/qualify_operator.py \
        --source-revision "$SRC_REV" --target-revision "$TGT_REV" \
        --artifact "$O/eos_${arm}/train154_lin.pt" \
        --premerge "$O/eos_${arm}/train154_premerge.pt" \
        --fixtures "$A/fixtures.pt" \
        --out "$O/operator_eos_${arm}.json"
    stamp "qualify_${arm}_done"
done

stamp gate_start
stage gate python3 research/kv_translate/run_dev_gate.py \
    --gold "$A/dev64_gold.json" \
    --arm eos_on="$O/eos_on/train154_lin.pt" \
    --arm eos_off="$O/eos_off/train154_lin.pt" \
    --arm anchor77="$A/train77_lin.pt" \
    --source-revision "$SRC_REV" --target-revision "$TGT_REV" \
    --verdict-scope quality_health \
    --out-dir "$O/gate"
stamp gate_done

stage rescore python3 research/kv_translate/rescore_gold.py \
    --rows "$O/gate/rows.jsonl" --manifest "$A/dev64_gold.json" \
    --out "$O/gate/rows.goldscored.jsonl"

# The advancement decision for this stage, applied to the scorer's
# measurements. The scorer's own verdict includes the 512 latency and
# native-suffix conjuncts, which this stage is told not to gate on.
stamp eos_gate_start
stage eos_gate python3 research/kv_translate/eos_gate.py \
    --rows "$O/gate/rows.jsonl" --arms eos_on,eos_off --anchor anchor77 \
    --out "$O/eos_gate.json"
stamp eos_gate_done

echo "ALL_OK" | tee -a "$O/drive.log"
touch "$O/DONE"
