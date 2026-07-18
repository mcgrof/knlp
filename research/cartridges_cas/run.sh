#!/bin/bash
# CAS driver: bootstrap (if needed) then run the phases selected in config.json
# (generated from the knlp .config by gen_config_json.py). Meant to run on a GPU
# host with vLLM available. All experiment policy comes from config.json.
#
#   Env: CART_ROOT (default /root/cartridges), PYTHON (CUDA torch),
#        VLLM (path to a vllm binary, for synthesis), OUT_DIR, RESULTS_DIR.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
CART_ROOT="${CART_ROOT:-/root/cartridges}"
OUT_DIR="${OUT_DIR:-/root/cas_out}"
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}"
RECORDS_DIR="${RECORDS_DIR:-$OUT_DIR/records}"
CFG="$HERE/config.json"

[ -f "$CFG" ] || "$PYTHON" "$HERE/gen_config_json.py"
jq_get() { "$PYTHON" -c "import json,sys;print(json.load(open('$CFG')).get(sys.argv[1]))" "$1"; }

MODEL=$(jq_get model); NP=$(jq_get num_patients); CONVOS=$(jq_get convos_per_patient)
KVT=$(jq_get kv_tokens); LR=$(jq_get lr); GB=$(jq_get global_batch)
STEPS=$(jq_get steps); EPOCHS=$(jq_get epochs); COMPILE=$(jq_get compile_flex)
PATIENTS=""; for i in $(seq -w 1 "$NP"); do PATIENTS="$PATIENTS patient_$i"; done
export CARTRIDGES_DIR="$CART_ROOT" CARTRIDGES_OUTPUT_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR"
export RECORDS_DIR WANDB_DISABLED=true WANDB_MODE=disabled
export CARTRIDGES_COMPILE_FLEX=$([ "$COMPILE" = "True" ] && echo 1 || echo 0)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT_DIR" "$RESULTS_DIR"; cd "$CART_ROOT"

free_gpu() { pkill -9 -f "vllm.*serve" 2>/dev/null || true; pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done; sleep 6; }

if [ "$(jq_get phase_synth)" = "True" ]; then
  echo "== SYNTH (vLLM $MODEL, $CONVOS/patient) =="
  "${VLLM:-vllm}" serve "$MODEL" --port 8000 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --enforce-eager > "$OUT_DIR/vllm.log" 2>&1 &
  for i in $(seq 1 90); do curl -s -m5 http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"' && break; sleep 10; done
  for P in $PATIENTS; do
    echo "  synth $P"; PATIENT=$P NUM_SAMPLES=$CONVOS VLLM_URL=http://localhost:8000/v1 \
      "$PYTHON" synth_pod.py > "$OUT_DIR/synth_$P.log" 2>&1
  done
  free_gpu
fi

"$PYTHON" cas_dump_records.py > "$OUT_DIR/records.log" 2>&1

if [ "$(jq_get phase_train_isolated)" = "True" ]; then
  echo "== TRAIN ISOLATED =="
  for P in $PATIENTS; do
    PARQ=$(ls -t "$OUT_DIR"/*/synth_qwen3_8b_lh_${P/patient_/p}_n*/artifact/dataset.parquet 2>/dev/null | head -1)
    echo "  train $P ($PARQ)"; PATIENT=$P DATA_PARQUET="$PARQ" STEPS=$STEPS KV_TOKENS=$KVT \
      LR=$LR EPOCHS=$EPOCHS GLOBAL_BS=$GB "$PYTHON" cas_train_isolated.py > "$OUT_DIR/train_$P.log" 2>&1
  done
fi

if [ "$(jq_get phase_collapse)" = "True" ]; then
  echo "== COLLAPSE EVAL =="
  CART_DIR="$OUT_DIR/carts" PATIENTS="$PATIENTS" MAX_Q=15 MAX_NEW=48 \
    OUT_JSON="$RESULTS_DIR/collapse.json" MODES="oracle collapse" "$PYTHON" cas_combine_eval.py
fi

if [ "$(jq_get phase_train_joint)" = "True" ]; then
  echo "== TRAIN JOINT (mixed-visibility) =="
  for P in $PATIENTS; do
    PARQ=$(ls -t "$OUT_DIR"/*/synth_qwen3_8b_lh_${P/patient_/p}_n*/artifact/dataset.parquet 2>/dev/null | head -1)
    DIST=$(echo $PATIENTS | tr ' ' '\n' | grep -v "^${P}$" | tr '\n' ' ')
    PATIENT=$P DATA_PARQUET="$PARQ" DISTRACTORS="$DIST" ISO_CART_DIR="$OUT_DIR/carts" \
      STEPS=$STEPS KV_TOKENS=$KVT LR=$LR EPOCHS=$EPOCHS GLOBAL_BS=$GB \
      "$PYTHON" cas_train_joint.py > "$OUT_DIR/joint_$P.log" 2>&1
  done
fi

if [ "$(jq_get phase_rescue)" = "True" ]; then
  echo "== RESCUE EVAL =="
  CART_DIR="$OUT_DIR/carts_joint" PATIENTS="$PATIENTS" MAX_Q=15 MAX_NEW=48 \
    OUT_JSON="$RESULTS_DIR/rescue.json" MODES="oracle collapse" "$PYTHON" cas_combine_eval.py
fi
echo "CAS_RUN_DONE results in $RESULTS_DIR"
