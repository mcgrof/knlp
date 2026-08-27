#!/bin/bash
# CAS driver: bootstrap (if needed) then run the phases selected in config.json
# (generated from the knlp .config by gen_config_json.py). Meant to run on a GPU
# host with vLLM available. All experiment policy comes from config.json.
#
#   Env: CART_ROOT (default /root/cartridges), PYTHON (CUDA torch),
#        VLLM (path to a vllm binary, for synthesis), OUT_DIR, RESULTS_DIR.
set -eu -o pipefail
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
# config expresses intent; an explicit env wins, because compiled
# FlexAttention is a property of the installed torch build rather than
# of the experiment (torch 2.13/CUDA 13 raises NoValidChoicesError
# lowering the flex kernel, and an eager fallback must not require
# editing the defconfig)
export CARTRIDGES_COMPILE_FLEX=${CARTRIDGES_COMPILE_FLEX:-$([ "$COMPILE" = "True" ] && echo 1 || echo 0)}
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

# records already staged (offline host) -> skip the network dump
ls "$RECORDS_DIR"/*.txt >/dev/null 2>&1 || \
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

if [ "$(jq_get phase_control_screen)" = "True" ]; then
  echo "== CONTROL-AWARE SCREEN (fixed-trajectory objective decomposition) =="
  CP=$(jq_get ctrl_patient)
  CPARQ="${DATA_PARQUET:?control screen needs DATA_PARQUET}"
  : "${CTRL_CART_INIT:?control screen needs CTRL_CART_INIT (starting cartridge .pt)}"
  CS_OUT="$OUT_DIR/control_screen"; mkdir -p "$CS_OUT"
  run_ctrl() {
    MODEL="$MODEL" ARM="$1" PATIENT="$CP" DATA_PARQUET="$CPARQ" \
      CART_INIT="$CTRL_CART_INIT" OPT_INIT="$CS_OUT/opt_init.pt" \
      SCHEDULE_JSON="$CS_OUT/schedule.json" STEPS=$(jq_get ctrl_steps) \
      ACCUM=$(jq_get ctrl_accum) LR=$(jq_get ctrl_lr) SEED=$(jq_get ctrl_seed) \
      CHECKPOINT_AT=$(jq_get ctrl_checkpoint_at) OUT_DIR="$CS_OUT" \
      "$PYTHON" control_aware_train.py 2>&1 | tee "$CS_OUT/train_$1.log"
  }
  # parity gates the matrix: legacy_raw must equal unique + anchors
  run_ctrl parity
  for CARM in $(jq_get ctrl_arms); do
    echo "  arm $CARM"
    run_ctrl "$CARM"
  done
  echo "  strict + forced-choice + probe eval of every checkpoint"
  # every arm's step0 is the identical starting cartridge: eval it once
  CCARTS=""
  CFIRST=1
  for CARM in $(jq_get ctrl_arms); do
    for f in "$CS_OUT/$CARM/${CP}"_step*.pt; do
      [ -f "$f" ] || continue
      s=$(basename "$f" .pt); s=${s##*_}
      if [ "$s" = "step0" ]; then
        # plain `[ ... ] && assign` would abort the script under set -e
        # the first time the test is false (every arm after the first)
        if [ "$CFIRST" = "1" ]; then
          CCARTS="${CCARTS:+$CCARTS,}start_step0=$f"
        fi
        continue
      fi
      CCARTS="${CCARTS:+$CCARTS,}${CARM}_${s}=$f"
    done
    CFIRST=0
  done
  MODEL="$MODEL" CARTS="$CCARTS" PATIENT="$CP" DATA_PARQUET="$CPARQ" \
    SCHEDULE_JSON="$CS_OUT/schedule.json" LONGHEALTH_JSON="${LONGHEALTH_JSON:-}" \
    MAX_Q=$(jq_get ctrl_max_q) PROBE_N=$(jq_get ctrl_probe_n) \
    OUT_JSON="$RESULTS_DIR/control_screen_eval.json" \
    "$PYTHON" control_aware_eval.py 2>&1 | tee "$CS_OUT/eval.log"
fi

if [ "$(jq_get phase_opt_ablation)" = "True" ]; then
  echo "== OPTIMIZER ABLATION (stored-objective cartridge, matched arms) =="
  KNLP_ROOT="$(cd "$HERE/../.." && pwd)"
  OP=$(jq_get opt_patient)
  # a staged parquet (DATA_PARQUET env) wins over synth-phase output
  # `|| true`: under set -e a failed discovery would abort here, before
  # the guard below could explain what is missing
  OPARQ="${DATA_PARQUET:-$(ls -t "$OUT_DIR"/*/synth_*_${OP/patient_/p}_n*/artifact/dataset.parquet 2>/dev/null | head -1 || true)}"
  [ -n "$OPARQ" ] || { echo "no parquet for $OP; set DATA_PARQUET"; exit 1; }
  ABL_OUT="$OUT_DIR/opt_ablation"; mkdir -p "$ABL_OUT"
  for OPT_ARM in $(jq_get opt_arms); do
    echo "  arm $OPT_ARM ($OPARQ)"
    MODEL="$MODEL" OPTIMIZER="$OPT_ARM" PATIENT="$OP" DATA_PARQUET="$OPARQ" \
      RECORDS_DIR="$RECORDS_DIR" KV_TOKENS=$KVT INIT_CART="$ABL_OUT/init_cart.pt" \
      STEPS=$(jq_get opt_steps) ACCUM=$(jq_get opt_accum) LR=$(jq_get opt_lr) \
      SEED=$(jq_get opt_seed) CHECKPOINT_AT=$(jq_get opt_checkpoint_at) \
      SOAP_PRECOND_FREQ=$(jq_get opt_soap_precond_freq) KNLP_ROOT="$KNLP_ROOT" \
      OUT_DIR="$ABL_OUT" "$PYTHON" cart_opt_ablation.py 2>&1 | tee "$ABL_OUT/train_$OPT_ARM.log"
  done
  echo "  strict re-eval of saved checkpoints"
  OCARTS="init=$ABL_OUT/init_cart.pt"
  for OPT_ARM in $(jq_get opt_arms); do
    for f in "$ABL_OUT/$OPT_ARM/${OP}"_step*.pt; do
      [ -f "$f" ] || continue
      s=$(basename "$f" .pt); s=${s##*_}
      OCARTS="$OCARTS,${OPT_ARM}_${s}=$f"
    done
  done
  CARTS="$OCARTS" RECORD="$RECORDS_DIR/$OP.txt" PATIENT="$OP" MODEL="$MODEL" \
    LONGHEALTH_JSON="${LONGHEALTH_JSON:-}" \
    OUT_JSON="$RESULTS_DIR/opt_ablation_reeval.json" "$PYTHON" opcart_reeval.py \
    2>&1 | tee "$ABL_OUT/reeval.log"
  "$PYTHON" cart_opt_report.py --ablation-dir "$ABL_OUT" \
    --reeval "$RESULTS_DIR/opt_ablation_reeval.json" \
    --out "$RESULTS_DIR/opt_ablation_report.md"
fi
echo "CAS_RUN_DONE results in $RESULTS_DIR"
