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
export PYTHONPATH="$CART_ROOT${PYTHONPATH:+:$PYTHONPATH}"
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

if [ "$(jq_get phase_paper_regime)" = "True" ]; then
  echo "== PAPER-REGIME ISOLATED CONFIRMATION =="
  : "${PAPER_DATA_DIR:?paper-regime phase needs PAPER_DATA_DIR}"
  PAPER_OUT="$OUT_DIR/paper_regime"
  PAPER_PATIENTS=$(jq_get paper_patients)
  PAPER_KV_DIVISOR=$(jq_get paper_kv_divisor)
  PAPER_EVAL_RUNS=$(jq_get paper_eval_runs)
  mkdir -p "$PAPER_OUT"

  [ "$GB" = "128" ] || { echo "paper regime requires global batch 128, got $GB"; exit 1; }
  [ "$LR" = "0.1" ] || { echo "paper regime requires LR 0.1, got $LR"; exit 1; }
  [ "$EPOCHS" = "80" ] || { echo "paper regime requires 80 epochs, got $EPOCHS"; exit 1; }
  [ "$STEPS" = "5000" ] || { echo "paper regime requires a 5000-step schedule, got $STEPS"; exit 1; }

  {
    echo "started=$(date -Is)"
    echo "knlp_commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "patients=$PAPER_PATIENTS"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
  } > "$PAPER_OUT/campaign.env"
  cp "$CFG" "$PAPER_OUT/config.json"
  nvidia-smi -q > "$PAPER_OUT/nvidia-smi-q.txt"

  PAPER_ACTIVE=""
  paper_record_failure() {
    rc=$?
    trap - EXIT
    if [ "$rc" -ne 0 ] && [ -n "$PAPER_ACTIVE" ]; then
      date -Is > "$PAPER_ACTIVE/FAILED"
    fi
    exit "$rc"
  }
  trap paper_record_failure EXIT

  for P in $PAPER_PATIENTS; do
    PARQ="$PAPER_DATA_DIR/$P.parquet"
    RECORD="$RECORDS_DIR/$P.txt"
    PD="$PAPER_OUT/$P"
    [ -s "$PARQ" ] || { echo "missing parquet: $PARQ"; exit 1; }
    [ -s "$RECORD" ] || { echo "missing record: $RECORD"; exit 1; }
    mkdir -p "$PD"
    PAPER_ACTIVE="$PD"
    rm -f "$PD/FAILED"
    if [ -f "$PD/DONE" ]; then
      echo "  skip $P: $PD/DONE exists"
      PAPER_ACTIVE=""
      continue
    fi

    sha256sum "$PARQ" "$RECORD" > "$PD/inputs.new.sha256"
    {
      echo "patient=$P"
      echo "model=$MODEL"
      echo "kv_tokens=auto"
      echo "kv_divisor=$PAPER_KV_DIVISOR"
      echo "lr=$LR"
      echo "global_batch=$GB"
      echo "epochs=$EPOCHS"
      echo "schedule_steps=$STEPS"
      echo "schedule=linear"
      echo "warmup_steps=200"
      echo "warmup_min_lr=2e-3"
      echo "alpha_f=0.02"
      echo "eval_runs=$PAPER_EVAL_RUNS"
    } > "$PD/recipe.new.env"

    if [ -f "$PD/TRAIN_DONE" ]; then
      cmp -s "$PD/inputs.sha256" "$PD/inputs.new.sha256" || {
        echo "refusing to resume $P: input hashes changed"
        exit 1
      }
      cmp -s "$PD/recipe.env" "$PD/recipe.new.env" || {
        echo "refusing to resume $P: recipe changed"
        exit 1
      }
    fi
    mv "$PD/inputs.new.sha256" "$PD/inputs.sha256"
    mv "$PD/recipe.new.env" "$PD/recipe.env"

    TRAIN_OUT="$PD/train"
    CART="$TRAIN_OUT/carts/$P.pt"
    if [ ! -f "$PD/TRAIN_DONE" ]; then
      echo "  train $P"
      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        PATIENT="$P" DATA_PARQUET="$PARQ" RECORDS_DIR="$RECORDS_DIR" \
        KV_TOKENS=auto KV_DIVISOR="$PAPER_KV_DIVISOR" LR="$LR" \
        GLOBAL_BS="$GB" EPOCHS="$EPOCHS" STEPS="$STEPS" \
        SCHED_STEPS="$STEPS" SCHEDULE=linear WARMUP_STEPS=200 \
        WARMUP_MIN_LR=2e-3 ALPHA_F=0.02 OUT_DIR="$TRAIN_OUT" \
        "$PYTHON" "$HERE/scripts/cas_train_isolated.py" 2>&1 | tee "$PD/train.log"
      [ -s "$CART" ] || { echo "training produced no cartridge: $CART"; exit 1; }
      grep -q "CAS_ISO_DONE $P" "$PD/train.log" || { echo "training did not complete for $P"; exit 1; }
      date -Is > "$PD/TRAIN_DONE"
    else
      echo "  skip $P training: $PD/TRAIN_DONE exists"
    fi
    [ -s "$CART" ] || { echo "training produced no cartridge: $CART"; exit 1; }

    EVAL_CARTS="$PD/eval_carts"
    if [ ! -f "$PD/EVAL_DONE" ]; then
      echo "  evaluate $P ($PAPER_EVAL_RUNS runs)"
      mkdir -p "$EVAL_CARTS"
      cp "$CART" "$EVAL_CARTS/$P.pt"
      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MODE=cart \
        CART_DIR="$EVAL_CARTS" PATIENTS="$P" MAX_Q=20 RUNS="$PAPER_EVAL_RUNS" \
        DEVICE=cuda MODEL="$MODEL" SAVE_RAW=1 \
        OUT_JSON="$PD/table15_runs${PAPER_EVAL_RUNS}.json" \
        "$PYTHON" "$HERE/scripts/cas_eval_table15.py" 2>&1 | tee "$PD/eval.log"
      grep -q "CAS_EVAL_T15_DONE mode=cart" "$PD/eval.log" || {
        echo "evaluation did not complete for $P"
        exit 1
      }
      date -Is > "$PD/EVAL_DONE"
    else
      echo "  skip $P evaluation: $PD/EVAL_DONE exists"
    fi

    find "$PD" -type f ! -name SHA256SUMS ! -name DONE ! -name FAILED -print0 | \
      sort -z | xargs -0 sha256sum > "$PD/SHA256SUMS"
    date -Is > "$PD/DONE"
    PAPER_ACTIVE=""
  done
  trap - EXIT
  date -Is > "$PAPER_OUT/DONE"
  echo "PAPER_REGIME_DONE $PAPER_OUT"
fi
echo "CAS_RUN_DONE results in $RESULTS_DIR"
