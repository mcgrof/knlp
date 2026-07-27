#!/bin/bash
# Train 5 ISOLATED cartridges from the decoupled-14B-q-gen synth data, with the
# paper-faithful training the iso5 baseline lacked: per-document cart size
# (KV_TOKENS=auto = doc_tokens/20 ~585), a linear warmup+decay LR schedule, and
# ENOUGH steps (~12 epochs vs iso5's ~0.2 epoch/200 steps -- the undertraining
# that dominated iso5's low oracle). Runs on the free GPUs (0-3 untouched; the
# serves hold 5/6). Two GPUs (4,7) x sequential patients. Link-independent (tmux).
set -u
CASV=/home/mcgrof/cas_venv/bin/python
CART=/home/mcgrof/cartridges
S=/home/mcgrof/cas_out/synth_decoupled
OUT=/home/mcgrof/cas_out/iso5_decoupled
REC=/home/mcgrof/cas_out/records
LOGS=/home/mcgrof/cas_out/logs
mkdir -p "$OUT" "$LOGS"

# training hyperparameters (env-overridable)
# STEPS counts OPTIMIZER steps (grad-accum ~17 micro-batches/step, ~36
# opt-steps/epoch at 2000 convos) -> 600 ~= 17 epochs, ~3x the iso5 baseline's
# epoch count, with loss already well-converged (0.43/ppl1.53 by step ~36).
export KV_TOKENS=${KV_TOKENS:-auto} SCHEDULE=${SCHEDULE:-linear} \
       LR=${LR:-0.02} STEPS=${STEPS:-600} EPOCHS=${EPOCHS:-40} \
       WARMUP_STEPS=${WARMUP_STEPS:-60} WARMUP_MIN_LR=${WARMUP_MIN_LR:-0.002} \
       ALPHA_F=${ALPHA_F:-0.05} GLOBAL_BS=${GLOBAL_BS:-16}

PATIENTS=(patient_01 patient_02 patient_03 patient_05 patient_06)

pq_for() {  # patient -> its decoupled synth parquet (newest)
  local p=$1 pstr=${1/patient_/p}
  ls "$S"/*/synth_decoupled_14bq_8ba_${pstr}_n*/artifact/dataset.parquet \
     2>/dev/null | head -1
}

train() {  # gpu patient
  local g=$1 p=$2 pq
  pq=$(pq_for "$p")
  if [ -z "$pq" ]; then
    echo "[train] $p NO PARQUET -- skipped $(date)" >> "$LOGS/train_iso_orch.log"; return
  fi
  if [ -f "$OUT/carts/$p.pt" ]; then
    echo "[train] $p already trained" >> "$LOGS/train_iso_orch.log"; return
  fi
  echo "[train] $p start gpu$g pq=$pq $(date)" >> "$LOGS/train_iso_orch.log"
  ( cd "$CART" && CUDA_VISIBLE_DEVICES=$g PATIENT=$p DATA_PARQUET="$pq" \
      RECORDS_DIR="$REC" OUT_DIR="$OUT" \
      "$CASV" cas_train_isolated.py > "$LOGS/train_iso_$p.log" 2>&1 )
  echo "[train] $p DONE $(date)" >> "$LOGS/train_iso_orch.log"
}

echo "[train] START $(date) KV=$KV_TOKENS SCHED=$SCHEDULE LR=$LR STEPS=$STEPS" \
  >> "$LOGS/train_iso_orch.log"
# 3-wide across free GPUs (0-2 left clear): GPU4: p01,p06  GPU7: p02,p05  GPU3: p03
( train 4 patient_01 && train 4 patient_06 ) &
( train 7 patient_02 && train 7 patient_05 ) &
( train 3 patient_03 ) &
wait
echo "[train] ALL DONE $(date) -> $OUT/carts" >> "$LOGS/train_iso_orch.log"
