#!/bin/bash
# Decoupled CAS self-study pilot: bot A = Qwen3-14B (q-gen, port 8105),
# bot B = Qwen3-8B (answer teacher, port 8005). Five LongHealth patients,
# 8000 convos each (~40k total = the paper's per-dataset regime, spread over
# the 5 docs), run SEQUENTIALLY against the two shared serves so each patient
# gets both serves' full batch parallelism. Link-independent (run in tmux).
# Idempotent: a patient whose parquet already exists is skipped.
set -u
CASV=$HOME/cas_venv/bin/python
CART=$HOME/cartridges
OUT=$HOME/cas_out/synth_decoupled
LOGS=$HOME/cas_out/logs
NCONV=${NCONV:-8000}
PATIENTS=(patient_01 patient_02 patient_03 patient_05 patient_06)
A_URL=http://localhost:8105/v1
A_MODEL=qwen3-14b-qgen
B_URL=http://localhost:8005/v1
B_MODEL=qwen3-8b-ans
mkdir -p "$OUT" "$LOGS"

echo "[pilot] start $(date) NCONV=$NCONV patients=${PATIENTS[*]}" >> "$LOGS/synth_decoupled_orch.log"
for P in "${PATIENTS[@]}"; do
  pstr=${P/patient_/p}
  pq=$(ls "$OUT"/*/synth_decoupled_14bq_8ba_${pstr}_n*/artifact/dataset.parquet 2>/dev/null | head -1)
  if [ -n "$pq" ]; then
    echo "[pilot] $P already done ($pq)" >> "$LOGS/synth_decoupled_orch.log"; continue
  fi
  echo "[pilot] $P start $(date)" >> "$LOGS/synth_decoupled_orch.log"
  ( cd "$CART" && PATIENT=$P NUM_SAMPLES=$NCONV \
      CLIENT_A_URL="$A_URL" CLIENT_A_MODEL="$A_MODEL" \
      CLIENT_B_URL="$B_URL" CLIENT_B_MODEL="$B_MODEL" \
      CARTRIDGES_DIR="$CART" CARTRIDGES_OUTPUT_DIR="$OUT" \
      WANDB_DISABLED=true WANDB_MODE=disabled \
      "$CASV" cas_synth_decoupled.py >> "$LOGS/synth_decoupled_$P.log" 2>&1 )
  echo "[pilot] $P DONE $(date)" >> "$LOGS/synth_decoupled_orch.log"
done
echo "[pilot] ALL DONE $(date) -> $OUT" >> "$LOGS/synth_decoupled_orch.log"
