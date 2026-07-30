#!/bin/bash
# Pro strategy #3 end-to-end: hard-negative entity-binding synth -> train -> eval.
# Serves on GPU 0 (14B qgen :8105) + GPU 1 (8B ans :8005); GPUs 4-7 left free for
# the Korea team; GPU 3 holds an unrelated straggler. After synth we stop ONLY our
# own serve PIDs (never a pattern-kill on the shared box) to free 0/1 for training.
set -u
CASV=/home/mcgrof/cas_venv/bin/python
CART=/home/mcgrof/cartridges
LOGS=/home/mcgrof/cas_out/logs
SYNTH=/home/mcgrof/cas_out/synth_hardneg
ISO=/home/mcgrof/cas_out/iso_hardneg
REC=/home/mcgrof/cas_out/records
EVAL_JSON=/home/mcgrof/cas_out/eval_hardneg.json
PATIENTS=(patient_01 patient_02 patient_03 patient_05 patient_06)
NCONV=${NCONV:-1500}
mkdir -p "$SYNTH" "$ISO" "$LOGS"
ORCH="$LOGS/hardneg_orch.log"
say(){ echo "[hardneg $(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }
gmem(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

# ---------------- Phase 0: serves ----------------
serve_up(){ curl -s -m 3 "http://localhost:$1/v1/models" >/dev/null 2>&1; }
: > "$LOGS/hardneg_serve.pids"
start_serve(){ # gpu port name model log
  local g=$1 port=$2 name=$3 model=$4 log=$5
  if serve_up "$port"; then say "serve :$port already UP"; return; fi
  say "start serve $name gpu$g :$port"
  VLLM_GPU=$g VLLM_PORT=$port VLLM_NAME=$name VLLM_MODEL=$model HF_HUB_OFFLINE=1 \
    setsid nohup /home/mcgrof/run_vllm_serve.sh >"$log" 2>&1 &
  echo "$!" >> "$LOGS/hardneg_serve.pids"
}
start_serve 0 8105 qwen3-14b-qgen Qwen/Qwen3-14B "$LOGS/serve_qgen.log"
start_serve 1 8005 qwen3-8b-ans   Qwen/Qwen3-8B  "$LOGS/serve_ans.log"
say "waiting for serves (model load + graph capture)..."
for i in $(seq 1 180); do
  serve_up 8105 && serve_up 8005 && { say "both serves UP (~$((i*10))s)"; break; }
  sleep 10
done
serve_up 8105 && serve_up 8005 || { say "SERVES FAILED -- see serve_*.log"; exit 1; }

# ---------------- Phase 1: synth ----------------
for P in "${PATIENTS[@]}"; do
  pstr=${P/patient_/p}
  pq=$(ls "$SYNTH"/*/synth_hardneg_14bq_8ba_${pstr}_n*/artifact/dataset.parquet 2>/dev/null | head -1)
  [ -n "$pq" ] && { say "synth $P already done"; continue; }
  say "synth $P start"
  ( cd "$CART" && PATIENT=$P NUM_SAMPLES=$NCONV PROB_THINKING=1.0 MAXTOK_B=2048 \
      MAX_BATCHES=8 WORKER_TIMEOUT=2400 \
      CLIENT_A_URL=http://localhost:8105/v1 CLIENT_A_MODEL=qwen3-14b-qgen \
      CLIENT_B_URL=http://localhost:8005/v1 CLIENT_B_MODEL=qwen3-8b-ans \
      CARTRIDGES_DIR="$CART" CARTRIDGES_OUTPUT_DIR="$SYNTH" \
      WANDB_DISABLED=true WANDB_MODE=disabled \
      "$CASV" cas_synth_hardneg.py >"$LOGS/synth_hardneg_$P.log" 2>&1 )
  say "synth $P rc=$?"
done

# ---------------- stop OUR serves (precise PIDs only) ----------------
say "stopping our serves to free gpu0/1"
while read -r pid; do
  [ -n "$pid" ] || continue
  pkill -TERM -P "$pid" 2>/dev/null   # our EngineCore child (PPID = our api server)
  kill  -TERM      "$pid" 2>/dev/null
done < "$LOGS/hardneg_serve.pids"
for i in $(seq 1 12); do
  [ "$(gmem 0)" -lt 3000 ] 2>/dev/null && [ "$(gmem 1)" -lt 3000 ] 2>/dev/null && break
  sleep 5
done
while read -r pid; do
  [ -n "$pid" ] || continue
  pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
done < "$LOGS/hardneg_serve.pids"
sleep 5
say "post-stop gpu mem: g0=$(gmem 0) g1=$(gmem 1) MiB"

# ---------------- Phase 2: train (corrected trainer already active) ----------------
export KV_TOKENS=auto SCHEDULE=linear LR=0.02 STEPS=600 EPOCHS=40 \
       WARMUP_STEPS=60 WARMUP_MIN_LR=0.002 ALPHA_F=0.05 GLOBAL_BS=16 PACK_LEN=4096
# pick up to two GPUs that are actually free (prefer freed 0/1, else 2)
TGPUS=(); for g in 0 1 2; do m=$(gmem "$g"); [ -n "$m" ] && [ "$m" -lt 3000 ] && TGPUS+=("$g"); done
[ ${#TGPUS[@]} -eq 0 ] && TGPUS=(2)
say "train GPUs: ${TGPUS[*]}"
pq_for(){ local pstr=${1/patient_/p}; ls "$SYNTH"/*/synth_hardneg_14bq_8ba_${pstr}_n*/artifact/dataset.parquet 2>/dev/null | head -1; }
train_one(){ # gpu patient
  local g=$1 p=$2 pq; pq=$(pq_for "$p")
  [ -z "$pq" ] && { say "train $p NO PARQUET -- skip"; return; }
  [ -f "$ISO/carts/$p.pt" ] && { say "train $p already done"; return; }
  say "train $p gpu$g"
  ( cd "$CART" && CUDA_VISIBLE_DEVICES=$g PATIENT=$p DATA_PARQUET="$pq" \
      RECORDS_DIR="$REC" OUT_DIR="$ISO" \
      "$CASV" cas_train_isolated.py > "$LOGS/train_hardneg_$p.log" 2>&1 )
  say "train $p rc=$?"
}
if [ ${#TGPUS[@]} -ge 2 ]; then
  gA=${TGPUS[0]}; gB=${TGPUS[1]}
  ( train_one "$gA" patient_01 && train_one "$gA" patient_03 && train_one "$gA" patient_06 ) &
  ( train_one "$gB" patient_02 && train_one "$gB" patient_05 ) &
  wait
else
  for p in "${PATIENTS[@]}"; do train_one "${TGPUS[0]}" "$p"; done
fi

# ---------------- Phase 3: eval (matched protocol) ----------------
EG=${TGPUS[0]}
say "eval on gpu$EG -> $EVAL_JSON"
( cd "$CART" && CUDA_VISIBLE_DEVICES=$EG CART_DIR="$ISO/carts" \
    PATIENTS="patient_01 patient_02 patient_03 patient_05 patient_06" \
    MAX_Q=20 MAX_COMPLETION=2048 RUNS=3 TOP_K=20 TOP_P=0.95 DEVICE=cuda:0 \
    OUT_JSON="$EVAL_JSON" CARTRIDGES_DIR="$CART" \
    "$CASV" cas_eval_matched.py > "$LOGS/eval_hardneg.log" 2>&1 )
say "eval rc=$?"
say "HARDNEG_ALL_DONE"
echo "===== eval_hardneg.json tail ====="
tail -20 "$EVAL_JSON" 2>/dev/null
