#!/bin/bash
# CAS self-study synth on the FREE half (GPUs 5/6/7). Uses graph-vLLM
# (flashinfer disabled, CUDA graphs ON) at 16384 ctx so a ~7.7K-token
# LongHealth doc + 512 output fits and throughput is ~2x enforce-eager.
# N patients are round-robined across the 3 GPUs; each GPU runs its
# patients sequentially. Idempotent: a patient whose parquet already
# exists is skipped. Fully link-independent (launched inside tmux).
set -u
CASV=$HOME/cas_venv/bin/python
CART=$HOME/cartridges
OUT=$HOME/cas_out/synth
LOGS=$HOME/cas_out/logs
NCONV=${NCONV:-20000}
GPUS=(5 6 7)
PATIENTS=(patient_01 patient_02 patient_03 patient_04 patient_05 patient_06)
mkdir -p "$OUT" "$LOGS"

# 1. one graph-vLLM per free GPU, launched SEQUENTIALLY. vLLM's EngineCore picks
# its internal torch.distributed rendezvous port with a racy get_open_port(); if
# three launch at once they collide (EADDRINUSE). Staggering -- wait for each to
# be serving before starting the next -- serializes that allocation. Widely-spaced
# HTTP ports (8005/8105/8205) push their derived internal ports apart too.
for i in 0 1 2; do
  g=${GPUS[$i]}; port=$((8005 + 100 * i))
  if curl -s -m2 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "[orch] vLLM GPU $g port $port already up $(date)" >> "$LOGS/orch.log"; continue
  fi
  echo "[orch] start vLLM GPU $g port $port $(date)" >> "$LOGS/orch.log"
  VLLM_GPU=$g VLLM_PORT=$port bash $HOME/run_vllm_graph.sh \
    >> "$LOGS/vllm_g$g.log" 2>&1 &
  # wait for THIS server before launching the next (serializes port allocation)
  for t in $(seq 1 90); do
    curl -s -m3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q '"id"' && break
    sleep 8
  done
  echo "[orch] vLLM GPU $g port $port ready $(date)" >> "$LOGS/orch.log"
done
echo "[orch] all vLLM ready $(date)" >> "$LOGS/orch.log"

# 3. per-GPU worker: synth its assigned patients sequentially against its port
run_gpu() {
  local i=$1; shift
  local port=$((8005 + 100 * i))
  local P pstr pq
  for P in "$@"; do
    pstr=${P/patient_/p}
    pq=$(ls "$OUT"/*/synth_qwen3_8b_lh_${pstr}_n*/artifact/dataset.parquet 2>/dev/null | head -1)
    if [ -n "$pq" ]; then
      echo "[orch] $P already done ($pq)" >> "$LOGS/orch.log"; continue
    fi
    echo "[orch] $P start on port $port $(date)" >> "$LOGS/orch.log"
    ( cd "$CART" && PATIENT=$P NUM_SAMPLES=$NCONV \
        VLLM_URL="http://localhost:$port/v1" \
        CARTRIDGES_DIR="$CART" CARTRIDGES_OUTPUT_DIR="$OUT" \
        WANDB_DISABLED=true WANDB_MODE=disabled \
        "$CASV" synth_pod.py >> "$LOGS/synth_$P.log" 2>&1 )
    echo "[orch] $P DONE $(date)" >> "$LOGS/orch.log"
  done
}

# GPU5(i=0): p01,p04   GPU6(i=1): p02,p05   GPU7(i=2): p03,p06
run_gpu 0 patient_01 patient_04 &
run_gpu 1 patient_02 patient_05 &
run_gpu 2 patient_03 patient_06 &
wait
echo "[orch] ALL PATIENTS DONE ($NCONV convos each) $(date)" >> "$LOGS/orch.log"
