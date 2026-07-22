#!/bin/bash
# STILL driver: read config.json (generated from the knlp .config by
# gen_config.py) and run the selected experiment on a single GPU. All experiment
# policy comes from config.json; nothing here hard-codes shapes or scale.
#
#   Env: PYTHON (a ROCm/CUDA torch interpreter), OUT_DIR (results), DEVICE
#        (torch device, default cuda:0), HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OUT_DIR="${OUT_DIR:-$HERE/out}"
CFG="$HERE/config.json"
export PYTHONPATH="$HERE/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[ -f "$CFG" ] || "$PYTHON" "$HERE/gen_config.py"
get() { "$PYTHON" -c "import json,sys;print(json.load(open('$CFG')).get(sys.argv[1]))" "$1"; }

EXP=$(get experiment); MODEL=$(get model)
CTX=$(get ctx_tokens); CHUNK=$(get chunk); TC=$(get t_compact)
NTR=$(get n_train); NEV=$(get n_eval); EP=$(get epochs); BS=$(get batch); SEED=$(get seed)
mkdir -p "$OUT_DIR"
S="$HERE/scripts"
echo "== STILL experiment: $EXP ($MODEL, device $DEVICE) =="

case "$EXP" in
  kernel)
    "$PYTHON" "$S/still_kernel.py" 2>&1 | tee "$OUT_DIR/kernel.log" ;;
  ledger)
    "$PYTHON" "$S/still_hbm_ledger.py" --out-dir "$OUT_DIR" --device "$DEVICE" \
      --chunk "$CHUNK" 2>&1 | tee "$OUT_DIR/ledger.log"
    "$PYTHON" "$S/still_ledger_report.py" "$OUT_DIR/hbm_ledger.json" \
      2>&1 | tee "$OUT_DIR/ledger_report.md" ;;
  baselines)
    "$PYTHON" "$S/still_baselines.py" --model "$MODEL" --device "$DEVICE" \
      --chunk "$CHUNK" --t-chunk "$TC" 2>&1 | tee "$OUT_DIR/baselines.log" ;;
  chunked)
    "$PYTHON" "$S/still_chunked_stream.py" --model "$MODEL" --device "$DEVICE" \
      --T "$CTX" --chunk "$CHUNK" --t-chunk "$TC" 2>&1 | tee "$OUT_DIR/chunked.log" ;;
  concurrency)
    "$PYTHON" "$S/still_concurrency.py" --model "$MODEL" --device "$DEVICE" \
      2>&1 | tee "$OUT_DIR/concurrency.log" ;;
  io)
    "$PYTHON" "$S/still_io_ssd.py" --model "$MODEL" --device "$DEVICE" \
      2>&1 | tee "$OUT_DIR/io_ssd.log" ;;
  ladder)
    "$PYTHON" "$S/still_gen_ladder.py" --model "$MODEL" --device "$DEVICE" \
      --n-train "$NTR" --n-eval "$NEV" --ctx-tokens "$CTX" --t-compact "$TC" \
      --epochs "$EP" --batch "$BS" --seed "$SEED" --out "$OUT_DIR" \
      2>&1 | tee "$OUT_DIR/ladder.log" ;;
  *)
    echo "unknown experiment: $EXP" >&2; exit 1 ;;
esac
echo "STILL_RUN_DONE results in $OUT_DIR"
