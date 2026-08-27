#!/bin/bash
# Matched-micro driver: run the phases selected in config.json (regenerated
# from the knlp .config by gen_config_json.py on every invocation, so a stale
# config.json cannot survive a defconfig change). One GPU. All experiment
# policy comes from config.json; hosts and paths come from the environment.
#
#   Env: PYTHON (torch-capable python, default python3),
#        DATA_DIR (token stream, default <knlp>/matched-micro-data),
#        OUT_DIR (run output, default <knlp>/matched-micro-runs),
#        RESULTS_DIR (summaries, default $OUT_DIR/results),
#        NESTED_LEARNING_SRC (nested_learning src/ dir, hope arm),
#        TITANS_PYTORCH_DIR (titans-pytorch checkout, titans arm),
#        ARMS_FILTER / SEEDS_FILTER (optional SUBSET of the configured
#          lists, for partitioning work across GPUs; an item not in
#          the configured list is an error — policy stays in Kconfig),
#        DRY_RUN=1 prints the planned commands without running them.
set -eu -o pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KNLP="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-$KNLP/matched-micro-data}"
OUT_DIR="${OUT_DIR:-$KNLP/matched-micro-runs}"
RESULTS_DIR="${RESULTS_DIR:-$OUT_DIR/results}"
CFG="$HERE/config.json"

"$PYTHON" "$HERE/gen_config_json.py"
jq_get() { "$PYTHON" -c "import json,sys;print(json.load(open('$CFG')).get(sys.argv[1]))" "$1"; }

[ "$(jq_get enabled)" = "True" ] || {
  echo "CONFIG_MATCHED_MICRO is not set — load a defconfig first, e.g.:"
  echo "  make defconfig-matched-micro-batch-probe"
  exit 1
}

[ -n "${NESTED_LEARNING_SRC:-}" ] &&
  export PYTHONPATH="$NESTED_LEARNING_SRC${PYTHONPATH:+:$PYTHONPATH}"
[ -n "${TITANS_PYTORCH_DIR:-}" ] &&
  export PYTHONPATH="$TITANS_PYTORCH_DIR${PYTHONPATH:+:$PYTHONPATH}"

ARMS=$(jq_get arms)
BATCHES=$(jq_get probe_batches)
STEPS=$(jq_get probe_steps)
TOKENS=$(jq_get data_tokens)
run() { if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY: $*"; else "$@"; fi; }

# intersect a configured list with an optional filter; a filter item
# absent from the configured list is a hard error, so the environment
# can only partition configured work, never add to it
subset() {
  full="$1" filt="$2"
  [ -z "$filt" ] && { echo "$full"; return; }
  out=""
  for x in $filt; do
    case " $full " in
    *" $x "*) out="$out $x" ;;
    *)
      echo "filter item '$x' not in configured list '$full'" >&2
      exit 1
      ;;
    esac
  done
  echo "$out"
}
ARMS=$(subset "$ARMS" "${ARMS_FILTER:-}")
mkdir -p "$OUT_DIR" "$RESULTS_DIR"

if [ ! -f "$DATA_DIR/tokens_gpt2.npy" ]; then
  echo "== PREPARE DATA ($TOKENS tokens -> $DATA_DIR) =="
  run "$PYTHON" "$KNLP/scripts/matched_micro_train.py" prepare-data \
    --data-dir "$DATA_DIR" --tokens "$TOKENS"
fi

if [ "$(jq_get phase_batch_probe)" = "True" ]; then
  echo "== BATCH PROBE (arms: $ARMS; batches: $BATCHES; $STEPS steps) =="
  for ARM in $ARMS; do
    for B in $BATCHES; do
      echo "-- probe $ARM batch $B"
      run "$PYTHON" "$KNLP/scripts/matched_micro_train.py" train \
        --arm "$ARM" --data-dir "$DATA_DIR" --steps "$STEPS" \
        --batch "$B" --out-dir "$OUT_DIR/probe-$ARM-b$B" --device cuda
    done
  done
  run "$PYTHON" "$HERE/probe_summary.py" --out-dir "$OUT_DIR" \
    --results-dir "$RESULTS_DIR" --arms "$ARMS" --batches "$BATCHES"
fi

if [ "$(jq_get phase_campaign)" = "True" ]; then
  CB=$(jq_get campaign_batch)
  CT=$(jq_get campaign_token_budget)
  CE=$(jq_get campaign_eval_every)
  CSEEDS=$(subset "$(jq_get campaign_seeds)" "${SEEDS_FILTER:-}")
  echo "== CAMPAIGN (arms: $ARMS; seeds: $CSEEDS; batch $CB; budget $CT) =="
  for SEED in $CSEEDS; do
    for ARM in $ARMS; do
      echo "-- campaign $ARM seed $SEED"
      run "$PYTHON" "$KNLP/scripts/matched_micro_train.py" train \
        --arm "$ARM" --data-dir "$DATA_DIR" --batch "$CB" \
        --token-budget "$CT" --seed "$SEED" --eval-every "$CE" \
        --save-checkpoint \
        --out-dir "$OUT_DIR/campaign-b$CB-seed$SEED-$ARM" --device cuda
    done
  done
  run "$PYTHON" "$HERE/campaign_summary.py" --out-dir "$OUT_DIR" \
    --results-dir "$RESULTS_DIR"
fi

echo MICRO_RUN_DONE
