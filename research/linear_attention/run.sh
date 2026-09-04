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

# only the training phases consume the token stream; an eval-only
# run must not spend half an hour tokenizing data it never reads
if [ "$(jq_get phase_batch_probe)" = "True" ] ||
  [ "$(jq_get phase_campaign)" = "True" ]; then
  if [ ! -f "$DATA_DIR/tokens_gpt2.npy" ]; then
    echo "== PREPARE DATA ($TOKENS tokens -> $DATA_DIR) =="
    run "$PYTHON" "$KNLP/scripts/matched_micro_train.py" prepare-data \
      --data-dir "$DATA_DIR" --tokens "$TOKENS"
  fi
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

if [ "$(jq_get phase_rank_eval)" = "True" ]; then
  RSEED=$(jq_get rank_seed)
  RPF=$(jq_get rank_per_family)
  RCTL=$(jq_get rank_controls)
  RNORM=$(jq_get rank_norm)
  RMAX=$(jq_get rank_max_tokens)
  RSHARDS=$(jq_get rank_shards)
  RTAG=$(jq_get rank_tag)
  RNOUPD=""
  [ "$(jq_get rank_no_memory_update)" = "True" ] && RNOUPD="--no-memory-update"
  [ "$(jq_get rank_no_context)" = "True" ] && RNOUPD="$RNOUPD --no-context"
  RCHUNK=$(jq_get rank_chunk_path)
  [ "$RCHUNK" != "0" ] && RNOUPD="$RNOUPD --chunk-path $RCHUNK"
  CKPT_DIR="${CKPT_DIR:-$OUT_DIR}"
  REVAL="$OUT_DIR/$RTAG"
  mkdir -p "$REVAL"
  echo "== RANK EVAL $RTAG (seed $RSEED, $RPF/family, norm $RNORM) $RNOUPD =="
  [ -f "$REVAL/episodes.jsonl" ] ||
    run "$PYTHON" "$KNLP/scripts/accountability_bench.py" generate \
      --seed "$RSEED" --per-family "$RPF" --controls "$RCTL" \
      --out "$REVAL/episodes.jsonl"
  for D in "$CKPT_DIR"/campaign-b*-seed*-*; do
    [ -d "$D" ] || continue
    NAME=$(basename "$D")
    ARM=${NAME##*-}
    # honor the configured arm list: an ablation that only applies to
    # one arm must not silently re-score the whole checkpoint set
    case " $ARMS " in
    *" $ARM "*) ;;
    *) continue ;;
    esac
    [ -f "$D/$ARM.pt" ] || { echo "-- rank-eval $NAME: no checkpoint, skip"; continue; }
    echo "-- rank-eval $NAME ($RSHARDS shard(s))"
    if [ "$RSHARDS" -le 1 ]; then
      run "$PYTHON" "$HERE/rank_eval.py" --checkpoint "$D/$ARM.pt" \
        --episodes "$REVAL/episodes.jsonl" --out-dir "$REVAL/$NAME" \
        --norm "$RNORM" --max-tokens "$RMAX" --self-check --device cuda $RNOUPD
    else
      # every query is scored independently from a fresh state, so
      # sharding the query set across processes is exact, not an
      # approximation.  Hope is host-bound, so the parallelism is what
      # makes a full sweep affordable.
      for I in $(seq 0 $((RSHARDS - 1))); do
        run "$PYTHON" "$HERE/rank_eval.py" --checkpoint "$D/$ARM.pt" \
          --episodes "$REVAL/episodes.jsonl" \
          --out-dir "$REVAL/$NAME/shard$I" \
          --norm "$RNORM" --max-tokens "$RMAX" --self-check --device cuda \
          --num-shards "$RSHARDS" --shard "$I" $RNOUPD &
      done
      wait
      if [ "${DRY_RUN:-0}" != "1" ]; then
        cat "$REVAL/$NAME"/shard*/predictions.jsonl > "$REVAL/$NAME/predictions.jsonl"
        cat "$REVAL/$NAME"/shard*/raw_logprobs.jsonl > "$REVAL/$NAME/raw_logprobs.jsonl"
      fi
    fi
    run "$PYTHON" "$KNLP/scripts/accountability_bench.py" score \
      --episodes "$REVAL/episodes.jsonl" \
      --predictions "$REVAL/$NAME/predictions.jsonl" \
      --out "$REVAL/$NAME/score.json"
  done
  run "$PYTHON" "$HERE/rank_summary.py" --out-dir "$OUT_DIR" \
    --results-dir "$RESULTS_DIR" --tag "$RTAG"
fi

echo MICRO_RUN_DONE
