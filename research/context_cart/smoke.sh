#!/usr/bin/env bash
# End-to-end CPU smoke of every stage on the tiny model with the real
# tokenizer. Usage: smoke.sh <python> <work-dir>
set -euo pipefail
PY=${1:?python interpreter}
WORK=${2:?work directory}
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD
CLI="$PY -m research.context_cart.cli"
DS=$WORK/dataset
RUN=$WORK/run
MODEL="--model tiny --device cpu --source-revision $(printf 'e%.0s' {1..40})"
rm -rf "$WORK"; mkdir -p "$WORK"
$CLI prepare --out "$DS" --count calibration=4 --count tiny=4 --count train=8 \
  --count validation=6 --count test_ordinary=4 --count test_shifted=4 --count diagnostic=2 $MODEL
$CLI evaluate --dataset "$DS" --pool calibration --run-dir "$RUN" --text-arms full,empty,tail,summary \
  --batch 3 --summary-max-new 8 $MODEL
$CLI train --set run_dir=\"$RUN/tiny\" --set dataset_dir=\"$DS\" --set mode=\"tiny\" --set pool=\"tiny\" \
  --set validation_pool=\"tiny\" --set validation_worlds=0 --set updates=3 --set validate_at=[3] \
  --set validate_at_start=true --set microbatch=2 --set effective_batch=4 --set device=\"cpu\" \
  --set model.name=\"tiny\" --set eval_batch=4 --source-revision "$(printf 'e%.0s' {1..40})"
$CLI train --set run_dir=\"$RUN/train_s61031\" --set dataset_dir=\"$DS\" --set updates=4 --set validate_at=[2,4] \
  --set warmup=1 --set microbatch=2 --set effective_batch=4 --set device=\"cpu\" --set model.name=\"tiny\" \
  --set validation_worlds=3 --set eval_batch=4 --source-revision "$(printf 'e%.0s' {1..40})"
$CLI train --set run_dir=\"$RUN/profile\" --set dataset_dir=\"$DS\" --set mode=\"profile\" --set profile_warmup=1 \
  --set profile_timed=2 --set microbatch=2 --set effective_batch=4 --set device=\"cpu\" --set model.name=\"tiny\" \
  --set validation_worlds=2 --set eval_batch=4 --source-revision "$(printf 'e%.0s' {1..40})"
SEL=$($PY -c "import json;print(json.load(open('$RUN/train_s61031/train_summary.json'))['selected_checkpoint'])")
CK=$RUN/train_s61031/ckpt_$SEL.pt
$CLI export --dataset "$DS" --pool calibration --checkpoint "$CK" --out "$RUN/replay/producer" \
  --namespace smoke --worlds 2 --replay-reference $MODEL
$CLI replay --artifacts "$RUN/replay/producer/cartridges" --questions "$DS/questions_calibration.jsonl" \
  --out "$RUN/replay/consumer_a" --range 0:4 $MODEL
$CLI replay --artifacts "$RUN/replay/producer/cartridges" --questions "$DS/questions_calibration.jsonl" \
  --out "$RUN/replay/consumer_b" --range 4:8 $MODEL
$CLI compare-replay --producer "$RUN/replay/producer" --consumers "$RUN/replay/consumer_a" "$RUN/replay/consumer_b" \
  --out "$RUN/replay/replay_acceptance.json"
for pool in validation test_ordinary; do
  $CLI export --dataset "$DS" --pool $pool --checkpoint "$CK" --out "$RUN/cartridges/learned_s61031/$pool" --namespace smoke $MODEL
  $CLI export --dataset "$DS" --pool $pool --checkpoint "$RUN/train_s61031/ckpt_init.pt" --out "$RUN/cartridges/untrained/$pool" --namespace smoke $MODEL
  $CLI evaluate --dataset "$DS" --pool $pool --run-dir "$RUN" --text-arms full,empty,tail,summary --batch 3 --summary-max-new 8 \
    --artifact-arm untrained="$RUN/cartridges/untrained/$pool/cartridges" \
    --artifact-arm learned_s61031="$RUN/cartridges/learned_s61031/$pool/cartridges" \
    --wrong-arm wrong_s61031="$RUN/cartridges/learned_s61031/$pool/cartridges" $MODEL
done
$CLI fit --dataset "$DS" --checkpoint "$CK" --run-dir "$RUN" --updates 3 --worlds 2 $MODEL
$CLI report --run-dir "$RUN"
echo SMOKE_OK
