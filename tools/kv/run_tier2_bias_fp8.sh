#!/bin/bash
# Tier-2 K-bias / FP8 runner. Phases: audit fake_quant long_context alpha serving phi_fii report all.
# Yields to any foreign GPU job (FogRFM) before each model-loading step. NO set -e.
# Usage: bash run_tier2_bias_fp8.sh <phase> [pass-through args...]
#   env: KBIAS_TIER2_ROOT (output root), KBIAS_PY (python), KBIAS_W7900=1 (HIP env + yield guard)
PHASE="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
T2="$HERE/tier2_bias_fp8"
ROOT="${KBIAS_TIER2_ROOT:-/data/knlp-key-results/k-bias-tier2-20260620}"
PY="${KBIAS_PY:-$HOME/envs/w7900-ml/bin/python3}"
CFG="${KBIAS_CFG:-$HERE/../configs/kv/k_bias_tier2_models.yaml}"
mkdir -p "$ROOT"
if [ "${KBIAS_W7900:-1}" = "1" ]; then export HIP_VISIBLE_DEVICES=0 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1; fi

wait_gpu () {
  [ "${KBIAS_W7900:-1}" = "1" ] || return 0
  while true; do
    n=$(ps -eo comm | grep -c "^python")
    [ "$n" -eq 0 ] && return 0
    echo "[$(date +%H:%M:%S)] GPU busy ($n python) -- yielding 120s" >> "$ROOT/tier2.log"; sleep 120
  done
}
# emit "short_name model_id [trust]" lines for tier!=large
models () {
  $PY - "$CFG" <<'PYE'
import sys, yaml
for m in yaml.safe_load(open(sys.argv[1]))["models"]:
    if m.get("tier") == "large": continue
    print(m["short_name"], m["model_id"])
PYE
}

run_audit () {  # tier-1 bias_audit + activation_audit (64 prompts), reuse the validated scripts
  local OUT="$ROOT/audit"; mkdir -p "$OUT"; rm -f "$OUT/model_bias_summary.csv" "$OUT/model_activation_summary.csv"
  models | while read sn mid; do
    wait_gpu; echo "[$(date +%H:%M:%S)] audit $sn" >> "$ROOT/tier2.log"
    $PY "$HERE/bias_audit.py" --model "$mid" --short-name "$sn" --device cpu --trust-remote-code --output-dir "$OUT" >> "$ROOT/tier2.log" 2>&1
    $PY "$HERE/k_bias_activation_audit.py" --model "$mid" --short-name "$sn" --num-prompts "${NP:-64}" --seq-len "${SL:-2048}" --device cuda:0 --trust-remote-code --output-dir "$OUT" >> "$ROOT/tier2.log" 2>&1
  done
}
run_alpha () {  # tier-1 alpha_sweep per model
  local OUT="$ROOT/alpha"; mkdir -p "$OUT"; rm -f "$OUT/alpha_sweep_summary.csv"
  models | while read sn mid; do
    wait_gpu; echo "[$(date +%H:%M:%S)] alpha $sn" >> "$ROOT/tier2.log"
    $PY "$HERE/k_bias_alpha_sweep.py" --model "$mid" --short-name "$sn" --alphas 0.0,0.25,0.5,1.0,1.5,2.0 --num-prompts "${NP:-32}" --seq-len "${SL:-2048}" --device cuda:0 --trust-remote-code --output-dir "$OUT" >> "$ROOT/tier2.log" 2>&1
  done
}

case "$PHASE" in
  audit)        NP="${NP:-64}" SL="${SL:-2048}"; run_audit ;;
  fake_quant)   wait_gpu; $PY "$T2/fp8_variant_probe.py" --models-file "$CFG" --skip-large --num-prompts "${NP:-64}" --output-dir "$ROOT/fake_quant" "$@" ;;
  long_context) wait_gpu; $PY "$T2/long_context_bias_probe.py" --models-file "$CFG" --output-dir "$ROOT/long_context" "$@" ;;
  alpha)        run_alpha ;;
  serving)      wait_gpu; $PY "$T2/hf_cache_prebias_eval.py" --models-file "$CFG" --output-dir "$ROOT/serving" "$@" ;;
  phi_fii)      wait_gpu; $PY "$T2/phi_fii_diagnose.py" --models-file "$CFG" --output-dir "$ROOT/phi_fii" "$@" ;;
  report)       $PY "$T2/report_tier2_bias_fp8.py" --root "$ROOT" ;;
  all)
    run_audit
    wait_gpu; $PY "$T2/fp8_variant_probe.py" --models-file "$CFG" --skip-large --num-prompts 64 --output-dir "$ROOT/fake_quant" >> "$ROOT/tier2.log" 2>&1
    wait_gpu; $PY "$T2/long_context_bias_probe.py" --models qwen25_7b dsr1_qwen_7b qwen25_14b phi2 --models-file "$CFG" --only qwen25_7b dsr1_qwen_7b qwen25_14b phi2 --seq-lens 2048,8192 --num-prompts 16 --output-dir "$ROOT/long_context" >> "$ROOT/tier2.log" 2>&1
    run_alpha
    wait_gpu; $PY "$T2/hf_cache_prebias_eval.py" --models-file "$CFG" --num-prompts 24 --output-dir "$ROOT/serving" >> "$ROOT/tier2.log" 2>&1
    wait_gpu; $PY "$T2/phi_fii_diagnose.py" --models-file "$CFG" --only phi2 phi4 --num-prompts 32 --output-dir "$ROOT/phi_fii" >> "$ROOT/tier2.log" 2>&1
    $PY "$T2/report_tier2_bias_fp8.py" --root "$ROOT" >> "$ROOT/tier2.log" 2>&1
    echo TIER2DONE > "$ROOT/tier2_status.txt" ;;
  *) echo "phases: audit fake_quant long_context alpha serving phi_fii report all"; exit 1 ;;
esac
