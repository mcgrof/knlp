#!/bin/bash
# Phi/FII FP8-failure diagnosis runner. Phases: metadata subspace kv_isolation scale_granularity
# rope_placement attention_score layer_sweep report all. FogRFM-yield guard; NO set -e.
# env: PHI_ROOT (output), PHI_PY (python), PHI_MODELS (space-sep model ids), PHI_W7900=1.
PHASE="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${PHI_ROOT:-/data/knlp-key-results/phi-failure-20260620}"
PY="${PHI_PY:-$HOME/envs/w7900-ml/bin/python3}"
# primary + controls (the K16/V8 paper panel). Llama-3.1-8B is gated; Mistral covers if absent.
MODELS="${PHI_MODELS:-microsoft/phi-2 microsoft/phi-4 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct mistralai/Mistral-7B-Instruct-v0.3}"
mkdir -p "$ROOT"
[ "${PHI_W7900:-1}" = "1" ] && export HIP_VISIBLE_DEVICES=0 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
wait_gpu () { [ "${PHI_W7900:-1}" = "1" ] || return 0
  while true; do n=$(ps -eo comm | grep -c "^python"); [ "$n" -eq 0 ] && return 0
    echo "[$(date +%H:%M:%S)] GPU busy ($n python) yielding 120s" >> "$ROOT/phi.log"; sleep 120; done; }
M () { echo $MODELS; }
case "$PHASE" in
  metadata)         $PY "$HERE/phi_metadata_audit.py" --models $(M) --output-dir "$ROOT/metadata" "$@" ;;
  subspace)         wait_gpu; $PY "$HERE/phi_subspace_activation_audit.py" --models $(M) --num-prompts "${NP:-64}" --output-dir "$ROOT/subspace" "$@" ;;
  kv_isolation)     wait_gpu; $PY "$HERE/phi_kv_isolation_probe.py" --models $(M) --num-prompts "${NP:-32}" --output-dir "$ROOT/kv_isolation" "$@" ;;
  scale_granularity)wait_gpu; $PY "$HERE/phi_scale_granularity_probe.py" --models $(M) --num-prompts "${NP:-32}" --output-dir "$ROOT/scale_granularity" "$@" ;;
  rope_placement)   wait_gpu; $PY "$HERE/phi_rope_placement_probe.py" --models $(M) --num-prompts "${NP:-32}" --output-dir "$ROOT/rope_placement" "$@" ;;
  attention_score)  wait_gpu; $PY "$HERE/phi_attention_score_diagnostics.py" --models $(M) --num-prompts "${NP:-16}" --output-dir "$ROOT/attention_score" "$@" ;;
  layer_sweep)      wait_gpu; $PY "$HERE/phi_layer_sweep.py" --models $(M) --num-prompts "${NP:-16}" --output-dir "$ROOT/layer_sweep" "$@" ;;
  report)           $PY "$HERE/report_phi_failure.py" --root "$ROOT" --primary microsoft/phi-2 ;;
  all)
    $PY "$HERE/phi_metadata_audit.py" --models $(M) --output-dir "$ROOT/metadata" >> "$ROOT/phi.log" 2>&1
    wait_gpu; $PY "$HERE/phi_kv_isolation_probe.py" --models $(M) --num-prompts 32 --output-dir "$ROOT/kv_isolation" >> "$ROOT/phi.log" 2>&1
    wait_gpu; $PY "$HERE/phi_subspace_activation_audit.py" --models $(M) --num-prompts 64 --output-dir "$ROOT/subspace" >> "$ROOT/phi.log" 2>&1
    wait_gpu; $PY "$HERE/phi_scale_granularity_probe.py" --models $(M) --num-prompts 32 --output-dir "$ROOT/scale_granularity" >> "$ROOT/phi.log" 2>&1
    wait_gpu; $PY "$HERE/phi_rope_placement_probe.py" --models $(M) --num-prompts 32 --output-dir "$ROOT/rope_placement" >> "$ROOT/phi.log" 2>&1
    wait_gpu; $PY "$HERE/phi_attention_score_diagnostics.py" --models $(M) --num-prompts 16 --output-dir "$ROOT/attention_score" >> "$ROOT/phi.log" 2>&1
    wait_gpu; $PY "$HERE/phi_layer_sweep.py" --models microsoft/phi-2 Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 --num-prompts 16 --output-dir "$ROOT/layer_sweep" >> "$ROOT/phi.log" 2>&1
    $PY "$HERE/report_phi_failure.py" --root "$ROOT" --primary microsoft/phi-2 >> "$ROOT/phi.log" 2>&1
    echo PHIDONE > "$ROOT/phi_status.txt" ;;
  *) echo "phases: metadata subspace kv_isolation scale_granularity rope_placement attention_score layer_sweep report all"; exit 1 ;;
esac
