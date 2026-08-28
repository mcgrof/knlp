#!/bin/bash
# Oracle + co-load Table-15 eval for the decoupled-14B-q-gen isolated carts.
# Oracle (COLLAPSE=0): each cart alone on its OWN patient's questions -- the
# single-cart quality the >=0.65 gate is about. Co-load (COLLAPSE=1): all 5
# carts resident, the interference/collapse measurement. Faithful Table-15
# protocol (option-text answer, temp 0.6, thinking, RUNS runs, fuzzy match).
# Flex GPU path (loads Qwen3-8B on DEVICE, reconstructs carts) -- independent of
# the serves. Prints a verdict vs the gate, the iso5 baseline, and the paper.
set -u
CASV=$HOME/cas_venv/bin/python
CART=$HOME/cartridges
CARTS=$HOME/cas_out/iso5_decoupled/carts
OUTD=$HOME/cas_out
LOGS=$HOME/cas_out/logs
PATIENTS="patient_01 patient_02 patient_03 patient_05 patient_06"
DEV=${DEV:-cuda:0} GPU=${GPU:-4} RUNS=${RUNS:-3} MAX_Q=${MAX_Q:-20}
mkdir -p "$LOGS"

echo "[eval] oracle start $(date)" >> "$LOGS/eval_oracle_orch.log"
CUDA_VISIBLE_DEVICES=$GPU MODE=cart CART_DIR="$CARTS" PATIENTS="$PATIENTS" \
  MAX_Q=$MAX_Q RUNS=$RUNS COLLAPSE=0 DEVICE="$DEV" \
  OUT_JSON="$OUTD/eval_t15_cart_iso5_decoupled.json" \
  CARTRIDGES_DIR="$CART" \
  "$CASV" "$CART/cas_eval_table15.py" > "$LOGS/eval_oracle.log" 2>&1
echo "[eval] oracle done $(date)" >> "$LOGS/eval_oracle_orch.log"

echo "[eval] collapse start $(date)" >> "$LOGS/eval_oracle_orch.log"
CUDA_VISIBLE_DEVICES=$GPU MODE=cart CART_DIR="$CARTS" PATIENTS="$PATIENTS" \
  MAX_Q=$MAX_Q RUNS=$RUNS COLLAPSE=1 DEVICE="$DEV" \
  OUT_JSON="$OUTD/eval_t15_cart_iso5_decoupled_collapse.json" \
  CARTRIDGES_DIR="$CART" \
  "$CASV" "$CART/cas_eval_table15.py" > "$LOGS/eval_collapse.log" 2>&1
echo "[eval] collapse done $(date)" >> "$LOGS/eval_oracle_orch.log"

# --- verdict ---
"$CASV" - <<PY
import json, os
def acc(p):
    try: return json.load(open(p))["summary"]["acc"]
    except Exception: return None
O="$OUTD"
new_o = acc(f"{O}/eval_t15_cart_iso5_decoupled.json")
new_c = acc(f"{O}/eval_t15_cart_iso5_decoupled_collapse.json")
iso_o = acc(f"{O}/eval_t15_cart_iso5.json")
nocx  = acc(f"{O}/eval_t15_nocontext.json")
full  = acc(f"{O}/eval_t15_fulldoc.json")
print("\n==================== PILOT VERDICT ====================")
print(f"  no-context baseline : {nocx}")
print(f"  full-document ceil  : {full}   (paper 0.874)")
print(f"  iso5 (8B q-gen)     : {iso_o}   <- prior single-cart oracle")
print(f"  NEW (14B q-gen)     : {new_o}   <- decoupled single-cart oracle")
print(f"  NEW co-load (N=5)   : {new_c}")
print(f"  paper isolated      : 0.736")
gate = 0.65
if new_o is not None:
    d = (new_o - iso_o) if iso_o is not None else None
    print(f"\n  gate >= {gate}: {'PASS' if new_o >= gate else 'FAIL'} (oracle {new_o})")
    if d is not None:
        print(f"  delta vs iso5: {d:+.3f} ({'>=+0.08 OK' if d>=0.08 else '<+0.08'})")
print("=======================================================")
PY
echo "[eval] verdict printed $(date)" >> "$LOGS/eval_oracle_orch.log"
