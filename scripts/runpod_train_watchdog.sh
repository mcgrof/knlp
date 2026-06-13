#!/usr/bin/env bash
# Hardened RunPod training watchdog.
#
# Why this exists: a prior Trellis run was time-capped and the watchdog did
# "mirror logs + terminate" -- but the trainer had saved NO checkpoint and the
# eval only ran after full training, so the cap destroyed ~10h of GPU work and
# produced no number. Money wasted.
#
# THE RULE THIS ENFORCES: never terminate a training pod on CAP or IDLE unless
# an EVALUABLE ARTIFACT has been mirrored off the pod (a *_evals.jsonl with at
# least one line, a rung_*.json, or an ALL_DONE marker). If none exists yet on a
# cap, REFUSE to terminate and extend by EXTEND_H (up to HARD_MAX_H) so the run
# reaches its next periodic checkpoint+eval instead of dying empty. Pair this
# with a trainer that checkpoints+evals periodically (trellis_ladder.py
# --ckpt_every/--eval_every) so the artifact is always close at hand.
#
# Usage:
#   runpod_train_watchdog.sh POD_JSON DEST_DIR REMOTE_RESULT_DIR \
#       [MAXH] [DONE_MARKER] [FATAL_MARKER] [LOGFILE]
# POD_JSON: {pod_id,ssh_ip,ssh_port}. DEST_DIR: local mirror (key-results).
# REMOTE_RESULT_DIR: pod dir holding checkpoints + *_evals.jsonl + rung_*.json.
set -uo pipefail
source ~/.enhance-bash 2>/dev/null
source ~/envs/runpod/bin/activate 2>/dev/null

J="${1:?pod json}"; DEST="${2:?dest dir}"; RDIR="${3:?remote result dir}"
MAXH="${4:-13}"; DONE_MARKER="${5:-ALL_DONE}"; FATAL_MARKER="${6:-ALL_DONE_FATAL}"
LOG="${7:-/root/pod_run.log}"
EXTEND_H=2; HARD_MAX_H=$((MAXH + 8))

ID=$(jq -r .pod_id "$J"); IP=$(jq -r .ssh_ip "$J"); PORT=$(jq -r .ssh_port "$J")
K=~/.ssh/runpod
SSH="ssh -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p $PORT root@$IP"
SCP="scp -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT"
mkdir -p "$DEST"
START=$(date +%s); last=0; stag=0; miss=0

mirror(){  # pull checkpoints + evals + logs (best-effort, bounded)
  timeout 600 $SCP -r "root@$IP:$RDIR" "$DEST/" 2>/dev/null || true
  timeout 60  $SCP "root@$IP:$LOG" "$DEST/" 2>/dev/null || true
}
have_artifact(){  # 0 == an evaluable artifact is present locally (any depth)
  local f
  f=$(find "$DEST" -name '*_evals.jsonl' -size +0c 2>/dev/null | head -1)
  [ -n "$f" ] && return 0
  find "$DEST" -name 'rung_*.json' 2>/dev/null | grep -q . && return 0
  return 1
}
terminate(){ echo "$1 $(date -u)" > "$DEST/outcome.txt"
  python ~/.claude/skills/runpod/terminate_pod.py "$ID" 2>&1 | tail -2; exit 0; }
fin(){ local reason="$1"; echo "[wd] $reason: mirror then decide"; mirror
  if [ "$reason" = "$DONE_MARKER" ] || [ "$reason" = "COMPLETE" ] || \
     [ "$reason" = "FATAL" ] || [ "$reason" = "SSH_UNREACHABLE" ]; then
    echo "[wd] $reason -> terminate"; terminate "$reason"
  fi
  if have_artifact; then echo "[wd] $reason + artifact present -> terminate"; terminate "$reason"
  else
    if [ "$MAXH" -lt "$HARD_MAX_H" ]; then
      MAXH=$((MAXH + EXTEND_H))
      echo "[wd] !! $reason but NO eval artifact yet -> REFUSE terminate, extend to ${MAXH}h"
      return 0
    fi
    echo "[wd] !! $reason, no artifact, hit HARD_MAX_H=${HARD_MAX_H}h -> terminate to stop burn"
    terminate "${reason}_NO_ARTIFACT"
  fi; }

while true; do
  sleep 240
  el=$(( ($(date +%s)-START)/3600 ))
  out=$(timeout 45 $SSH "tail -2 $LOG 2>/dev/null; echo __U__; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1; echo __S__; stat -c%s $LOG 2>/dev/null" 2>/dev/null)
  rc=$?
  mirror
  if [ $rc -ne 0 ] || [ -z "$out" ]; then miss=$((miss+1)); echo "[wd] t=${el}h ssh-miss=$miss"; [ "$miss" -ge 6 ] && fin SSH_UNREACHABLE; [ "$el" -ge "$MAXH" ] && fin CAP; continue; fi
  miss=0
  echo "$out" | grep -q "$FATAL_MARKER" && fin FATAL
  echo "$out" | grep -q "$DONE_MARKER" && fin "$DONE_MARKER"
  util=$(echo "$out" | sed -n '/__U__/,/__S__/p' | grep -oE '^[0-9]+' | head -1); util=${util:-99}
  size=$(echo "$out" | tail -1 | grep -oE '[0-9]+' | tail -1); size=${size:-0}
  if [ "$size" = "$last" ] && [ "${util:-99}" -lt 5 ]; then stag=$((stag+1)); else stag=0; fi
  last=$size
  echo "[wd] t=${el}h util=${util}% size=$size stag=$stag MAXH=${MAXH}"
  [ "$stag" -ge 3 ] && fin CRASH_IDLE
  [ "$el" -ge "$MAXH" ] && fin CAP
done
