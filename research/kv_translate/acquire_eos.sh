#!/bin/bash
# The corrected terminal-supervision pair, end to end on one rented machine.
#
# Every stage's exit is tested, the allocation id is recorded before anything
# is attempted with it, trained weights are exported before the pod is
# released, and release is verified even when the run failed.
set -u
S=/tmp/claude-1000/-home-mcgrof-devel-knlp/4d528f86-eccf-4ac4-a7e4-c6af5fa8ef68/scratchpad
R=/data/knlp-key-results/kv-translate-20260919
D=$R/eos-repair-20260922
ATTEMPT="${ATTEMPT:-attempt-1}"
# Outputs are scoped by attempt so a retry cannot overwrite a previous
# attempt's trained weights, which are the expensive part of this stage.
OUTDIR="$D/$ATTEMPT"
CAP_SECONDS="${CAP_SECONDS:-7200}"   # stage E ceiling
mkdir -p $OUTDIR
LOG=$OUTDIR/acquisition.log
export ALLOC_RECORD=$S/.eos_alloc STAGE_FAILED=$S/.eos_failed
: > $STAGE_FAILED
source "$(dirname "$0")/drive_remote.sh"
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }

T0=$(date +%s.%N)
say "$ATTEMPT: waiting on prune for a machine"
OUT=$(ssh prune 'REC=$HOME/.kvt_allocation bash ~/grab_pod.sh' 2>&1 | tee -a $LOG)
POD=$(printf '%s' "$OUT" | grep -oP 'POD_ID=\K[0-9a-f]{32}' | tail -1)
if [ -z "$POD" ]; then
    # The grabber records what it created even when it cannot resolve an id.
    POD=$(ssh prune 'cat $HOME/.kvt_allocation 2>/dev/null' | tr -d '[:space:]')
    [ "$POD" = "UNRESOLVED_AFTER_CREATE" ] && POD=""
fi
if [ -z "$POD" ]; then
    say "no machine obtained"
    SWEEP_CMD="ssh prune '. ~/envs/prime/bin/activate && prime pods list 2>/dev/null' | grep -oE '[0-9a-f]{32}' | xargs -r -I{} ssh prune '. ~/envs/prime/bin/activate && yes | prime pods terminate {}'" release_allocation >>$LOG 2>&1
    exit 1
fi
record_allocation "$POD" | tee -a $LOG
python3 -c "
import json; json.dump({'attempt_id':'$ATTEMPT','pod_id':'$POD','t_request':$T0,'state':'CREATED','gpu_ever_active':None,'stage':'E'}, open('$OUTDIR/allocation.json','w'), indent=2)"

terminate () {
    ssh prune ". ~/envs/prime/bin/activate && yes | prime pods terminate $POD" >>$LOG 2>&1
    sleep 8
    REM=$(ssh prune '. ~/envs/prime/bin/activate && prime pods list 2>/dev/null' | grep -c "$POD" || true)
    T=$(date +%s.%N)
    python3 -c "
import json; json.dump({'attempt_id':'$ATTEMPT','pod_id':'$POD','stage':'E','state':'$1',
 't_request':$T0,'t_released':$T,'occupied_seconds':$T-$T0,'gpu_ever_active':$2,
 'release_verified':($REM==0),'gpu':'A100-SXM4-40GB','cap_seconds':$CAP_SECONDS},
 open('$OUTDIR/allocation.json','w'), indent=2)"
    say "$1; occupied $(echo "$T - $T0" | bc)s of $CAP_SECONDS; released=$([ $REM -eq 0 ] && echo yes || echo NO)"
}

IP=""
for i in $(seq 1 60); do
    js=$(ssh prune ". ~/envs/prime/bin/activate && prime pods status $POD -o json" 2>/dev/null)
    read -r st ip <<<"$(echo "$js" | python3 -c "
import json,sys
raw=sys.stdin.read(); b=raw.find('{')
d=json.loads(raw[b:]) if b>=0 else {}
print(d.get('status',''), d.get('ip','') or '-')" 2>/dev/null)"
    [ "$st" = "ACTIVE" ] && [ "$ip" != "-" ] && { IP="$ip"; break; }
    [ "$st" = "FAILED" ] && break
    sleep 10
done
[ -z "$IP" ] && { say "never became reachable"; terminate RESOURCE_BLOCKED_NEVER_ACTIVE false; exit 1; }
say "ACTIVE at $IP after $(echo "$(date +%s.%N) - $T0" | bc)s"

SSHOPT="-i $HOME/.ssh/runpod -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=15 -o ConnectTimeout=20"
POD_SH () { ssh $SSHOPT ubuntu@$IP "$@"; }
for i in $(seq 1 30); do POD_SH 'true' 2>/dev/null && break; sleep 10; done

( POD_SH 'mkdir -p /home/ubuntu/ws/kvt/artifacts /home/ubuntu/ws/kvt/out' &&
  rsync -a -e "ssh $SSHOPT" $HOME/devel/knlp/research $HOME/devel/knlp/tools ubuntu@$IP:/home/ubuntu/ws/kvt/ &&
  rsync -a -e "ssh $SSHOPT" \
      $R/h1-corrected-20260920/arms/unw_k16.pt \
      $R/h1-corrected-20260920/arms/unw_k16.pt.manifest.json \
      $R/probe-20260922/probe_init.k.pt $R/probe-20260922/probe_init.v.pt \
      $R/rescue-20260921/manifests.json $R/rescue-20260921/dev64_gold.json \
      $R/rescue-20260921/a6000-run/train77_lin.pt \
      $R/rescue-20260921/a6000-run/train77_lin.pt.manifest.json \
      $R/qualify-20260922/fixtures.pt \
      $R/eos-repair-20260922/CONTRACT.json \
      ubuntu@$IP:/home/ubuntu/ws/kvt/artifacts/ ) >>$LOG 2>&1 &
SHIP=$!

POD_SH 'bash -s' < $S/bootstrap.sh > $OUTDIR/bootstrap.log 2>&1
BRC=$?
say "bootstrap exit $BRC"
wait_checked $SHIP upload >>$LOG 2>&1; SRC=$?
say "upload exit $SRC"

pull () { rsync -a -e "ssh $SSHOPT" ubuntu@$IP:/home/ubuntu/ws/kvt/out/ $OUTDIR/ >>$LOG 2>&1 || true; }

[ $BRC -ne 0 ] && { tail -4 $OUTDIR/bootstrap.log | tee -a $LOG; pull; terminate INVALID_BOOTSTRAP_FAILED true; exit 1; }
[ $SRC -ne 0 ] && { pull; terminate INVALID_UPLOAD_FAILED true; exit 1; }

POD_SH "cd /home/ubuntu/ws/kvt && T0=$T0 EXPERIMENT=kv-translate-eos-repair ATTEMPT=$ATTEMPT \
    DEADLINE=6600 bash research/kv_translate/measure_eos.sh" >>$OUTDIR/measure.log 2>&1
MRC=$?
say "measure exit $MRC"

# Weights come off the machine before it is released, whatever the exit was.
say "exporting trained artifacts"
rsync -a -e "ssh $SSHOPT" ubuntu@$IP:/home/ubuntu/ws/kvt/out/ $OUTDIR/ >>$LOG 2>&1 || say "export incomplete"
ls -la $OUTDIR/eos_on/*.pt $OUTDIR/eos_off/*.pt >>$LOG 2>&1 || true

[ $MRC -eq 0 ] && terminate COMPLETED_VALID true || terminate INVALID_OR_PARTIAL true
exit $MRC
