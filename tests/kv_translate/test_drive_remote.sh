#!/bin/bash
# Failure injection for the remote orchestration. Each case makes one stage
# fail and asserts that the run stops there and records which stage it was.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../research/kv_translate/drive_remote.sh"
T=$(mktemp -d)
export ALLOC_RECORD="$T/alloc" STAGE_FAILED="$T/failed"
source "$SRC"
fails=0
check () { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2' want '$3'"; fails=$((fails+1)); fi; }

# The allocation id is recorded before any stage runs, and survives a failure
# in the very next stage -- which is the case that stranded a machine before.
record_allocation "deadbeef" >/dev/null
check "id recorded immediately" "$(allocation_id)" "deadbeef"
stage_checked bootstrap false >/dev/null 2>&1
check "bootstrap failure recorded" "$(cat $T/failed)" "bootstrap"
check "id still available after failure" "$(allocation_id)" "deadbeef"
RELEASE_CMD="echo released" release_allocation | grep -q deadbeef
check "release used the recorded id" "$?" "0"

# Each named stage, failing in turn.
for s in bootstrap missing_helper qualifier output_flush; do
    : > "$T/failed"
    if stage_checked "$s" false >/dev/null 2>&1; then
        echo "FAIL $s: a failing stage returned success"; fails=$((fails+1))
    else
        check "$s stops the run" "$(cat $T/failed)" "$s"
    fi
done

# A missing helper is a nonzero exit, not a silent skip.
: > "$T/failed"
stage_checked missing_helper "$T/does_not_exist" >/dev/null 2>&1
check "missing helper is a failure" "$(cat $T/failed)" "missing_helper"

# A passing stage must not be recorded as failed.
: > "$T/failed"
stage_checked good true >/dev/null 2>&1
check "passing stage records nothing" "$(cat $T/failed)" ""

# A backgrounded stage's status is tested, not merely awaited.
: > "$T/failed"
( exit 3 ) & bg=$!
wait_checked $bg upload >/dev/null 2>&1
check "background failure is caught" "$(cat $T/failed)" "upload"
: > "$T/failed"
( exit 0 ) & bg=$!
wait_checked $bg upload >/dev/null 2>&1
check "background success is clean" "$(cat $T/failed)" ""

# With no id recorded at all, cleanup must still try something.
: > "$ALLOC_RECORD"
out=$(SWEEP_CMD="echo swept" release_allocation)
case "$out" in *swept*) echo "ok   sweeps when no id was recorded";; *) echo "FAIL no-id sweep"; fails=$((fails+1));; esac

# An unresolved create is treated as no id, so it sweeps rather than passing a
# placeholder to a terminate command.
printf 'UNRESOLVED_AFTER_CREATE\n' > "$ALLOC_RECORD"
out=$(SWEEP_CMD="echo swept" release_allocation)
case "$out" in *swept*) echo "ok   unresolved create sweeps";; *) echo "FAIL unresolved sweep"; fails=$((fails+1));; esac

rm -rf "$T"
echo "---"
if [ $fails -eq 0 ]; then echo "ALL DRIVER FAILURE CASES PASS"; exit 0; else echo "$fails FAILURES"; exit 1; fi
