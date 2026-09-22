#!/bin/bash
# Remote-stage orchestration, as functions so the failure paths can be tested
# without renting anything.
#
# Two defects in this lane came from this layer rather than from the driver it
# launches. One step's exit status was awaited and never tested, so a failed
# dependency install was followed by a measurement. And the cleanup path read
# the allocation id from a file the launcher writes only on success, so the one
# failure that stranded a machine was the one failure it could not clean up.
# Both are addressed here, and both are exercised by tests.

ALLOC_RECORD="${ALLOC_RECORD:-/tmp/kvt_allocation_record}"

# Write the allocation id the instant it is known, before anything is attempted
# with it. Cleanup reads this file and nothing else.
record_allocation () {
    local id="$1"
    printf '%s\n' "$id" > "$ALLOC_RECORD"
    echo "allocation recorded: $id"
}

allocation_id () {
    [ -s "$ALLOC_RECORD" ] && cat "$ALLOC_RECORD" || printf ''
}

# Run a stage and test how it exited. Awaiting is not checking.
stage_checked () {
    local name="$1"; shift
    "$@"
    local rc=$?
    echo "stage $name exit $rc"
    if [ $rc -ne 0 ]; then
        printf '%s\n' "$name" > "${STAGE_FAILED:-/tmp/kvt_stage_failed}"
        return $rc
    fi
    return 0
}

# Wait on a background job and test its status too. A bare `wait` discards it.
wait_checked () {
    local pid="$1" name="$2"
    wait "$pid"
    local rc=$?
    echo "stage $name exit $rc"
    if [ $rc -ne 0 ]; then
        printf '%s\n' "$name" > "${STAGE_FAILED:-/tmp/kvt_stage_failed}"
        return $rc
    fi
    return 0
}

# Release whatever was recorded. Falls back to sweeping by name only when no id
# was recorded at all, which should now be impossible.
release_allocation () {
    local id
    id="$(allocation_id)"
    if [ -n "$id" ] && [ "$id" != "UNRESOLVED_AFTER_CREATE" ]; then
        echo "releasing $id"
        ${RELEASE_CMD:-true} "$id"
    else
        echo "no allocation id recorded; sweeping by name"
        ${SWEEP_CMD:-true}
    fi
}
