#!/bin/bash
# Stop at the first failing stage. The run that produced the objective probe
# printed a failing status and carried on, which is why every number from it
# was provisional. A driver that continues past a failed prerequisite is not
# a driver, it is a log.
set -u
O="${O:?set O to the output directory}"
mkdir -p "$O"

stage () {
    n="$1"; shift
    echo "=== $n start $(date -u +%H:%M:%S)" | tee -a "$O/drive.log"
    "$@" >> "$O/$n.log" 2>&1
    rc=$?
    echo "=== $n exit $rc $(date -u +%H:%M:%S)" | tee -a "$O/drive.log"
    if [ $rc -ne 0 ]; then
        echo "STOP: $n failed with $rc; nothing downstream runs" | tee -a "$O/drive.log"
        printf '%s\n' "$n" > "$O/FAILED"
        exit $rc
    fi
}
