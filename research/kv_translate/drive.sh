#!/bin/bash
# Run a sequence of stages, stopping at the first failure.
#
# The version this replaces printed each stage's exit status and continued
# regardless, so an objective probe was trained and scored on a device whose
# sentinel had just failed both of its required checks. A driver that reports
# a failure it does not act on is worse than one that never checked.
set -o pipefail
FILTER='^Loading|it/s\]|Fetching|Token indices|Generating'

run() {
  local name="$1"; shift
  echo "=== $name ==="
  stdbuf -oL "$@" 2>&1 | stdbuf -oL grep -vE "$FILTER"
  local rc=${PIPESTATUS[0]}
  echo "exit $rc"
  if [ "$rc" -ne 0 ]; then
    echo "STOPPING: stage '$name' failed with status $rc; later stages would"
    echo "be measurements on a configuration that did not qualify."
    exit "$rc"
  fi
}
