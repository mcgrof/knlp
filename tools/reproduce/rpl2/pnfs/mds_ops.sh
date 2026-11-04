#!/bin/bash
# Print the lab metadata server's per-operation counters. Run as root.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
"$here/labctl.sh" mdsns curl -s "http://$IP_MDS:9090/metrics" | grep -E '^pnfs_mds_nfs_op\{' || true
