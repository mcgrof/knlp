#!/bin/bash
# Run native_reader.py as the owner inside a client's namespaces. Run as root.
#   run.sh <client> <native_reader args...>
# <client> is cli (the lab client), c<k> (an extra lab client) or remote
# (the client of remote-client.sh on this host). The interpreter is the one
# LMCache was built with ($R/python, written by setup.sh) and torch's C++
# libraries are put on LD_LIBRARY_PATH for the extension.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
who=$1
shift
. "$here/env.sh"
PY=${RPL2P_PYTHON:-$(cat "$R/python")}
TORCH_LIB=$("$PY" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
if [ "$who" = remote ]; then
	exec "$here/remote-client.sh" cli env LD_LIBRARY_PATH="$TORCH_LIB" "$PY" "$here/native_reader.py" \
		--src "$R/src/lmcache" "$@"
fi
RPL2P_CLI=$who exec "$here/labctl.sh" cli env LD_LIBRARY_PATH="$TORCH_LIB" "$PY" "$here/native_reader.py" \
	--src "$R/src/lmcache" "$@"
