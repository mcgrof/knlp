#!/bin/bash
# Single-node RonDB (one management node, one data node) inside the MDS
# network namespace, listening on its loopback only. Memory is fixed by
# config.ini (AutomaticMemoryConfig=0); the transient unit adds a cgroup
# MemoryMax on top. Data lives under the task run directory.
set -euo pipefail
. "$(dirname "$0")/env.sh"
D=$RUN/rondb
mkdir -p "$D/mgm" "$D/data"
cat >"$D/config.ini" <<EOF
[ndb_mgmd]
NodeId=65
HostName=127.0.0.1
PortNumber=1186
DataDir=$D/mgm

[ndbd default]
NoOfReplicas=1
AutomaticMemoryConfig=0
AutomaticThreadConfig=0
MaxNoOfExecutionThreads=4
DataDir=$D/data
FileSystemPath=$D/data
DataMemory=$RONDB_DATA_MEMORY
SharedGlobalMemory=256M
RedoBuffer=64M
TransactionMemory=$RONDB_TRANSACTION_MEMORY
IndexMemory=64M

[ndbd]
NodeId=1
HostName=127.0.0.1

EOF
for n in $(seq 101 116); do
	printf '[api]\nNodeId=%s\nHostName=127.0.0.1\n\n' "$n" >>"$D/config.ini"
done
export LD_LIBRARY_PATH=$LIBPATH

pids=()
cleanup() {
	set +e
	"$RONDB/bin/ndb_mgm" -c 127.0.0.1:1186 -e shutdown >/dev/null 2>&1
	sleep 3
	kill "${pids[@]}" 2>/dev/null
	wait
}
trap cleanup EXIT TERM INT

"$RONDB/bin/ndb_mgmd" -f "$D/config.ini" --configdir="$D/mgm" \
	--initial --nodaemon >"$D/mgmd.log" 2>&1 &
pids+=($!)
sleep 3
"$RONDB/bin/ndbmtd" --ndb-connectstring=127.0.0.1:1186 --ndb-nodeid=1 \
	--initial --nodaemon >"$D/ndbmtd.log" 2>&1 &
pids+=($!)
for _ in $(seq 120); do
	if "$RONDB/bin/ndb_mgm" -c 127.0.0.1:1186 -e show 2>/dev/null |
		grep -q 'Nodegroup: 0'; then
		break
	fi
	sleep 2
done
"$RONDB/bin/ndb_mgm" -c 127.0.0.1:1186 -e show
"$RONDB/bin/ndb_index_stat" --sys-create-if-not-exist -c 127.0.0.1:1186
echo READY
wait "${pids[1]}"
