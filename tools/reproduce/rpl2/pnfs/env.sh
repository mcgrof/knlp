# Task-owned parameters for the pNFS read-path lab.
# Sourced by labctl.sh and the *-inner.sh helpers. Every path, network
# namespace, interface, unit and port named here belongs to this task;
# nothing outside these names is created, edited or restarted.

# Every lab instance is named by RPL2P_PREFIX (default rpl2p, at most 6
# characters) and RPL2P_NET (default 10.247.0); RPL2P_ROOT has no default,
# so a run can never fall back to another operator's directory.
R=${RPL2P_ROOT:?set RPL2P_ROOT to the task root of this lab}
RUN=$R/run
BIN=$R/bin
STORE=$R/storage
LEDGER=$RUN/ledger
LATTICE_SRC=$R/src/pnfs-lattice
LATTICE_BUILD=$R/build
MDS_BIN=$LATTICE_BUILD/src/mds/pnfs-mds
RONDB=$R/rondb
NTIRPC_PREFIX=$R/prefix
LIBPATH=$RONDB/lib:$RONDB/lib64:$NTIRPC_PREFIX/lib

# Private link-local style subnet; verified absent from the host
# routing table before first use.
NET=${RPL2P_NET:-10.247.0}
NETMASK=24
MTU=${RPL2P_MTU:-9000}
NS_PREFIX=${RPL2P_PREFIX:-rpl2p}
[ ${#NS_PREFIX} -le 6 ] || { echo "RPL2P_PREFIX longer than 6" >&2; exit 2; }
# veth names: <prefix>a-<role> in the role namespace, <prefix>b-<role> on
# the bridge (at most 6 + 2 + 4 = 12 characters).
VA=${NS_PREFIX}a-
VB=${NS_PREFIX}b-
TRACE_INSTANCE=/sys/kernel/tracing/instances/$NS_PREFIX
NS_BR=${NS_PREFIX}-br
NS_MDS=${NS_PREFIX}-mds
# RPL2P_CLI selects the client: cli (the measured client) or c<k>, the
# extra clients of labctl.sh clients-up.
CLI=${RPL2P_CLI:-cli}
NS_CLI=${NS_PREFIX}-$CLI
IP_MDS=$NET.10
# c<k> are the extra clients of labctl.sh clients-up, at $NET.(100+k).
case $CLI in
cli) IP_CLI=$NET.30 ;;
c[1-9] | c[1-9][0-9]) IP_CLI=$NET.$((100 + ${CLI#c})) ;;
*) echo "bad RPL2P_CLI" >&2; exit 2 ;;
esac
# Data server i lives in namespace rpl2p-ds<i> at $NET.(20+i).
ds_ns() { echo "${NS_PREFIX}-ds$1"; }
ds_ip() { echo "$NET.$((20 + $1))"; }
ds_export() { echo "$STORE/ds$1"; }
DS_MOUNTD_PORT=20048
DS_NFSD_THREADS=${RPL2P_DS_THREADS:-8}

MDS_WORKERS=${RPL2P_MDS_WORKERS:-8}
CLI_MNT=$RUN/$CLI/mnt

# Absolute task memory caps (cgroup MemoryMax per transient unit).
# One DS: 6G + 4G + 1G + 3G = 14G, plus 1G per extra client.
MEM_RONDB=6G
MEM_MDS=4G
MEM_DS=1G
MEM_CLI=3G
MEM_CLIN=1G
RONDB_DATA_MEMORY=2G
RONDB_TRANSACTION_MEMORY=1G

UNIT_PREFIX=$NS_PREFIX
# Host side of the optional remote-client path (labctl.sh remote-up).
IP_HOST=$NET.1
HOST_VETH=${VA}host
OWNER=${RPL2P_OWNER:-$(stat -c %U "$R")}
