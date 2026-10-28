#!/bin/bash
# Bring the task-owned pNFS lab up and down on one host. Run as root.
#
#   labctl.sh up [ds_count] [extra-mds-conf]   network, DS, RonDB, MDS, client
#   labctl.sh down                             ledger-driven teardown + checks
#   labctl.sh status
#   labctl.sh cli <cmd...>                     run a command (as the owner
#                                              user) inside the client mount
#                                              and network namespaces
#   labctl.sh mdsns <cmd...>                   run as root in the MDS netns
#   labctl.sh dsns <i> <cmd...>                run as root in a DS netns
#   labctl.sh remount                          fresh client mount
#   labctl.sh clients-up <n> | clients-down    extra clients c1..cn
#   labctl.sh remote-up | remote-down          route for clients on other hosts
#   labctl.sh stop-ds <i> | start-ds <i> | stop-mds | start-mds
#
# Only names from env.sh are created. The host knfsd, /etc/exports and the
# host rpcbind are never modified; `down` compares them with the baseline.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"

ledger() { echo "$(date -u +%FT%TZ) $*" >>"$LEDGER"; }

wait_ready() {
	local unit=$1 t=${2:-120} inv
	inv=$(systemctl show -p InvocationID --value "$unit")
	for _ in $(seq "$t"); do
		if journalctl _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager 2>/dev/null |
			grep -qx READY; then
			return 0
		fi
		if ! systemctl is-active -q "$unit"; then
			journalctl -u "$unit" -o cat --no-pager | tail -40
			echo "unit $unit exited before READY" >&2
			return 1
		fi
		sleep 1
	done
	journalctl -u "$unit" -o cat --no-pager | tail -40
	echo "unit $unit not READY after ${t}s" >&2
	return 1
}

run_unit() {
	local unit=$1 mem=$2
	shift 2
	# Transient units do not inherit the caller's environment; pass every
	# lab parameter explicitly so inner scripts resolve this lab's paths.
	systemd-run --unit="$unit" --collect -p MemoryMax="$mem" \
		-p KillMode=mixed -p TimeoutStopSec=30 \
		-E RPL2P_ROOT="$R" -E RPL2P_PREFIX="$NS_PREFIX" -E RPL2P_NET="$NET" \
		-E RPL2P_OWNER="$OWNER" -E RPL2P_CLI="${RPL2P_CLI:-cli}" \
		-E RPL2P_REMOTE_CLIENTS="${RPL2P_REMOTE_CLIENTS:-}" "$@" >/dev/null
	ledger "unit $unit"
}

baseline() {
	rpcinfo -p >"$RUN/host-rpcinfo.$1"
	iptables -S >"$RUN/host-iptables.$1"
	exportfs -v >"$RUN/host-exportfs.$1"
	sha256sum /etc/exports >"$RUN/host-etc-exports.$1"
	cat /proc/fs/nfsd/threads >"$RUN/host-nfsd-threads.$1"
}

net_up() {
	local nds=$1
	if ip netns list | grep -q "^$NS_PREFIX-"; then
		echo "a lab with prefix $NS_PREFIX already exists on this host; choose RPL2P_PREFIX" >&2
		exit 1
	fi
	if ip -4 route | grep -q "^$NET\."; then
		echo "subnet $NET.0/$NETMASK already routed on host" >&2
		exit 1
	fi
	ip netns add "$NS_BR"
	ledger "netns $NS_BR"
	ip -n "$NS_BR" link add br0 type bridge
	ip -n "$NS_BR" link set br0 mtu "$MTU" up
	ip -n "$NS_BR" link set lo up
	local roles=("mds:$NS_MDS:$IP_MDS" "cli:$NS_CLI:$IP_CLI")
	for i in $(seq 0 $((nds - 1))); do
		roles+=("ds$i:$(ds_ns "$i"):$(ds_ip "$i")")
	done
	local r name ns ip
	for r in "${roles[@]}"; do
		IFS=: read -r name ns ip <<<"$r"
		ip netns add "$ns"
		ledger "netns $ns"
		ip link add "$VA$name" mtu "$MTU" type veth peer name "$VB$name" mtu "$MTU"
		ip link set "$VA$name" netns "$ns"
		ip link set "$VB$name" netns "$NS_BR"
		ip -n "$NS_BR" link set "$VB$name" master br0 up
		ip -n "$ns" addr add "$ip/$NETMASK" dev "$VA$name"
		ip -n "$ns" link set "$VA$name" up
		ip -n "$ns" link set lo up
	done
}

up() {
	local nds=${1:-1} extra=${2:-}
	mkdir -p "$RUN/cli" "$RUN/logs" "$STORE"
	chown "$OWNER": "$RUN" "$RUN/cli" "$STORE"
	: >>"$LEDGER"
	baseline before
	date '+%Y-%m-%d %H:%M:%S' >"$RUN/started"
	echo "$nds" >"$RUN/nds"
	echo "$extra" >"$RUN/mds-extra"
	net_up "$nds"
	for i in $(seq 0 $((nds - 1))); do
		run_unit "$UNIT_PREFIX-ds$i" "$MEM_DS" ip netns exec "$(ds_ns "$i")" \
			unshare -m --propagation private "$BIN/ds-inner.sh" "$i"
		wait_ready "$UNIT_PREFIX-ds$i" 60
	done
	run_unit "$UNIT_PREFIX-rondb" "$MEM_RONDB" ip netns exec "$NS_MDS" \
		unshare -m --propagation private "$BIN/rondb-inner.sh"
	wait_ready "$UNIT_PREFIX-rondb" 300
	run_unit "$UNIT_PREFIX-mds" "$MEM_MDS" ip netns exec "$NS_MDS" \
		unshare -m --propagation private "$BIN/mds-inner.sh" "$nds" $extra
	for _ in $(seq 60); do
		ip netns exec "$NS_MDS" ss -ltn | grep -q ":2049 " && break
		systemctl is-active -q "$UNIT_PREFIX-mds" || break
		sleep 1
	done
	ip netns exec "$NS_MDS" ss -ltn | grep ":2049 " || {
		journalctl -u "$UNIT_PREFIX-mds" -o cat --no-pager | tail -40
		tail -40 "$RUN/mds/pnfs-mds.log" 2>/dev/null
		exit 1
	}
	run_unit "$UNIT_PREFIX-cli" "$MEM_CLI" ip netns exec "$NS_CLI" \
		unshare -m --propagation private "$BIN/cli-inner.sh"
	wait_ready "$UNIT_PREFIX-cli" 60
	status
}

save_logs() {
	local u
	for u in $(systemctl list-units --all --plain --no-legend "$UNIT_PREFIX-*" | awk '{print $1}'); do
		journalctl -u "$u" --since "$(cat "$RUN/started" 2>/dev/null || echo today)" -o short-iso --no-pager \
			>"$RUN/logs/${u%.service}.journal" 2>&1 || true
	done
}

# Optional path for clients on other hosts: one host veth into the lab
# bridge, default routes from the lab namespaces back through it, and
# accept rules for the lab subnet in Docker's DOCKER-USER chain (Docker
# sets the FORWARD policy to DROP). Remote clients add a route to $NET.0/24
# via this host and must be listed in RPL2P_REMOTE_CLIENTS before `up` so
# the data servers export to them. Nothing persists across a reboot.
# Extra clients c1..cN for the many-client sweep: each is its own NFS
# client identity in its own namespace, with a 1 GiB-capped holder unit.
clients_up() {
	local n=$1 k
	for k in $(seq 1 "$n"); do
		(
			export RPL2P_CLI=c$k
			. "$here/env.sh"
			mkdir -p "$RUN/c$k"
			chown "$OWNER": "$RUN/c$k"
			ip netns add "$NS_CLI"
			ledger "netns $NS_CLI"
			ip link add "${VA}c$k" mtu "$MTU" type veth peer name "${VB}c$k" mtu "$MTU"
			ip link set "${VA}c$k" netns "$NS_CLI"
			ip link set "${VB}c$k" netns "$NS_BR"
			ip -n "$NS_BR" link set "${VB}c$k" master br0 up
			ip -n "$NS_CLI" addr add "$IP_CLI/$NETMASK" dev "${VA}c$k"
			ip -n "$NS_CLI" link set "${VA}c$k" up
			ip -n "$NS_CLI" link set lo up
			run_unit "$UNIT_PREFIX-c$k" "$MEM_CLIN" ip netns exec "$NS_CLI" \
				unshare -m --propagation private "$BIN/cli-inner.sh"
			wait_ready "$UNIT_PREFIX-c$k" 60
		)
	done
}

clients_down() {
	local u
	for u in $(systemctl list-units --all --plain --no-legend "$UNIT_PREFIX-c[0-9]*" | awk '{print $1}'); do
		systemctl stop "$u" || true
	done
	local ns
	for ns in $(ip netns list | awk '{print $1}' | grep -E "^$NS_PREFIX-c[0-9]+$" || true); do
		ip netns del "$ns"
		ledger "netns-del $ns"
	done
}

remote_up() {
	ip netns list | grep -q "^$NS_BR\b" || { echo "lab is not up" >&2; exit 1; }
	[ "$(cat /proc/sys/net/ipv4/ip_forward)" = 1 ] || { echo "ip_forward is off" >&2; exit 1; }
	ip link add "$HOST_VETH" mtu "$MTU" type veth peer name "${VB}host" mtu "$MTU"
	ledger "host-veth $HOST_VETH"
	ip link set "${VB}host" netns "$NS_BR"
	ip -n "$NS_BR" link set "${VB}host" master br0 up
	ip addr add "$IP_HOST/$NETMASK" dev "$HOST_VETH"
	ip link set "$HOST_VETH" up
	local ns
	for ns in $(ip netns list | awk '{print $1}' | grep "^$NS_PREFIX-" | grep -v "^$NS_BR\$"); do
		ip -n "$ns" route replace default via "$IP_HOST"
	done
	iptables -I DOCKER-USER -s "$NET.0/$NETMASK" -j ACCEPT
	iptables -I DOCKER-USER -d "$NET.0/$NETMASK" -j ACCEPT
	ledger "iptables DOCKER-USER $NET.0/$NETMASK"
	echo "remote path up: route $NET.0/$NETMASK via this host's LAN address"
}

remote_down() {
	while iptables -C DOCKER-USER -s "$NET.0/$NETMASK" -j ACCEPT 2>/dev/null; do
		iptables -D DOCKER-USER -s "$NET.0/$NETMASK" -j ACCEPT
	done
	while iptables -C DOCKER-USER -d "$NET.0/$NETMASK" -j ACCEPT 2>/dev/null; do
		iptables -D DOCKER-USER -d "$NET.0/$NETMASK" -j ACCEPT
	done
	ip link show "$HOST_VETH" >/dev/null 2>&1 && ip link del "$HOST_VETH"
	return 0
}

down() {
	save_logs
	remote_down
	local u
	clients_down
	for u in cli mds rondb; do
		systemctl stop "$UNIT_PREFIX-$u" 2>/dev/null || true
	done
	for u in $(systemctl list-units --all --plain --no-legend "$UNIT_PREFIX-ds*" | awk '{print $1}'); do
		systemctl stop "$u" || true
	done
	local ns
	for ns in $(ip netns list | awk '{print $1}' | grep "^$NS_PREFIX-" || true); do
		ip netns del "$ns"
		ledger "netns-del $ns"
	done
	baseline after
	local leak=0
	if findmnt -rn -o TARGET | grep -q "^$R"; then
		echo "LEAK: mounts under $R remain:" >&2
		findmnt -rn -o TARGET | grep "^$R" >&2
		leak=1
	fi
	if systemctl list-units --all --plain --no-legend "$UNIT_PREFIX-*" | grep -q .; then
		echo "LEAK: units remain" >&2
		leak=1
	fi
	for f in rpcinfo exportfs etc-exports nfsd-threads iptables; do
		if ! cmp -s "$RUN/host-$f.before" "$RUN/host-$f.after"; then
			echo "HOST CHANGE: $f differs from baseline" >&2
			diff "$RUN/host-$f.before" "$RUN/host-$f.after" >&2 || true
			leak=1
		fi
	done
	ip -br link | grep -E "^($VA|$VB)" && leak=1
	[ "$leak" = 0 ] && echo "TEARDOWN CLEAN" && ledger "teardown clean"
	return "$leak"
}

status() {
	systemctl list-units --all --plain --no-legend "$UNIT_PREFIX-*" || true
	ip netns list | grep "^$NS_PREFIX-" || true
	for u in $(systemctl list-units --plain --no-legend "$UNIT_PREFIX-*" | awk '{print $1}'); do
		printf '%s memory.current=%s\n' "$u" \
			"$(systemctl show -p MemoryCurrent --value "$u")"
	done
}

cli_pid() { cat "$RUN/$CLI/holder.pid"; }

remount() {
	systemctl stop "$UNIT_PREFIX-cli"
	for _ in $(seq 30); do
		systemctl list-units --all --plain --no-legend "$UNIT_PREFIX-cli.service" | grep -q . || break
		sleep 1
	done
	run_unit "$UNIT_PREFIX-cli" "$MEM_CLI" ip netns exec "$NS_CLI" \
		unshare -m --propagation private "$BIN/cli-inner.sh" "$@"
	wait_ready "$UNIT_PREFIX-cli" 60
}

stop_unit() {
	systemctl stop "$1" || true
	for _ in $(seq 30); do
		systemctl list-units --all --plain --no-legend "$1.service" | grep -q . || return 0
		sleep 1
	done
	echo "unit $1 did not stop" >&2
	return 1
}

start_ds() {
	run_unit "$UNIT_PREFIX-ds$1" "$MEM_DS" ip netns exec "$(ds_ns "$1")" \
		unshare -m --propagation private "$BIN/ds-inner.sh" "$1"
	wait_ready "$UNIT_PREFIX-ds$1" 60
}

start_mds() {
	local extra
	extra=$(cat "$RUN/mds-extra")
	run_unit "$UNIT_PREFIX-mds" "$MEM_MDS" ip netns exec "$NS_MDS" \
		unshare -m --propagation private "$BIN/mds-inner.sh" "$(cat "$RUN/nds")" $extra
	for _ in $(seq 60); do
		ip netns exec "$NS_MDS" ss -ltn | grep -q ":2049 " && return 0
		sleep 1
	done
	echo "MDS not listening" >&2
	return 1
}

case ${1:-} in
up) shift; up "$@" ;;
remote-up) remote_up ;;
clients-up) clients_up "$2" ;;
clients-down) clients_down ;;
remote-down) remote_down ;;
stop-ds) stop_unit "$UNIT_PREFIX-ds$2" ;;
start-ds) start_ds "$2" ;;
stop-mds) stop_unit "$UNIT_PREFIX-mds" ;;
start-mds) start_mds ;;
remount) shift; remount "$@" ;;
down) down ;;
status) status ;;
cli)
	shift
	exec nsenter -t "$(cli_pid)" -n -m setpriv --reuid="$OWNER" --regid="$OWNER" \
		--init-groups --reset-env -- "$@"
	;;
cliroot) shift; exec nsenter -t "$(cli_pid)" -n -m "$@" ;;
mdsns) shift; exec ip netns exec "$NS_MDS" "$@" ;;
dsns) shift; i=$1; shift; exec ip netns exec "$(ds_ns "$i")" "$@" ;;
*) sed -n 2,13p "$0"; exit 2 ;;
esac
