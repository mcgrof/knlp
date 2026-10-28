#!/bin/bash
# A pNFS client on another host for the lab on RPL2P_LAB_HOST. Run as root
# on the client host, with the same RPL2P_PREFIX and RPL2P_NET as the lab
# and RPL2P_ROOT set to a directory on this host:
#   remote-client.sh up <lab-host-lan-ip> [mount options]
#   remote-client.sh cli <cmd...>     run as the owner in the mount namespace
#   remote-client.sh down
# `up` adds a route to the lab subnet via the lab host and holds the NFSv4.2
# mount in a private mount namespace under a transient unit, so nothing is
# mounted in the host namespace. `down` removes both.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
U=$UNIT_PREFIX-rcli
D=$RUN/rcli
MNT=$D/mnt
case ${1:-} in
up)
	lab=${2:?lab host LAN address}
	opts=${3:-vers=4.2,sec=sys,hard,proto=tcp,rsize=1048576,wsize=1048576}
	mkdir -p "$MNT"
	chown "$OWNER": "$RUN" "$D" 2>/dev/null || true
	ip route replace "$NET.0/$NETMASK" via "$lab"
	echo "$lab" >"$D/lab-host"
	systemd-run --unit="$U" --collect -p MemoryMax=4G -p KillMode=mixed \
		unshare -m --propagation private "$here/remote-holder.sh" "$MNT" "$IP_MDS" "$opts" "$D/holder.pid" >/dev/null
	inv=$(systemctl show -p InvocationID --value "$U")
	for _ in $(seq 60); do
		journalctl _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | grep -qx READY && break
		systemctl is-active -q "$U" || break
		sleep 1
	done
	journalctl _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | grep -qx READY || {
		journalctl _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | tail -20
		exit 1
	}
	echo "remote client up: $MNT (pid $(cat "$D/holder.pid"))"
	;;
cli)
	shift
	exec nsenter -t "$(cat "$D/holder.pid")" -m setpriv --reuid="$OWNER" --regid="$OWNER" \
		--init-groups --reset-env -- "$@"
	;;
cliroot)
	shift
	exec nsenter -t "$(cat "$D/holder.pid")" -m "$@"
	;;
down)
	systemctl stop "$U" 2>/dev/null || true
	ip route del "$NET.0/$NETMASK" 2>/dev/null || true
	findmnt -rn -o TARGET | grep -q "^$MNT" && { echo "LEAK: $MNT still mounted" >&2; exit 1; }
	ip -4 route | grep -q "^$NET\\." && { echo "LEAK: route remains" >&2; exit 1; }
	echo "REMOTE CLIENT DOWN CLEAN"
	;;
*)
	sed -n 2,10p "$0"
	exit 2
	;;
esac
