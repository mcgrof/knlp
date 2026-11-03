#!/bin/bash
# Measure LMCache's native file connector on a filesystem you already run,
# as an ordinary user. Nothing is mounted, remounted, exported or
# configured, no cache is dropped, and every file goes into one
# subdirectory, rpl2-site, of the directory you give. Remove that
# subdirectory to undo a run.
#
#   site.sh <step> <out-dir>
#
# Steps (each writes JSON records under <out-dir>/<step>):
#   doctor       filesystem type, NFS mount options and pNFS layout driver,
#                free space, O_DIRECT and RWF_NOWAIT probes
#   fixture      seal 64 x 8 MiB and 64 x 256 KiB objects (once)
#   fingerprint  NFS RPCs per object, first touch and repeat, stock and held
#   screen       paired blocks: stock (held_fd_capacity 0) against held,
#                1 x 8 MiB and 16 x 8 MiB per request
#   sweep        N reader processes on this host at once, stock and held
#   readmodes    read_mode pool, inline and adaptive at each in-flight depth
#   kernel       the nfs_open_scale selftest, when its binary is given
# Settings come from the environment (the Makefile sets them):
#   RPL2P_ROOT  root built by setup.sh (python, LMCache, helpers)
#   SITE_DIR    existing writable directory on the filesystem under test
#   CAP BLOCKS REQS CLIENTS SECS INFLIGHT MODES BUSY       run shape
#   MDS_METRICS optional Lattice MDS metrics URL, read before and after
#   DATA_SERVERS DRIVES LABEL                              recorded only
#   KERNEL_SELFTEST KERNEL_THREADS                         kernel step
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
step=${1:?step}
OUT=${2:?out dir}
R=${RPL2P_ROOT:?}
SITE_DIR=${SITE_DIR:?set the directory under test}
CAP=${CAP:-64}
BLOCKS=${BLOCKS:-3}
REQS=${REQS:-200}
CLIENTS=${CLIENTS:-1 2 4 8 16}
SECS=${SECS:-20}
INFLIGHT=${INFLIGHT:-1 4 16}
MODES=${MODES:-pool inline adaptive}
BUSY=${BUSY:-16}
W=$SITE_DIR/rpl2-site
PY=$(cat "$R/python")
TORCH_LIB=$("$PY" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
reader() {
	LD_LIBRARY_PATH="$TORCH_LIB" "$PY" "$here/native_reader.py" --src "$R/src/lmcache" "$@"
}
metrics() {
	[ -n "${MDS_METRICS:-}" ] || return 0
	curl -fsS --max-time 5 "$MDS_METRICS" | grep -E '^pnfs_mds_nfs_op\{' >"$1" || true
}
d=$OUT/$step
[ -e "$d" ] && { echo "$d exists; runs are preserved" >&2; exit 1; }
[ -d "$SITE_DIR" ] || { echo "SITE_DIR must be an existing directory" >&2; exit 1; }
# Probe by creating a file: on NFS, access(2) can report a directory as
# read-only while the server accepts creates in it.
{ mkdir -p "$W" && : >"$W/.probe" && rm -f "$W/.probe"; } 2>/dev/null ||
	{ echo "cannot create files in $W" >&2; exit 1; }
mkdir -p "$d"

case $step in
doctor)
	mkdir -p "$W"
	HERE="$here" "$PY" - "$W" "$d/doctor.json" "${DATA_SERVERS:-}" "${DRIVES:-}" "${LABEL:-}" <<-'EOF'
	import json, os, shutil, subprocess, sys
	sys.path.insert(0, os.environ["HERE"])
	import mountstats
	w, out, ds, drives, label = sys.argv[1:6]
	rec = {"dir": w, "label": label, "data_servers": ds, "drives": drives}
	rec["fstype"] = subprocess.run(["stat", "-f", "-c", "%T", w], capture_output=True, text=True).stdout.strip()
	rec["free_bytes"] = shutil.disk_usage(w).free
	rec["need_bytes"] = 64 * (8 << 20) + 64 * (256 << 10)
	ms = mountstats.snapshot(w)
	rec["nfs_header"] = ms["header"] if ms else None
	probe = os.path.join(w, "probe")
	with open(probe, "wb") as f:
	    f.write(os.urandom(1 << 20))
	    f.flush()
	    os.fsync(f.fileno())
	for name, flags in (("o_direct", os.O_RDONLY | os.O_DIRECT), ("buffered", os.O_RDONLY)):
	    try:
	        fd = os.open(probe, flags)
	        import mmap
	        buf = mmap.mmap(-1, 1 << 20)
	        os.preadv(fd, [buf], 0)
	        if name == "buffered":
	            try:
	                os.preadv(fd, [buf], 0, os.RWF_NOWAIT)
	                rec["rwf_nowait"] = "accepted"
	            except OSError as e:
	                rec["rwf_nowait"] = f"refused: errno {e.errno} {e.strerror}"
	        os.close(fd)
	        rec[name] = "ok"
	    except OSError as e:
	        rec[name] = f"failed: errno {e.errno} {e.strerror}"
	os.unlink(probe)
	json.dump(rec, open(out, "w"), indent=1)
	print(json.dumps(rec, indent=1))
	if rec["free_bytes"] < 2 * rec["need_bytes"]:
	    sys.exit("not enough free space for the fixtures")
	EOF
	;;
fixture)
	for spec in "8388608 rpl2-8m" "262144 rpl2-256k"; do
		set -- $spec
		if [ -e "$W/$2/RPL2-MANIFEST.json" ]; then
			echo "fixture $2 already sealed" | tee -a "$d/fixture.log"
		else
			reader init --base "$W/$2" --size "$1" --count 64 --seed 1 | tee -a "$d/fixture.log"
		fi
	done
	;;
fingerprint)
	for held in 0 "$CAP"; do
		reader fingerprint --base "$W/rpl2-8m" --hot 8 --batch 1 --requests 100 \
			--held "$held" --mountstats --client site --out "$d/held$held.json"
	done
	;;
screen)
	for block in $(seq 1 "$BLOCKS"); do
		if [ $((block % 2)) = 1 ]; then order=(0 "$CAP"); else order=("$CAP" 0); fi
		for shape in b1 b16; do
			case $shape in b1) hot=8 batch=1 ;; b16) hot=16 batch=16 ;; esac
			for held in "${order[@]}"; do
				reader read --base "$W/rpl2-8m" --hot "$hot" --batch "$batch" --requests "$REQS" \
					--held "$held" --block "$block" --mountstats --client site \
					--out "$d/$shape-held$held-k$block.json" | tee -a "$d/screen.log"
			done
		done
	done
	python3 "$here/summarize.py" screen "$d"
	;;
sweep)
	pos=0
	for n in $CLIENTS; do
		pos=$((pos + 1))
		if [ $((pos % 2)) = 1 ]; then order=(0 "$CAP"); else order=("$CAP" 0); fi
		for held in "${order[@]}"; do
			cell=$d/n$n-held$held
			mkdir -p "$cell"
			metrics "$cell/mds-ops.before"
			start=$(($(date +%s) + 3))
			pids=()
			for k in $(seq 1 "$n"); do
				reader read --base "$W/rpl2-256k" --hot 16 --batch 16 --held "$held" \
					--start-at "$start" --seconds "$SECS" --client "p$k" \
					$([ "$k" = 1 ] && echo --mountstats) \
					--out "$cell/c$k.json" >"$cell/c$k.log" 2>&1 &
				pids+=($!)
			done
			rc=0
			for p in "${pids[@]}"; do wait "$p" || rc=1; done
			metrics "$cell/mds-ops.after"
			echo "$(date -Is) n=$n held=$held rc=$rc" | tee -a "$d/sweep.log"
			[ "$rc" = 0 ] || { echo "a reader failed in $cell" >&2; exit 1; }
		done
	done
	;;
readmodes)
	# Page-cache reads of 16 hot 256 KiB objects: one pass warms the
	# cache, then each mode runs at each depth with held descriptors, so
	# only the read mode differs.  Bytes are verified after each timed
	# window, so hashing does not compete with the copies being timed.
	reader read --base "$W/rpl2-256k" --hot 16 --batch 16 --requests 4 --odirect 0 \
		--held "$CAP" --client warm --out "$d/warm.json" >/dev/null
	for p in $INFLIGHT; do
		for mode in $MODES; do
			reader read --base "$W/rpl2-256k" --hot 16 --batch 16 --seconds "$SECS" \
				--odirect 0 --held "$CAP" --read-mode "$mode" --busy-target "$BUSY" \
				--pipeline "$p" --verify-every 0 --mountstats --client site \
				--out "$d/$mode-p$p.json" | tee -a "$d/readmodes.log"
		done
	done
	;;
kernel)
	if [ -z "${KERNEL_SELFTEST:-}" ]; then
		echo '{"status": "skipped", "reason": "no kernel selftest given"}' >"$d/kernel.json"
		echo "kernel step skipped: set KERNEL_SELFTEST to the selftest binary"
		exit 0
	fi
	# One shared credential, as an LMCache reader pool has.  The mode with
	# a credential per thread (-u) needs root; run it by hand as root.
	mkdir -p "$W/opens"
	"$KERNEL_SELFTEST" -d "$W/opens" -t "${KERNEL_THREADS:-1,2,4,8}" -f 64 -s 4 |
		tee "$d/kernel.log"
	uname -r >"$d/kernel-release"
	;;
*)
	echo "unknown step $step" >&2
	exit 1
	;;
esac
