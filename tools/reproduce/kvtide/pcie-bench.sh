#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Compare read-only raw-PCIe and kernel NVMe paths on the same controllers.

. "$(dirname "$0")/lib.sh"

is_pcie || kvtide_die "select CONFIG_KVTIDE_MODE_PCIE"
[ "${CONFIG_KVTIDE_PCIE_ALLOW_REBIND:-n}" = y ] || \
	kvtide_die "CONFIG_KVTIDE_PCIE_ALLOW_REBIND must be enabled"
[ -n "${CONFIG_KVTIDE_PCIE_BDFS:-}" ] || \
	kvtide_die "CONFIG_KVTIDE_PCIE_BDFS is empty"
[ "${CONFIG_KVTIDE_PCIE_NSID}" = 1 ] || \
	kvtide_die "current xnvmeperf parity mode requires namespace 1"

PCIE_BIND="$KVTIDE_DIR/pcie-bind.sh"
SPDK_PERF="$PCIE_SPDK_SRC/build/bin/spdk_nvme_perf"
XNVME_PERF=""
URING_PERF="$KVTIDE_SRC/uring_nvm_perf"
SYSFS_ROOT=${KVTIDE_SYSFS_ROOT:-/sys}
CLK_TCK=$(getconf CLK_TCK)
RUN_DIR="$KVTIDE_RESULTS/kvtide-pcie-$(date +%Y%m%d-%H%M%S)"
CSV="$RUN_DIR/results.csv"
RAW_DIR="$RUN_DIR/raw"
KERNEL_DEVS=""
NAMESPACE_BYTES=""
LBA_SIZE=""
NORMAL_MAX_BYTES=""
DEVICE_COUNT=$(wc -w <<< "${CONFIG_KVTIDE_PCIE_BDFS}")
ORIGINAL_HUGEPAGES=$(cat /proc/sys/vm/nr_hugepages)

mkdir -p "$RAW_DIR"
if want_pcie_spdk; then
	[ -x "$SPDK_PERF" ] || \
		kvtide_die "spdk_nvme_perf is missing; run make kvtide-build"
fi
if want_pcie_upcie || want_pcie_linux; then
	XNVME_PERF=$(pcie_xnvmeperf)
	[ -x "$XNVME_PERF" ] || \
		kvtide_die "xnvmeperf is missing; run make kvtide-build"
fi
if want_pcie_linux; then
	[ -x "$URING_PERF" ] || \
		kvtide_die "uring_nvm_perf is missing; run make kvtide-build"
fi

restore_kernel_driver() {
	local status=$?

	trap - EXIT
	if ! "$PCIE_BIND" kernel; then
		echo "kvtide: CRITICAL: failed to return every NVMe controller to the kernel" >&2
		status=1
	fi
	if ! printf '%s\n' "$ORIGINAL_HUGEPAGES" | \
		sudo tee /proc/sys/vm/nr_hugepages >/dev/null; then
		echo "kvtide: CRITICAL: failed to restore the hugepage count" >&2
		status=1
	fi
	exit "$status"
}

trap restore_kernel_driver EXIT

driver_name() {
	local path="$SYSFS_ROOT/bus/pci/devices/$1/driver"

	[ -L "$path" ] && basename "$(readlink -f "$path")" || echo unbound
}

ng_for_bdf() {
	local bdf=$1 entry path name

	for entry in "$SYSFS_ROOT"/class/nvme-generic/ng*; do
		[ -e "$entry" ] || continue
		path=$(readlink -f "$entry/device")
		name=${entry##*/}
		case "$path" in
		*"/$bdf/"*)
			case "$name" in
			ng*n"${CONFIG_KVTIDE_PCIE_NSID}")
				echo "/dev/$name"
				return 0
				;;
			esac
			;;
		esac
	done
	return 1
}

block_for_ng() {
	local ng=${1##*/}

	echo "/dev/nvme${ng#ng}"
}

cpu_list() {
	local wanted=$1 cpu out="" count=0

	for cpu in ${CONFIG_KVTIDE_PCIE_CPUS}; do
		out="${out}${out:+,}$cpu"
		count=$((count + 1))
		[ "$count" -eq "$wanted" ] && {
			echo "$out"
			return 0
		}
	done
	return 1
}

cpu_mask() {
	python3 - "$1" <<'PY'
import sys

mask = 0
for cpu in sys.argv[1].split(","):
    mask |= 1 << int(cpu)
print(hex(mask))
PY
}

validate_bench_config() {
	local cpu online profile nth threads size sizes qd qds
	local seen=" " cpu_count=0
	local total_queues=$((DEVICE_COUNT * CONFIG_KVTIDE_PCIE_NQUEUES))

	for cpu in ${CONFIG_KVTIDE_PCIE_CPUS}; do
		case "$cpu" in
		''|*[!0-9]*) kvtide_die "invalid logical CPU '$cpu'" ;;
		esac
		case "$seen" in
		*" $cpu "*) kvtide_die "logical CPU $cpu appears more than once" ;;
		esac
		[ -d "$SYSFS_ROOT/devices/system/cpu/cpu$cpu" ] || \
			kvtide_die "logical CPU $cpu does not exist"
		online="$SYSFS_ROOT/devices/system/cpu/cpu$cpu/online"
		[ ! -r "$online" ] || [ "$(cat "$online")" = 1 ] || \
			kvtide_die "logical CPU $cpu is offline"
		seen="$seen$cpu "
		cpu_count=$((cpu_count + 1))
	done
	[ "$cpu_count" -gt 0 ] || kvtide_die "CONFIG_KVTIDE_PCIE_CPUS is empty"
	if ! want_pcie_upcie && ! want_pcie_spdk && ! want_pcie_linux; then
		kvtide_die "enable at least one physical PCIe benchmark arm"
	fi
	if want_vfio && want_pcie_upcie; then
		case "${CONFIG_KVTIDE_PCIE_XNVME_VFIO_MODE:-auto}" in
		auto|iommufd|type1) ;;
		*) kvtide_die "xNVMe VFIO mode must be auto, iommufd or type1" ;;
		esac
	fi
	if want_pcie_fixed || want_pcie_premap; then
		[ "${CONFIG_KVTIDE_PCIE_NQUEUES}" -eq 1 ] || \
			kvtide_die "fixed and premap arms require one queue per controller"
	fi
	[ -n "${CONFIG_KVTIDE_PCIE_PROFILES}" ] || \
		kvtide_die "CONFIG_KVTIDE_PCIE_PROFILES is empty"
	for profile in ${CONFIG_KVTIDE_PCIE_PROFILES}; do
		case "$profile" in
		aisio)
			threads=${CONFIG_KVTIDE_PCIE_AISIO_THREADS}
			sizes=${CONFIG_KVTIDE_PCIE_AISIO_SIZES}
			qds=${CONFIG_KVTIDE_PCIE_AISIO_QDS}
			;;
		kv)
			threads=${CONFIG_KVTIDE_PCIE_KV_THREADS}
			sizes=${CONFIG_KVTIDE_PCIE_KV_SIZES}
			qds=${CONFIG_KVTIDE_PCIE_KV_QDS}
			;;
		*) kvtide_die "unknown physical PCIe profile '$profile'" ;;
		esac
		[ -n "$threads" ] && [ -n "$sizes" ] && [ -n "$qds" ] || \
			kvtide_die "$profile has an empty size, depth, or thread list"
		for nth in $threads; do
			case "$nth" in
			''|*[!0-9]*|0) kvtide_die "invalid $profile thread count '$nth'" ;;
			esac
			[ "$nth" -le "$cpu_count" ] || \
				kvtide_die "$profile asks for $nth threads but only $cpu_count CPUs are configured"
			[ "$nth" -le "$total_queues" ] || \
				kvtide_die "$profile asks for $nth threads but only $total_queues queues are configured"
		done
		for size in $sizes; do
			case "$size" in
			''|*[!0-9]*|0) kvtide_die "invalid $profile I/O size '$size'" ;;
			esac
		done
		for qd in $qds; do
			case "$qd" in
			''|*[!0-9]*|0) kvtide_die "invalid $profile queue depth '$qd'" ;;
			esac
			if want_pcie_upcie || want_pcie_linux; then
				[ $((qd & (qd - 1))) -eq 0 ] || \
					kvtide_die "xnvmeperf requires a power-of-two queue depth, not $qd"
			fi
		done
	done
}

busy_jiffies() {
	awk '/^cpu[0-9]+ / {busy += $2+$3+$4+$7+$8+$9} END {print busy}' \
		/proc/stat
}

record_metadata() {
	local bdf ng block bytes lba max_bytes cpu path

	cp "$KVTIDE_TOP/.config" "$RUN_DIR/knlp.config"
	{
		echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "kernel=$(uname -r)"
		echo "kernel_commit=$(cat /proc/version)"
		echo "command_line=$(cat /proc/cmdline)"
		echo "xnvme_ref=${CONFIG_KVTIDE_PCIE_XNVME_REF}"
		echo "xnvme_commit=$(git -C "$PCIE_XNVME_SRC" rev-parse HEAD 2>/dev/null || echo unavailable)"
		echo "spdk_ref=${CONFIG_KVTIDE_PCIE_SPDK_REF}"
		echo "spdk_commit=$(git -C "$PCIE_SPDK_SRC" rev-parse HEAD 2>/dev/null || echo unavailable)"
		echo "userspace_driver=$(want_vfio && echo vfio-pci || echo uio_pci_generic)"
		echo "xnvme_vfio_mode=${CONFIG_KVTIDE_PCIE_XNVME_VFIO_MODE:-auto}"
		echo "goal=Evaluate whether Linux premap can match or beat current SPDK"
		echo "evidence=No performance conclusion is implied by this configuration"
	} > "$RUN_DIR/run.meta"
	lscpu > "$RUN_DIR/lscpu.txt"
	numactl --hardware > "$RUN_DIR/numa.txt"
	lspci -Dnnk > "$RUN_DIR/lspci.txt"
	lspci -Dtv > "$RUN_DIR/lspci-tree.txt"
	lsblk -O > "$RUN_DIR/lsblk.txt"
	if want_pcie_cuda; then
		nvidia-smi -q > "$RUN_DIR/nvidia-smi.txt"
		nvidia-smi topo -m > "$RUN_DIR/nvidia-topology.txt"
	fi
	{
		while read -r bdf ng block bytes lba max_bytes; do
			echo "### $bdf"
			echo "driver=$(driver_name "$bdf")"
			echo "namespace=$ng bytes=$bytes lba=$lba ordinary_max_bytes=$max_bytes"
			sudo nvme id-ctrl "$block" 2>&1 || true
		done < "$RUN_DIR/namespaces.tsv"
	} > "$RUN_DIR/nvme-id-ctrl.txt"
	{
		for cpu in ${CONFIG_KVTIDE_PCIE_CPUS}; do
			path="$SYSFS_ROOT/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor"
			printf 'cpu%s=' "$cpu"
			[ ! -r "$path" ] || cat "$path"
			[ -r "$path" ] || echo unavailable
		done
	} > "$RUN_DIR/cpu-governors.txt"
}

record_device_state() {
	local label=$1 bdf ng block bytes lba max_bytes

	cp /proc/interrupts "$RUN_DIR/interrupts-$label.txt"
	while read -r bdf ng block bytes lba max_bytes; do
		sudo nvme smart-log "$block" \
			> "$RUN_DIR/smart-${label}-${bdf//:/_}.txt" 2>&1 || true
	done < "$RUN_DIR/namespaces.tsv"
}

want_vfio() {
	[ "${CONFIG_KVTIDE_PCIE_DRIVER_VFIO:-n}" = y ]
}

validate_driver_environment() {
	local bdf group groups=0

	if want_vfio; then
		for bdf in ${CONFIG_KVTIDE_PCIE_BDFS}; do
			[ -L "$SYSFS_ROOT/bus/pci/devices/$bdf/iommu_group" ] || \
				kvtide_die "$bdf has no IOMMU group for vfio-pci"
		done
		return
	fi
	for group in "$SYSFS_ROOT"/kernel/iommu_groups/*; do
		[ ! -e "$group" ] || groups=$((groups + 1))
	done
	case " $(cat /proc/cmdline) " in
	*" iommu=off "*|*" intel_iommu=off "*|*" amd_iommu=off "*) return ;;
	esac
	[ "$groups" -eq 0 ] || \
		kvtide_die "UIO parity mode requires an IOMMU-disabled boot"
}

inventory_kernel_devices() {
	local bdf ng block bytes lba max_bytes
	local first_bytes="" first_lba="" first_max_bytes=""

	KERNEL_DEVS=""
	: > "$RUN_DIR/namespaces.tsv"
	for bdf in ${CONFIG_KVTIDE_PCIE_BDFS}; do
		ng=$(ng_for_bdf "$bdf") || \
			kvtide_die "cannot resolve namespace ${CONFIG_KVTIDE_PCIE_NSID} for $bdf"
		block=$(block_for_ng "$ng")
		[ -b "$block" ] || kvtide_die "$block for $ng is not a block device"
		bytes=$(sudo blockdev --getsize64 "$block")
		lba=$(sudo blockdev --getss "$block")
		max_bytes=$(($(cat "$SYSFS_ROOT/class/block/${block##*/}/queue/max_hw_sectors_kb") * 1024))
		[ -z "$first_bytes" ] || [ "$bytes" = "$first_bytes" ] || \
			kvtide_die "all namespaces must have the same size"
		[ -z "$first_lba" ] || [ "$lba" = "$first_lba" ] || \
			kvtide_die "all namespaces must have the same LBA size"
		[ -z "$first_max_bytes" ] || [ "$max_bytes" = "$first_max_bytes" ] || \
			kvtide_die "all namespaces must have the same ordinary I/O limit"
		first_bytes=$bytes
		first_lba=$lba
		first_max_bytes=$max_bytes
		KERNEL_DEVS="$KERNEL_DEVS $ng"
		printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
			"$bdf" "$ng" "$block" "$bytes" "$lba" "$max_bytes" \
			>> "$RUN_DIR/namespaces.tsv"
		sudo nvme id-ns -H "$block" > "$RUN_DIR/id-ns-${bdf//:/_}.txt"
	done
	KERNEL_DEVS=${KERNEL_DEVS# }
	NAMESPACE_BYTES=$first_bytes
	LBA_SIZE=$first_lba
	NORMAL_MAX_BYTES=$first_max_bytes
}

validate_premap() {
	local cpu probe log parsed iops mibps failed
	local -a command

	# The allocation command can fall back to an ordinary fixed buffer when
	# it cannot retain an IOVA.  Complete one read beyond the ordinary queue
	# limit so a run labeled "premap" cannot silently measure that fallback.
	probe=$(( ((NORMAL_MAX_BYTES / LBA_SIZE) + 1) * LBA_SIZE ))
	[ "$probe" -le "$NAMESPACE_BYTES" ] || \
		kvtide_die "namespace is too small to validate premapping"
	cpu=$(cpu_list 1) || kvtide_die "the CPU list is empty"
	log="$RUN_DIR/premap-validation.log"
	command=(sudo "$URING_PERF" --mode premap --qd 1
		--iosize "$probe" --lba-size "$LBA_SIZE"
		--namespace-bytes "$NAMESPACE_BYTES" --seconds 1 --cpus "$cpu")
	for dev in $KERNEL_DEVS; do command+=("$dev"); done
	{
		printf 'ordinary_max_bytes=%s\n' "$NORMAL_MAX_BYTES"
		printf 'probe_bytes=%s\n' "$probe"
		printf 'command:'
		printf ' %q' "${command[@]}"
		printf '\n'
	} > "$RUN_DIR/premap-validation.meta"
	"${command[@]}" > "$log" 2>&1 || \
		kvtide_die "premap validation failed; inspect $log"
	parsed=$(parse_uring "$log")
	read -r iops mibps failed <<< "$parsed"
	[ -n "${iops:-}" ] && [ "${failed:-1}" = 0 ] || \
		kvtide_die "premap validation returned failed I/O; inspect $log"
	kvtide_log "premap crossed the ordinary $NORMAL_MAX_BYTES-byte limit with a $probe-byte read"
}

parse_xnvme() {
	awk '/^[[:space:]]+Total:/ {print $(NF-2), $(NF-1), $NF}' "$1" | tail -1
}

parse_spdk() {
	awk '/^Total / {print $3, $4, 0}' "$1" | tail -1
}

parse_uring() {
	awk '
		/^mode=/ {
			for (i=1; i<=NF; i++) {
				split($i, f, "="); v[f[1]]=f[2]
			}
			print v["iops"], v["MiBps"], v["failed"]
		}' "$1" | tail -1
}

run_command() {
	local log=$1; shift
	local b0 b1 t0 t1 wall cores status=0 parsed iops mibps failed

	b0=$(busy_jiffies)
	t0=$(date +%s.%N)
	"$@" > "$log" 2>&1 || status=$?
	t1=$(date +%s.%N)
	b1=$(busy_jiffies)
	wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.6f", b-a}')
	cores=$(awk -v a="$b0" -v b="$b1" -v w="$wall" -v hz="$CLK_TCK" \
		'BEGIN {printf "%.6f", ((b-a)/hz)/w}')
	case "$CURRENT_ARM" in
	upcie|upcie-cuda|linux) parsed=$(parse_xnvme "$log") ;;
	spdk) parsed=$(parse_spdk "$log") ;;
	fixed|premap) parsed=$(parse_uring "$log") ;;
	esac
	read -r iops mibps failed <<< "$parsed"
	if [ "$status" -ne 0 ] || [ -z "${iops:-}" ]; then
		iops=ERR mibps=ERR failed=ERR
	elif ! [[ "$failed" =~ ^[0-9]+$ ]] || [ "$failed" -ne 0 ]; then
		status=1
	fi
	printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
		"$CURRENT_PROFILE" "$CURRENT_ARM" "$CURRENT_TOOL" \
		"$CURRENT_DRIVER" randread \
		"$CURRENT_SIZE" "$CURRENT_QD" "$CURRENT_THREADS" "$CURRENT_REP" \
		"$iops" "$mibps" "$failed" "$wall" "$cores" >> "$CSV"
	return "$status"
}

run_cell() {
	local arm=$1 size=$2 qd=$3 threads=$4 rep=$5 cpus mask log
	local -a command

	case "$arm" in
	linux|fixed)
		if [ "$size" -gt "$NORMAL_MAX_BYTES" ]; then
			kvtide_log "skip $arm: $size bytes exceeds the ordinary \
$NORMAL_MAX_BYTES-byte limit"
			return
		fi
		;;
	esac

	CURRENT_ARM=$arm
	CURRENT_SIZE=$size
	CURRENT_QD=$qd
	CURRENT_THREADS=$threads
	CURRENT_REP=$rep
	cpus=$(cpu_list "$threads") || \
		kvtide_die "$threads CPUs requested but the configured list is too short"
	mask=$(cpu_mask "$cpus")
	log="$RAW_DIR/${CURRENT_PROFILE}-${arm}-${size}-q${qd}-t${threads}-r${rep}.log"

	case "$arm" in
	upcie|upcie-cuda)
		CURRENT_TOOL=xnvmeperf
		command=(sudo)
		if want_vfio && \
		   [ "${CONFIG_KVTIDE_PCIE_XNVME_VFIO_MODE:-auto}" != auto ]; then
			command+=(env "XNVME_UPCIE_VFIO_MODE=${CONFIG_KVTIDE_PCIE_XNVME_VFIO_MODE}")
		fi
		command+=("$XNVME_PERF" run --iopattern randread
			--qdepth "$qd" --iosize "$size" --runtime "${CONFIG_KVTIDE_PCIE_SECS}"
			--cpumask "$mask" --nqueues "${CONFIG_KVTIDE_PCIE_NQUEUES}"
			--be "$arm")
		[ "$arm" != upcie-cuda ] || \
			command+=(--gpu_id "${CONFIG_KVTIDE_PCIE_GPU_ID:-0}")
		for bdf in ${CONFIG_KVTIDE_PCIE_BDFS}; do command+=("$bdf"); done
		;;
	spdk)
		CURRENT_TOOL=spdk_nvme_perf
		command=(sudo "$SPDK_PERF" -q "$qd" -o "$size" -w randread
			-t "${CONFIG_KVTIDE_PCIE_SECS}" -c "[$cpus]")
		for bdf in ${CONFIG_KVTIDE_PCIE_BDFS}; do
			command+=(-r "trtype:PCIe traddr:$bdf ns:${CONFIG_KVTIDE_PCIE_NSID}")
		done
		;;
	linux)
		CURRENT_TOOL=xnvmeperf
		command=(sudo "$XNVME_PERF" run --iopattern randread
			--qdepth "$qd" --iosize "$size" --runtime "${CONFIG_KVTIDE_PCIE_SECS}"
			--cpumask "$mask" --nqueues "${CONFIG_KVTIDE_PCIE_NQUEUES}"
			--be linux --direct)
		for dev in $KERNEL_DEVS; do command+=("$dev"); done
		;;
	fixed|premap)
		CURRENT_TOOL=uring_nvm_perf
		if [ "${CONFIG_KVTIDE_PCIE_NQUEUES}" -ne 1 ]; then
			kvtide_log "skip $arm: uring_nvm_perf currently requires one queue per device"
			return
		fi
		if [ "$threads" -gt "$(wc -w <<< "$KERNEL_DEVS")" ]; then
			kvtide_log "skip $arm: thread count exceeds device count"
			return
		fi
		command=(sudo "$URING_PERF" --mode "$arm" --qd "$qd"
			--iosize "$size" --lba-size "$LBA_SIZE"
			--namespace-bytes "$NAMESPACE_BYTES"
			--seconds "${CONFIG_KVTIDE_PCIE_SECS}" --cpus "$cpus")
		for dev in $KERNEL_DEVS; do command+=("$dev"); done
		;;
	esac

	{
		printf 'command:'
		printf ' %q' "${command[@]}"
		printf '\n'
	} > "$log.command"
	if ! run_command "$log" "${command[@]}"; then
		kvtide_die "cell failed; inspect $log; raw results remain in $RUN_DIR"
	fi
}

run_profile() {
	local profile=$1 arm=$2 sizes qds threads size qd nth rep warmup

	CURRENT_PROFILE=$profile
	case "$profile" in
	aisio)
		sizes=${CONFIG_KVTIDE_PCIE_AISIO_SIZES}
		qds=${CONFIG_KVTIDE_PCIE_AISIO_QDS}
		threads=${CONFIG_KVTIDE_PCIE_AISIO_THREADS}
		;;
	kv)
		sizes=${CONFIG_KVTIDE_PCIE_KV_SIZES}
		qds=${CONFIG_KVTIDE_PCIE_KV_QDS}
		threads=${CONFIG_KVTIDE_PCIE_KV_THREADS}
		;;
	*) kvtide_die "unknown physical PCIe profile '$profile'" ;;
	esac
	for size in $sizes; do
		if [ $((size % LBA_SIZE)) -ne 0 ]; then
			kvtide_die "$profile size $size is not aligned to the $LBA_SIZE-byte namespace LBA"
		fi
		for qd in $qds; do
			for nth in $threads; do
				for warmup in $(seq 1 "${CONFIG_KVTIDE_PCIE_WARMUPS}"); do
					run_cell "$arm" "$size" "$qd" "$nth" "warmup-$warmup"
				done
				for rep in $(seq 1 "${CONFIG_KVTIDE_PCIE_REPS}"); do
					run_cell "$arm" "$size" "$qd" "$nth" "$rep"
				done
			done
		done
	done
}

run_arm() {
	local arm=$1 profile

	for profile in ${CONFIG_KVTIDE_PCIE_PROFILES}; do
		run_profile "$profile" "$arm"
	done
}

echo "profile,arm,tool,driver,pattern,iosize,qd,threads,rep,iops,MiBps,failed,wall_seconds,system_cpu_cores" > "$CSV"

validate_bench_config
"$PCIE_BIND" kernel
inventory_kernel_devices
validate_driver_environment
record_metadata
record_device_state before

if want_vfio; then
	CURRENT_DRIVER=vfio-pci
	"$PCIE_BIND" vfio
else
	CURRENT_DRIVER=uio_pci_generic
	"$PCIE_BIND" uio
fi
printf '%s\n' "${CONFIG_KVTIDE_PCIE_HUGEPAGES}" | \
	sudo tee /proc/sys/vm/nr_hugepages >/dev/null

want_pcie_upcie && run_arm upcie
want_pcie_cuda && run_arm upcie-cuda
want_pcie_spdk && run_arm spdk

if want_pcie_linux; then
	"$PCIE_BIND" kernel
	inventory_kernel_devices
	CURRENT_DRIVER=nvme
	want_pcie_premap && validate_premap
	run_arm linux
	want_pcie_fixed && run_arm fixed
	want_pcie_premap && run_arm premap
fi

"$PCIE_BIND" kernel
inventory_kernel_devices
record_device_state after
printf '%s\n' "$ORIGINAL_HUGEPAGES" | \
	sudo tee /proc/sys/vm/nr_hugepages >/dev/null
trap - EXIT
ln -sfn "$(basename "$RUN_DIR")" "$KVTIDE_RESULTS/kvtide-pcie-latest"
kvtide_log "physical PCIe bench done: $CSV"
