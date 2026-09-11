#!/bin/bash
# SPDX-License-Identifier: MIT
#
# Rebind only the explicitly configured disposable NVMe controllers.

. "$(dirname "$0")/lib.sh"

[ "$#" -eq 1 ] || kvtide_die "usage: pcie-bind.sh <kernel|vfio|uio>"
is_pcie || kvtide_die "select CONFIG_KVTIDE_MODE_PCIE"
[ "${CONFIG_KVTIDE_PCIE_ALLOW_REBIND:-n}" = y ] || \
	kvtide_die "set CONFIG_KVTIDE_PCIE_ALLOW_REBIND=y after checking every BDF"
[ -n "${CONFIG_KVTIDE_PCIE_BDFS:-}" ] || \
	kvtide_die "CONFIG_KVTIDE_PCIE_BDFS is empty"

SYSFS_ROOT=${KVTIDE_SYSFS_ROOT:-/sys}
DEV_ROOT=${KVTIDE_DEV_ROOT:-/dev}

case "$1" in
kernel) target=nvme ;;
vfio) target=vfio-pci ;;
uio) target=uio_pci_generic ;;
*) kvtide_die "driver mode must be kernel, vfio or uio" ;;
esac

prepare_vfio() {
	# vfio-pci does not depend on the legacy type1 IOMMU module. Load it
	# explicitly so xNVMe can use its type1 fallback when the host does not
	# expose a vfio device cdev through iommufd.
	sudo modprobe vfio-pci
	sudo modprobe vfio_iommu_type1 2>/dev/null || true
}

prepare_target() {
	case "$target" in
	nvme) sudo modprobe nvme ;;
	vfio-pci) prepare_vfio ;;
	uio_pci_generic) sudo modprobe uio_pci_generic ;;
	esac
}

validate_vfio_transport() {
	local bdf cdev_ready=y
	local mode=${CONFIG_KVTIDE_PCIE_XNVME_VFIO_MODE:-auto}

	[ -c "$DEV_ROOT/iommu" ] || cdev_ready=n
	for bdf in ${CONFIG_KVTIDE_PCIE_BDFS}; do
		[ -d "$SYSFS_ROOT/bus/pci/devices/$bdf/vfio-dev" ] || \
			cdev_ready=n
	done
	case "$mode" in
	auto)
		[ "$cdev_ready" = n ] || return 0
		[ -d "$SYSFS_ROOT/module/vfio_iommu_type1" ] || \
			kvtide_die "vfio-pci has no usable cdev or type1 transport"
		;;
	iommufd)
		[ "$cdev_ready" = y ] || \
			kvtide_die "xNVMe iommufd mode needs a cdev for every controller"
		;;
	type1)
		[ -d "$SYSFS_ROOT/module/vfio_iommu_type1" ] || \
			kvtide_die "xNVMe type1 mode needs vfio_iommu_type1"
		;;
	*) kvtide_die "xNVMe VFIO mode must be auto, iommufd or type1" ;;
	esac
}

PREMAP_ORDER_PATH="$SYSFS_ROOT/module/nvme_core/parameters/iobuf_pool_order"
PREMAP_FOLIOS_PATH="$SYSFS_ROOT/module/nvme_core/parameters/iobuf_pool_folios"
if [ "$target" = nvme ] && want_pcie_premap; then
	[ -e "$PREMAP_ORDER_PATH" ] && [ -e "$PREMAP_FOLIOS_PATH" ] || \
		kvtide_die "running kernel lacks nvme_core premap pool parameters"
	[ "$(cat "$PREMAP_ORDER_PATH")" = "${CONFIG_KVTIDE_PCIE_IOBUF_ORDER}" ] || \
		kvtide_die "boot with nvme_core.iobuf_pool_order=${CONFIG_KVTIDE_PCIE_IOBUF_ORDER}"
	[ "$(cat "$PREMAP_FOLIOS_PATH")" = "${CONFIG_KVTIDE_PCIE_IOBUF_FOLIOS}" ] || \
		kvtide_die "boot with nvme_core.iobuf_pool_folios=${CONFIG_KVTIDE_PCIE_IOBUF_FOLIOS}"
fi

validate_bdf() {
	case "$1" in
	[[:xdigit:]][[:xdigit:]][[:xdigit:]][[:xdigit:]]:[[:xdigit:]][[:xdigit:]]:[[:xdigit:]][[:xdigit:]].[0-7]) ;;
	*) kvtide_die "invalid PCI address '$1'" ;;
	esac
	[ -d "$SYSFS_ROOT/bus/pci/devices/$1" ] || \
		kvtide_die "PCI device $1 does not exist"
	class=$(cat "$SYSFS_ROOT/bus/pci/devices/$1/class")
	[ "$class" = 0x010802 ] || \
		kvtide_die "$1 is class $class, not an NVMe controller"
}

block_devices() {
	local bdf=$1 entry path

	for entry in "$SYSFS_ROOT"/class/block/*; do
		[ -e "$entry" ] || continue
		path=$(readlink -f "$entry")
		case "$path" in
		*"/$bdf/"*) echo "/dev/${entry##*/}" ;;
		esac
	done
}

assert_unused() {
	local bdf=$1 dev name

	while read -r dev; do
		[ -n "$dev" ] || continue
		name=${dev##*/}
		if lsblk -nrpo MOUNTPOINTS "$dev" 2>/dev/null | grep -q '[^[:space:]]'; then
			kvtide_die "$bdf backs a mounted filesystem through $dev"
		fi
		if awk -v dev="$dev" 'NR > 1 && $1 == dev {found=1} END {exit !found}' \
			/proc/swaps; then
			kvtide_die "$bdf backs active swap on $dev"
		fi
		if compgen -G "$SYSFS_ROOT/class/block/$name/holders/*" >/dev/null; then
			kvtide_die "$bdf has a block holder through $dev"
		fi
		if sudo fuser -s "$dev"; then
			kvtide_die "$bdf has a process holding $dev open"
		fi
	done < <(block_devices "$bdf")
}

bind_one() {
	local bdf=$1 devdir current="" override bind

	validate_bdf "$bdf"
	devdir="$SYSFS_ROOT/bus/pci/devices/$bdf"
	if [ -L "$devdir/driver" ]; then
		current=$(basename "$(readlink -f "$devdir/driver")")
	fi
	[ "$current" = "$target" ] && {
		kvtide_log "$bdf already bound to $target"
		return
	}
	[ "$current" != nvme ] || assert_unused "$bdf"

	override="$devdir/driver_override"
	bind="$SYSFS_ROOT/bus/pci/drivers/$target/bind"
	printf '%s' "$target" | sudo tee "$override" >/dev/null
	if [ -n "$current" ]; then
		printf '%s' "$bdf" | sudo tee \
			"$SYSFS_ROOT/bus/pci/drivers/$current/unbind" >/dev/null
	fi
	printf '%s' "$bdf" | sudo tee "$bind" >/dev/null
	printf '\n' | sudo tee "$override" >/dev/null
	kvtide_log "$bdf: ${current:-unbound} -> $target"
}

prepare_target
for bdf in ${CONFIG_KVTIDE_PCIE_BDFS}; do
	bind_one "$bdf"
done

[ "$target" != vfio-pci ] || validate_vfio_transport

if [ "$target" = nvme ]; then
	command -v udevadm >/dev/null 2>&1 && sudo udevadm settle
fi
