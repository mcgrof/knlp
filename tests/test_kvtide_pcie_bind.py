import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
KVTIDE = ROOT / "tools" / "reproduce" / "kvtide"
BDF = "0000:01:00.0"


def _write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run_bind(tmp_path, type1_available, cdev_available=False):
    repo = tmp_path / "repo"
    scripts = repo / "tools" / "reproduce" / "kvtide"
    scripts.mkdir(parents=True)
    for name in ("lib.sh", "pcie-bind.sh"):
        shutil.copy2(KVTIDE / name, scripts / name)

    (repo / ".config").write_text(
        "CONFIG_KVTIDE=y\n"
        "CONFIG_KVTIDE_MODE_PCIE=y\n"
        f'CONFIG_KVTIDE_PCIE_BDFS="{BDF}"\n'
        "CONFIG_KVTIDE_PCIE_ALLOW_REBIND=y\n"
        "CONFIG_KVTIDE_PCIE_PREMAP=n\n"
        'CONFIG_KVTIDE_SRC_DIR="/tmp/kvtide-test"\n',
        encoding="utf-8",
    )

    sysfs = tmp_path / "sys"
    dev = sysfs / "bus" / "pci" / "devices" / BDF
    nvme_driver = sysfs / "bus" / "pci" / "drivers" / "nvme"
    vfio_driver = sysfs / "bus" / "pci" / "drivers" / "vfio-pci"
    for driver in (nvme_driver, vfio_driver):
        driver.mkdir(parents=True)
        (driver / "bind").touch()
        (driver / "unbind").touch()
    dev.mkdir(parents=True)
    (dev / "class").write_text("0x010802\n", encoding="utf-8")
    (dev / "driver_override").touch()
    (dev / "driver").symlink_to(nvme_driver)
    (sysfs / "class" / "block").mkdir(parents=True)

    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    if cdev_available:
        (dev_root / "iommu").symlink_to("/dev/null")
        (dev / "vfio-dev").mkdir()

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    _write_executable(fakebin / "sudo", '#!/bin/sh\nexec "$@"\n')
    _write_executable(
        fakebin / "modprobe",
        "#!/bin/sh\n"
        'printf "%s\\n" "$1" >> "$KVTIDE_MODPROBE_LOG"\n'
        'if [ "$1" = vfio_iommu_type1 ]; then\n'
        '  [ "$KVTIDE_TYPE1_AVAILABLE" = y ] || exit 1\n'
        '  mkdir -p "$KVTIDE_SYSFS_ROOT/module/vfio_iommu_type1"\n'
        "fi\n",
    )

    log = tmp_path / "modprobe.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fakebin}:{env['PATH']}",
            "KVTIDE_SYSFS_ROOT": str(sysfs),
            "KVTIDE_DEV_ROOT": str(dev_root),
            "KVTIDE_MODPROBE_LOG": str(log),
            "KVTIDE_TYPE1_AVAILABLE": "y" if type1_available else "n",
        }
    )
    result = subprocess.run(
        [scripts / "pcie-bind.sh", "vfio"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, log.read_text(encoding="utf-8").splitlines()


def test_vfio_binding_loads_the_type1_transport(tmp_path):
    result, modules = _run_bind(tmp_path, type1_available=True)

    assert result.returncode == 0, result.stderr
    assert modules == ["vfio-pci", "vfio_iommu_type1"]


def test_vfio_binding_rejects_a_host_without_a_transport(tmp_path):
    result, modules = _run_bind(tmp_path, type1_available=False)

    assert result.returncode != 0
    assert modules == ["vfio-pci", "vfio_iommu_type1"]
    assert "vfio-pci needs vfio_iommu_type1" in result.stderr


def test_vfio_binding_accepts_device_cdevs(tmp_path):
    result, modules = _run_bind(tmp_path, type1_available=False, cdev_available=True)

    assert result.returncode == 0, result.stderr
    assert modules == ["vfio-pci", "vfio_iommu_type1"]
