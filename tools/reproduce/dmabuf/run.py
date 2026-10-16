#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare pinned DMA-BUF sources and run explicitly selected validation stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEATURES = (
    "IO_URING",
    "DMA_SHARED_BUFFER",
    "DMABUF_HEAPS",
    "DMABUF_HEAPS_SYSTEM",
    "UDMABUF",
    "PCI_P2PDMA",
    "BLK_DEV_NVME",
    "EXT4_FS",
    "XFS_FS",
)
STAGES = (
    "plan",
    "fetch",
    "kernel-config",
    "kernel-build",
    "kernel-install",
    "selftest-build",
    "doctor",
    "peer-check",
    "storage-test",
    "lmcache-install",
    "cpu-test",
    "mp-host-test",
    "gpu-test",
    "pd-lmcache-test",
    "pd-mp-test",
    "pd-vllm-test",
    "pd-timeline-check",
)

PD_LMCACHE_TESTS = (
    "tests/v1/storage_backend/test_raw_block_handoff.py",
    "tests/v1/storage_backend/test_raw_block_pd_tracker.py",
    "tests/v1/storage_backend/test_storage_pd_protocol.py",
    "tests/v1/storage_backend/test_storage_pd_proxy.py",
    "tests/v1/storage_backend/test_storage_pd_proxy_stream.py",
    "tests/v1/storage_backend/test_storage_pd_ack_control.py",
    "tests/v1/storage_backend/test_storage_pd_ack_integration.py",
    "tests/v1/storage_backend/test_storage_pd_notification_queue.py",
    "tests/v1/storage_backend/test_storage_pd_trace.py",
    "tests/v1/storage_backend/test_storage_pd_timeline.py",
    "tests/v1/storage_backend/test_storage_pd_file_launcher.py",
    "tests/v1/test_storage_pd_restore_reservations.py",
    "tests/v1/test_storage_pd_restore_coverage.py",
    "tests/v1/test_v1_adapter_storage_pd_construction.py",
)
PD_MP_TESTS = (
    "tests/v1/distributed/test_raw_block_mp_handoff.py",
    "tests/v1/multiprocess/test_storage_pd_http.py",
)
PD_VLLM_TESTS = (
    "tests/v1/kv_connector/unit/test_nixl_storage_coordinator.py",
    "tests/v1/kv_connector/unit/test_nixl_storage_connector.py",
    "tests/v1/kv_connector/unit/test_nixl_storage_adapter.py",
)


def read_config(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith("CONFIG_KNLP_DMABUF_"):
            continue
        key, value = line.split("=", 1)
        result[key.removeprefix("CONFIG_KNLP_DMABUF_")] = (
            json.loads(value) if value.startswith('"') else int(value)
        )
    return result


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def verify_source(path: Path, revision: str) -> None:
    if git(path, "rev-parse", "HEAD") != revision:
        raise RuntimeError(f"{path}: revision differs from sources.json")
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"{path}: tracked source edits; refusing to reuse")


def fetch_source(path: Path, source: dict) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".fetch-") as tmp:
            stage = Path(tmp) / "source"
            subprocess.run(["git", "init", "-q", str(stage)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(stage),
                    "fetch",
                    "--depth=1",
                    source["url"],
                    source["commit"],
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(stage), "checkout", "-q", "--detach", "FETCH_HEAD"],
                check=True,
            )
            verify_source(stage, source["commit"])
            stage.rename(path)
    verify_source(path, source["commit"])


def check_junit(path: Path) -> None:
    tree = ET.parse(path)
    cases = list(tree.iter("testcase"))
    if not cases or any(
        list(tree.iter(tag)) for tag in ("failure", "error", "skipped")
    ):
        raise RuntimeError(f"{path}: empty, failed or skipped checks are not a pass")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
        return result.hexdigest()


class Workflow:
    def __init__(self, config: dict, sources: dict):
        self.config = config
        self.sources = sources
        self.root = Path(config.get("WORKDIR", ".cache/io-uring-dmabuf")).resolve()
        self.build = self.root / "build/kernel"
        self.results = self.root / "results"

    def source(self, name: str) -> Path:
        path = self.root / "src" / name
        verify_source(path, self.sources[name]["commit"])
        return path.resolve()

    def execute(self, label: str, argv: list, *, cwd=None, env=None) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        log = self.results / f"{label}.log"
        with log.open("w") as output:
            output.write(shlex.join(map(str, argv)) + "\n")
            output.flush()
            result = subprocess.run(
                list(map(str, argv)),
                check=False,
                cwd=cwd,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        print(f"{label}: exit={result.returncode}; log={log}")
        if result.returncode:
            with log.open() as output:
                print("".join(deque(output, maxlen=30)), file=sys.stderr)
            raise RuntimeError(f"{label} failed; retained files and logs")

    def plan(self) -> None:
        print("Pinned sources:")
        print(json.dumps(self.sources, indent=2))
        print("\nExplicit stages: " + ", ".join(STAGES[1:]))
        print("Plain make prints this plan. No fetch, install, reboot or I/O ran.")
        print("See tools/reproduce/dmabuf/README.md for dependencies and scope.")

    def fetch(self) -> None:
        for name, source in self.sources.items():
            fetch_source(self.root / "src" / name, source)
        print("All four source revisions verified.")

    def kernel_config(self) -> None:
        src = self.source("linux")
        if (self.build / ".config").exists():
            raise RuntimeError("kernel .config exists; preserve/review it explicitly")
        base = self.config.get("BASE_CONFIG")
        if base:
            data = Path(base).read_bytes()
        elif Path(f"/boot/config-{platform.release()}").is_file():
            data = Path(f"/boot/config-{platform.release()}").read_bytes()
        else:
            with gzip.open("/proc/config.gz", "rb") as source:
                data = source.read()
        self.build.mkdir(parents=True, exist_ok=True)
        (self.build / ".config").write_bytes(data)
        args = [src / "scripts/config", "--file", self.build / ".config"]
        for feature in FEATURES:
            args += ["--enable", feature]
        args += [
            "--set-str",
            "SYSTEM_TRUSTED_KEYS",
            "",
            "--set-str",
            "SYSTEM_REVOCATION_KEYS",
            "",
            "--set-str",
            "LOCALVERSION",
            "-knlp-dmabuf",
            "--disable",
            "LOCALVERSION_AUTO",
        ]
        self.execute("kernel-features", args)
        self.execute(
            "kernel-config", ["make", "-C", src, f"O={self.build}", "olddefconfig"]
        )
        text = (self.build / ".config").read_text()
        for feature in FEATURES:
            if not any(f"CONFIG_{feature}={v}\n" in text for v in ("y", "m")):
                raise RuntimeError(
                    f"olddefconfig dropped {feature}; check dependencies"
                )
        shutil.copyfile(self.build / ".config", self.results / "kernel.config")
        print(
            "Base platform config retained; review NIC/root-storage drivers and signing."
        )

    def kernel_build(self) -> None:
        if not (self.build / ".config").is_file():
            raise RuntimeError("run kernel-config first")
        jobs = int(self.config.get("JOBS", 4))
        if not 1 <= jobs <= 256:
            raise RuntimeError("JOBS must be between 1 and 256")
        receipt = self.results / "kernel-build.json"
        receipt.unlink(missing_ok=True)
        self.execute(
            "kernel-build",
            ["make", "-C", self.source("linux"), f"O={self.build}", f"-j{jobs}"],
        )
        receipt.write_text(
            json.dumps(
                {
                    "source": self.sources["linux"]["commit"],
                    "config": digest(self.build / ".config"),
                    "vmlinux": digest(self.build / "vmlinux"),
                },
                indent=2,
            )
            + "\n"
        )

    def kernel_install(self) -> None:
        receipt = self.results / "kernel-build.json"
        if not receipt.is_file():
            raise RuntimeError("run kernel-build first")
        built = json.loads(receipt.read_text())
        if built != {
            "source": self.sources["linux"]["commit"],
            "config": digest(self.build / ".config"),
            "vmlinux": digest(self.build / "vmlinux"),
        }:
            raise RuntimeError(
                "kernel build inputs/output changed; rebuild before install"
            )
        self.execute(
            "kernel-install",
            [
                "sudo",
                "make",
                "-C",
                self.source("linux"),
                f"O={self.build}",
                "INSTALL_MOD_STRIP=1",
                "modules_install",
                "install",
            ],
        )
        print("Keep the rescue kernel. Select the test kernel and reboot manually.")

    def selftest_build(self) -> None:
        src = self.source("linux")
        binary = self.root / "bin/dmabuf-block-io"
        binary.parent.mkdir(parents=True, exist_ok=True)
        self.execute(
            "selftest-build",
            [
                "cc",
                "-O2",
                "-Wall",
                "-Wextra",
                src / "tools/testing/selftests/dmabuf-heaps/dmabuf-block-io.c",
                "-o",
                binary,
            ],
        )

    def doctor(self) -> None:
        limits = {}
        for file in Path("/sys/block").glob("nvme*/queue/max_hw_dmabuf_sectors_kb"):
            if int(file.read_text()) > 0:
                limits[file.parent.parent.name] = int(file.read_text())
        if not limits:
            raise RuntimeError("no NVMe queue exposes a positive DMA-BUF ceiling")
        expected = self.build / "include/config/kernel.release"
        if expected.exists() and expected.read_text().strip() != platform.release():
            raise RuntimeError(
                "running kernel differs from the local build; reboot/check it"
            )
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / "kernel-doctor.json").write_text(
            json.dumps(
                {
                    "running_kernel": platform.release(),
                    "dmabuf_ceilings_kib": limits,
                    "built_release_checked": expected.exists(),
                    "transfer_tested": False,
                },
                indent=2,
            )
            + "\n"
        )
        print("Kernel prerequisites found; no GPU route or transfer is qualified.")

    def peer_check(self) -> None:
        gpu, nvme = self.config.get("GPU"), self.config.get("NVME")
        if not gpu or not nvme:
            raise RuntimeError("configure exact GPU and NVME PCI addresses first")
        self.execute(
            "peer-policy",
            [
                sys.executable,
                self.source("linux")
                / "tools/testing/selftests/dmabuf-heaps/pci-peer-check.py",
                "--gpu",
                gpu,
                "--nvme",
                nvme,
                "--verbose",
                "--json",
                "--no-nvidia-query",
            ],
        )
        print("Policy diagnostic only; inspect unknown/rejected outcomes in the log.")

    def scratch(self) -> Path:
        value = self.config.get("SCRATCH", "")
        if not value or not Path(value).is_dir():
            raise RuntimeError("SCRATCH must name an existing owned directory")
        path = Path(value).resolve()
        fs = subprocess.check_output(
            ["findmnt", "-no", "FSTYPE", "-T", str(path)], text=True
        ).strip()
        if fs not in ("ext4", "xfs"):
            raise RuntimeError("SCRATCH must be on XFS/ext4 backed by qualified NVMe")
        return Path(tempfile.mkdtemp(prefix="knlp-dmabuf-", dir=path))

    def storage_test(self) -> None:
        self.doctor()
        self.source("linux")
        if not (self.root / "bin/dmabuf-block-io").is_file():
            raise RuntimeError("run selftest-build first")
        mode = self.config.get("EXPORTER", "sysheap")
        if mode not in ("sysheap", "udmabuf"):
            raise RuntimeError("EXPORTER must be sysheap or udmabuf")
        scratch = self.scratch()
        for size in (4096, 8 << 20):
            path = scratch / f"host-{size}.bin"
            with path.open("xb") as output:
                output.write(bytes(size))
                output.flush()
                os.fsync(output.fileno())
            self.execute(
                f"storage-{size}",
                [self.root / "bin/dmabuf-block-io", path, mode, str(size), "0"],
            )
        print(f"Host-memory payload tests passed; files retained at {scratch}.")

    def lmcache_install(self) -> None:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError(
                "use a dedicated virtualenv with matching Torch/toolchain"
            )
        src = self.source("lmcache")
        self.execute(
            "lmcache-install",
            [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", src],
        )
        self.execute(
            "raw-block-install",
            [sys.executable, "-m", "pip", "install", src / "rust/raw_block"],
        )

    def tests(self, stage: str) -> None:
        project = "vllm" if stage == "pd-vllm-test" else "lmcache"
        src = self.source(project)
        env = dict(
            os.environ,
            LMCACHE_RUN_HOST_DMABUF_IO="0",
            LMCACHE_RUN_DMABUF_ORDERING="0",
            LMCACHE_RUN_VULKAN_RM="0",
        )
        # Both projects have an examples package; expose only the selected root.
        project_roots = {
            (self.root / "src" / name).resolve() for name in ("lmcache", "vllm")
        }
        extra_paths = [
            entry
            for entry in env.get("PYTHONPATH", "").split(os.pathsep)
            if entry and Path(entry).resolve() not in project_roots
        ]
        env["PYTHONPATH"] = os.pathsep.join([str(src), *extra_paths])
        cpu_stages = ("cpu-test", "pd-mp-test", "pd-lmcache-test", "pd-vllm-test")
        if stage in cpu_stages:
            env.update(
                CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES=""
            )
        imports = "import pathlib,lmcache,lmcache_rust_raw_block_io; "
        if project == "vllm":
            imports = "import pathlib,vllm; "
        self.execute(
            f"{stage}-imports",
            [
                sys.executable,
                "-c",
                imports
                + f"assert pathlib.Path({project}.__file__).resolve().is_relative_to({str(src)!r})",
            ],
            env=env,
        )
        if stage in cpu_stages:
            paths = {
                "cpu-test": PD_MP_TESTS,
                "pd-mp-test": PD_MP_TESTS,
                "pd-lmcache-test": PD_LMCACHE_TESTS,
                "pd-vllm-test": PD_VLLM_TESTS,
            }[stage]
        else:
            self.doctor()
            scratch = self.scratch()
            if stage == "mp-host-test":
                paths = ["tests/v1/distributed/test_mp_registration.py"]
                env["LMCACHE_RUN_HOST_DMABUF_IO"] = "1"
            else:
                provider = self.config.get("GPU_PROVIDER", "native")
                if provider not in ("native", "vulkan_rm"):
                    raise RuntimeError("GPU_PROVIDER must be native or vulkan_rm")
                paths = ["tests/v1/gpu_connector/test_vllm_staging_ordering_dmabuf.py"]
                env.update(
                    LMCACHE_RUN_DMABUF_ORDERING="1",
                    LMCACHE_DMABUF_TEST_DIR=str(scratch),
                    LMCACHE_DMABUF_TEST_PROVIDER=provider,
                )
        report = self.results / f"{stage}.xml"
        argv = [sys.executable, "-m", "pytest", "-xvs", *paths, f"--junitxml={report}"]
        if stage == "pd-vllm-test":
            # Select CPU contracts explicitly; hardware cases are separate gates.
            argv += [
                "--confcutdir=tests/v1/kv_connector/unit",
                "-k",
                (
                    "not dmabuf_exporters_hand_out_fds "
                    "and not cuda_exporter_is_device_memory_in_dmabuf_chunks "
                    "and not write_then_read_blocks_through_the_namespace "
                    "and not (plan_pack_and_unpack_round_trip_through_the_arena and cuda)"
                ),
            ]
        if stage == "mp-host-test":
            argv += [f"--basetemp={scratch / 'pytest'}"]
        self.execute(stage, argv, cwd=src, env=env)
        check_junit(report)
        print(f"{stage}: selected checks passed with no skips; scope is in README.md")

    def pd_timeline_check(self, timeline: Path | None, required_path: str) -> None:
        if timeline is None or not timeline.is_file():
            raise RuntimeError("set PD_TIMELINE to an existing serving JSONL receipt")
        self.execute(
            "pd-timeline-check",
            [
                sys.executable,
                self.source("lmcache")
                / "examples/disagg_prefill/storage_pd_timeline_check.py",
                timeline.resolve(),
                "--require-path",
                required_path,
                "--json-out",
                self.results / "pd-timeline-check.json",
            ],
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(".config"))
    parser.add_argument("--sources", type=Path, default=HERE / "sources.json")
    parser.add_argument("--timeline", type=Path)
    parser.add_argument(
        "--require-path",
        default="dmabuf_fixed",
        choices=(
            "dmabuf_fixed",
            "host_fixed",
            "bounce",
            "regular",
            "uring_cmd",
            "uring_cmd_fixed",
        ),
    )
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args(argv)
    try:
        workflow = Workflow(
            read_config(args.config), json.loads(args.sources.read_text())
        )
        if args.stage == "pd-timeline-check":
            workflow.pd_timeline_check(args.timeline, args.require_path)
        elif args.stage.endswith("-test") and args.stage != "storage-test":
            workflow.tests(args.stage)
        else:
            getattr(workflow, args.stage.replace("-", "_"))()
    except (
        OSError,
        ValueError,
        RuntimeError,
        ET.ParseError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"dmabuf: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
