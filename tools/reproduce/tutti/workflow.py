#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pinned Tutti build and byte-verified synthetic NVMe measurements."""

import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
TOP = HERE.parents[2]
PIN = "5e48c2ab6deea2ec9972a5f1beb56854f62bf149"
DEFAULTS = {
    "TUTTI_GIT": "https://github.com/xPU-IO/Tutti.git",
    "TUTTI_REV": PIN,
    "TUTTI_WORK_DIR": "./work/tutti",
    "TUTTI_OUTPUT_DIR": "./results/tutti",
    "TUTTI_CUDA_ARCH": "90",
    "TUTTI_JOBS": "8",
    "TUTTI_BUILD_MODULE": "y",
    "TUTTI_IO_KIB": "64 256 1024 4096",
    "TUTTI_BATCH_DEPTHS": "1 8 32",
    "TUTTI_SPAN_MIB": "256",
    "TUTTI_WARMUPS": "1",
    "TUTTI_REPEATS": "5",
    "TUTTI_SEED": "42",
    "TUTTI_OVERLAP_LAYERS": "4",
    "TUTTI_OVERLAP_CHUNKS": "16",
    "TUTTI_OVERLAP_TENSOR_KIB": "512",
    "TUTTI_OVERLAP_COMPUTE_US": "1000",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def config(path):
    """Parse Kconfig literals without executing shell expressions."""
    values = DEFAULTS.copy()
    for line in Path(path).read_text().splitlines():
        disabled = re.fullmatch(r"# CONFIG_(TUTTI\w*) is not set", line)
        if disabled:
            values[disabled[1]] = "n"
            continue
        match = re.fullmatch(r"CONFIG_(TUTTI\w*)=(.*)", line)
        if not match:
            continue
        key, value = match.groups()
        if value.startswith('"'):
            # Kconfig strings only escape quotes and backslashes.
            require(re.fullmatch(r'"(?:[^"\\]|\\["\\])*"', value), f"bad string: {key}")
            value = re.sub(r'\\(["\\])', r"\1", value[1:-1])
        else:
            require(re.fullmatch(r"(?:y|n|[0-9]+)", value), f"bad value: {key}")
        values[key] = value
    require(values.get("TUTTI") == "y", "run make defconfig-tutti-build first")
    require(
        re.fullmatch(r"[0-9a-f]{40}", values["TUTTI_REV"]),
        "pin a full Tutti commit SHA",
    )
    require(
        re.fullmatch(r"[0-9]+(?:;(?:[0-9]+))*", values["TUTTI_CUDA_ARCH"]),
        "bad CUDA architecture",
    )
    return values


def path_value(c, name):
    p = Path(c[name]).expanduser()
    return (TOP / p).resolve() if not p.is_absolute() else p.resolve()


def integer(c, key, low, high):
    value = int(c[key])
    require(low <= value <= high, f"{key} must be in [{low}, {high}]")
    return value


def cells(c, smoke=False):
    span = (16 if smoke else integer(c, "TUTTI_SPAN_MIB", 1, 1048576)) << 20
    sizes = [64] if smoke else [int(v) for v in c["TUTTI_IO_KIB"].split()]
    depths = [1, 8] if smoke else [int(v) for v in c["TUTTI_BATCH_DEPTHS"].split()]
    require(sizes and depths, "size and batch-depth lists must be nonempty")
    require(
        len(set(sizes)) == len(sizes) and len(set(depths)) == len(depths),
        "duplicate matrix entries",
    )
    rows = []
    for kib, depth in itertools.product(sizes, depths):
        size = kib * 1024
        require(
            size > 0 and size % 4096 == 0 and span % size == 0,
            "sizes must be 4 KiB aligned and divide the span",
        )
        require(
            1 <= depth <= min(4096, span // size),
            "batch depth exceeds object count or runtime limit",
        )
        rows.append({"io_bytes": size, "span_bytes": span, "batch_depth": depth})
    random.Random(integer(c, "TUTTI_SEED", 0, 2147483647)).shuffle(rows)
    return rows


def run(argv, **kwargs):
    print("+ " + " ".join(map(str, argv)), file=sys.stderr, flush=True)
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def capture(argv):
    try:
        p = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
        return {"command": argv, "returncode": p.returncode, "output": p.stdout}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "unavailable": str(error)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measurement_rows(text):
    # Tutti and dependency diagnostics may also use stdout.
    prefix = "KNLP_TUTTI_JSON "
    return [
        json.loads(line[len(prefix) :])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source(c):
    return path_value(c, "TUTTI_WORK_DIR") / "src"


def verify_source(c):
    src = source(c)
    head = run(
        ["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = run(
        ["git", "-C", src, "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(
        head == c["TUTTI_REV"],
        f"source is {head}, expected {c['TUTTI_REV']}; use a new workspace",
    )
    require(
        not dirty,
        "Tutti checkout has local changes; preserve them and use a new workspace",
    )
    return head


def fetch(c):
    src = source(c)
    if src.exists():
        verify_source(c)
        return
    src.parent.mkdir(parents=True, exist_ok=True)
    # No reset/clean of an existing checkout; incomplete fetches stay reviewable.
    run(["git", "init", src])
    run(["git", "-C", src, "remote", "add", "origin", c["TUTTI_GIT"]])
    run(["git", "-C", src, "fetch", "--depth=1", "origin", c["TUTTI_REV"]])
    run(["git", "-C", src, "checkout", "--detach", "FETCH_HEAD"])
    verify_source(c)


def build_commands(c):
    src = source(c)
    build = src / "build"
    return [
        [
            "cmake",
            "--preset",
            "default",
            "-S",
            str(src),
            "-B",
            str(build),
            "-DTUTTI_ACCELERATOR=CUDA",
            f"-DCMAKE_CUDA_ARCHITECTURES={c['TUTTI_CUDA_ARCH']}",
            f"-DTUTTI_BUILD_KERNEL_MODULE={'ON' if c.get('TUTTI_BUILD_MODULE') == 'y' else 'OFF'}",
            f"-DCMAKE_PROJECT_Tutti_INCLUDE={HERE / 'bench.cmake'}",
        ],
        [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            str(integer(c, "TUTTI_JOBS", 1, 256)),
        ],
    ]


def doctor(c):
    required = ["git", "cmake", "make", "c++", "nvcc", "pkg-config", "nvidia-smi"]
    missing = [name for name in required if shutil.which(name) is None]
    if (
        c.get("TUTTI_BUILD_MODULE") == "y"
        and not Path(f"/lib/modules/{platform.release()}/build").exists()
    ):
        missing.append("running-kernel headers")
    require(not missing, "missing build prerequisites: " + ", ".join(missing))
    require(platform.system() == "Linux", "Tutti hardware requires Linux")
    run(["nvidia-smi", "-L"])
    print(
        "Build prerequisites found. CMake checks libraries and CUDA >= 12.6. "
        "This is not a P2P, SNVMe or DMA-BUF qualification."
    )


def build(c):
    verify_source(c)
    for command in build_commands(c):
        run(command, cwd=source(c))
    files = [
        source(c) / "build/bin" / n
        for n in ("knlp_tutti_io", "tutti_layerwise_kv_overlap")
    ]
    require(all(p.is_file() for p in files), "benchmark binaries missing after build")
    write_json(
        path_value(c, "TUTTI_WORK_DIR") / "build.json",
        {
            "revision": c["TUTTI_REV"],
            "commands": build_commands(c),
            "transfer_source_sha256": sha(HERE / "transfer.cpp"),
            "cmake_hook_sha256": sha(HERE / "bench.cmake"),
            "binaries": {p.name: sha(p) for p in files},
            "cmake_cache_sha256": sha(source(c) / "build/CMakeCache.txt"),
        },
    )


def verify_build(c):
    verify_source(c)
    stamp = json.loads((path_value(c, "TUTTI_WORK_DIR") / "build.json").read_text())
    require(
        stamp["revision"] == c["TUTTI_REV"] and stamp["commands"] == build_commands(c),
        "build settings changed; rebuild",
    )
    require(
        stamp["transfer_source_sha256"] == sha(HERE / "transfer.cpp")
        and stamp["cmake_hook_sha256"] == sha(HERE / "bench.cmake"),
        "benchmark source changed; rebuild",
    )
    require(
        stamp["cmake_cache_sha256"] == sha(source(c) / "build/CMakeCache.txt"),
        "CMake cache changed; rebuild",
    )
    for name, digest in stamp["binaries"].items():
        require(
            sha(source(c) / "build/bin" / name) == digest, "binary changed; rebuild"
        )
    return stamp


def validate_runtime(c):
    require(
        c.get("TUTTI_ALLOW_FILE_WRITES") == "y",
        "enable TUTTI_ALLOW_FILE_WRITES for an owned test directory",
    )
    require(
        c.get("TUTTI_DIRECTORY") and c.get("TUTTI_RUNTIME_CONFIG"),
        "configure TUTTI_DIRECTORY and TUTTI_RUNTIME_CONFIG",
    )
    directory = path_value(c, "TUTTI_DIRECTORY")
    require(
        directory.is_dir() and directory != Path("/"),
        "test directory must already exist",
    )
    require(
        not any(directory.is_relative_to(p) for p in ("/dev", "/proc", "/sys")),
        "test directory cannot be a device or virtual filesystem",
    )
    # JSON is also YAML; accept a JSON runtime file without PyYAML installed.
    runtime = path_value(c, "TUTTI_RUNTIME_CONFIG")
    try:
        spec = json.loads(runtime.read_text())
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:
            raise ValueError("install PyYAML to read the runtime YAML") from error
        spec = yaml.safe_load(runtime.read_text())
    require(isinstance(spec, dict), "runtime config must be a mapping")
    require(
        spec.get("accelerator", {}).get("profile") == "CUDA", "runtime must select CUDA"
    )
    require(
        type(spec.get("runtime", {}).get("accel_id")) is int
        and spec["runtime"]["accel_id"] >= 0,
        "runtime needs an explicit GPU id",
    )
    storage = spec.get("storage", {})
    require(
        len(storage.get("datapaths", [])) == 1
        and storage["datapaths"][0].get("type") == "local-nvme",
        "initial matrix requires one local-nvme datapath",
    )
    resources = storage.get("resources", [])
    allocation = resources[0].get("allocation", {}) if len(resources) == 1 else {}
    devices = allocation.get("device_ids", [])
    require(
        len(resources) == 1
        and resources[0].get("type") == "nvme"
        and allocation.get("selection") == "explicit"
        and len(devices) == 1
        and type(devices[0]) is int
        and devices[0] >= 0,
        "initial matrix requires one explicitly selected NVMe device",
    )
    return directory, runtime, spec


def validate_rows(rows, cell, repeats, seed, access):
    require(len(rows) == repeats * 2, "missing or extra measurement rows")
    seen = set()
    for row in rows:
        require(
            row.get("schema_version") == 1
            and row.get("backend") == "tutti"
            and row.get("workload") == "synthetic-transfer"
            and row.get("scheduling") == "submit-and-drain",
            "wrong measurement schema/backend",
        )
        require(
            all(type(row.get(k)) is int and row[k] == v for k, v in cell.items()),
            "measurement geometry differs",
        )
        require(
            row.get("seed") == seed and row.get("access") == access,
            "measurement order differs",
        )
        rep, op = row.get("repeat"), row.get("operation")
        require(
            type(rep) is int and 0 <= rep < repeats and op in ("read", "write"),
            "invalid repetition or operation",
        )
        require((rep, op) not in seen, "duplicate measurement")
        seen.add((rep, op))
        require(
            row.get("correctness") == "full-readback"
            and row.get("verified_bytes") == cell["span_bytes"]
            and row.get("completed_bytes") == cell["span_bytes"]
            and row.get("operations") == cell["span_bytes"] // cell["io_bytes"],
            "incomplete I/O or verification",
        )
        for key in (
            "wall_seconds",
            "process_cpu_seconds",
            "batch_p50_us",
            "batch_p99_us",
        ):
            value = row.get(key)
            require(
                type(value) in (int, float)
                and math.isfinite(value)
                and (value >= 0 if key == "process_cpu_seconds" else value > 0),
                f"invalid {key}",
            )
        expected_batches = math.ceil(row["operations"] / cell["batch_depth"])
        require(
            row.get("batch_count") == expected_batches
            and row["batch_p99_us"] >= row["batch_p50_us"],
            "invalid batch statistics",
        )


def benchmark(c, config_path, smoke=False, overlap=False):
    stamp = verify_build(c)
    directory, runtime, spec = validate_runtime(c)
    matrix = cells(c, smoke)
    repeats = 1 if smoke else integer(c, "TUTTI_REPEATS", 1, 1000)
    warmups = 0 if smoke else integer(c, "TUTTI_WARMUPS", 0, 100)
    seed = integer(c, "TUTTI_SEED", 0, 2147483647)
    output = path_value(c, "TUTTI_OUTPUT_DIR")
    output.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".knlp-tutti.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=time.strftime("%Y%m%dT%H%M%SZ-", time.gmtime()), dir=output
        )
    )
    payload_dir = Path(tempfile.mkdtemp(prefix="knlp-tutti-", dir=directory))
    shutil.copyfile(runtime, run_dir / "runtime.yaml")
    shutil.copyfile(config_path, run_dir / "knlp.config")
    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "backend": "tutti",
        "kind": "upstream-synthetic-overlap" if overlap else "synthetic-transfer",
        "build": stamp,
        "runtime_spec": spec,
        "payload_directory": str(payload_dir),
        "runtime_sha256": sha(runtime),
        "config_sha256": sha(config_path),
        "workflow_sha256": sha(Path(__file__)),
        "knlp": capture(["git", "-C", str(TOP), "rev-parse", "HEAD"]),
        "knlp_status": capture(["git", "-C", str(TOP), "status", "--porcelain"]),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("TUTTI_") or k == "CUDA_VISIBLE_DEVICES"
        },
        "inventory": [
            capture(cmd)
            for cmd in [
                ["uname", "-a"],
                ["nvidia-smi", "-q"],
                ["nvidia-smi", "topo", "-m"],
                ["lspci", "-Dnn"],
                ["findmnt", "-J", "-T", str(directory)],
                ["lsblk", "-J", "-o", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL"],
            ]
        ],
        "cells": matrix,
        "repeats": repeats,
        "warmups": warmups,
        "seed": seed,
        "access": (
            "random-permutation" if c.get("TUTTI_RANDOM") == "y" else "sequential"
        ),
        "cpu_scope": "benchmark process and its threads only; daemon/IRQ/system CPU separate",
        "timing_scope": "submit/complete plus final CUDA stream sync; setup and verification excluded",
        "commands": [],
        "files": {},
    }
    write_json(run_dir / "manifest.json", manifest)
    try:
        all_rows = []
        if overlap:
            layers = integer(c, "TUTTI_OVERLAP_LAYERS", 1, 1000)
            chunks = integer(c, "TUTTI_OVERLAP_CHUNKS", 2, 4096)
            tensor = integer(c, "TUTTI_OVERLAP_TENSOR_KIB", 4, 1048576)
            require(tensor % 4 == 0, "overlap tensor must be 4 KiB aligned")
            command = [
                source(c) / "build/bin/tutti_layerwise_kv_overlap",
                "--single",
                "--config",
                run_dir / "runtime.yaml",
                "--directory",
                payload_dir,
                "--layers",
                layers,
                "--ctx-tokens",
                chunks * 256,
                "--chunk-tokens",
                256,
                "--tensor-kb",
                tensor,
                "--hit-pct",
                50,
                "--requests",
                repeats,
                "--compute-us",
                integer(c, "TUTTI_OVERLAP_COMPUTE_US", 1, 1000000),
                "--verify",
            ]
            manifest["commands"].append(list(map(str, command)))
            with (run_dir / "overlap.log").open("w") as log:
                run(command, stdout=log, stderr=subprocess.STDOUT)
            text = (run_dir / "overlap.log").read_text()
            require(
                "layerwise_kv_overlap: PASSED" in text and "Phase H: verified" in text,
                "missing upstream overlap completion",
            )
            manifest["verification_scope"] = (
                "upstream sampled first-byte checks; not full payload or serving validation"
            )
        else:
            for index, cell in enumerate(matrix):
                name = f"cell-{index:03d}"
                command = [
                    source(c) / "build/bin/knlp_tutti_io",
                    "--config",
                    run_dir / "runtime.yaml",
                    "--file",
                    payload_dir / f"{name}.bin",
                    "--io-bytes",
                    cell["io_bytes"],
                    "--span-bytes",
                    cell["span_bytes"],
                    "--batch-depth",
                    cell["batch_depth"],
                    "--warmups",
                    warmups,
                    "--repeats",
                    repeats,
                    "--seed",
                    seed,
                ]
                if c.get("TUTTI_RANDOM") == "y":
                    command.append("--random")
                manifest["commands"].append(list(map(str, command)))
                write_json(run_dir / "manifest.json", manifest)
                with (run_dir / f"{name}.stdout").open("w") as stdout, (
                    run_dir / f"{name}.log"
                ).open("w") as stderr:
                    run(command, stdout=stdout, stderr=stderr)
                rows = measurement_rows((run_dir / f"{name}.stdout").read_text())
                validate_rows(rows, cell, repeats, seed, manifest["access"])
                all_rows.extend(rows)
            (run_dir / "measurements.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in all_rows)
            )
        # Upstream failures leave their payload files for diagnosis.
        payload_dir.rmdir()
        manifest["status"] = "PASS"
    except BaseException as error:
        manifest["status"] = "FAIL"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["files"] = {
            p.name: sha(p)
            for p in run_dir.iterdir()
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(run_dir / "manifest.json", manifest)
        print(f"Run receipt: {run_dir}", file=sys.stderr)
        lock.close()
    if not overlap:
        report(run_dir)
    return run_dir


def report(run_dir):
    run_dir = Path(run_dir)
    # A rejected re-analysis must not leave an apparently current PASS summary.
    for name in ("summary.json", "summary.md"):
        (run_dir / name).unlink(missing_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    require(
        manifest["status"] == "PASS" and manifest["kind"] == "synthetic-transfer",
        "report requires a successful transfer run",
    )
    require("measurements.jsonl" in manifest["files"], "unhashed measurements")
    require(
        manifest["cells"]
        and len({tuple(sorted(c.items())) for c in manifest["cells"]})
        == len(manifest["cells"]),
        "empty or duplicate matrix",
    )
    for name, digest in manifest["files"].items():
        require(
            Path(name).name == name and sha(run_dir / name) == digest,
            "receipt file changed",
        )
    rows = [
        json.loads(line)
        for line in (run_dir / "measurements.jsonl").read_text().splitlines()
    ]
    require(
        len(rows) == len(manifest["cells"]) * manifest["repeats"] * 2,
        "incomplete matrix",
    )
    summary = []
    for cell in manifest["cells"]:
        selected = [r for r in rows if all(r.get(k) == v for k, v in cell.items())]
        validate_rows(
            selected, cell, manifest["repeats"], manifest["seed"], manifest["access"]
        )
        for operation in ("write", "read"):
            group = [r for r in selected if r["operation"] == operation]
            rates = [
                r["completed_bytes"] / r["wall_seconds"] / (1 << 20) for r in group
            ]
            cpu = [r["process_cpu_seconds"] * 1e6 / r["operations"] for r in group]
            summary.append(
                {
                    **cell,
                    "operation": operation,
                    "n": len(group),
                    "median_MiB_s": statistics.median(rates),
                    "min_MiB_s": min(rates),
                    "max_MiB_s": max(rates),
                    "median_process_cpu_us_op": statistics.median(cpu),
                }
            )
    write_json(run_dir / "summary.json", summary)
    lines = [
        "# Tutti synthetic transfer measurements",
        "",
        "All measured round trips passed full byte verification. These are submit-and-drain batches;",
        "batch depth is not observed NVMe queue depth. CPU covers the benchmark process only.",
        "No LMCache comparison, serving result, steady-state media claim or paper reproduction is implied.",
        "",
        "| I/O KiB | Batch | Operation | n | Median MiB/s | Min–max MiB/s | Process CPU µs/op |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['io_bytes'] // 1024} | {r['batch_depth']} | {r['operation']} | {r['n']} | "
            f"{r['median_MiB_s']:.1f} | {r['min_MiB_s']:.1f}–{r['max_MiB_s']:.1f} | {r['median_process_cpu_us_op']:.2f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(run_dir / "summary.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "plan",
            "doctor",
            "fetch",
            "build",
            "pipeline",
            "smoke",
            "bench",
            "overlap",
            "report",
        ),
    )
    parser.add_argument("--config", type=Path, default=TOP / ".config")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    try:
        if args.stage == "report":
            require(
                args.run is not None and str(args.run) != ".",
                "pass --run or make tutti-report RUN=/absolute/run",
            )
            report(args.run)
            return
        c = config(args.config)
        if args.stage == "plan":
            print(
                json.dumps(
                    {
                        "revision": c["TUTTI_REV"],
                        "build_commands": build_commands(c),
                        "cells": cells(c),
                        "writes_enabled": c.get("TUTTI_ALLOW_FILE_WRITES") == "y",
                        "run_after_build": c.get("TUTTI_RUN") == "y",
                    },
                    indent=2,
                )
            )
            return
        if args.stage == "doctor":
            doctor(c)
            return
        workspace = path_value(c, "TUTTI_WORK_DIR")
        workspace.mkdir(parents=True, exist_ok=True)
        with (workspace / ".workflow.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.stage == "fetch":
                fetch(c)
            elif args.stage == "build":
                build(c)
            elif args.stage == "pipeline":
                if c.get("TUTTI_RUN") == "y":
                    validate_runtime(c)
                    cells(c)
                doctor(c)
                fetch(c)
                build(c)
                if c.get("TUTTI_RUN") == "y":
                    benchmark(c, args.config)
            else:
                benchmark(
                    c,
                    args.config,
                    smoke=args.stage == "smoke",
                    overlap=args.stage == "overlap",
                )
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        KeyError,
        TypeError,
    ) as error:
        print(f"tutti: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
