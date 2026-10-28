# SPDX-License-Identifier: MIT
"""Doctor, smoke, run and report commands for the mounted-filesystem baseline.

The run writes a small set of immutable objects to a directory the user
names, then reads each object back as one whole request: open, read and
close. Every request is timed and its bytes are checked against the
manifest digest after the timed region. Records go to an output directory
the user names, which must be outside this source tree.

Nothing here needs root, drops caches or changes kernel or service
settings. Cache state is therefore recorded as uncontrolled: buffered
reads may be served by the client page cache, while direct reads bypass
it on filesystems that honor O_DIRECT.
"""

import argparse
import hashlib
import json
import mmap
import platform
import subprocess
import sys
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

from . import objects, records

ROOT = Path(__file__).resolve().parents[3]
PREFIX = "CONFIG_KNLP_RPL2_"
DEFAULTS = {
    "profile": "smoke",
    "target_dir": "",
    "output_dir": "",
    "object_bytes": 8 * 1024 * 1024,
    "objects": 16,
    "requests": 64,
    "seed": 1,
    "io_mode": "direct",
}


def configuration(path):
    cfg = dict(DEFAULTS)
    if path is None or not Path(path).exists():
        return cfg
    lines = Path(path).read_text().splitlines()
    if "CONFIG_KNLP_REPRODUCE_RPL2=y" not in lines:
        raise SystemExit("select an rpl2 defconfig first, for example rpl2-smoke")
    values = {}
    for line in lines:
        if line.startswith(PREFIX) and "=" in line:
            key, value = line.split("=", 1)
            values[key.removeprefix(PREFIX).lower()] = (
                json.loads(value) if value.startswith('"') else value
            )
    if values.get("mounted_fs") == "y":
        cfg["profile"] = "mounted-fs"
    if values.get("io_buffered") == "y":
        cfg["io_mode"] = "buffered"
    for name in ("target_dir", "output_dir"):
        if name in values:
            cfg[name] = values[name]
    for name in ("object_bytes", "objects", "requests", "seed"):
        if name in values:
            cfg[name] = int(values[name])
    return cfg


def inside_tree(path):
    resolved = Path(path).expanduser().resolve()
    return resolved == ROOT or ROOT in resolved.parents


def source():
    info = {"repository": "knlp", "commit": "unknown", "modified": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", "tools/reproduce"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        info["modified"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def runtime():
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
    }


def _need(value, what, flag):
    if not value:
        raise SystemExit(
            f"no {what} given: pass {flag}=/path to make, or set it with "
            "make menuconfig"
        )
    return Path(value).expanduser().resolve()


def doctor(cfg):
    """Print one line per check; return the process exit status."""
    failed = False

    def line(state, message):
        nonlocal failed
        failed |= state == "FAIL"
        print(f"{state:5} {message}")

    version = sys.version_info
    line(
        "PASS" if version >= (3, 10) else "FAIL",
        f"Python {platform.python_version()} (3.10 or newer is required)",
    )
    if cfg["profile"] == "smoke":
        line("INFO", "smoke profile: no target directory is used")
        return 1 if failed else 0
    target = _need(cfg["target_dir"], "target directory", "RPL2_TARGET")
    output = _need(cfg["output_dir"], "output directory", "RPL2_OUTPUT")
    for name, path in (("target", target), ("output", output)):
        if inside_tree(path):
            line("FAIL", f"{name} {path} is inside the source tree")
    if not target.is_dir():
        line("FAIL", f"target {target} is not an existing directory")
        return 1
    mount = objects.mount_of(target)
    version = f", NFS version {mount['nfs_version']}" if "nfs_version" in mount else ""
    line("INFO", f"target filesystem type {mount['type']}{version}")
    probe = target / ".rpl2-write-probe"
    try:
        probe.write_bytes(b"x")
        probe.unlink()
        line("PASS", f"target {target} is writable")
    except OSError as exc:
        line("FAIL", f"target {target} is not writable: {exc.strerror}")
        return 1
    name = objects.set_name(cfg["seed"], cfg["object_bytes"], cfg["objects"])
    need = cfg["object_bytes"] * cfg["objects"]
    if (target / name / objects.MANIFEST).exists():
        line("PASS", f"object set {name} exists and will be reused after checks")
    else:
        have = objects.free_bytes(target)
        line(
            "PASS" if have > need * 1.1 else "FAIL",
            f"free space {have} bytes for {need} bytes of objects plus 10%",
        )
    try:
        objects.check_size(cfg["object_bytes"], cfg["io_mode"])
        line("PASS", f"object size {cfg['object_bytes']} bytes")
    except objects.ObjectError as exc:
        line("FAIL", str(exc))
    supported, reason = objects.probe_direct(target)
    if cfg["io_mode"] == "direct":
        line("PASS" if supported else "SKIP", f"direct I/O: {reason}")
        if not supported:
            print("      the run will record a skipped direct-I/O baseline;")
            print("      choose the buffered defconfig option to measure instead")
    else:
        line("INFO", f"direct I/O (not selected): {reason}")
    if output.exists() and not output.is_dir():
        line("FAIL", f"output {output} exists and is not a directory")
    else:
        line("PASS", f"output directory {output}")
    return 1 if failed else 0


def smoke():
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent / "tests"), top_level_dir=str(ROOT)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def write_line(stream, record):
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


def run(cfg, stamp=None):
    """Write or reuse the object set, time every request, write records."""
    target = _need(cfg["target_dir"], "target directory", "RPL2_TARGET")
    output = _need(cfg["output_dir"], "output directory", "RPL2_OUTPUT")
    for path in (target, output):
        if inside_tree(path):
            raise SystemExit(f"{path} is inside the source tree; choose another")
    size, mode = cfg["object_bytes"], cfg["io_mode"]
    objects.check_size(size, mode)
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output / f"rpl2-{stamp}"
    run_dir.mkdir(parents=True)
    header = {
        "schema": records.SCHEMA,
        "type": "header",
        "status": "ok",
        "source": source(),
        "runtime": runtime(),
        "filesystem": objects.mount_of(target),
        "io_mode_requested": mode,
        "io_mode_actual": mode,
        "object_bytes": size,
        "objects": cfg["objects"],
        "requests": cfg["requests"],
        "seed": cfg["seed"],
        "cache_state": "uncontrolled",
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = run_dir / "records.jsonl"
    if mode == "direct":
        supported, reason = objects.probe_direct(target)
        if not supported:
            header |= {"status": "skipped", "io_mode_actual": "none"}
            header["reason"] = f"direct I/O unsupported: {reason}"
            with path.open("x") as stream:
                write_line(stream, header)
            print(f"SKIP  {header['reason']}")
            print(f"      header written to {path}")
            return 0
    directory, manifest = objects.prepare(target, cfg["seed"], size, cfg["objects"])
    items = manifest["objects"]
    buffer = mmap.mmap(-1, size)
    bad = 0
    with path.open("x") as stream:
        write_line(stream, header)
        for index in range(cfg["requests"]):
            item = items[index % len(items)]
            result = {"validation": "error"}
            start = time.perf_counter_ns()
            try:
                count = objects.read_object(
                    directory / item["file"], size, mode, buffer
                )
                end = time.perf_counter_ns()
            except OSError as exc:
                end = time.perf_counter_ns()
                count = 0
                result["error"] = exc.strerror or str(exc)
            else:
                if count != size:
                    result["validation"] = "short_read"
                elif hashlib.sha256(buffer[:size]).hexdigest() == item["sha256"]:
                    result["validation"] = "match"
                else:
                    result["validation"] = "mismatch"
            bad += result["validation"] != "match"
            write_line(
                stream,
                {
                    "schema": records.SCHEMA,
                    "type": "request",
                    "request": index,
                    "key": item["key"],
                    "generation": item["generation"],
                    "object_bytes": size,
                    "bytes_read": count,
                    "io_mode": mode,
                    "start_ns": start,
                    "end_ns": end,
                    "latency_ns": end - start,
                }
                | result,
            )
    buffer.close()
    print(f"wrote {cfg['requests']} request records to {path}")
    if bad:
        print(f"FAIL  {bad} requests did not return the expected bytes")
        return 1
    return 0


def latest(output):
    runs = sorted(Path(output).expanduser().glob("rpl2-*/records.jsonl"))
    if not runs:
        raise SystemExit(f"no rpl2 records under {output}")
    return runs[-1]


def report(path):
    header, requests = records.load(path)
    summary = records.summarize(header, requests)
    destination = Path(path).with_name("summary.json")
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary written to {destination}")
    if summary["status"] == "ok" and summary["verified_requests"] != len(requests):
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("doctor", "smoke", "run", "report"))
    parser.add_argument("--config", type=Path, help="Kconfig .config to read")
    parser.add_argument("--target", default="", help="existing directory to use")
    parser.add_argument("--output", default="", help="directory for records")
    parser.add_argument("--records", default="", help="records.jsonl to report")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return smoke()
    cfg = configuration(args.config)
    if args.target:
        cfg["target_dir"] = args.target
    if args.output:
        cfg["output_dir"] = args.output
    if args.command == "doctor":
        return doctor(cfg)
    if args.command == "run":
        if cfg["profile"] != "mounted-fs":
            raise SystemExit("run needs the rpl2-mounted-fs defconfig")
        return run(cfg)
    path = args.records or latest(
        _need(cfg["output_dir"], "output directory", "RPL2_OUTPUT")
    )
    return report(path)


if __name__ == "__main__":
    sys.exit(main())
