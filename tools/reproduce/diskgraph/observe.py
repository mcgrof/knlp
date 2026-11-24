# SPDX-License-Identifier: MIT
"""Optional read-only NVMe observation while the live CPU workload runs."""

from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import subprocess
import time


def command(*argv):
    return subprocess.check_output(argv, text=True).strip()


@contextmanager
def observe(monitor, index, run):
    if monitor is None:
        yield
        return
    source, fs = command("findmnt", "-T", str(index), "-no", "SOURCE,FSTYPE").split()
    if fs != "xfs":
        raise ValueError("The validated optional FIEMAP join requires local XFS")
    device = Path(source).resolve().name
    partition = Path("/sys/class/block") / device
    leaf = (
        command("lsblk", "-ndo", "PKNAME", source)
        if (partition / "partition").exists()
        else device
    )
    if not re.fullmatch(r"nvme[0-9]+n[0-9]+", leaf):
        raise ValueError("The optional observer requires a local NVMe namespace")
    start = int((partition / "start").read_text()) if leaf != device else 0
    (run / "partition-start-sectors").write_text(f"{start}\n")
    files = [str(f) for f in index.rglob("*") if f.name == "edges.bin" and f.is_file()]
    with (run / "fiemap.txt").open("x") as output:
        subprocess.run(
            [shutil.which("filefrag") or "/usr/sbin/filefrag", "-b512", "-v", *files],
            stdout=output,
            check=True,
        )
    pid = run / "monitor.pid"
    log = run / "observer.log"
    capture = run / "nvme.jsonl"
    with log.open("x") as stderr:
        process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "sh",
                "-c",
                'umask 022; echo "$$" > "$1"; exec "$2" --disk "$3" --dur 300 --jsonl "$4"',
                "sh",
                str(pid),
                str(monitor.resolve()),
                leaf,
                str(capture),
            ],
            stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 15
            while "attached" not in log.read_text():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"NVMe observer failed to attach. See {log}")
                time.sleep(0.05)
            yield
            if process.poll() is not None:
                raise RuntimeError("NVMe observer ended before the workload")
            time.sleep(0.2)
        finally:
            if process.poll() is None and pid.is_file():
                subprocess.run(
                    ["sudo", "-n", "kill", "-INT", str(int(pid.read_text()))],
                    check=True,
                )
            if process.wait(timeout=15):
                raise RuntimeError(f"NVMe observer failed. See {log}")
