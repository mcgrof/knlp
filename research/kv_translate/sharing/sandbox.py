#!/usr/bin/env python3
"""Run untrusted test code where it can damage nothing and reach nothing.

Scoring a code task means executing what a model wrote, inside a repository
someone else wrote. Both get a private view of the machine: the file system
read-only, the home directory replaced by an empty one, no network, and one
writable directory, which is a disposable copy of the repository. A wall
clock limit and a memory limit end anything that runs away.

The sandbox is bubblewrap, which needs no privileges. There is no fallback
to running unconfined. If the sandbox cannot start, the result says so and
is never read as a failed test.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

PASS, FAIL, TIMEOUT, MEMORY, BROKEN = (
    "pass",
    "fail",
    "timeout",
    "memory",
    "sandbox_error",
)


def available():
    return shutil.which("bwrap") is not None


def command(argv, *, writable, cwd, readable=(), network=False, hide=(), env=None):
    """The bubblewrap invocation for one confined process."""
    cmd = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]  # fmt: skip
    if not network:
        cmd.append("--unshare-net")
    # credentials and other people's data live under these; none of it is
    # needed to run a test, so none of it is visible
    for path in (os.path.expanduser("~"), "/root", *hide):
        if os.path.isdir(path):
            cmd += ["--tmpfs", path]
    for path in readable:
        cmd += ["--ro-bind", path, path]
    cmd += ["--bind", writable, writable, "--chdir", cwd]
    merged = {"HOME": "/tmp", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    merged.update(env or {})
    for k, v in merged.items():
        cmd += ["--setenv", k, v]
    return cmd + list(argv)


def _tree_rss(pid):
    """Resident memory of a process and everything below it, in bytes."""
    total, seen, stack = 0, set(), [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/statm") as f:
                total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            with open(f"/proc/{p}/task/{p}/children") as f:
                stack += [int(c) for c in f.read().split()]
        except (OSError, ValueError, IndexError):
            continue
    return total


def run(argv, *, writable, cwd, timeout, memory_bytes, tail=4000, **kwargs):
    """Run one confined command. Returns its status and what it printed."""
    if not available():
        return {"status": BROKEN, "detail": "bubblewrap is not installed"}
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            command(argv, writable=writable, cwd=cwd, **kwargs),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as e:
        return {"status": BROKEN, "detail": f"could not start the sandbox: {e}"}
    status, peak = None, 0
    chunks = []
    os.set_blocking(proc.stdout.fileno(), False)
    while True:
        try:
            data = proc.stdout.read()
        except OSError:
            data = None
        if data:
            chunks.append(data)
        if proc.poll() is not None:
            break
        rss = _tree_rss(proc.pid)
        peak = max(peak, rss)
        if rss > memory_bytes:
            status = MEMORY
        elif time.time() - t0 > timeout:
            status = TIMEOUT
        if status:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            break
        time.sleep(0.05)
    try:
        rest = proc.stdout.read()
        if rest:
            chunks.append(rest)
    except OSError:
        pass
    out = b"".join(chunks).decode("utf-8", "replace")
    if status is None:
        # bubblewrap reports its own failures on its own lines
        if proc.returncode != 0 and out.lstrip().startswith("bwrap:"):
            status = BROKEN
        else:
            status = PASS if proc.returncode == 0 else FAIL
    return {
        "status": status,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "peak_rss_bytes": peak,
        "output_tail": out[-tail:],
    }
