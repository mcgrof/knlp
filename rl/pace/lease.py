"""A file-based lease that lets a person reclaim the GPU from training.

The training host has one GPU that is shared between unattended
learning runs and interactive use. Rather than schedule around a
calendar, the runs honour a hold flag: while the hold exists no
worker may use the GPU, and a running worker checkpoints and exits at
its next safe point. A supervisor waits for the hold to clear and
restarts the worker from its checkpoint.

State lives in one directory (``$KNLP_GPU_LEASE_DIR``, default
``~/.local/state/knlp/gpu-lease``):

    hold.json            present while the GPU is reserved for a person
    workers/<pid>.json   one file per live worker or supervisor

Everything here is plain files and POSIX signals, so any shell, agent
or cron job can drive it, and a stale file never blocks anything:
a worker record whose process is gone is ignored and removed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

# Exit status a worker uses to say "I stopped because of a hold and can
# be resumed from my checkpoint". 75 is EX_TEMPFAIL in sysexits.h.
EXIT_YIELD = 75

# Signal the pause command sends to workers so they yield promptly
# instead of waiting for their next natural check.
YIELD_SIGNAL = signal.SIGUSR1


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def default_state_dir() -> Path:
    env = os.environ.get("KNLP_GPU_LEASE_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    return Path(base) / "knlp" / "gpu-lease"


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


class GpuLease:
    """Hold flag plus worker registry in one state directory."""

    def __init__(self, state_dir: Optional[Path] = None):
        self.dir = Path(state_dir) if state_dir else default_state_dir()
        self.workers_dir = self.dir / "workers"
        self.hold_path = self.dir / "hold.json"
        self.workers_dir.mkdir(parents=True, exist_ok=True)

    # ---- hold -------------------------------------------------------

    def hold(self) -> Optional[dict]:
        """Return the hold record, or None when the GPU is available."""
        return _read_json(self.hold_path)

    def request_hold(self, reason: str = "", by: str = "") -> dict:
        record = {
            "reason": reason or "interactive use",
            "by": by or os.environ.get("USER", "unknown"),
            "host": socket.gethostname(),
            "since": _now(),
        }
        _write_json(self.hold_path, record)
        return record

    def release_hold(self) -> bool:
        try:
            self.hold_path.unlink()
        except FileNotFoundError:
            return False
        return True

    def wait_until_clear(
        self, poll: float = 5.0, timeout: Optional[float] = None
    ) -> bool:
        """Block while a hold exists. Returns False on timeout."""
        start = time.monotonic()
        while self.hold() is not None:
            if timeout is not None and time.monotonic() - start > timeout:
                return False
            time.sleep(poll)
        return True

    # ---- workers ----------------------------------------------------

    def _worker_path(self, pid: int) -> Path:
        return self.workers_dir / f"{pid}.json"

    def register(self, name: str, role: str = "worker", **extra: Any) -> dict:
        record = {
            "pid": os.getpid(),
            "name": name,
            "role": role,
            "host": socket.gethostname(),
            "started": _now(),
            "state": "running",
            "cmd": sys.argv,
        }
        record.update(extra)
        _write_json(self._worker_path(os.getpid()), record)
        return record

    def update(self, **fields: Any) -> None:
        path = self._worker_path(os.getpid())
        record = _read_json(path) or {"pid": os.getpid()}
        record.update(fields)
        record["updated"] = _now()
        _write_json(path, record)

    def unregister(self) -> None:
        try:
            self._worker_path(os.getpid()).unlink()
        except FileNotFoundError:
            pass

    def workers(self, prune: bool = True) -> list[dict]:
        """Live worker records; stale ones (dead pid) are dropped."""
        out = []
        for path in sorted(self.workers_dir.glob("*.json")):
            record = _read_json(path)
            if record is None:
                continue
            pid = int(record.get("pid", 0))
            if not _pid_alive(pid):
                if prune:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                continue
            out.append(record)
        return out

    def gpu_workers(self) -> list[dict]:
        return [w for w in self.workers() if w.get("role") == "worker"]

    def signal_workers(self, sig: int = YIELD_SIGNAL) -> int:
        n = 0
        for w in self.gpu_workers():
            try:
                os.kill(int(w["pid"]), sig)
                n += 1
            except OSError:
                pass
        return n

    def wait_until_idle(
        self, poll: float = 2.0, timeout: Optional[float] = None
    ) -> bool:
        """Block until no GPU worker is alive. Returns False on timeout."""
        start = time.monotonic()
        while self.gpu_workers():
            if timeout is not None and time.monotonic() - start > timeout:
                return False
            time.sleep(poll)
        return True


class YieldRequest:
    """Collects yield requests from the hold flag and from signals.

    A trainer creates one, calls ``check()`` at every safe point (an
    update boundary), and on True saves its checkpoint and exits with
    ``EXIT_YIELD``. ``install_signals()`` makes SIGUSR1 and SIGTERM set
    the flag so the pause command does not have to wait for the next
    natural check.
    """

    def __init__(self, lease: GpuLease, min_interval: float = 2.0):
        self.lease = lease
        self.min_interval = min_interval
        self._last = 0.0
        self._flag = False
        self.reason: Optional[str] = None

    def _on_signal(self, signum: int, frame: Any) -> None:
        self._flag = True
        self.reason = f"signal {signal.Signals(signum).name}"

    def install_signals(self) -> None:
        signal.signal(YIELD_SIGNAL, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def check(self, force: bool = False) -> bool:
        if self._flag:
            return True
        now = time.monotonic()
        if not force and now - self._last < self.min_interval:
            return False
        self._last = now
        hold = self.lease.hold()
        if hold is not None:
            self.reason = "hold: " + str(hold.get("reason", ""))
            self._flag = True
        return self._flag


def run_supervised(
    name: str,
    launch: Callable[[], int],
    lease: Optional[GpuLease] = None,
    max_restarts: int = 1000,
    poll: float = 5.0,
    log: Callable[[str], None] = print,
) -> int:
    """Run ``launch`` until it returns something other than EXIT_YIELD.

    ``launch`` starts the worker and returns its exit status. While a
    hold exists the supervisor waits before (re)launching. Returns the
    worker's final exit status.
    """
    lease = lease or GpuLease()
    lease.register(name, role="supervisor")
    try:
        restarts = 0
        while True:
            hold = lease.hold()
            if hold is not None:
                lease.update(state="waiting", hold=hold)
                log(f"[pace] hold by {hold.get('by')}: {hold.get('reason')}; waiting")
                lease.wait_until_clear(poll=poll)
                log("[pace] hold released; launching")
            lease.update(state="running", restarts=restarts)
            rc = launch()
            if rc != EXIT_YIELD:
                lease.update(state="finished", rc=rc)
                return rc
            restarts += 1
            log(f"[pace] worker yielded (restart {restarts})")
            if restarts > max_restarts:
                log("[pace] too many restarts; giving up")
                return rc
    finally:
        lease.unregister()


def iter_status(lease: GpuLease) -> Iterator[str]:
    hold = lease.hold()
    if hold:
        yield (
            f"HOLD   since {hold.get('since')} by {hold.get('by')}@{hold.get('host')}: "
            f"{hold.get('reason')}"
        )
    else:
        yield "FREE   no hold; training may use the GPU"
    workers = lease.workers()
    if not workers:
        yield "workers: none"
    for w in workers:
        yield (
            f"{w.get('role', '?'):10s} pid={w.get('pid')} state={w.get('state')} "
            f"name={w.get('name')} started={w.get('started')}"
            + (f" step={w['step']}" if "step" in w else "")
        )
