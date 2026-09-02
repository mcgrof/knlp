"""The GPU lease: hold, worker registry, yield request, supervisor."""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from rl.pace.lease import EXIT_YIELD, GpuLease, YieldRequest, run_supervised


def test_hold_round_trip(tmp_path):
    lease = GpuLease(tmp_path)
    assert lease.hold() is None
    rec = lease.request_hold("testing", by="pytest")
    assert rec["reason"] == "testing"
    assert lease.hold()["by"] == "pytest"
    assert lease.release_hold() is True
    assert lease.hold() is None
    assert lease.release_hold() is False


def test_worker_registry_prunes_dead_pids(tmp_path):
    lease = GpuLease(tmp_path)
    lease.register("me", role="worker")
    assert [w["name"] for w in lease.gpu_workers()] == ["me"]
    # forge a record for a pid that cannot be alive
    stale = lease.workers_dir / "999999999.json"
    stale.write_text('{"pid": 999999999, "name": "ghost", "role": "worker"}')
    assert [w["name"] for w in lease.gpu_workers()] == ["me"]
    assert not stale.exists()
    lease.unregister()
    assert lease.gpu_workers() == []


def test_yield_request_sees_hold_and_signal(tmp_path):
    lease = GpuLease(tmp_path)
    req = YieldRequest(lease, min_interval=0.0)
    assert req.check() is False
    lease.request_hold("go away")
    assert req.check() is True
    assert "hold" in req.reason
    lease.release_hold()
    req2 = YieldRequest(lease, min_interval=0.0)
    req2._on_signal(signal.SIGUSR1, None)
    assert req2.check() is True
    assert "SIGUSR1" in req2.reason


def test_supervisor_restarts_after_yield(tmp_path):
    lease = GpuLease(tmp_path)
    calls = []

    def launch():
        calls.append(time.monotonic())
        return EXIT_YIELD if len(calls) < 3 else 0

    rc = run_supervised(
        "t", launch, lease=lease, max_restarts=5, poll=0.01, log=lambda s: None
    )
    assert rc == 0
    assert len(calls) == 3
    assert lease.workers() == []


def test_supervisor_waits_for_hold_release(tmp_path):
    lease = GpuLease(tmp_path)
    lease.request_hold("busy")
    seen_hold = []

    def launch():
        seen_hold.append(lease.hold())
        return 0

    # release the hold from a helper process shortly after start
    helper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import time
                from rl.pace.lease import GpuLease
                time.sleep(0.3)
                GpuLease({str(tmp_path)!r}).release_hold()
                """),
        ],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    rc = run_supervised("t", launch, lease=lease, poll=0.05, log=lambda s: None)
    helper.wait(timeout=10)
    assert rc == 0
    assert seen_hold == [None]


def test_ctl_cli_status_pause_resume(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.getcwd(), "KNLP_GPU_LEASE_DIR": str(tmp_path)}

    def ctl(*a):
        return subprocess.run(
            [sys.executable, "-m", "rl.pace.ctl", *a],
            env=env,
            capture_output=True,
            text=True,
        )

    out = ctl("status")
    assert out.returncode == 0 and "FREE" in out.stdout
    out = ctl("pause", "--reason", "a person needs the GPU")
    assert out.returncode == 0 and "hold set" in out.stdout
    out = ctl("status")
    assert "HOLD" in out.stdout and "a person needs the GPU" in out.stdout
    out = ctl("resume")
    assert out.returncode == 0 and "released" in out.stdout
    assert "FREE" in ctl("status").stdout
