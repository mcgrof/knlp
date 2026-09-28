# SPDX-License-Identifier: GPL-2.0
"""The sandbox: no network, no home directory, one writable place, and limits."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap

import pytest

from research.kv_translate.sharing import sandbox

needs_sandbox = pytest.mark.skipif(
    not sandbox.available(), reason="bubblewrap is not installed"
)


@needs_sandbox
def test_the_sandbox_has_no_network_no_home_and_one_writable_place(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    probe = textwrap.dedent("""
        import os, socket
        out = []
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=2); out.append("NET")
        except OSError:
            out.append("nonet")
        out.append("home" if os.listdir(os.path.expanduser("~")) else "nohome")
        try:
            open("/usr/x_probe", "w"); out.append("WROTE")
        except OSError:
            out.append("readonly")
        open("ok.txt", "w").write("1"); out.append("wrote_work")
        print(" ".join(out))
        """)
    home = os.path.expanduser("~")
    r = sandbox.run(
        [sys.executable, "-c", probe],
        writable=str(work),
        cwd=str(work),
        timeout=30,
        memory_bytes=2**30,
        env={"HOME": home},
    )
    assert r["status"] == sandbox.PASS, r
    assert r["output_tail"].split() == ["nonet", "nohome", "readonly", "wrote_work"]
    assert (work / "ok.txt").exists()


@needs_sandbox
def test_runaway_code_is_stopped_and_named(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    kw = dict(writable=str(work), cwd=str(work))
    slow = sandbox.run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=1,
        memory_bytes=2**30,
        **kw,
    )
    assert slow["status"] == sandbox.TIMEOUT and slow["seconds"] < 10
    big = sandbox.run(
        [
            sys.executable,
            "-c",
            "x = bytearray(600 * 2**20); import time; time.sleep(20)",
        ],
        timeout=30,
        memory_bytes=200 * 2**20,
        **kw,
    )
    assert big["status"] == sandbox.MEMORY
    bad = sandbox.run(
        [sys.executable, "-c", "raise SystemExit(3)"],
        timeout=10,
        memory_bytes=2**30,
        **kw,
    )
    assert bad["status"] == sandbox.FAIL and bad["returncode"] == 3


def test_without_a_sandbox_nothing_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    r = sandbox.run(
        ["true"], writable=str(tmp_path), cwd=str(tmp_path), timeout=1, memory_bytes=1
    )
    assert r["status"] == sandbox.BROKEN
