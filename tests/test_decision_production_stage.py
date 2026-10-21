# SPDX-License-Identifier: MIT
"""Exercise collector failure retention with real, isolated child processes."""

import json
import subprocess
import sys

import pytest

from tools.kv.decision import production_stage as stage


class Sampler:
    policy = {"driver_envelope_bytes": 100}
    rows = [{"driver_used_bytes": 50}]

    def check(self):
        pass

    def window(self, start, end, *, enforce):
        return {"eligible": True, "enforced": enforce}


class Server:
    def poll(self):
        return None


def test_failed_fixed_collector_is_retained_once_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "pinned", lambda argv, cpus: argv)
    counter = tmp_path / "attempt-count"
    script = (
        "import pathlib,sys; "
        "p=pathlib.Path(sys.argv[1]); "
        "p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
        "print('retained failure', flush=True); sys.exit(17)"
    )
    with pytest.raises(RuntimeError, match="collector failed: 17"):
        stage.child(
            [sys.executable, "-c", script, str(counter)],
            tmp_path,
            "quality",
            [0],
            Sampler(),
            Server(),
            enforce=True,
        )
    receipt = json.loads((tmp_path / "quality-command.json").read_text())
    assert receipt["returncode"] == 17
    assert receipt["application_retries"] == 0
    assert receipt["ended_ns"] > receipt["started_ns"]
    assert receipt["resources"]["enforced"]
    assert counter.read_text() == "x"
    assert "retained failure" in (tmp_path / "quality.log").read_text()


def test_resource_overrun_stops_the_owned_collector_and_retains_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stage, "pinned", lambda argv, cpus: argv)
    sampler = Sampler()
    sampler.rows = [{"driver_used_bytes": 101}]
    with pytest.raises(RuntimeError, match="common ceiling"):
        stage.child(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            "quality",
            [0],
            sampler,
            Server(),
            enforce=True,
        )
    receipt = json.loads((tmp_path / "quality-command.json").read_text())
    assert receipt["returncode"] == -15
    assert "common ceiling" in receipt["error"]


def test_terminate_only_the_new_command_group():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    assert stage.terminate(process) == -15
    assert process.poll() == -15
