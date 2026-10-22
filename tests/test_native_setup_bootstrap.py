# SPDX-License-Identifier: MIT
"""Run the complete bootstrap control flow with external operations stubbed."""

import hashlib
import json
from pathlib import Path
import types

import pytest

from tools.kv.native import october_setup as setup


@pytest.mark.parametrize(
    "recipe", ["historical-compact", "current", "recent-transfer", "decision-http"]
)
@pytest.mark.parametrize("corrupt", [False, True])
def test_host_python_completes_bootstrap_and_keeps_wheel_gate(
    tmp_path, monkeypatch, recipe, corrupt
):
    # Run this test under Python 3.10 too. Removing the API also detects an
    # accidental dependency on it when the test runner is a newer Python.
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    root = tmp_path / "installation"
    pins = setup.recipe_pins(recipe)
    commands = []
    wheel = b"known wheel fixture" * 100000
    monkeypatch.setattr(setup, "WHEEL_SHA", hashlib.sha256(wheel).hexdigest())
    monkeypatch.setattr(setup, "EVENTS", [])
    monkeypatch.setattr("sys.argv", ["installer", str(root), "--recipe", recipe])

    def download(url, destination):
        assert url == setup.WHEEL
        destination.write_bytes(wheel + (b"changed" if corrupt else b""))

    def external(argv, **kwargs):
        commands.append(argv)
        if argv[:3] == ["uv", "pip", "freeze"]:
            kwargs["stdout"].write("fixture==1\n")
        return types.SimpleNamespace(returncode=0)

    def revision(argv, **kwargs):
        assert argv[:2] == ["git", "-C"]
        assert argv[3:] == ["rev-parse", "HEAD"]
        return pins[Path(argv[2]).name][1] + "\n"

    monkeypatch.setattr(setup.urllib.request, "urlretrieve", download)
    monkeypatch.setattr(setup.subprocess, "run", external)
    monkeypatch.setattr(setup.subprocess, "check_output", revision)
    if corrupt:
        with pytest.raises(RuntimeError, match="binary digest differs"):
            setup.main()
        assert not (root / "setup.json").exists()
        assert not any("-e" in argv for argv in commands)
        return

    setup.main()
    receipt = json.loads((root / "setup.json").read_text())
    assert receipt["recipe"] == recipe
    assert receipt["wheel_sha256"] == setup.WHEEL_SHA
    assert receipt["sources"] == {k: list(v) for k, v in pins.items()}
    assert receipt["source_transfer"] is None
    assert (root / "requirements.lock").read_text() == "fixture==1\n"
    assert sum("-e" in argv for argv in commands) == 2
    assert any(
        argv[0] == str(root / "env/bin/python")
        and "torch.cuda.is_available()" in argv[-1]
        for argv in commands
    )
