# SPDX-License-Identifier: MIT
"""Verify portable configuration and failure propagation without GPU allocation."""

import json
from pathlib import Path
import subprocess
import sys

import kconfiglib
import pytest

ROOT = Path(__file__).resolve().parents[4]
PRESETS = sorted((ROOT / "defconfigs").glob("context-artifacts-*"))


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_complete_roundtrip_and_dispatch(tmp_path, preset):
    config = tmp_path / ".config"
    saved = tmp_path / "saved"
    subprocess.run(
        [
            "make",
            "defconfig",
            f"DEFCONFIG={preset.name}",
            f"CONFIG_FILE={config}",
            f"PYTHON={sys.executable}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["make", "olddefconfig", f"CONFIG_FILE={config}", f"PYTHON={sys.executable}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    kconf = kconfiglib.Kconfig(str(ROOT / "Kconfig"), warn_to_stderr=False)
    kconf.load_config(str(config))
    expected = {
        name: sym.str_value
        for name, sym in kconf.syms.items()
        if name.startswith("KNLP_CA_") or name == "KNLP_CONTEXT_ARTIFACTS"
    }
    subprocess.run(
        [
            "make",
            "savedefconfig",
            f"DEFCONFIG={saved}",
            "DEFCONFIG_DIR=/",
            f"CONFIG_FILE={config}",
            f"PYTHON={sys.executable}",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    kconf.load_config(str(saved))
    assert expected == {name: kconf.syms[name].str_value for name in expected}
    kconf.write_config(str(config))
    command = subprocess.check_output(
        ["make", "-n", f"CONFIG_FILE={config}", f"PYTHON={sys.executable}"],
        cwd=ROOT,
        text=True,
    )
    assert "context_artifacts.run doctor" in command
    assert "context_artifacts.run test" in command
    assert "gpt2/train.py" not in command
    conflicting = subprocess.run(
        ["make", "-n", f"CONFIG_FILE={config}", "CONFIG_KNLP_REPRODUCE_DECODE=y"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert conflicting.returncode != 0
    assert "Conflicting" in conflicting.stderr


def test_unprepared_default_fails_closed(tmp_path):
    config = tmp_path / ".config"
    config.write_text(
        (ROOT / "defconfigs/context-artifacts-memorilla-files").read_text()
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.reproduce.context_artifacts.run",
            "test",
            "--config",
            str(config),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "BLOCKED"
