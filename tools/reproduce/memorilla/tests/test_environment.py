# SPDX-License-Identifier: MIT
"""Installed wheels must preserve source identity without requiring a .git file."""

from types import SimpleNamespace

import pytest

from tools.reproduce.memorilla.environment import verify_package


def test_wheel_code_is_checked_against_pinned_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    package = source / "example"
    installed = tmp_path / "site-packages" / "example"
    package.mkdir(parents=True)
    installed.mkdir(parents=True)
    for directory in [package, installed]:
        (directory / "__init__.py").write_text("VALUE = 7\n")
    monkeypatch.setattr(
        "tools.reproduce.memorilla.environment.subprocess.check_output",
        lambda *a, **kw: "pinned\n",
    )
    monkeypatch.setattr(
        "tools.reproduce.memorilla.environment.subprocess.run", lambda *a, **kw: None
    )
    module = SimpleNamespace(__name__="example", __file__=installed / "__init__.py")
    result = verify_package(module, source, "pinned")
    assert result["installation"] == "wheel" and result["python_files"] == 1
    extra = installed / "shadow.py"
    extra.write_text("VALUE = 9\n")
    with pytest.raises(ValueError, match="unexpected Python"):
        verify_package(module, source, "pinned")
    extra.unlink()
    (installed / "__init__.py").write_text("VALUE = 8\n")
    with pytest.raises(ValueError, match="differs"):
        verify_package(module, source, "pinned")
    with pytest.raises(ValueError, match="revision"):
        verify_package(module, source, "different")
