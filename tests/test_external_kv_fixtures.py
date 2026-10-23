from pathlib import Path

import pytest

from tools.kv.native.current import fixture_root


def test_external_captured_input_root_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("KNLP_KV_FIXTURES", raising=False)
    with pytest.raises(ValueError, match="set KNLP_KV_FIXTURES"):
        fixture_root()
    monkeypatch.setenv("KNLP_KV_FIXTURES", str(tmp_path))
    with pytest.raises(ValueError, match="current directory"):
        fixture_root()
    (tmp_path / "current").mkdir()
    assert fixture_root() == tmp_path / "current"


def test_captured_inputs_cannot_be_kept_in_source(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("KNLP_KV_FIXTURES", str(root / "tools/kv/native/fixtures"))
    with pytest.raises(ValueError, match="outside the checkout"):
        fixture_root()
