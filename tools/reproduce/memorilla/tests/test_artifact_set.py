# SPDX-License-Identifier: MIT
"""Pinned readiness records must not accept partial or changed imports."""

from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("safetensors")
from tools.reproduce.context_artifacts.artifacts import ArtifactError
from tools.reproduce.memorilla import artifact_set


def test_atomic_install_is_idempotent_and_never_overwrites(tmp_path):
    path = tmp_path / "sets" / "ready.json"
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(lambda _: artifact_set.install_bytes(path, b"complete"), range(8))
        )
    assert path.read_bytes() == b"complete"
    with pytest.raises(ArtifactError, match="differs"):
        artifact_set.install_bytes(path, b"different")
    assert path.read_bytes() == b"complete"
    assert not list(path.parent.glob(".import-*"))


def test_pinned_set_rejects_changed_source_collections_and_content(tmp_path):
    identity = {"index": "original", "checkpoint": "original"}
    key = artifact_set.publish(tmp_path, identity, {1: "a" * 64, 3: "b" * 64})
    assert artifact_set.load(tmp_path, identity, key, [1, 3]) == {
        1: "a" * 64,
        3: "b" * 64,
    }
    with pytest.raises(ArtifactError, match="source or checkpoint"):
        artifact_set.load(tmp_path, identity | {"index": "changed"}, key, [1, 3])
    with pytest.raises(ArtifactError, match="collections"):
        artifact_set.load(tmp_path, identity, key, [1])
    path = tmp_path / "sets" / f"{key}.json"
    body = json.loads(path.read_text())
    body["ids"]["1"] = "c" * 64
    path.write_text(json.dumps(body))
    with pytest.raises(ArtifactError, match="pin"):
        artifact_set.load(tmp_path, identity, key, [1, 3])


def test_failed_sync_has_no_ready_file(tmp_path, monkeypatch):
    def failed(_):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(artifact_set.os, "fsync", failed)
    path = tmp_path / "ready.json"
    with pytest.raises(OSError, match="fsync"):
        artifact_set.install_bytes(path, b"complete")
    assert not path.exists()
    assert not list(tmp_path.glob(".import-*"))


def test_extent_check_is_separate_from_payload_hashing(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"abcd")
    artifact_set.check_extent(SimpleNamespace(path=path, offset=1, length=3))
    with pytest.raises(ArtifactError, match="truncated"):
        artifact_set.check_extent(SimpleNamespace(path=path, offset=1, length=4))
