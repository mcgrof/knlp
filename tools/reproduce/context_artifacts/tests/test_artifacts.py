# SPDX-License-Identifier: MIT
"""CPU contract tests; these do not qualify model serving or device storage."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from ..artifacts import (
    ArtifactError,
    ConsumerIdentity,
    ContextArtifactManifest,
    ContextArtifactRegistry,
    DocumentMetadata,
    InputMetadata,
    PrefixMetadata,
    TensorSpec,
    attachment_key,
    digest_file,
    inspect_payload,
)

CONSUMER = ConsumerIdentity("decoder", "a" * 40)


def fixture(tmp_path, kind="document_embeddings", value=1, length=17):
    path = tmp_path / f"{kind}-{value}-{length}.safetensors"
    if kind == "document_embeddings":
        tensors = {"embeddings": np.full((length, 4), value, dtype=np.float16)}
        metadata = DocumentMetadata(
            ConsumerIdentity("encoder", "b" * 40),
            "c" * 64,
            hashlib.sha256(str(value).encode()).hexdigest(),
            length,
            4,
        )
    elif kind == "input_embeddings":
        tensors = {"embeddings": np.full((length, 4), value, dtype=np.float32)}
        metadata = InputMetadata(
            "d" * 64,
            "e" * 64,
            hashlib.sha256(str(value).encode()).hexdigest(),
            "f" * 64,
            "0" * 64,
            "system_message",
            tuple(range(length)),
            4,
        )
    else:
        physical = (length + 15) // 16 * 16
        tensors = {
            f"{plane}.{layer}": np.full((physical, 2, 4), value, dtype=np.float16)
            for layer in range(2)
            for plane in ("key", "value")
        }
        metadata = PrefixMetadata(
            CONSUMER,
            "c" * 64,
            "d" * 64,
            2,
            2,
            4,
            length,
            physical,
            16,
            tuple(range(length)),
            "THD",
            "none",
        )
    save_file(tensors, str(path))
    manifest = ContextArtifactManifest(
        1,
        "test",
        kind,
        "fixture",
        "1" * 40,
        "2" * 64,
        (CONSUMER,),
        digest_file(path),
        inspect_payload(path),
        metadata,
    )
    return path, manifest


@pytest.mark.parametrize(
    "kind", ["document_embeddings", "input_embeddings", "prefix_kv"]
)
@pytest.mark.parametrize("length", [1, 15, 16, 17, 32, 33])
def test_roundtrip_eviction_and_fresh_process(tmp_path, kind, length):
    path, manifest = fixture(tmp_path, kind, length=length)
    registry = ContextArtifactRegistry(tmp_path / "store")
    artifact_id = registry.register(manifest, path)
    assert registry.register(manifest, path) == artifact_id
    assert registry.load(artifact_id, CONSUMER) == path.read_bytes()
    with pytest.raises(ArtifactError, match="evict"):
        registry.restore(artifact_id, CONSUMER)
    assert registry.evict(artifact_id)
    assert not registry.resident
    assert registry.restore(artifact_id, CONSUMER) == path.read_bytes()
    registry.close()
    code = """import hashlib, sys
from tools.reproduce.context_artifacts.artifacts import ContextArtifactRegistry, ConsumerIdentity
r = ContextArtifactRegistry(sys.argv[1])
b = r.restore(sys.argv[2], ConsumerIdentity("decoder", "a" * 40))
print(hashlib.sha256(b).hexdigest())
r.close()
"""
    result = subprocess.check_output(
        [sys.executable, "-c", code, str(tmp_path / "store"), artifact_id], text=True
    )
    assert result.strip() == manifest.payload_digest


@pytest.mark.parametrize(
    "kind", ["document_embeddings", "input_embeddings", "prefix_kv"]
)
def test_equal_shape_payload_isolation_and_order(tmp_path, kind):
    path_a, a = fixture(tmp_path, kind, 1)
    path_b, b = fixture(tmp_path, kind, 2)
    registry = ContextArtifactRegistry(tmp_path / "store")
    assert a.artifact_id != b.artifact_id
    for path, manifest in [(path_a, a), (path_b, b), (path_a, a), (path_b, b)]:
        registry.register(manifest, path)
        assert registry.load(manifest.artifact_id, CONSUMER) == path.read_bytes()
    assert attachment_key([a, b], CONSUMER, "0" * 64, "1" * 64) != attachment_key(
        [b, a], CONSUMER, "0" * 64, "1" * 64
    )
    registry.close()


def test_query_and_version_identity(tmp_path):
    _, manifest = fixture(tmp_path, "input_embeddings")
    changed_query = dataclasses.replace(
        manifest, metadata=dataclasses.replace(manifest.metadata, query_digest="9" * 64)
    )
    changed_checkpoint = dataclasses.replace(manifest, checkpoint_digest="9" * 64)
    changed_namespace = dataclasses.replace(manifest, namespace="other")
    assert (
        len(
            {
                m.artifact_id
                for m in (
                    manifest,
                    changed_query,
                    changed_checkpoint,
                    changed_namespace,
                )
            }
        )
        == 4
    )


def test_corrupt_import_never_published(tmp_path):
    path, manifest = fixture(tmp_path)
    path.write_bytes(path.read_bytes()[:-1] + b"!")
    registry = ContextArtifactRegistry(tmp_path / "store")
    with pytest.raises(ArtifactError, match="digest"):
        registry.register(manifest, path)
    assert not list(registry.objects.iterdir())
    assert registry.db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
    registry.close()


def test_corrupt_durable_payload_rejected_even_when_resident(tmp_path):
    path, manifest = fixture(tmp_path)
    registry = ContextArtifactRegistry(tmp_path / "store")
    registry.register(manifest, path)
    registry.load(manifest.artifact_id, CONSUMER)
    payload = registry.objects / manifest.artifact_id / "payload.safetensors"
    payload.write_bytes(payload.read_bytes()[:-1] + b"!")
    with pytest.raises(ArtifactError, match="digest"):
        registry.load(manifest.artifact_id, CONSUMER)
    registry.close()


def test_missing_incompatible_and_manifest_corruption(tmp_path):
    path, manifest = fixture(tmp_path)
    registry = ContextArtifactRegistry(tmp_path / "store")
    with pytest.raises(ArtifactError, match="missing"):
        registry.load(manifest.artifact_id, CONSUMER)
    registry.register(manifest, path)
    with pytest.raises(ArtifactError, match="consumer"):
        registry.load(manifest.artifact_id, ConsumerIdentity("decoder", "b" * 40))
    (registry.objects / manifest.artifact_id / "manifest.json").write_text("{}")
    with pytest.raises(ArtifactError, match="manifest"):
        registry.resolve(manifest.artifact_id, CONSUMER)
    registry.close()


def test_interrupted_payload_is_invisible_and_retry_recovers(tmp_path, monkeypatch):
    path, manifest = fixture(tmp_path)
    registry = ContextArtifactRegistry(tmp_path / "store")
    real_fsync = os.fsync
    monkeypatch.setattr(
        os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("interrupted"))
    )
    with pytest.raises(OSError, match="interrupted"):
        registry.register(manifest, path)
    assert registry.db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
    assert not list(registry.objects.iterdir())
    monkeypatch.setattr(os, "fsync", real_fsync)
    registry.register(manifest, path)
    assert registry.load(manifest.artifact_id, CONSUMER) == path.read_bytes()
    registry.close()


@pytest.mark.parametrize(
    "change",
    [
        {"logical_tokens": 16},
        {"positions": (1, 3)},
        {"layout": "unknown"},
        {"physical_tokens": 64},
        {"quantization": "fp8"},
        {"model": ConsumerIdentity("wrong", "a" * 40)},
    ],
)
def test_invalid_prefix_contract(tmp_path, change):
    _, manifest = fixture(tmp_path, "prefix_kv")
    manifest = dataclasses.replace(
        manifest, metadata=dataclasses.replace(manifest.metadata, **change)
    )
    with pytest.raises(ArtifactError):
        manifest.validate()


def test_document_metadata_has_no_kv_requirement(tmp_path):
    _, manifest = fixture(tmp_path)
    manifest.validate()
    assert (
        ContextArtifactManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
        == manifest
    )
    with pytest.raises(ArtifactError, match="typed"):
        dataclasses.replace(manifest, representation="prefix_kv").validate()


def test_payload_ranges_and_bfloat16(tmp_path):
    path = tmp_path / "bfloat.safetensors"
    header = {"embeddings": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 8)
    assert inspect_payload(path) == (TensorSpec("embeddings", (2, 2), "BF16"),)
    path.write_bytes(path.read_bytes() + b"!")
    with pytest.raises(ArtifactError, match="payload"):
        inspect_payload(path)


def test_concurrent_publication_and_orphan_retry(tmp_path):
    path, manifest = fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    code = """import json, os, sys
from pathlib import Path
import tools.reproduce.context_artifacts.artifacts as module
m = module.ContextArtifactManifest.from_dict(json.loads(Path(sys.argv[2]).read_text()))
r = module.ContextArtifactRegistry(sys.argv[1])
if sys.argv[4] == "interrupt":
    original = module._fsync_dir
    def interrupted(path):
        if path == r.objects and (path / m.artifact_id).is_dir():
            os._exit(23)
        original(path)
    module._fsync_dir = interrupted
print(r.register(m, sys.argv[3]))
r.close()
"""
    argv = [
        sys.executable,
        "-c",
        code,
        str(tmp_path / "store"),
        str(manifest_path),
        str(path),
    ]
    assert subprocess.run([*argv, "interrupt"]).returncode == 23
    registry = ContextArtifactRegistry(tmp_path / "store")
    with pytest.raises(ArtifactError, match="missing"):
        registry.resolve(manifest.artifact_id, CONSUMER)
    registry.close()
    children = [
        subprocess.Popen([*argv, "publish"], stdout=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    for child in children:
        stdout, _ = child.communicate(timeout=20)
        assert child.returncode == 0
        assert stdout.strip() == manifest.artifact_id
    registry = ContextArtifactRegistry(tmp_path / "store")
    assert registry.load(manifest.artifact_id, CONSUMER) == path.read_bytes()
    registry.close()
