# SPDX-License-Identifier: MIT
"""Synthetic sets exercise the real catalog and installed LMCache allocator."""

import inspect
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lmcache.v1.context_artifacts")
catalog_module = pytest.importorskip("vllm.artifact_registry")
from safetensors.torch import save_file

from tools.reproduce.context_artifacts.artifacts import ArtifactError
from tools.reproduce.memorilla.run import AUTHOR
from tools.reproduce.memorilla.worker import ArtifactEmbeddingStore


class Native:
    config = "synthetic"
    split = "test"

    def __init__(self, directory):
        self.directory = directory
        directory.mkdir()
        self.collection_ids = torch.tensor([0, 1, 2])
        self._length = torch.tensor([40000, 48000, 36000])
        self.values = {
            i: torch.arange(int(n) * 8, dtype=torch.int32)
            .remainder(1024)
            .to(torch.float16)
            .reshape(int(n), 8)
            / 64
            for i, n in enumerate(self._length)
        }
        save_file(
            {"collection_ids": self.collection_ids, "length": self._length},
            str(directory / "index.safetensors"),
        )
        save_file(
            {"embeddings": torch.zeros(3, 8, dtype=torch.float16)},
            str(directory / "questions.safetensors"),
        )
        self.reads = 0

    def documents(self, key):
        self.reads += 1
        return self.values[int(key)]

    def question(self, row):
        return torch.full((8,), row / 64, dtype=torch.float16)


@pytest.fixture
def imported(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    checkout = Path(inspect.getfile(catalog_module.ArtifactRegistry)).parents[1]
    (source / "vllm").symlink_to(checkout, target_is_directory=True)
    native = Native(tmp_path / "native")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "memory.pt").write_bytes(
        b"synthetic reader identity, not model weights"
    )
    (checkpoint / "config.json").write_text('{"embedding_dim": 8}')
    cfg = {"root": str(tmp_path), "pool_bytes": 1048576, "source_revision": AUTHOR}
    store = ArtifactEmbeddingStore(native, cfg, checkpoint, tmp_path / "import")
    key = store.set_id
    ids = dict(store.ids)
    store.close()
    yield native, checkpoint, cfg | {
        "artifact_sets": {native.config: key}
    }, ids, tmp_path


def test_attach_never_reconstructs_and_checks_every_loaded_bank(imported):
    native, checkpoint, cfg, ids, root = imported
    native.reads = 0
    store = ArtifactEmbeddingStore.attach(native, cfg, checkpoint, root / "attach")
    assert native.reads == 0
    assert store.registry.storage.read_count == 0
    for key, expected in native.values.items():
        actual = store.documents(key)
        assert torch.equal(actual, expected)
    assert store.registry.storage.read_count == 3
    assert not any(store.registry.storage.active.values())
    store.close()


@pytest.mark.parametrize("damage", ["checkpoint", "index", "descriptor", "truncate"])
def test_changed_import_refuses_before_exposure(imported, damage):
    native, checkpoint, cfg, ids, root = imported
    if damage == "checkpoint":
        (checkpoint / "memory.pt").write_bytes(b"different")
    elif damage == "index":
        (native.directory / "index.safetensors").write_bytes(b"different")
    elif damage == "descriptor":
        key = cfg["artifact_sets"][native.config]
        path = root / "artifacts" / native.config / "sets" / f"{key}.json"
        body = json.loads(path.read_text())
        body["ids"]["0"] = "f" * 64
        path.write_text(json.dumps(body))
    else:
        path = next((root / "artifacts" / native.config).glob("*.safetensors"))
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises((ArtifactError, ValueError)):
        ArtifactEmbeddingStore.attach(native, cfg, checkpoint, root / "refused")


def test_same_length_corruption_is_detected_on_first_payload_acquisition(imported):
    native, checkpoint, cfg, ids, root = imported
    path = root / "artifacts" / native.config
    target = next(path.glob("*.safetensors"))
    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    store = ArtifactEmbeddingStore.attach(native, cfg, checkpoint, root / "attach")
    with pytest.raises(ValueError, match="checksum"):
        for key in native.values:
            store.documents(key)
    assert not any(store.registry.storage.active.values())
    store.close()


def test_missing_catalog_entry_is_not_republished_by_attach(imported):
    native, checkpoint, cfg, ids, root = imported
    db = sqlite3.connect(root / "artifacts" / native.config / "cartridges.sqlite3")
    try:
        with db:
            db.execute("DELETE FROM context_artifacts WHERE id=?", (ids[0],))
    finally:
        db.close()
    with pytest.raises(KeyError):
        ArtifactEmbeddingStore.attach(native, cfg, checkpoint, root / "missing")


def test_new_process_restores_import_without_a_document_source_method(imported):
    native, checkpoint, cfg, ids, root = imported
    args = root / "restart-args.json"
    args.write_text(
        json.dumps(
            {"cfg": cfg, "checkpoint": str(checkpoint), "native": str(native.directory)}
        )
    )
    program = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from tools.reproduce.memorilla.worker import ArtifactEmbeddingStore
value=json.loads(Path(sys.argv[1]).read_text())
index=load_file(str(Path(value['native'])/'index.safetensors'))
native=SimpleNamespace(config='synthetic',split='test',directory=Path(value['native']),collection_ids=index['collection_ids'],_length=index['length'])
root=Path(value['cfg']['root'])
store=ArtifactEmbeddingStore.attach(native,value['cfg'],Path(value['checkpoint']),root/'restarted')
for key,length in zip(native.collection_ids.tolist(),native._length.tolist(),strict=True):
    expected=torch.arange(length*8,dtype=torch.int32).remainder(1024).to(torch.float16).reshape(length,8)/64
    assert torch.equal(store.documents(key),expected)
loads=store.registry.storage.read_count
store.close()
assert store.registry.storage.allocator.memcheck()
(root/'restart-result.json').write_text(json.dumps({'collections':len(store.ids),'loads':loads}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(args)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    (root / "restart.stdout").write_text(completed.stdout)
    (root / "restart.stderr").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((root / "restart-result.json").read_text()) == {
        "collections": 3,
        "loads": 3,
    }


def test_fifo_prefetch_progresses_when_two_banks_do_not_fit_together(imported):
    from tools.reproduce.memorilla.input_batch import LeasedMemoryCollator
    from tools.reproduce.memorilla.tests.test_input_batch import (
        Tokenizer,
        collator,
        items,
    )
    from memorilla.data import MemoryCollator

    native, checkpoint, cfg, ids, root = imported
    store = ArtifactEmbeddingStore.attach(native, cfg, checkpoint, root / "prefetch")
    actual = collator(
        LeasedMemoryCollator,
        store,
        clients=1,
        prefetch=True,
        final_dtype=torch.bfloat16,
    )
    expected = collator(MemoryCollator, native)(items())
    batch = actual(items())
    for key in expected:
        assert torch.equal(
            batch[key].view(torch.uint8), expected[key].view(torch.uint8)
        )
    stats = actual.stats()
    assert stats["prefetch_batches"] == 1
    assert stats["active_prefetch_loans"] == 0
    assert 1 <= stats["peak_prefetch_loans"] <= 2
    assert not any(store.registry.storage.active.values())
    actual.close()
    store.close()


def test_prefetch_copy_failure_drains_and_releases_every_loan(imported, monkeypatch):
    import tools.reproduce.memorilla.input_batch as module
    from tools.reproduce.memorilla.tests.test_input_batch import collator, items

    native, checkpoint, cfg, ids, root = imported
    store = ArtifactEmbeddingStore.attach(
        native, cfg, checkpoint, root / "prefetch-fault"
    )
    actual = collator(
        module.LeasedMemoryCollator,
        store,
        clients=1,
        prefetch=True,
        final_dtype=torch.bfloat16,
    )

    def fail(*args, **kwargs):
        raise OSError("injected foreground copy failure")

    monkeypatch.setattr(module, "copy_fp16_cpu", fail)
    with pytest.raises(OSError, match="foreground copy"):
        actual(items())
    assert actual.stats()["active_prefetch_loans"] == 0
    assert not any(store.registry.storage.active.values())
    actual.close()
    store.close()
