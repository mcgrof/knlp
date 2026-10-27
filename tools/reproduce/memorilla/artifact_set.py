# SPDX-License-Identifier: MIT
"""Content-addressed import receipts for immutable document-bank sets."""

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from tools.reproduce.context_artifacts.artifacts import (
    ArtifactError,
    canonical,
    digest,
    digest_file,
)


def identity(native, cfg, checkpoint, checkpoint_hash):
    """Bind source index, question vectors and reader configuration."""
    from .run import DATA, ENCODER, READER

    config = json.loads((checkpoint / "config.json").read_text())
    return {
        "source_revision": cfg["source_revision"],
        "data_revision": DATA,
        "reader_revision": READER,
        "encoder_revision": ENCODER,
        "benchmark": native.config,
        "split": native.split,
        "checkpoint_sha256": checkpoint_hash,
        "reader_config_sha256": digest_file(checkpoint / "config.json"),
        "embedding_dim": config["embedding_dim"],
        "index_sha256": digest_file(native.directory / "index.safetensors"),
        "questions_sha256": digest_file(native.directory / "questions.safetensors"),
    }


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def install_bytes(path, content):
    """Install a complete durable file without overwriting an existing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".import-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise ArtifactError("existing immutable import file differs")
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def save_bank(source, vectors):
    """Create a safetensors payload atomically; publication verifies it next."""
    from safetensors.torch import save_file

    with tempfile.NamedTemporaryFile(
        dir=source.parent, prefix=".bank-", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        save_file({"embeddings": vectors.contiguous()}, str(temporary))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, source)
        except FileExistsError:
            pass
        _sync_directory(source.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish(directory, source_identity, ids):
    """Publish readiness only after every immutable catalog entry exists."""
    body = {
        "schema": 1,
        "identity": source_identity,
        "ids": {str(k): v for k, v in ids.items()},
    }
    encoded = canonical(body)
    set_id = hashlib.sha256(encoded).hexdigest()
    install_bytes(directory / "sets" / f"{set_id}.json", encoded + b"\n")
    return set_id


def load(directory, source_identity, set_id, collections):
    """Read a pinned complete import; never reconstruct absent payloads."""
    digest(set_id, "artifact set ID")
    path = directory / "sets" / f"{set_id}.json"
    body = json.loads(path.read_text())
    if hashlib.sha256(canonical(body)).hexdigest() != set_id:
        raise ArtifactError("artifact set identity differs from its pin")
    if (
        set(body) != {"schema", "identity", "ids"}
        or body["schema"] != 1
        or body["identity"] != source_identity
    ):
        raise ArtifactError("artifact set source or checkpoint differs")
    ids = body["ids"]
    if not isinstance(ids, dict) or set(ids) != {str(k) for k in collections}:
        raise ArtifactError("artifact set collections differ from the source index")
    for value in ids.values():
        digest(value, "artifact ID")
    return {int(k): v for k, v in ids.items()}


def check_extent(plan):
    """Check location metadata; payload acquisition still verifies all bytes."""
    info = Path(plan.path).stat()
    if not stat.S_ISREG(info.st_mode) or plan.offset + plan.length > info.st_size:
        raise ArtifactError("imported payload range is missing or truncated")
