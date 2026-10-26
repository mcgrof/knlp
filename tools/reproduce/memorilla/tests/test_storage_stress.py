# SPDX-License-Identifier: MIT
"""Concurrent reads must preserve identities and drain cancelled work."""

import threading
from types import SimpleNamespace

import pytest
import torch

from tools.reproduce.memorilla.storage_stress import (
    concurrent_replay,
    execute,
    prepare_file_cache,
    tensor_record,
)


def test_existing_capture_is_rejected_before_loading_inputs(tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("retained evidence\n")
    with pytest.raises(FileExistsError, match="new storage-stress output"):
        execute(SimpleNamespace(output=tmp_path))
    assert sentinel.read_text() == "retained evidence\n"


def test_concurrent_replay_retries_capacity_and_preserves_each_request():
    expected = {
        k: tensor_record(torch.full((2, 3), k, dtype=torch.float16)) for k in range(3)
    }
    lock = threading.Lock()
    attempts = 0

    def load(key):
        nonlocal attempts
        with lock:
            attempts += 1
            if attempts <= 3:
                raise MemoryError("busy")
        return torch.full((2, 3), key, dtype=torch.float16)

    order = [i % 3 for i in range(300)]
    rows = concurrent_replay(load, expected, order, clients=8)
    assert [r["collection_id"] for r in rows] == order
    assert all(r["status"] == "completed" for r in rows)
    assert sum(r["capacity_retries"] for r in rows) == 3


def test_real_queued_cancellation_drains_running_reads():
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    rows = concurrent_replay(
        lambda _: tensor,
        {0: tensor_record(tensor)},
        [0] * 100,
        clients=2,
        cancel_every=2,
    )
    assert len(rows) == 100
    assert sum(r["status"] == "cancelled" for r in rows) >= 48
    assert all(r["cancellation_accepted"] for r in rows if r["status"] == "cancelled")
    assert any(r["status"] == "completed" for r in rows)


def test_wrong_bank_is_not_tolerated_as_numerical_drift():
    tensor = torch.zeros((2, 3), dtype=torch.float16)
    with pytest.raises(ValueError, match="wrong collection"):
        concurrent_replay(
            lambda _: tensor + 1, {0: tensor_record(tensor)}, [0] * 8, clients=2
        )


def test_cache_control_rejects_files_outside_owned_root(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    path = tmp_path / "other.bin"
    path.write_bytes(b"x" * 8192)
    with pytest.raises(ValueError, match="inside"):
        prepare_file_cache([path], "cold", root)
    result = prepare_file_cache([path], "unchanged", root)
    assert 0 <= result["resident_pages"] <= result["pages"]
    assert path.read_bytes() == b"x" * 8192
