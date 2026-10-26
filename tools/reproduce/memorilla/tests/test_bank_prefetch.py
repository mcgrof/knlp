# SPDX-License-Identifier: MIT
"""Concurrency must preserve complete collator batches and bank identities."""

import threading

import pytest

from tools.reproduce.memorilla.bank_prefetch import ConcurrentBanks
from tools.reproduce.memorilla.run import configuration


def test_completion_order_does_not_reorder_repeated_banks():
    completed = threading.Event()

    class Store:
        def documents(self, key):
            if key == "slow":
                assert completed.wait(2)
            else:
                completed.set()
            return key

        def question(self, row):
            return row + 1

    banks = ConcurrentBanks(Store(), 3)
    banks.prepare(["slow", "fast", "slow"])
    assert [banks.documents(key) for key in ["slow", "fast", "slow"]] == [
        "slow",
        "fast",
        "slow",
    ]
    assert banks.question(2) == 3
    banks.finish()
    banks.close()


def test_capacity_retry_is_bounded_and_unconsumed_batch_is_rejected():
    class Store:
        attempts = 0

        def documents(self, key):
            self.attempts += 1
            if self.attempts == 1:
                raise MemoryError("busy")
            return key

    banks = ConcurrentBanks(Store(), 1)
    banks.prepare([4])
    assert banks.retries == 1
    with pytest.raises(RuntimeError, match="complete bank batch"):
        banks.finish()
    assert banks.documents(4) == 4
    with pytest.raises(RuntimeError, match="unprepared"):
        banks.documents(5)
    banks.close()

    class Full:
        def documents(self, key):
            raise MemoryError("full")

    banks = ConcurrentBanks(Full(), 1, timeout=0)
    with pytest.raises(TimeoutError, match="deadline"):
        banks.prepare([1])
    banks.close()


def test_native_clients_use_factory_handles_on_worker_threads():
    caller = threading.get_ident()
    owners = []

    class Handle:
        def __init__(self):
            self.owner = threading.get_ident()
            owners.append(self.owner)

        def documents(self, key):
            assert threading.get_ident() == self.owner != caller
            return key

    banks = ConcurrentBanks(None, 3, factory=Handle)
    banks.prepare(list(range(30)))
    assert [banks.documents(key) for key in range(30)] == list(range(30))
    banks.close()
    assert 1 <= len(owners) <= 3


def test_original_author_rejects_concurrent_override(tmp_path):
    path = tmp_path / "config"
    path.write_text(
        "CONFIG_KNLP_REPRODUCE_MEMORILLA=y\n"
        "CONFIG_KNLP_MEM_AUTHOR=y\nCONFIG_KNLP_MEM_CLIENTS=8\n"
    )
    with pytest.raises(ValueError, match="author mode rejects concurrent"):
        configuration(path)
