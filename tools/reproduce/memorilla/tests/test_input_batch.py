# SPDX-License-Identifier: MIT
"""Owned batch assembly must match the unchanged author collator."""

import threading

import pytest

torch = pytest.importorskip("torch")
author = pytest.importorskip("memorilla.data")
from tools.reproduce.memorilla.input_batch import LeasedMemoryCollator


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return repr(messages)

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.arange(len(texts) * 3).reshape(len(texts), 3),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }


class Store:
    def __init__(self, fail=None):
        self.values = {
            i: torch.arange(n * 8, dtype=torch.float16).reshape(n, 8) / (i + 1)
            for i, n in enumerate([3, 7, 2])
        }
        self.capacity = threading.Semaphore(1)
        self.active = 0
        self.completed = []
        self.fail = fail

    def documents(self, key):
        return self.values[key].clone()

    def collection_shape(self, key):
        return self.values[key].shape

    def copy_documents(self, key, destination):
        with self.capacity:
            self.active += 1
            try:
                destination.copy_(self.values[key])
                if key == self.fail:
                    raise OSError("injected copy failure")
                self.completed.append(key)
            finally:
                self.active -= 1

    def question(self, row):
        return torch.full((8,), row / 7, dtype=torch.float16)


def items():
    return [
        dict(config="test", split="test", row=i, collection_id=k, question=f"q{i}")
        for i, k in enumerate([1, 0, 1, 2])
    ]


def collator(cls, store, **kwargs):
    return cls(
        Tokenizer(),
        {("test", "test"): store},
        num_memories=2,
        max_length=64,
        train=False,
        **kwargs,
    )


@pytest.mark.parametrize("clients", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_complete_batch_matches_author_with_duplicates_and_padding(clients, dtype):
    store = Store()
    observed = collator(LeasedMemoryCollator, store, clients=clients, final_dtype=dtype)
    expected = collator(author.MemoryCollator, store)(items())
    batch = observed(items())
    for name, tensor in expected.items():
        assert batch[name].dtype == tensor.dtype
        assert torch.equal(batch[name], tensor)
    assert store.active == 0
    assert sorted(store.completed) == [0, 1, 1, 2]
    for value in store.values.values():
        value.zero_()
    assert torch.equal(batch["doc_embeds"], expected["doc_embeds"])
    observed.close()
    observed.close()
    with pytest.raises(RuntimeError, match="closed"):
        observed(items())


def test_failed_batch_drains_running_copies_and_can_be_reused():
    store = Store(fail=0)
    observed = collator(LeasedMemoryCollator, store, clients=8)
    with pytest.raises(OSError, match="copy failure"):
        observed(items())
    assert store.active == 0
    store.fail = None
    batch = observed(items())
    assert not batch["doc_padding_mask"][0].any()
    observed.close()


def test_native_view_protocol_has_the_same_batch_values():
    values = Store()

    class Native:
        def documents(self, key):
            return values.values[key]

        def question(self, row):
            return values.question(row)

    observed = collator(LeasedMemoryCollator, Native(), clients=8)
    expected = collator(author.MemoryCollator, Native())(items())
    batch = observed(items())
    for key in expected:
        assert torch.equal(batch[key], expected[key])
    observed.close()


@pytest.mark.parametrize("clients", [1, 8])
def test_self_owning_mapped_views_bypass_the_arena_copy_protocol(clients):
    store = Store()
    store.borrowed_documents = lambda key: store.values[key]

    def forbidden(*args):
        raise AssertionError("mapped tensors must use their own backing reference")

    store.copy_documents = forbidden
    observed = collator(
        LeasedMemoryCollator, store, clients=clients, final_dtype=torch.bfloat16
    )
    expected = collator(author.MemoryCollator, store)(items())
    actual = observed(items())
    for key in expected:
        assert torch.equal(
            actual[key].view(torch.uint8), expected[key].view(torch.uint8)
        )
    assert not store.completed
    observed.close()


def test_invalid_geometry_refuses_before_copy():
    store = Store()
    store.collection_shape = lambda key: (2.5, 8)
    observed = collator(LeasedMemoryCollator, store)
    with pytest.raises(ValueError, match="geometry"):
        observed(items())
    assert not store.completed
    observed.close()


def test_admitted_geometry_cache_is_bounded_and_small_banks_stay_direct():
    store = Store()
    observed = collator(
        LeasedMemoryCollator,
        store,
        clients=1,
        prefetch=True,
        final_dtype=torch.bfloat16,
        max_geometries=2,
    )
    observed(items())
    assert observed.stats()["geometry_entries"] == 2
    assert observed.stats()["prefetch_batches"] == 0
    assert observed._reader is None
    observed.close()


def test_direct_cast_preserves_special_values_and_padding_bits():
    store = Store()
    special = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            65504,
            2**-24,
            -(2**-24),
        ],
        dtype=torch.float16,
    )
    for tensor in store.values.values():
        tensor[:] = special
    observed = collator(
        LeasedMemoryCollator, store, clients=8, final_dtype=torch.bfloat16
    )
    expected = collator(author.MemoryCollator, store)(items())
    actual = observed(items())
    for name in expected:
        assert torch.equal(
            actual[name].view(torch.uint8), expected[name].view(torch.uint8)
        )
    observed.close()
