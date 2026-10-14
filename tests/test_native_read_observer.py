# SPDX-License-Identifier: MIT
"""Exercise actual wrapper arguments without an asymmetric-only helper."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from tools.kv.native import instrument


@pytest.mark.parametrize("arm", ["k16_v16", "k16_v8", "k8_v8"])
def test_wrapper_read_receipts_cover_symmetric_and_mixed_views(
    arm, tmp_path, monkeypatch
):
    expected = {
        "k16_v16": {"torch.float16"},
        "k16_v8": {"torch.float16", "torch.float8_e4m3fn"},
        "k8_v8": {"torch.float8_e4m3fn"},
    }[arm]
    if arm == "k8_v8":
        backing = torch.zeros(8, dtype=torch.uint8)
        cache = backing.view(torch.float8_e4m3fn)
    else:
        key = torch.zeros(4, dtype=torch.float16)
        value = torch.zeros(
            4, dtype=torch.float16 if arm == "k16_v16" else torch.float8_e4m3fn
        )
        cache = (key, value)
        backing = cache
    layer = SimpleNamespace(_k_scale=None, kv_cache=backing)
    model = SimpleNamespace(named_modules=lambda: [("layers.0.attn", layer)])

    class Prefill:
        _backend = "fa3"

        def run(self, query, paged_kv_cache):
            return query

    class Decode:
        _backend = "fa2"
        use_tensor_cores = True

        def run(self, query, paged_kv_cache):
            return query

    original_prefill, original_decode = Prefill.run, Decode.run
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=Prefill,
            BatchDecodeWithPagedKVCacheWrapper=Decode,
        ),
    )
    instrument.observe_wrappers(model, tmp_path, expected_types=expected)
    query = torch.zeros(4, dtype=torch.float16)
    try:
        assert Prefill().run(query, cache) is query
        assert Decode().run(query, paged_kv_cache=cache) is query
        assert Prefill().run(query, cache) is query
        events = [
            json.loads(line)
            for line in (tmp_path / "kernel-read-events.jsonl").read_text().splitlines()
        ]
        assert len(events) == 2
        assert {x["wrapper"] for x in events} == {
            "BatchPrefillWithPagedKVCacheWrapper",
            "BatchDecodeWithPagedKVCacheWrapper",
        }
        assert all({t["dtype"] for t in x["cache"]} == expected for x in events)
        assert json.loads((tmp_path / "kernel-read-0.json").read_text())
        outside = (
            tuple(t.clone() for t in cache)
            if isinstance(cache, tuple)
            else cache.clone()
        )
        with pytest.raises(RuntimeError, match="outside model cache storage"):
            Decode().run(query, outside)
        wrong = (
            tuple(t.view(torch.uint8) for t in cache)
            if isinstance(cache, tuple)
            else backing
        )
        with pytest.raises(RuntimeError, match="read dtype"):
            Decode().run(query, wrong)
        with pytest.raises(RuntimeError, match="already installed"):
            instrument.observe_wrappers(model, tmp_path, expected_types=expected)
    finally:
        instrument.restore_wrappers(model)
    assert Prefill.run is original_prefill and Decode.run is original_decode
    assert not hasattr(model, "_study_wrapper_originals")
