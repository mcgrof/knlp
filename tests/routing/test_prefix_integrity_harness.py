# SPDX-License-Identifier: GPL-2.0
"""End-to-end harness tests (selector mode, CPU, no torch).

Builds a synthetic metadata-only cartridge (just a meta.json) and runs the full
validate path for the reference adapters, asserting each lands in the right
plain-language classification. This is the doc's headline demonstration: an
offline KRI prior is SAFE_FOR_PREFIX_OFFLOAD, while the same-shaped query-aware
selector is DANGEROUS_FOR_PREFIX_SHARING when keyed by prefix_hash alone.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from routing.prefix_integrity.harness import run_validate
from routing.prefix_integrity.datatypes import Classification, Status


def _make_cartridge(tmp_path):
    d = os.path.join(tmp_path, "cart")
    os.makedirs(d, exist_ok=True)
    meta = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "bfloat16",
        "budget_tokens": 4096,
        "n_layers": 28,
        "n_heads": 4,
        "head_dim": 128,
        "prefix_token_ids_sha256": "abc123",
    }
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    # touch a fake .pt so _resolve finds it (never loaded on the metadata path)
    open(os.path.join(d, "cartridge.pt"), "wb").close()
    return d


def _queries(n=12):
    return [
        {"id": f"q{i}", "query": f"question number {i} about the document"}
        for i in range(n)
    ]


def test_full_is_safe_for_offload(tmp_path):
    cart = _make_cartridge(str(tmp_path))
    res = run_validate(cart, "full", _queries(), budget_k=256, pins="A1R2")
    assert res["identity"]["num_blocks"] == 256  # 4096 / 16
    assert res["status"] == Status.PASS.value
    assert res["classification"] == Classification.SAFE_FOR_PREFIX_OFFLOAD.value
    assert res["metrics"]["pre_mean"] == 1.0


def test_offline_kri_prior_is_safe(tmp_path):
    cart = _make_cartridge(str(tmp_path))
    # anchor_recency stands in for an offline, query-independent prior here:
    # deterministic and the same selection for every request. Geometry is
    # prefix-safe regardless of budget.
    res = run_validate(cart, "anchor_recency", _queries(), budget_k=16, pins="A1R2")
    assert res["metrics"]["manifest_stability"] == 1  # query independent
    assert res["classification"] == Classification.SAFE_FOR_PREFIX_OFFLOAD.value
    # tight budget keeps few blocks -> low PRE -> honest WARN, not a failure
    assert res["status"] == Status.WARN.value
    assert res["metrics"]["pre_mean"] < 0.7
    # a generous budget passes cleanly
    res2 = run_validate(cart, "anchor_recency", _queries(), budget_k=200, pins="A1R2")
    assert res2["status"] == Status.PASS.value
    assert res2["classification"] == Classification.SAFE_FOR_PREFIX_OFFLOAD.value


def test_query_aware_prefix_keyed_is_dangerous(tmp_path):
    cart = _make_cartridge(str(tmp_path))
    res = run_validate(cart, "query_aware", _queries(), budget_k=16, pins="A1R2")
    # selection varies per query -> manifest unstable across queries
    assert res["metrics"]["manifest_stability"] > 1
    assert res["status"] == Status.FAIL.value
    assert res["classification"] == Classification.DANGEROUS_FOR_PREFIX_SHARING.value
    assert res["danger_score"] > 0.4


def test_query_aware_with_declared_query_hash_is_extended_key(tmp_path):
    cart = _make_cartridge(str(tmp_path))
    res = run_validate(
        cart,
        "query_aware",
        _queries(),
        budget_k=16,
        pins="A1R2",
        adapter_config={"declare_query_hash": True},
    )
    assert (
        res["classification"] == Classification.SAFE_ONLY_WITH_EXTENDED_CACHE_KEY.value
    )


def test_random_per_query_is_unstable(tmp_path):
    cart = _make_cartridge(str(tmp_path))
    res = run_validate(cart, "random", _queries(), budget_k=16, pins="A1R2")
    assert res["metrics"]["manifest_stability"] > 1


def test_semantic_drift_folds_into_verdict(tmp_path):
    cart = _make_cartridge(str(tmp_path))
    # full selection is geometry-safe, but a bad codec drift should still fail
    res = run_validate(
        cart,
        "full",
        _queries(),
        budget_k=256,
        pins="A1R2",
        semantic={"kl": 0.5, "top1": 0.4, "repairable": False},
    )
    assert res["status"] == Status.FAIL.value
    assert any("drift" in v for v in res["violations"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
