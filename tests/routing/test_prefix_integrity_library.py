# SPDX-License-Identifier: GPL-2.0
"""Contract classification of ten leaderboard KV-compression methods.

Runs each method through the CPU harness against a synthetic cartridge and
asserts the plain-language verdict its design earns. The headline: only the
structural, query-independent method (StreamingLLM) is prefix-cache clean; the
adaptive ones are query-dependent and land DANGEROUS under a prefix-only key,
demoting to SAFE_ONLY_WITH_EXTENDED_CACHE_KEY when they declare query_hash.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from routing.prefix_integrity.harness import run_validate
from routing.prefix_integrity.datatypes import Classification


def _cart(tmp_path):
    d = os.path.join(tmp_path, "cart")
    os.makedirs(d, exist_ok=True)
    json.dump(
        {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "dtype": "bfloat16",
            "budget_tokens": 4096,
            "n_layers": 28,
            "n_heads": 4,
            "head_dim": 128,
            "prefix_token_ids_sha256": "abc",
        },
        open(os.path.join(d, "meta.json"), "w"),
    )
    open(os.path.join(d, "cartridge.pt"), "wb").close()
    return d


def _q(n=12):
    return [{"id": f"q{i}", "query": f"clinical question {i}"} for i in range(n)]


ADAPTIVE = ["snapkv", "h2o", "tova", "pyramidkv", "expected_attention", "adakv"]


def test_streaming_llm_is_prefix_cache_clean(tmp_path):
    r = run_validate(
        _cart(str(tmp_path)), "streaming_llm", _q(), budget_k=200, pins="A1R2"
    )
    assert r["metrics"]["manifest_stability"] == 1  # query independent
    assert r["metrics"]["partial_block_rate"] == 0.0  # whole block
    assert r["classification"] == Classification.SAFE_FOR_PREFIX_OFFLOAD.value


def test_knorm_query_independent_but_partial(tmp_path):
    r = run_validate(_cart(str(tmp_path)), "knorm", _q(), budget_k=16, pins="A1R2")
    assert r["metrics"]["manifest_stability"] == 1  # query independent
    assert r["metrics"]["partial_block_rate"] > 0.0  # token-level -> partial
    assert r["classification"] == Classification.SAFE_ONLY_WITH_CUSTOM_CONNECTOR.value


def test_kivi_codec_needs_custom_connector(tmp_path):
    r = run_validate(_cart(str(tmp_path)), "kivi", _q(), budget_k=256, pins="A1R2")
    assert r["metrics"]["manifest_stability"] == 1
    assert r["classification"] == Classification.SAFE_ONLY_WITH_CUSTOM_CONNECTOR.value


def test_quest_is_routing_only(tmp_path):
    r = run_validate(_cart(str(tmp_path)), "quest", _q(), budget_k=16, pins="A1R2")
    assert r["metrics"]["manifest_stability"] > 1  # query dependent
    assert r["metrics"]["partial_block_rate"] == 0.0  # whole pages
    assert (
        r["classification"] == Classification.ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE.value
    )


def test_adaptive_methods_are_dangerous(tmp_path):
    cart = _cart(str(tmp_path))
    for name in ADAPTIVE:
        r = run_validate(cart, name, _q(), budget_k=16, pins="A1R2")
        assert r["metrics"]["manifest_stability"] > 1, name
        assert (
            r["classification"] == Classification.DANGEROUS_FOR_PREFIX_SHARING.value
        ), f"{name}: {r['classification']}"


def test_declaring_query_hash_rescues_to_extended_key(tmp_path):
    cart = _cart(str(tmp_path))
    for name in ADAPTIVE:
        r = run_validate(
            cart,
            name,
            _q(),
            budget_k=16,
            pins="A1R2",
            adapter_config={"declare_query_hash": True},
        )
        # once query_hash is in the key, the query-dependence is no longer fatal;
        # the binding constraint becomes the partial blocks (custom connector).
        assert r["classification"] in (
            Classification.SAFE_ONLY_WITH_EXTENDED_CACHE_KEY.value,
            Classification.SAFE_ONLY_WITH_CUSTOM_CONNECTOR.value,
        ), f"{name}: {r['classification']}"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
