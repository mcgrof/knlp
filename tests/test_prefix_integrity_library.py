# SPDX-License-Identifier: GPL-2.0
"""Check published-method adapters against the CPU prefix contract."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from routing.prefix_integrity.harness import run_validate
from routing.prefix_integrity.library_adapters import LEADERBOARD, LIBRARY


def _cartridge(tmp_path):
    path = tmp_path / "prefix"
    path.mkdir()
    (path / "cartridge.pt").touch()
    (path / "meta.json").write_text(
        json.dumps(
            {
                "model": "test/model",
                "dtype": "bfloat16",
                "n_layers": 2,
                "n_heads": 4,
                "head_dim": 8,
                "budget_tokens": 4096,
                "prefix_token_ids_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def _queries():
    return [{"id": f"q{i}", "query": f"suffix {i}"} for i in range(16)]


def test_every_catalogued_method_has_an_adapter():
    assert set(LEADERBOARD) == set(LIBRARY)


def test_snapstream_fails_prefix_only_sharing(tmp_path):
    result = run_validate(
        cartridge=str(_cartridge(tmp_path)),
        adapter_spec="snapstream",
        queries=_queries(),
        budget_k=16,
        block_size=16,
        pins="A1R2K13",
        repeats=3,
    )

    assert result["status"] == "FAIL"
    assert result["classification"] == "DANGEROUS_FOR_PREFIX_SHARING"
    assert result["metrics"]["manifest_stability"] == 16
    assert result["metrics"]["same_query_artifact_count"] == 1
    assert result["metrics"]["partial_block_rate"] > 0


def test_snapstream_needs_both_identity_and_custom_layout(tmp_path):
    result = run_validate(
        cartridge=str(_cartridge(tmp_path)),
        adapter_spec="routing.prefix_integrity.library_adapters:make",
        adapter_config={"name": "snapstream", "declare_query_hash": True},
        queries=_queries(),
        budget_k=16,
        block_size=16,
        pins="A1R2K13",
        repeats=3,
    )

    assert result["classification"] == "SAFE_ONLY_WITH_CUSTOM_CONNECTOR"
    assert result["cache_key_fields"] == ["prefix_hash", "query_hash"]
