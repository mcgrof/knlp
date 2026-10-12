# SPDX-License-Identifier: GPL-2.0
"""Separate HiSparse backing-history integrity from query-local selected reads."""

import json

import pytest

from routing.prefix_integrity.harness import run_validate
from routing.prefix_integrity.library_adapters import HiSparseWorkingSet


def _inputs(tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    (prefix / "cartridge.pt").touch()
    (prefix / "meta.json").write_text(
        json.dumps(
            {
                "model": "synthetic/contract-only",
                "dtype": "bfloat16",
                "n_layers": 2,
                "n_heads": 4,
                "head_dim": 8,
                "budget_tokens": 64,
                "prefix_token_ids_sha256": "0" * 64,
            }
        )
    )
    # Complete anchor block and one explicit selected middle token. The geometry
    # is synthetic; these positions are not a trace from a trained indexer.
    queries = [
        {"id": f"q{i}", "selected_positions": [0, 1, 2, 3, 8 + i]} for i in range(16)
    ]
    return prefix, queries


def _run(tmp_path, name, **config):
    prefix, queries = _inputs(tmp_path)
    return run_validate(
        str(prefix),
        "routing.prefix_integrity.library_adapters:make",
        queries,
        budget_k=2,
        block_size=4,
        repeats=3,
        adapter_config={"name": name, **config},
    )


def test_hisparse_host_history_is_not_truncated_by_device_budget(tmp_path):
    result = _run(tmp_path, "hisparse")
    assert result["status"] == "PASS"
    assert result["classification"] == "SAFE_FOR_PREFIX_OFFLOAD"
    assert result["metrics"]["pre_mean"] == 1
    assert result["metrics"]["manifest_stability"] == 1
    assert result["metrics"]["same_query_artifact_count"] == 1
    assert all(blocks == list(range(16)) for blocks in result["per_query_selected"])
    assert "kl" not in result["metrics"]


def test_hisparse_selected_reads_are_routing_not_saved_prefix(tmp_path):
    result = _run(tmp_path, "hisparse_working_set")
    assert result["classification"] == "ROUTING_ONLY_NOT_PREFIX_CACHE_SAFE"
    assert result["metrics"]["manifest_stability"] > 1
    assert result["metrics"]["same_query_artifact_count"] == 1
    assert result["metrics"]["partial_block_rate"] > 0


def test_hisparse_saving_only_selected_reads_fails_prefix_sharing(tmp_path):
    result = _run(tmp_path, "hisparse_working_set", declare_prefix_cache=True)
    assert result["status"] == "FAIL"
    assert result["classification"] == "DANGEROUS_FOR_PREFIX_SHARING"
    assert any("query-dependent" in v for v in result["violations"])


@pytest.mark.parametrize("positions", [[-1], [64], [True], [1, 1], "1,2"])
def test_hisparse_rejects_invalid_logical_positions(positions):
    with pytest.raises(ValueError):
        HiSparseWorkingSet().manifest({"selected_positions": positions}, 16, 2, 4)


def test_hisparse_requires_named_selection_input():
    with pytest.raises(KeyError):
        HiSparseWorkingSet().manifest({}, 16, 2, 4)


def test_hisparse_rejects_selection_over_budget():
    with pytest.raises(ValueError):
        HiSparseWorkingSet().manifest({"selected_positions": list(range(9))}, 16, 2, 4)
