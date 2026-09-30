# SPDX-License-Identifier: GPL-2.0
"""Cheap contracts for the bounded sparse-replay workload driver."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from research.kv_translate.sharing.run_repair_screen import (  # noqa: E402
    arms_for,
    choose_contexts,
    expand_block_indices,
    mask_from_indices,
    uniform_token_indices,
    validate_kri_prior,
)


def test_uniform_token_budget_is_exact_and_spans_the_prefix():
    selected = uniform_token_indices(16384, 4096)
    assert selected.numel() == 4096
    assert selected[:4].tolist() == [0, 4, 8, 12]
    assert int(selected[-1]) == 16380
    assert len(set(selected.tolist())) == 4096
    mask = mask_from_indices(16384, selected, "cpu")
    assert int(mask.sum()) == 4096


def test_uniform_selection_refuses_an_impossible_budget():
    with pytest.raises(ValueError, match="outside"):
        uniform_token_indices(10, 11)


def test_block_selection_expands_to_an_exact_contiguous_token_budget():
    blocks = uniform_token_indices(1024, 256)
    selected = expand_block_indices(blocks, 16, total_tokens=16384)
    assert selected.numel() == 4096
    assert selected[:16].tolist() == list(range(16))
    assert selected[16:32].tolist() == list(range(64, 80))
    assert selected[-16:].tolist() == list(range(16320, 16336))
    assert int(mask_from_indices(16384, selected, "cpu").sum()) == 4096


def test_block_selection_refuses_repeats_and_out_of_range_blocks():
    with pytest.raises(ValueError, match="repeat"):
        expand_block_indices([2, 2], 16, total_tokens=64)
    with pytest.raises(ValueError, match="outside"):
        expand_block_indices([4], 16, total_tokens=64)


def test_kri_arms_are_explicitly_admitted():
    assert len(arms_for({"kri_enabled": False})) == 5
    admitted = arms_for({"kri_enabled": True})
    assert admitted[-2:] == ("mapped_uniform_blocks", "mapped_kri_blocks")


def test_one_oracle_diagnostic_replaces_the_screen_arms():
    cfg = {
        "kri_enabled": True,
        "native_donor_arm": "mapped_kri_blocks_native_donor",
    }
    assert arms_for(cfg) == ("mapped_kri_blocks_native_donor",)


def test_context_selection_is_sorted_by_group_with_one_row_per_group():
    rows = []
    for domain in ("document", "code"):
        for group in ("g3", "g1", "g2", "g1"):
            rows.append(
                {
                    "domain": domain,
                    "length_bucket": 16384,
                    "source_group_id": f"{domain}-{group}",
                    "context_id": f"{domain}-{group}-{len(rows)}",
                }
            )
        rows.append(
            {
                "domain": domain,
                "length_bucket": 4096,
                "source_group_id": f"{domain}-ignored",
                "context_id": f"{domain}-ignored",
            }
        )
    selected = choose_contexts(rows, 16384, 2)
    assert [r["source_group_id"] for r in selected] == [
        "document-g1",
        "document-g2",
        "code-g1",
        "code-g2",
    ]


def test_context_selection_refuses_an_underfilled_domain():
    rows = [
        {
            "domain": "document",
            "length_bucket": 16384,
            "source_group_id": "d1",
            "context_id": "d1",
        },
        {
            "domain": "code",
            "length_bucket": 16384,
            "source_group_id": "c1",
            "context_id": "c1",
        },
    ]
    with pytest.raises(ValueError, match="wanted 2 document"):
        choose_contexts(rows, 16384, 2)


def test_kri_prior_geometry_and_budget_are_part_of_its_contract():
    cfg = {
        "kri_block_size": 16,
        "kri_selected_blocks": 2,
        "selected_tokens": 32,
    }
    record = {"prefix_tokens_sha256": "ab" * 32}
    prior = {
        "identity": "run-id",
        "prefix_tokens_sha256": record["prefix_tokens_sha256"],
        "block_size": 16,
        "selected_blocks": [1, 4],
        "selected_tokens": 32,
    }
    assert validate_kri_prior(prior, cfg, record, identity="run-id") is prior
    for field, bad, message in (
        ("block_size", 8, "block size"),
        ("selected_blocks", [1], "block count"),
        ("selected_tokens", 16, "token count"),
    ):
        changed = {**prior, field: bad}
        with pytest.raises(ValueError, match=message):
            validate_kri_prior(changed, cfg, record, identity="run-id")
