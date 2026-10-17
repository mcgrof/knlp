# SPDX-License-Identifier: MIT
import pytest
import torch

from tools.kv.native.routing_oracle import raw_suffix, require_equal, require_placement


def test_raw_suffix_retains_zero_repeats_remap_tail_and_decode_token_offset():
    expected = raw_suffix(
        [0, 2, 35, 52],
        [17, 33, 17],
        [[7, 6, 0], [3, 0, 2], [5, 3, 0]],
        decode_requests=1,
    )
    assert expected == {
        "lengths": [33, 17],
        "source_ids": [3, 0, 2, 5, 3],
        "query_offset": 2,
        "query_indptr": [0, 33, 50],
        "page_indptr": [0, 3, 5],
        "tails": [1, 1],
    }


def test_fixed_faults_are_rejected_from_independent_expected_mapping():
    old = raw_suffix(
        [0, 2, 35, 52],
        [17, 33, 17],
        [[7, 6, 0], [3, 0, 2], [5, 3, 0]],
        decode_requests=1,
    )
    new = raw_suffix(
        [0, 2, 35, 52],
        [17, 33, 17],
        [[6, 7, 0], [2, 3, 0], [0, 5, 0]],
        decode_requests=1,
    )
    ids = torch.tensor(old["source_ids"])
    for wrong in (
        torch.tensor([7, 6, 3, 0, 2]),
        torch.tensor([7, 0, 2, 5, 3]),
        torch.tensor(new["source_ids"]),
    ):
        with pytest.raises(RuntimeError):
            require_equal(wrong, ids, "injected page fault")
    q = torch.zeros(52, 3, 4)
    out = torch.zeros_like(q)
    require_placement(q[2:], out[2:], q, out, 2)
    with pytest.raises(RuntimeError):
        require_placement(q[2:], out[1:-1], q, out, 2)


def test_nonfinite_equality_rejected():
    x = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError):
        require_equal(x, x, "nonfinite")
