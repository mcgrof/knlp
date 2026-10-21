# SPDX-License-Identifier: MIT
"""Raw page routing and GQA oracles must not inherit worker-derived answers."""

import pytest
import torch

from tools.kv.decision.production_numerics import attention, comparison, slots, within


def test_permuted_pages_mixed_suffix_and_changing_request_order():
    pages = [[9, 2], [7, 4]]
    # One decode after position 16, then three fresh-prefill positions.
    assert slots([0, 1, 4], [18, 3], pages, [17, 0, 1, 2]) == [33, 112, 113, 114]
    assert slots([0, 3, 4], [3, 18], pages[::-1], [0, 1, 2, 17]) == [112, 113, 114, 33]


@pytest.mark.parametrize("bad", ["position", "table", "alias", "offset", "page"])
def test_corrupted_raw_mapping_rejected(bad):
    offsets, lengths, pages, positions = [0, 1, 2], [18, 3], [[9, 2], [7]], [17, 2]
    if bad == "position":
        positions[0] = 16
    elif bad == "table":
        pages[0] = [9]
    elif bad == "alias":
        lengths[1] = 18
        positions[1] = 17
        pages[1] = [9, 2]
    elif bad == "offset":
        offsets[-1] = 3
    else:
        pages[0][1] = -1
    with pytest.raises(ValueError):
        slots(offsets, lengths, pages, positions)


def test_explicit_gqa_attention_matches_independent_head_loop():
    torch.manual_seed(17)
    query = torch.randn(8, 16)
    keys = torch.randn(11, 2, 16)
    values = torch.randn(11, 2, 16)
    actual = attention(query, keys, values, softmax_scale=0.25)
    expected = torch.stack(
        [
            ((keys[:, h // 4].double() @ query[h].double()) * 0.25).softmax(0)
            @ values[:, h // 4].double()
            for h in range(8)
        ]
    )
    assert torch.allclose(actual.double(), expected, rtol=2e-6, atol=2e-7)
    row = comparison(actual, expected)
    assert within(
        row,
        {
            "relative_l2": 2e-6,
            "normalized_maximum_error": 2e-6,
            "maximum_absolute_error": 2e-7,
        },
    )
    wrong = actual.flip(0)
    assert not within(
        comparison(wrong, expected),
        {
            "relative_l2": 0.02,
            "normalized_maximum_error": 0.05,
            "maximum_absolute_error": 0.001,
        },
    )


def test_near_zero_absolute_rule_and_nonfinite_failure():
    row = comparison(torch.tensor([1e-5]), torch.tensor([0.0]))
    assert within(
        row,
        {
            "relative_l2": 0.02,
            "normalized_maximum_error": 0.05,
            "maximum_absolute_error": 0.001,
        },
    )
    with pytest.raises(ValueError):
        comparison(torch.tensor([float("nan")]), torch.zeros(1))
