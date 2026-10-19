"""A model-declared token mask must not hide real numerical failures."""

import pytest
import torch

from tools.kv.native.recent_probe import valid_native_logits


def test_only_exact_declared_suppression_is_allowed():
    logits = torch.tensor([[0.2, -float("inf"), 0.4, -float("inf")]])
    before = logits.clone()
    assert valid_native_logits(logits, [1, 3])
    assert torch.equal(logits, before)
    assert not valid_native_logits(logits, [1])
    assert not valid_native_logits(logits, [])
    assert not valid_native_logits(torch.zeros(1, 4), [1, 3])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_and_positive_infinity_are_never_masked(bad):
    for index in range(4):
        logits = torch.tensor([[0.2, -float("inf"), 0.4, -float("inf")]])
        logits[0, index] = bad
        assert not valid_native_logits(logits, [1, 3])


def test_invalid_suppression_ids_fail():
    for ids in [[-1], [4], [1, 1], [1.5]]:
        with pytest.raises(ValueError):
            valid_native_logits(torch.zeros(1, 4), ids)
