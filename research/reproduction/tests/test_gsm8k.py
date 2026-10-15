# SPDX-License-Identifier: MIT
"""Check causal variable-length decoding and numeric-answer scoring."""

import pytest
import torch

from research.reproduction.gsm8k import generate, number
from research.tinystories_depth.model import Model


def test_answer_matching_separates_format_and_numeric_errors():
    assert number("calculation 4 then #### 1,200", True) == number("1200.0")
    assert number("the answer is -2.5") == "-2.5"
    assert number("there is no number") is None
    assert number("the answer is 4", True) is None


def test_future_padding_does_not_change_greedy_continuations():
    torch.set_num_threads(2)
    model = Model("M", 2, 89104, vocab=97).eval()
    prompts = [[2, 8, 9], [4, 5, 11, 6, 3]]
    batched, _, _ = generate(model, prompts, "cpu", eos=96, max_new=64, context=80)
    for prompt, actual in zip(prompts, batched):
        single, _, _ = generate(model, [prompt], "cpu", eos=96, max_new=64, context=80)
        assert actual == single[0]


def test_generation_rejects_context_exhaustion_without_truncation():
    with pytest.raises(ValueError, match="do not truncate"):
        generate(None, [[1] * 200], "cpu", context=256)
