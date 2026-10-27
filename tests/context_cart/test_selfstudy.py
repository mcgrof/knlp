# SPDX-License-Identifier: MIT
"""Self-Study conversion follows the native message and target convention."""

from __future__ import annotations

import numpy as np
import pytest

from research.context_cart import selfstudy as S


def _row(assistant_ids, with_end: bool, k: int = 3):
    t = len(assistant_ids)
    block = {
        "token_idx": [i for i in range(t) for _ in range(k)],
        "token_id": [
            assistant_ids[i] if j == 0 else 1000 + j for i in range(t) for j in range(k)
        ],
        "logprobs": [-0.1 if j == 0 else -3.0 for _ in range(t) for j in range(k)],
        "shape": [t, k],
    }
    ids = list(assistant_ids) + ([151645] if with_end else [])
    if with_end:
        # the captured ids include the end-of-turn token, and the capture
        # holds one target row for it like any other token
        t += 1
        block = {
            "token_idx": [i for i in range(t) for _ in range(k)],
            "token_id": [
                ids[i] if j == 0 else 1000 + j for i in range(t) for j in range(k)
            ],
            "logprobs": [-0.1 if j == 0 else -3.0 for _ in range(t) for j in range(k)],
            "shape": [t, k],
        }
    block_full = block
    return {
        "messages": [
            {
                "role": "user",
                "content": "What is the dose?",
                "token_ids": [11, 12, 13],
                "top_logprobs": None,
            },
            {
                "role": "assistant",
                "content": "x",
                "token_ids": ids,
                "top_logprobs": block_full,
            },
        ],
        "metadata": {"note_id": "text_1"},
        "type": "question",
    }


def test_element_matches_native_layout_and_targets():
    row = _row([21, 22, 23, 24], with_end=False)
    e = S.element_from_row(row, 0, max_tokens=2048)
    # <|im_start|>user\n ids <|im_end|>\n <|im_start|>assistant\n ids
    user = S.MESSAGE_START["user"] + [11, 12, 13] + S.MESSAGE_END + S.MESSAGE_EXTRA_END
    assistant = S.MESSAGE_START["assistant"] + [21, 22, 23, 24]
    assert e.input_ids == user + assistant
    offset = len(user) + len(S.MESSAGE_START["assistant"])
    assert e.target_positions == [offset + i for i in range(4)]
    assert e.target_ids.shape == (4, 3) and e.target_logprobs.shape == (4, 3)
    assert e.target_ids[0].tolist() == [21, 1001, 1002]
    assert not e.truncated and e.scored_tokens == 4
    assert abs(e.teacher_mass() - (np.exp(-0.1) + 2 * np.exp(-3.0))) < 1e-6
    # the predicted token sits at the recorded position
    for p, ids in zip(e.target_positions, e.target_ids):
        assert e.input_ids[p] == ids[0]


def test_truncation_drops_targets_past_the_cut():
    row = _row([21, 22, 23, 24], with_end=False)
    e = S.element_from_row(row, 0, max_tokens=12)
    assert e.truncated and len(e.input_ids) == 12
    assert all(p < 12 for p in e.target_positions)
    # user turn is 8 tokens and the assistant header 3, so only the first
    # assistant token (position 11) survives a 12-token cut
    assert e.target_positions == [11]
    assert e.target_ids.shape[0] == 1


def test_target_rows_must_match_assistant_tokens():
    row = _row([21, 22, 23], with_end=False)
    row["messages"][1]["token_ids"] = [21, 22]
    with pytest.raises(ValueError):
        S.element_from_row(row, 0, 2048)


def test_prompt_groups_and_split_are_deterministic():
    rows = []
    for g in range(6):
        for _ in range(2):
            r = _row([30 + g], with_end=True)
            r["messages"][0]["content"] = f"question {g}"
            rows.append(r)
    groups = S.prompt_groups(rows)
    assert len(groups) == 6 and all(len(v) == 2 for v in groups.values())
    a = S.select_groups(groups, 7)
    assert a == S.select_groups(groups, 7) and a != S.select_groups(groups, 8)
    assert sorted(a) == sorted(groups)
    audit = S.overlap_audit(
        ["question 3 about dose"], ["dose question 3?", "unrelated"]
    )
    assert audit[0]["jaccard"] > audit[1]["jaccard"]
    assert S.cartridge_rows(12628) == 632 and S.cartridge_rows(100) == 256
