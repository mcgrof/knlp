# SPDX-License-Identifier: GPL-2.0
"""The report that compares two scoring paths, on hand-made inputs."""

from __future__ import annotations

import pytest


def test_the_parity_report_counts_moved_answers_and_refuses_mismatched_inputs():
    from research.kv_translate.published import parity

    a = {(1, 2): ([2, 4], [-3.0, -5.0]), (3,): ([1, 1], [-1.0, -1.5])}
    same = parity.compare(a, a)
    assert same["largest_difference_nats"] == 0.0
    assert same["answers_changed"] == 0
    # the first context changes its raw answer but not its per-token answer
    b = {(1, 2): ([2, 4], [-5.2, -5.0]), (3,): ([1, 1], [-1.0, -1.5])}
    moved = parity.compare(a, b)
    assert moved["contexts"] == 2
    assert moved["largest_difference_nats"] == pytest.approx(2.2)
    assert moved["answers_changed"] == 1
    assert moved["answers_changed_length_normalised"] == 0
    with pytest.raises(ValueError, match="same contexts"):
        parity.compare(a, {(1, 2): a[(1, 2)]})
